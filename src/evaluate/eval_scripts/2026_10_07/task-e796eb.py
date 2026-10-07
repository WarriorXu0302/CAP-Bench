import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-e796eb"
TASK_DESCRIPTION = 'I plan to attend a music event in Miami from **next Friday to next Sunday**.\n\nFirst, search Eventbrite for "Music" events in Miami, FL, occurring during this period. Select an event with a clearly specified venue address.\n\nNext, use Google Flights to search for round-trip flights from New York (JFK) to Miami (MIA), departing on **next Thursday** and returning on **next Monday**. Requirements: Filter for "Nonstop" flights and select the lowest-priced option where the outbound flight\'s arrival time is before 2:00 PM.\n\nFinally, on Airbnb, search for accommodation for the period from **next Friday to next Sunday**. Filter criteria: "Entire home", "Kitchen" included, and priced under $400/night. Select 3 candidate listings. For each, use Google Maps to calculate the walking time from its approximate location to the event venue.\n\nOutput: Event Name, Venue Address, Event Link; Flight Airline, Price, Outbound Arrival Time, Flight Link; Titles, Prices, Walking Times, and Airbnb Links for the 3 accommodation listings.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventbriteInfo(BaseModel):
    """Event information extracted from the answer"""
    event_name: Optional[str] = None
    venue_address: Optional[str] = None
    event_link: Optional[str] = None


class FlightInfo(BaseModel):
    """Flight information extracted from the answer"""
    airline: Optional[str] = None
    price: Optional[str] = None
    outbound_arrival_time: Optional[str] = None
    flight_link: Optional[str] = None
    nonstop_mentioned: Optional[bool] = None


class AirbnbListing(BaseModel):
    """Single Airbnb listing"""
    title: Optional[str] = None
    price: Optional[str] = None
    walking_time: Optional[str] = None
    link: Optional[str] = None


class AirbnbInfo(BaseModel):
    """Airbnb information extracted from the answer"""
    listings: Optional[List[AirbnbListing]] = Field(default_factory=list)
    entire_home_mentioned: Optional[bool] = None
    kitchen_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eventbrite_from_answer() -> str:
    return """
Extract the Eventbrite event details from the answer:

- event_name: the name of the selected music event
- venue_address: the physical venue address
- event_link: the Eventbrite event URL

If any field is missing, set it to null.
"""


def prompt_extract_flight_from_answer() -> str:
    return """
Extract the Google Flights information from the answer:

- airline: the airline name for the selected flight
- price: the flight price exactly as stated
- outbound_arrival_time: the arrival time of the outbound flight
- flight_link: the Google Flights URL or link
- nonstop_mentioned: true if the answer mentions "nonstop" or "direct" flight, false otherwise

If any field is missing, set it to null.
"""


def prompt_extract_airbnb_from_answer() -> str:
    return """
Extract the Airbnb accommodation details from the answer:

- listings: a list of up to 3 listings, each containing:
  - title: the listing title
  - price: the price per night
  - walking_time: the walking time to the event venue
  - link: the Airbnb listing URL
- entire_home_mentioned: true if the answer mentions filtering for "entire home" or similar
- kitchen_mentioned: true if the answer mentions filtering for "kitchen" amenity

If any field is missing, set it to null or empty list.
"""


# --------------------------------------------------------------------------- #
# Helper functions for lenient checks                                         #
# --------------------------------------------------------------------------- #
def ci_contains(text: Optional[str], substr: str) -> bool:
    if not text:
        return False
    return substr.lower() in text.lower()


def has_any_ci(text: Optional[str], substrs: List[str]) -> bool:
    return any(ci_contains(text, s) for s in substrs)


def contains_digits(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\d', text))


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for street indicators and state/zip patterns
    return (contains_digits(text) and
            (has_any_ci(text, ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 'drive', 'dr', 'miami', 'fl']) or
             re.search(r'\d{5}', text) is not None))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://') or text.startswith('www.')


def parse_time_before_2pm(time_text: Optional[str]) -> bool:
    """Check if a time string represents a time before 2:00 PM (14:00)"""
    if not time_text:
        return False

    # Try various time formats
    patterns = [
        r'(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)',
        r'(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)',
        r'(\d{1,2}):(\d{2})'
    ]

    for pattern in patterns:
        match = re.search(pattern, time_text.lower())
        if match:
            try:
                if len(match.groups()) >= 3:
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    period = match.group(3).lower()

                    if 'p' in period and hour != 12:
                        hour += 12
                    elif 'a' in period and hour == 12:
                        hour = 0

                    return hour < 14 or (hour == 14 and minute == 0)
                elif len(match.groups()) >= 2:
                    hour = int(match.group(1))
                    period = match.group(2).lower()

                    if 'p' in period and hour != 12:
                        hour += 12
                    elif 'a' in period and hour == 12:
                        hour = 0

                    return hour < 14
                else:
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if len(match.groups()) > 1 else 0
                    return hour < 14 or (hour == 14 and minute == 0)
            except:
                continue

    return False


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\$€£]\s*\d+|(\d+)\s*(usd|dollars|dollar)', text.lower()))


def extract_price_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r'[\$€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            return None
    return None


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()

    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract information from the answer --------------------- #
    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_from_answer(),
        template_class=EventbriteInfo,
        extraction_name="eventbrite_info"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_from_answer(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_from_answer(),
        template_class=AirbnbInfo,
        extraction_name="airbnb_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music event search and venue address extraction",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - List information extraction
    event_name_ok = bool(eventbrite_info.event_name and eventbrite_info.event_name.strip())
    venue_address_ok = looks_like_address(eventbrite_info.venue_address)
    event_link_ok = looks_like_url(eventbrite_info.event_link)

    evaluator.add_custom_node(
        result=bool(event_name_ok and venue_address_ok),
        id="eventbrite_extract_event_venue",
        desc="[Action Node] eventbrite.com:F3:A8 - Extract event with clearly specified venue address",
        parent=eventbrite_node,
        critical=False
    )

    # Check Eventbrite search context
    eventbrite_search_ok = (has_any_ci(answer, ['eventbrite']) and
                           has_any_ci(answer, ['music', 'miami']))
    evaluator.add_custom_node(
        result=bool(eventbrite_search_ok),
        id="eventbrite_search_context",
        desc="Searched Eventbrite for Music events in Miami",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights round-trip search with filters",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - Nonstop filter
    nonstop_ok = (flight_info.nonstop_mentioned or
                 has_any_ci(answer, ['nonstop', 'non-stop', 'direct flight']))
    evaluator.add_custom_node(
        result=bool(nonstop_ok),
        id="flights_filter_nonstop",
        desc="[Action Node] google.comflights:F3:A5 - Filter for Nonstop flights",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A7 - Time slider for arrival before 2 PM
    arrival_time_ok = parse_time_before_2pm(flight_info.outbound_arrival_time)
    evaluator.add_custom_node(
        result=bool(arrival_time_ok),
        id="flights_filter_arrival_time",
        desc="[Action Node] google.comflights:F3:A7 - Filter outbound arrival time before 2:00 PM",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F1:P1 - Select lowest price
    price_ok = looks_like_price(flight_info.price)
    lowest_price_context = has_any_ci(answer, ['lowest', 'cheapest', 'lowest price', 'best price'])
    evaluator.add_custom_node(
        result=bool(price_ok and lowest_price_context),
        id="flights_select_lowest_price",
        desc="[Perception Node] google.comflights:F1:P1 - Select lowest-priced flight option",
        parent=flights_node,
        critical=False
    )

    # Check flight route context
    flight_route_ok = (has_any_ci(answer, ['jfk', 'new york']) and
                      has_any_ci(answer, ['mia', 'miami']))
    evaluator.add_custom_node(
        result=bool(flight_route_ok),
        id="flights_route_context",
        desc="Flight route JFK to MIA mentioned",
        parent=flights_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search with filters",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A12 - Entire home filter
    entire_home_ok = (airbnb_info.entire_home_mentioned or
                     has_any_ci(answer, ['entire home', 'entire place']))
    evaluator.add_custom_node(
        result=bool(entire_home_ok),
        id="airbnb_filter_entire_home",
        desc="[Action Node] airbnb.com:F2:A12 - Filter for Entire home",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Kitchen amenity filter
    kitchen_ok = (airbnb_info.kitchen_mentioned or
                 has_any_ci(answer, ['kitchen']))
    evaluator.add_custom_node(
        result=bool(kitchen_ok),
        id="airbnb_filter_kitchen",
        desc="[Action Node] airbnb.com:F2:A11 - Filter for Kitchen amenity",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price slider under $400/night
    listings = airbnb_info.listings if airbnb_info.listings else []
    prices_under_400 = []
    for listing in listings:
        price_val = extract_price_value(listing.price)
        if price_val is not None:
            prices_under_400.append(price_val < 400)

    price_filter_ok = (len(prices_under_400) > 0 and all(prices_under_400)) or has_any_ci(answer, ['under $400', 'below $400', 'less than $400'])
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="airbnb_filter_price",
        desc="[Action Node] airbnb.com:F2:A13 - Filter for price under $400/night",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P10 - Select 3 listings
    three_listings_ok = (len(listings) == 3 and
                        all(listing.title and listing.price and listing.link for listing in listings))
    evaluator.add_custom_node(
        result=bool(three_listings_ok),
        id="airbnb_select_three_listings",
        desc="[Perception Node] airbnb.com:F1:P10 - Select 3 candidate listings",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A38 - Google Maps walking time calculation
    walking_times_ok = (len(listings) > 0 and
                       all(listing.walking_time and contains_digits(listing.walking_time) for listing in listings))
    maps_mention = has_any_ci(answer, ['google maps', 'walking time', 'walking distance'])
    evaluator.add_custom_node(
        result=bool(walking_times_ok and maps_mention),
        id="airbnb_walking_time_calculation",
        desc="[Action Node] airbnb.com:F4:A38 - Calculate walking time from listings to event venue using Google Maps",
        parent=airbnb_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
