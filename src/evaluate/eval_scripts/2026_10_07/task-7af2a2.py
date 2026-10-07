import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-7af2a2"
TASK_DESCRIPTION = 'I’m a die-hard Coldplay fan based in New York (JFK), and I want to travel to see their upcoming tour dates in Europe.\n\nFirst, check Spotify for Coldplay’s tour schedule in Europe and select 3 different cities. If there are no available European dates on the current page, then find the 3 nearest upcoming tour cities worldwide (not limited to Europe) and clearly note this in the results.\n\nNext, for each of those 3 cities, use Google Flights to search round-trip flights that align with the concert timing (arrive 1 day before, depart 1 day after), prioritizing nonstop options, and identify the city with the lowest airfare. If a city has no nonstop flights, you may choose the option with the shortest layover and note that explicitly.\n\nFinally, for the winning city, use Airbnb to find accommodation for the concert period. Requirements: it must be labeled **“Guest Favorite”**, priced at **$300/night or less**, and verified in **map view** to be very close to the concert venue (walkable or a short drive).\n\nOutput: a flight price comparison across the 3 cities, plus the final selected Airbnb listing name, nightly price, estimated total trip cost (**flight + 2 nights of lodging**), and listing URL.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourCities(BaseModel):
    """Tour cities extracted from the answer"""
    city1: Optional[str] = None
    city1_date: Optional[str] = None
    city2: Optional[str] = None
    city2_date: Optional[str] = None
    city3: Optional[str] = None
    city3_date: Optional[str] = None
    europe_note: Optional[str] = None


class FlightPrices(BaseModel):
    """Flight prices extracted from the answer"""
    city1_price: Optional[str] = None
    city2_price: Optional[str] = None
    city3_price: Optional[str] = None
    lowest_city: Optional[str] = None
    nonstop_note: Optional[str] = None


class AirbnbListing(BaseModel):
    """Airbnb listing details extracted from the answer"""
    listing_name: Optional[str] = None
    nightly_price: Optional[str] = None
    is_guest_favorite: Optional[bool] = None
    total_trip_cost: Optional[str] = None
    listing_url: Optional[str] = None
    venue_proximity_note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_cities() -> str:
    return """
Extract the 3 Coldplay tour cities and their dates from the answer that the user found on Spotify.

Return:
- city1, city1_date: first city name and concert date
- city2, city2_date: second city name and concert date
- city3, city3_date: third city name and concert date
- europe_note: any note about whether these are European cities or worldwide cities (if no European dates were available)

If any field is missing, set it to null.
"""


def prompt_extract_flight_prices() -> str:
    return """
Extract the flight price comparison from the answer for the 3 cities.

Return:
- city1_price: flight price for the first city (include currency if present)
- city2_price: flight price for the second city (include currency if present)
- city3_price: flight price for the third city (include currency if present)
- lowest_city: which city had the lowest airfare
- nonstop_note: any note about nonstop vs. layover flights

If any field is missing, set it to null.
"""


def prompt_extract_airbnb_listing() -> str:
    return """
Extract the final Airbnb listing details from the answer for the winning city.

Return:
- listing_name: the name of the Airbnb listing
- nightly_price: the price per night (include currency/units if present)
- is_guest_favorite: true if the listing is mentioned as "Guest Favorite", false otherwise
- total_trip_cost: the estimated total trip cost (flight + 2 nights of lodging)
- listing_url: the URL of the listing
- venue_proximity_note: any note about the listing's proximity to the concert venue

If any field is missing, set it to null (or false for is_guest_favorite).
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
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    has_num = contains_digits(text)
    has_currency = has_any_ci(text, ['$', 'usd', 'dollar', '€', 'eur', '£', 'gbp'])
    return has_num and has_currency


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


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
    tour_info = await evaluator.extract(
        prompt=prompt_extract_tour_cities(),
        template_class=TourCities,
        extraction_name="tour_cities"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_prices(),
        template_class=FlightPrices,
        extraction_name="flight_prices"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_listing(),
        template_class=AirbnbListing,
        extraction_name="airbnb_listing"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify tour schedule for Coldplay",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F9:A22 - Navigate to Coldplay tour schedule
    spotify_action_ok = has_any_ci(answer, ['spotify']) and has_any_ci(answer, ['coldplay', 'tour'])
    evaluator.add_custom_node(
        result=bool(spotify_action_ok),
        id="spotify_action_tour_schedule",
        desc="[Action Node] open.spotify.com:F9:A22 - Navigate to Spotify and browse Coldplay's tour schedule",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F9:P14 - Extract 3 cities from tour list
    cities_extracted = all([tour_info.city1, tour_info.city2, tour_info.city3])
    dates_extracted = all([tour_info.city1_date, tour_info.city2_date, tour_info.city3_date])
    evaluator.add_custom_node(
        result=bool(cities_extracted and dates_extracted),
        id="spotify_perception_cities",
        desc="[Perception Node] open.spotify.com:F9:P14 - Extract 3 different cities and their concert dates from the tour list",
        parent=spotify_node,
        critical=False
    )

    # Optional: Check if Europe note is mentioned when no European dates available
    europe_context_ok = has_any_ci(answer, ['europe', 'european']) or (tour_info.europe_note is not None)
    evaluator.add_custom_node(
        result=bool(europe_context_ok),
        id="spotify_europe_context",
        desc="Mentions Europe context or notes about worldwide cities if no European dates available",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights search for round-trip flights to 3 cities",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F1:A1 - Input JFK and target cities in autocomplete
    flights_action_ok = (has_any_ci(answer, ['google flights', 'flights']) and
                         has_any_ci(answer, ['jfk', 'new york']) and
                         cities_extracted)
    evaluator.add_custom_node(
        result=bool(flights_action_ok),
        id="google_flights_action_search",
        desc="[Action Node] google.comflights:F1:A1 - Input JFK as departure and target cities as destinations in Google Flights autocomplete",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F1:P1 - Extract and compare flight prices
    price1_ok = looks_like_price(flight_info.city1_price)
    price2_ok = looks_like_price(flight_info.city2_price)
    price3_ok = looks_like_price(flight_info.city3_price)
    lowest_identified = flight_info.lowest_city is not None and len(flight_info.lowest_city.strip()) > 0
    evaluator.add_custom_node(
        result=bool(price1_ok and price2_ok and price3_ok and lowest_identified),
        id="google_flights_perception_prices",
        desc="[Perception Node] google.comflights:F1:P1 - Extract flight prices for all 3 cities and identify the lowest airfare city",
        parent=flights_node,
        critical=False
    )

    # Optional: Check timing alignment (arrive 1 day before, depart 1 day after)
    timing_context_ok = has_any_ci(answer, ['1 day before', 'day before', '1 day after', 'day after', 'concert timing'])
    evaluator.add_custom_node(
        result=bool(timing_context_ok),
        id="google_flights_timing_alignment",
        desc="Mentions flight timing alignment with concert (arrive 1 day before, depart 1 day after)",
        parent=flights_node,
        critical=False
    )

    # Optional: Check nonstop preference or layover note
    nonstop_context_ok = has_any_ci(answer, ['nonstop', 'non-stop', 'direct', 'layover'])
    evaluator.add_custom_node(
        result=bool(nonstop_context_ok),
        id="google_flights_nonstop_context",
        desc="Mentions nonstop flight preference or layover information",
        parent=flights_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search for the winning city",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A12 - Filter by Guest Favorite
    guest_favorite_ok = airbnb_info.is_guest_favorite == True
    evaluator.add_custom_node(
        result=bool(guest_favorite_ok),
        id="airbnb_action_guest_favorite",
        desc="[Action Node] airbnb.com:F2:A12 - Filter listings to show only 'Guest Favorite' properties",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Set price slider to $300/night or less
    nightly_price_num = extract_float(airbnb_info.nightly_price)
    price_within_budget = nightly_price_num is not None and nightly_price_num <= 300
    evaluator.add_custom_node(
        result=bool(price_within_budget),
        id="airbnb_action_price_slider",
        desc="[Action Node] airbnb.com:F2:A13 - Set price range slider to $300/night or less",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A38 - Use map view to search near concert venue
    map_context_ok = has_any_ci(answer, ['map', 'map view', 'venue', 'stadium', 'concert venue'])
    evaluator.add_custom_node(
        result=bool(map_context_ok),
        id="airbnb_action_map_search",
        desc="[Action Node] airbnb.com:F4:A38 - Use map view to search for listings near the concert venue",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F4:P16 - Verify listing proximity to venue
    proximity_verified = airbnb_info.venue_proximity_note is not None or has_any_ci(answer, ['close', 'near', 'walkable', 'short drive', 'proximity', 'distance'])
    evaluator.add_custom_node(
        result=bool(proximity_verified),
        id="airbnb_perception_proximity",
        desc="[Perception Node] airbnb.com:F4:P16 - Verify the selected listing is very close to the concert venue (walkable or short drive)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A43 - Interact with map markers to view listing details
    listing_name_ok = airbnb_info.listing_name is not None and len(airbnb_info.listing_name.strip()) > 0
    listing_url_ok = is_valid_url(airbnb_info.listing_url)
    evaluator.add_custom_node(
        result=bool(listing_name_ok and listing_url_ok),
        id="airbnb_action_map_marker_interaction",
        desc="[Action Node] airbnb.com:F4:A43 - Click or hover on map markers to view listing details and obtain listing name and URL",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Output completeness section
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required outputs are present",
        parent=root,
        critical=False
    )

    # Flight price comparison for 3 cities
    evaluator.add_custom_node(
        result=bool(price1_ok and price2_ok and price3_ok),
        id="output_flight_comparison",
        desc="Output includes flight price comparison across all 3 cities",
        parent=output_node,
        critical=False
    )

    # Final listing details
    evaluator.add_custom_node(
        result=bool(listing_name_ok),
        id="output_listing_name",
        desc="Output includes the final Airbnb listing name",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(looks_like_price(airbnb_info.nightly_price)),
        id="output_nightly_price",
        desc="Output includes the nightly price",
        parent=output_node,
        critical=False
    )

    # Total trip cost calculation
    total_cost_ok = looks_like_price(airbnb_info.total_trip_cost) and has_any_ci(answer, ['total', 'trip cost', 'flight + 2 nights', 'lodging'])
    evaluator.add_custom_node(
        result=bool(total_cost_ok),
        id="output_total_trip_cost",
        desc="Output includes estimated total trip cost (flight + 2 nights of lodging)",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(listing_url_ok),
        id="output_listing_url",
        desc="Output includes the Airbnb listing URL",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
