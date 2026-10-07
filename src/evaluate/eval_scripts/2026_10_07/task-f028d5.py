import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-f028d5"
TASK_DESCRIPTION = 'I’m planning to attend an expo at ExCeL London next month. Please help me arrange the itinerary:\n\nFirst, search on Google Flights for round-trip tickets from New York (JFK) to London (LHR). Choose an outbound flight on a Tuesday in the middle of next month, and a return flight on Sunday of the same week. The flight must be nonstop, and the outbound flight must arrive in London before 11:00 AM so I can attend the event in the afternoon. Filter for eligible Star Alliance flights. If there are no nonstop options on that date, switch to adjacent dates within the same week; if there are still no nonstop options, select the option with the shortest layover and clearly note it.\n\nThen go to Booking.com to find hotels matching the above itinerary (check-in on the outbound date, check-out on the return date). Filters: 4 stars or above, rating 8.5 or higher, and within 1 mile of ExCeL London.\n\nNext, use Google Maps to verify: calculate the actual walking distance from the selected hotel to ExCeL London to ensure it is indeed within 1 mile; and plan a public transit route from the hotel to the Victoria and Albert Museum on Saturday of the same week.\n\nFinally, check the V&A Museum’s official website to confirm the opening hours for that Saturday.\n\nPlease output:\n1. Flight airline, flight number, departure/arrival times, and Google Flights link;\n2. Hotel name, star rating, review score, and Booking link;\n3. Google Maps-verified walking distance from hotel to venue, public transit duration to the museum, and route link;\n4. V&A Museum Saturday opening hours.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    """Flight details extracted from the answer"""
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    outbound_departure_time: Optional[str] = None
    outbound_arrival_time: Optional[str] = None
    return_departure_time: Optional[str] = None
    return_arrival_time: Optional[str] = None
    is_nonstop: Optional[bool] = None
    google_flights_link: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    star_rating: Optional[str] = None
    review_score: Optional[str] = None
    booking_link: Optional[str] = None


class MapsInfo(BaseModel):
    """Google Maps route details extracted from the answer"""
    walking_distance_to_excel: Optional[str] = None
    transit_duration_to_vanda: Optional[str] = None
    maps_link: Optional[str] = None


class MuseumInfo(BaseModel):
    """V&A Museum opening hours extracted from the answer"""
    saturday_opening_hours: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight_info() -> str:
    return """
Extract the flight details from the answer for the Google Flights booking (JFK to LHR):

- airline: the airline name
- flight_number: the flight number
- outbound_departure_time: outbound departure time
- outbound_arrival_time: outbound arrival time (should be before 11:00 AM London time)
- return_departure_time: return departure time
- return_arrival_time: return arrival time
- is_nonstop: whether the flight is nonstop (true/false)
- google_flights_link: the Google Flights URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_hotel_info() -> str:
    return """
Extract the hotel details from the answer for the Booking.com reservation:

- hotel_name: the name of the hotel
- star_rating: the star rating (should be 4 or above)
- review_score: the review score (should be 8.5 or higher)
- booking_link: the Booking.com URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_maps_info() -> str:
    return """
Extract the Google Maps route details from the answer:

- walking_distance_to_excel: the walking distance from hotel to ExCeL London (should be within 1 mile)
- transit_duration_to_vanda: the public transit duration from hotel to V&A Museum
- maps_link: the Google Maps URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_museum_info() -> str:
    return """
Extract the V&A Museum opening hours for Saturday from the answer:

- saturday_opening_hours: the opening hours for Saturday

If missing, set it to null.
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


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Match time patterns like "10:30 AM", "10:30", "1030", etc.
    patterns = [r'\d{1,2}:\d{2}', r'\d{1,2}\s*(?:AM|PM|am|pm)', r'\d{3,4}']
    return any(re.search(p, text) for p in patterns)


def arrival_before_11am(time_text: Optional[str]) -> bool:
    if not time_text:
        return False
    # Extract hour from various time formats
    match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?', time_text)
    if not match:
        return False
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3).upper() if match.group(3) else None

    # Convert to 24-hour format if meridiem is present
    if meridiem == 'PM' and hour != 12:
        hour += 12
    elif meridiem == 'AM' and hour == 12:
        hour = 0

    # Check if before 11:00 AM (11:00 in 24-hour format)
    return hour < 11 or (hour == 11 and minute == 0)


def is_star_alliance_airline(airline: Optional[str]) -> bool:
    if not airline:
        return False
    # Common Star Alliance airlines
    star_alliance_airlines = [
        'united', 'lufthansa', 'air canada', 'ana', 'singapore airlines',
        'turkish airlines', 'swiss', 'austrian', 'brussels airlines',
        'scandinavian', 'sas', 'lot polish', 'tap air portugal', 'avianca',
        'copa airlines', 'ethiopian', 'egyptair', 'south african airways',
        'thai airways', 'air china', 'air india', 'air new zealand', 'asiana'
    ]
    return has_any_ci(airline, star_alliance_airlines)


def check_star_rating(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    return num is not None and num >= 4.0


def check_review_score(score_text: Optional[str]) -> bool:
    if not score_text:
        return False
    num = extract_float(score_text)
    return num is not None and num >= 8.5


def check_distance_within_mile(distance_text: Optional[str]) -> bool:
    if not distance_text:
        return False
    num = extract_float(distance_text)
    if num is None:
        return False
    # Check if it's in miles and within 1 mile
    if has_any_ci(distance_text, ['mile', 'mi']):
        return num <= 1.0
    # If in feet, convert to miles (5280 feet = 1 mile)
    elif has_any_ci(distance_text, ['feet', 'ft']):
        return num <= 5280
    # If in kilometers, convert to miles (1.609 km = 1 mile)
    elif has_any_ci(distance_text, ['km', 'kilometer']):
        return num <= 1.609
    return False


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
    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_info(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    maps_info = await evaluator.extract(
        prompt=prompt_extract_maps_info(),
        template_class=MapsInfo,
        extraction_name="maps_info"
    )

    museum_info = await evaluator.extract(
        prompt=prompt_extract_museum_info(),
        template_class=MuseumInfo,
        extraction_name="museum_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights booking for JFK to LHR",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F3:A7 - Time range filtering (arrival before 11:00 AM)
    arrival_time_ok = arrival_before_11am(flight_info.outbound_arrival_time)
    evaluator.add_custom_node(
        result=bool(arrival_time_ok),
        id="flights_action_time_filter",
        desc="[Action Node] google.comflights:F3:A7 - Filter to ensure outbound arrival before 11:00 AM",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A6 - Star Alliance filter
    is_star_alliance = is_star_alliance_airline(flight_info.airline)
    evaluator.add_custom_node(
        result=bool(is_star_alliance),
        id="flights_action_alliance_filter",
        desc="[Action Node] google.comflights:F3:A6 - Filter for Star Alliance flights",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F1:P1 - Extract departure/arrival times
    times_extracted = (looks_like_time(flight_info.outbound_departure_time) and
                      looks_like_time(flight_info.outbound_arrival_time))
    evaluator.add_custom_node(
        result=bool(times_extracted),
        id="flights_perception_times",
        desc="[Perception Node] google.comflights:F1:P1 - Extract departure and arrival times",
        parent=flights_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    has_flight_details = bool(flight_info.airline and flight_info.flight_number)
    evaluator.add_custom_node(
        result=bool(has_flight_details),
        id="flights_has_details",
        desc="Provides airline and flight number",
        parent=flights_node,
        critical=False
    )

    has_google_flights_link = bool(flight_info.google_flights_link and
                                   has_any_ci(flight_info.google_flights_link, ['google.com/flights', 'flights.google']))
    evaluator.add_custom_node(
        result=bool(has_google_flights_link),
        id="flights_has_link",
        desc="Provides Google Flights link",
        parent=flights_node,
        critical=False
    )

    # 3.2 Booking.com section
    booking_node = evaluator.add_sequential(
        id="booking_com_section",
        desc="Booking.com hotel search near ExCeL London",
        parent=root,
        critical=False
    )

    # [Action Node] booking.com:F3:A12 - Multi-dimension filtering (4+ stars, 8.5+ rating)
    star_ok = check_star_rating(hotel_info.star_rating)
    score_ok = check_review_score(hotel_info.review_score)
    filters_applied = star_ok and score_ok
    evaluator.add_custom_node(
        result=bool(filters_applied),
        id="booking_action_filters",
        desc="[Action Node] booking.com:F3:A12 - Filter for 4+ stars and 8.5+ review score",
        parent=booking_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    has_hotel_name = bool(hotel_info.hotel_name and hotel_info.hotel_name.strip())
    evaluator.add_custom_node(
        result=bool(has_hotel_name),
        id="booking_has_name",
        desc="Provides hotel name",
        parent=booking_node,
        critical=False
    )

    has_booking_link = bool(hotel_info.booking_link and
                           has_any_ci(hotel_info.booking_link, ['booking.com']))
    evaluator.add_custom_node(
        result=bool(has_booking_link),
        id="booking_has_link",
        desc="Provides Booking.com link",
        parent=booking_node,
        critical=False
    )

    mentions_excel = has_any_ci(answer, ['excel london', 'excel centre'])
    evaluator.add_custom_node(
        result=bool(mentions_excel),
        id="booking_mentions_excel",
        desc="Mentions proximity to ExCeL London",
        parent=booking_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning and verification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input (walking distance to ExCeL)
    distance_ok = check_distance_within_mile(maps_info.walking_distance_to_excel)
    evaluator.add_custom_node(
        result=bool(distance_ok),
        id="maps_action_walking_route",
        desc="[Action Node] maps.google.com:F2:A5 - Calculate walking distance to ExCeL London (within 1 mile)",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode switching (public transit to V&A)
    has_transit_duration = bool(maps_info.transit_duration_to_vanda and
                               contains_digits(maps_info.transit_duration_to_vanda))
    evaluator.add_custom_node(
        result=bool(has_transit_duration),
        id="maps_action_transit_route",
        desc="[Action Node] maps.google.com:F2:A7 - Plan public transit route to V&A Museum",
        parent=maps_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    has_maps_link = bool(maps_info.maps_link and
                        has_any_ci(maps_info.maps_link, ['google.com/maps', 'maps.google']))
    evaluator.add_custom_node(
        result=bool(has_maps_link),
        id="maps_has_link",
        desc="Provides Google Maps link",
        parent=maps_node,
        critical=False
    )

    mentions_vanda = has_any_ci(answer, ['victoria and albert', 'v&a', 'v & a'])
    evaluator.add_custom_node(
        result=bool(mentions_vanda),
        id="maps_mentions_vanda",
        desc="Mentions Victoria and Albert Museum routing",
        parent=maps_node,
        critical=False
    )

    # 3.4 V&A Museum section
    museum_node = evaluator.add_sequential(
        id="vanda_museum_section",
        desc="V&A Museum opening hours verification",
        parent=root,
        critical=False
    )

    # [Perception Node] vam.ac.uk:F1:P1 - Extract Saturday opening hours
    has_saturday_hours = bool(museum_info.saturday_opening_hours and
                             museum_info.saturday_opening_hours.strip())
    evaluator.add_custom_node(
        result=bool(has_saturday_hours),
        id="museum_perception_hours",
        desc="[Perception Node] vam.ac.uk:F1:P1 - Extract Saturday opening hours from V&A website",
        parent=museum_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    mentions_saturday = has_any_ci(answer, ['saturday'])
    evaluator.add_custom_node(
        result=bool(mentions_saturday),
        id="museum_mentions_saturday",
        desc="Mentions Saturday in the context of museum hours",
        parent=museum_node,
        critical=False
    )

    mentions_vam_website = has_any_ci(answer, ['vam.ac.uk', 'v&a website', 'v&a official', 'museum website'])
    evaluator.add_custom_node(
        result=bool(mentions_vam_website),
        id="museum_mentions_website",
        desc="Mentions checking the V&A official website",
        parent=museum_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
