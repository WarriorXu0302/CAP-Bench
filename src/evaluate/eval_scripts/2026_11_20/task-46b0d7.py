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
TASK_ID = "task-46b0d7"
TASK_DESCRIPTION = 'I’m planning to go to Miami this weekend for an EDM party, and I need you to arrange a complete itinerary. I’ll be departing from New York and want to find an electronic music event on Saturday night (prioritize the nearest upcoming weekend within the next two weeks; if there is no suitable event that weekend, check the following weekend and state that in the output).\n\nAfter finding the event, book a flight that arrives in Miami at least 4 hours before the event starts (if it’s a connecting flight, the layover must be at least 2 hours). Finally, find an Airbnb near the venue, with a walking time of no more than 15 minutes to the venue.\n\nPlease provide:\n- Event name\n- Venue name and address\n- Event start time\n- Eventbrite link\n- Flight number\n- Departure and arrival times\n- If connecting, include layover city and layover duration\n- Time difference between flight arrival and event start\n- Airbnb listing link\n- Airbnb address\n- Walking distance and walking time from Airbnb to venue\n- Google Maps walking route link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from the answer"""
    event_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    event_start_time: Optional[str] = None
    eventbrite_link: Optional[str] = None


class FlightInfo(BaseModel):
    """Flight details extracted from the answer"""
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    is_connecting: Optional[bool] = None
    layover_city: Optional[str] = None
    layover_duration: Optional[str] = None
    time_difference_to_event: Optional[str] = None


class AirbnbInfo(BaseModel):
    """Airbnb details extracted from the answer"""
    airbnb_link: Optional[str] = None
    airbnb_address: Optional[str] = None
    walking_distance: Optional[str] = None
    walking_time: Optional[str] = None
    google_maps_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the event details from the answer for the Miami EDM party:

Return:
- event_name: the name of the event exactly as written
- venue_name: the venue name exactly as written
- venue_address: the full venue address exactly as written
- event_start_time: the event start time exactly as written
- eventbrite_link: the Eventbrite event link exactly as written

If any field is missing in the answer, set it to null.
"""


def prompt_extract_flight_from_answer() -> str:
    return """
Extract the flight details from the answer for the New York to Miami flight:

Return:
- flight_number: the flight number exactly as written
- departure_time: the departure time exactly as written
- arrival_time: the arrival time in Miami exactly as written
- is_connecting: true if it's a connecting flight, false if direct, null if not specified
- layover_city: the layover city if it's a connecting flight, null otherwise
- layover_duration: the layover duration exactly as written if it's a connecting flight, null otherwise
- time_difference_to_event: the time difference between flight arrival and event start exactly as written

If any field is missing in the answer, set it to null.
"""


def prompt_extract_airbnb_from_answer() -> str:
    return """
Extract the Airbnb details from the answer:

Return:
- airbnb_link: the Airbnb listing link exactly as written
- airbnb_address: the Airbnb address exactly as written
- walking_distance: the walking distance from Airbnb to venue exactly as written
- walking_time: the walking time from Airbnb to venue exactly as written
- google_maps_link: the Google Maps walking route link exactly as written

If any field is missing in the answer, set it to null.
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


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_eventbrite_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return looks_like_url(text) and ci_contains(text, 'eventbrite')


def looks_like_airbnb_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return looks_like_url(text) and ci_contains(text, 'airbnb')


def looks_like_google_maps_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return looks_like_url(text) and (ci_contains(text, 'google.com/maps') or ci_contains(text, 'maps.google'))


def contains_miami(text: Optional[str]) -> bool:
    return has_any_ci(text, ['miami', '迈阿密'])


def contains_edm_keywords(text: Optional[str]) -> bool:
    return has_any_ci(text, ['edm', 'electronic', 'techno', 'house', 'trance', 'dubstep', 'electro'])


def looks_like_saturday(text: Optional[str]) -> bool:
    return has_any_ci(text, ['saturday', '周六', 'sat'])


def looks_like_evening_time(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'\b([6-9]|1[0-9]|2[0-3])\s*:?\s*\d{2}.*pm\b',
        r'\b(18|19|20|21|22|23)\s*:?\s*\d{2}\b',
        r'\b[6-9]\s*pm\b',
        r'\b1[0-2]\s*pm\b'
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def extract_duration_hours(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    hour_patterns = [
        r'(\d+\.?\d*)\s*hour',
        r'(\d+\.?\d*)\s*hr',
        r'(\d+)\s*小时'
    ]
    for pattern in hour_patterns:
        m = re.search(pattern, text.lower())
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return None


def extract_walking_time_minutes(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    patterns = [
        r'(\d+\.?\d*)\s*min',
        r'(\d+\.?\d*)\s*分钟'
    ]
    for pattern in patterns:
        m = re.search(pattern, text.lower())
        if m:
            try:
                return float(m.group(1))
            except:
                pass
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
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_from_answer(),
        template_class=EventInfo,
        extraction_name="event_info"
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
        desc="Eventbrite event search and selection for Miami EDM party",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location selector for Miami
    miami_location_ok = contains_miami(event_info.venue_address) or contains_miami(answer)
    evaluator.add_custom_node(
        result=bool(miami_location_ok),
        id="eventbrite_action_location",
        desc="[Action Node] eventbrite.com:F1:A12 - Select Miami in location dropdown",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date selector for Saturday
    saturday_ok = looks_like_saturday(event_info.event_start_time) or has_any_ci(answer, ['saturday', '周六'])
    evaluator.add_custom_node(
        result=bool(saturday_ok),
        id="eventbrite_action_date",
        desc="[Action Node] eventbrite.com:F1:A1 - Select Saturday date in date picker",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A13 - Category navigation for Music/Nightlife
    edm_category_ok = contains_edm_keywords(event_info.event_name) or has_any_ci(answer, ['music', 'nightlife'])
    evaluator.add_custom_node(
        result=bool(edm_category_ok),
        id="eventbrite_action_category",
        desc="[Action Node] eventbrite.com:F1:A13 - Select Music or Nightlife category in navigation menu",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A2 - Event format filter for Party/Festival
    party_format_ok = has_any_ci(event_info.event_name, ['party', 'festival', 'fest', 'rave']) or has_any_ci(answer, ['party', 'festival'])
    evaluator.add_custom_node(
        result=bool(party_format_ok),
        id="eventbrite_action_format",
        desc="[Action Node] eventbrite.com:F2:A2 - Select Party or Festival in format filter",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - Click event card to view details
    event_details_ok = bool(event_info.event_name and event_info.venue_name and event_info.eventbrite_link)
    evaluator.add_custom_node(
        result=bool(event_details_ok),
        id="eventbrite_action_click_card",
        desc="[Action Node] eventbrite.com:F3:A8 - Click event card to enter detail page",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Identify available events (not sold out)
    not_sold_out_ok = not has_any_ci(answer, ['sold out'])
    evaluator.add_custom_node(
        result=bool(not_sold_out_ok),
        id="eventbrite_perception_available",
        desc="[Perception Node] eventbrite.com:F1:P1 - Verify selected event is not marked as Sold Out",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - Understand event list information (time, type matching)
    evening_time_ok = looks_like_evening_time(event_info.event_start_time)
    evaluator.add_custom_node(
        result=bool(evening_time_ok and saturday_ok),
        id="eventbrite_perception_list_info",
        desc="[Perception Node] eventbrite.com:F1:P3 - Extract and verify event is Saturday evening (after 6 PM)",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P6 - Extract visual features indicating EDM style
    edm_style_ok = contains_edm_keywords(event_info.event_name)
    evaluator.add_custom_node(
        result=bool(edm_style_ok),
        id="eventbrite_perception_visual",
        desc="[Perception Node] eventbrite.com:F3:P6 - Verify event matches electronic music style from name/description",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P8 - Extract venue information from Location block
    venue_complete_ok = bool(event_info.venue_name and event_info.venue_address)
    evaluator.add_custom_node(
        result=bool(venue_complete_ok),
        id="eventbrite_perception_venue",
        desc="[Perception Node] eventbrite.com:F3:P8 - Extract complete venue name and address from Location section",
        parent=eventbrite_node,
        critical=False
    )

    # Verify Eventbrite link format
    eventbrite_link_ok = looks_like_eventbrite_url(event_info.eventbrite_link)
    evaluator.add_custom_node(
        result=bool(eventbrite_link_ok),
        id="eventbrite_link_valid",
        desc="Verify Eventbrite link is a valid URL",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Google Flights section
    flights_node = evaluator.add_sequential(
        id="flights_section",
        desc="Google Flights search and selection for NYC to Miami",
        parent=root,
        critical=False
    )

    # Basic flight information present
    flight_basic_ok = bool(flight_info.flight_number and flight_info.departure_time and flight_info.arrival_time)
    evaluator.add_custom_node(
        result=bool(flight_basic_ok),
        id="flights_basic_info",
        desc="Flight number, departure time, and arrival time are provided",
        parent=flights_node,
        critical=False
    )

    # Flight route mentions NYC/New York to Miami
    route_ok = has_any_ci(answer, ['new york', 'nyc', 'jfk', 'lga', 'ewr']) and contains_miami(answer)
    evaluator.add_custom_node(
        result=bool(route_ok),
        id="flights_route",
        desc="Flight route is from New York area to Miami",
        parent=flights_node,
        critical=False
    )

    # Time buffer verification (at least 4 hours before event)
    time_buffer_hours = extract_duration_hours(flight_info.time_difference_to_event)
    time_buffer_ok = time_buffer_hours is not None and time_buffer_hours >= 4.0
    evaluator.add_custom_node(
        result=bool(time_buffer_ok),
        id="flights_time_buffer",
        desc="Flight arrives at least 4 hours before event start time",
        parent=flights_node,
        critical=False
    )

    # If connecting flight, verify layover duration >= 2 hours
    if flight_info.is_connecting:
        layover_hours = extract_duration_hours(flight_info.layover_duration)
        layover_ok = layover_hours is not None and layover_hours >= 2.0
        layover_city_ok = bool(flight_info.layover_city)
        evaluator.add_custom_node(
            result=bool(layover_ok and layover_city_ok),
            id="flights_layover_requirements",
            desc="If connecting flight, layover is at least 2 hours and layover city is specified",
            parent=flights_node,
            critical=False
        )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb search for accommodation near venue",
        parent=root,
        critical=False
    )

    # Airbnb link validity
    airbnb_link_ok = looks_like_airbnb_url(airbnb_info.airbnb_link)
    evaluator.add_custom_node(
        result=bool(airbnb_link_ok),
        id="airbnb_link_valid",
        desc="Airbnb listing link is a valid URL",
        parent=airbnb_node,
        critical=False
    )

    # Airbnb address provided
    airbnb_address_ok = bool(airbnb_info.airbnb_address)
    evaluator.add_custom_node(
        result=bool(airbnb_address_ok),
        id="airbnb_address",
        desc="Airbnb address is provided",
        parent=airbnb_node,
        critical=False
    )

    # Walking time verification (<= 15 minutes)
    walking_minutes = extract_walking_time_minutes(airbnb_info.walking_time)
    walking_time_ok = walking_minutes is not None and walking_minutes <= 15.0
    evaluator.add_custom_node(
        result=bool(walking_time_ok),
        id="airbnb_walking_time",
        desc="Walking time from Airbnb to venue is 15 minutes or less",
        parent=airbnb_node,
        critical=False
    )

    # Walking distance provided
    walking_distance_ok = bool(airbnb_info.walking_distance and contains_digits(airbnb_info.walking_distance))
    evaluator.add_custom_node(
        result=bool(walking_distance_ok),
        id="airbnb_walking_distance",
        desc="Walking distance from Airbnb to venue is provided",
        parent=airbnb_node,
        critical=False
    )

    # Google Maps link validity
    maps_link_ok = looks_like_google_maps_url(airbnb_info.google_maps_link)
    evaluator.add_custom_node(
        result=bool(maps_link_ok),
        id="airbnb_maps_link",
        desc="Google Maps walking route link is a valid URL",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Overall completeness
    all_required_fields_ok = (
        event_details_ok and
        flight_basic_ok and
        airbnb_link_ok and
        airbnb_address_ok and
        walking_time_ok
    )
    evaluator.add_custom_node(
        result=bool(all_required_fields_ok),
        id="overall_completeness",
        desc="All required information fields are provided",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
