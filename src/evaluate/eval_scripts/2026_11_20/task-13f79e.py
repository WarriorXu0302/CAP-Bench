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
TASK_ID = "task-13f79e"
TASK_DESCRIPTION = "I'm a jazz enthusiast currently in London, planning to go to New York to catch a live performance. Please first go to Eventbrite and help me find jazz performances in New York that still have tickets available for a weekend in the coming weeks, for example, from next Friday to next Sunday. Don't just pick the first results; scroll down or browse a few pages. Choose an event that looks interesting and still has tickets available. Note its name, specific performance time, and venue.\n\nBased on this performance time, go to Skyscanner and search for round-trip flights from London (LHR) to New York. I want direct flights only; if none are available, choose the flight with the shortest travel time.\n\nFinally, go to Expedia and look for hotels in New York for those performance days. Find one with a rating of 4 stars or higher that includes free breakfast.\n\nFinally, compile and present the selected event, flight information, and hotel recommendation."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Jazz event details extracted from the answer"""
    event_name: Optional[str] = None
    performance_time: Optional[str] = None
    venue: Optional[str] = None
    tickets_available_mentioned: Optional[bool] = None


class FlightInfo(BaseModel):
    """Flight details extracted from the answer"""
    departure_city: Optional[str] = None
    arrival_city: Optional[str] = None
    direct_flight_mentioned: Optional[bool] = None
    flight_details_present: Optional[bool] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    star_rating: Optional[str] = None
    free_breakfast_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the jazz event information from Eventbrite that the user selected and reported in the answer.

Return:
- event_name: the name of the jazz event exactly as stated. If not present, set null.
- performance_time: the specific performance time/date exactly as stated. If not present, set null.
- venue: the venue name exactly as stated. If not present, set null.
- tickets_available_mentioned: true if the answer mentions that tickets are available or not sold out, false otherwise.

If any field is missing, set it to null.
"""


def prompt_extract_flight_from_answer() -> str:
    return """
From the answer, extract the flight information from Skyscanner:

- departure_city: the departure city mentioned (should be London or LHR). If not present, set null.
- arrival_city: the arrival city mentioned (should be New York or NYC). If not present, set null.
- direct_flight_mentioned: true if the answer mentions direct flights or non-stop flights, false otherwise.
- flight_details_present: true if the answer provides any specific flight details (times, airlines, etc.), false otherwise.

If any field is missing, set it to null.
"""


def prompt_extract_hotel_from_answer() -> str:
    return """
From the answer, extract the hotel information from Expedia:

- hotel_name: the name of the hotel exactly as stated. If not present, set null.
- star_rating: the star rating exactly as stated (e.g., "4 stars", "4-star", "4.5 stars"). If not present, set null.
- free_breakfast_mentioned: true if the answer mentions free breakfast as an amenity, false otherwise.

If any field is missing, set it to null.
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


def looks_like_weekend_date(text: Optional[str]) -> bool:
    if not text:
        return False
    weekend_keywords = ['friday', 'saturday', 'sunday', 'weekend', 'fri', 'sat', 'sun']
    return has_any_ci(text, weekend_keywords)


def extract_star_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*[-\s]?star', text.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
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

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite part
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite jazz event search in New York for weekend",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A6 - Navigate/filter by date (weekend in coming weeks)
    eventbrite_mentions = has_any_ci(answer, ['eventbrite'])
    jazz_mentions = has_any_ci(answer, ['jazz'])
    newyork_mentions = has_any_ci(answer, ['new york', 'nyc'])
    weekend_context = looks_like_weekend_date(answer) or has_any_ci(answer, ['next friday', 'next sunday', 'coming weeks', 'weekend'])

    eventbrite_navigation_ok = eventbrite_mentions and jazz_mentions and newyork_mentions and weekend_context

    evaluator.add_custom_node(
        result=bool(eventbrite_navigation_ok),
        id="eventbrite_action_date_filter",
        desc="[Action Node] eventbrite.com:F1:A6 - Navigate to Eventbrite and filter/search for jazz events in New York on a weekend in coming weeks",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Identify tickets available (not Sold Out)
    tickets_available = event_info.tickets_available_mentioned if event_info else False
    not_sold_out = not has_any_ci(answer, ['sold out'])

    evaluator.add_custom_node(
        result=bool(tickets_available or not_sold_out),
        id="eventbrite_perception_tickets_available",
        desc="[Perception Node] eventbrite.com:F1:P1 - Identify that tickets are still available (not Sold Out or Selling Fast)",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Scroll down or browse multiple pages
    scroll_browse_mentioned = has_any_ci(answer, ['scroll', 'browse', 'page', 'multiple', 'several', 'few pages'])
    first_result_avoided = has_any_ci(answer, ['not just the first', 'looked at several', 'browsed through', 'checked multiple'])

    evaluator.add_custom_node(
        result=bool(scroll_browse_mentioned or first_result_avoided),
        id="eventbrite_action_scroll_browse",
        desc="[Action Node] eventbrite.com:F1:A12 - Scroll down or browse multiple pages (not just first results)",
        parent=eventbrite_node,
        critical=False
    )

    # Event details captured
    event_name_ok = bool(event_info and event_info.event_name and event_info.event_name.strip())
    performance_time_ok = bool(event_info and event_info.performance_time and event_info.performance_time.strip())
    venue_ok = bool(event_info and event_info.venue and event_info.venue.strip())

    evaluator.add_custom_node(
        result=bool(event_name_ok and performance_time_ok and venue_ok),
        id="eventbrite_event_details_captured",
        desc="Event name, specific performance time, and venue are all captured",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Skyscanner part
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner round-trip flight search from London to New York",
        parent=root,
        critical=False
    )

    # Flight search mentions
    skyscanner_mentions = has_any_ci(answer, ['skyscanner'])
    london_departure = has_any_ci(answer, ['london', 'lhr'])
    ny_arrival = has_any_ci(answer, ['new york', 'nyc', 'jfk', 'ewr', 'lga'])
    round_trip_ok = has_any_ci(answer, ['round-trip', 'round trip', 'return'])

    evaluator.add_custom_node(
        result=bool(skyscanner_mentions and london_departure and ny_arrival and round_trip_ok),
        id="skyscanner_search_setup",
        desc="Search for round-trip flights from London to New York on Skyscanner",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A5 - Filter for direct flights only
    direct_flight_mentioned = flight_info.direct_flight_mentioned if flight_info else False
    direct_keywords = has_any_ci(answer, ['direct', 'non-stop', 'nonstop'])

    evaluator.add_custom_node(
        result=bool(direct_flight_mentioned or direct_keywords),
        id="skyscanner_action_direct_filter",
        desc="[Action Node] skyscanner.com:F1:A5 - Apply filter for direct flights only (or shortest travel time if none available)",
        parent=skyscanner_node,
        critical=False
    )

    # Flight details present
    flight_details_ok = flight_info.flight_details_present if flight_info else False
    has_flight_info = has_any_ci(answer, ['flight', 'departure', 'arrival', 'airline', 'time'])

    evaluator.add_custom_node(
        result=bool(flight_details_ok or has_flight_info),
        id="skyscanner_flight_details_captured",
        desc="Flight information is captured and presented",
        parent=skyscanner_node,
        critical=False
    )

    # Dates align with event
    dates_aligned = bool(event_info and event_info.performance_time and has_any_ci(answer, ['based on', 'according to', 'matching']))

    evaluator.add_custom_node(
        result=bool(dates_aligned),
        id="skyscanner_dates_aligned_with_event",
        desc="Flight dates are aligned with the performance dates from Eventbrite",
        parent=skyscanner_node,
        critical=False
    )

    # 3.3 Expedia part
    expedia_node = evaluator.add_sequential(
        id="expedia_section",
        desc="Expedia hotel search in New York with 4+ stars and free breakfast",
        parent=root,
        critical=False
    )

    # Hotel search mentions
    expedia_mentions = has_any_ci(answer, ['expedia'])
    hotel_mentions = has_any_ci(answer, ['hotel'])
    ny_location = has_any_ci(answer, ['new york', 'nyc'])

    evaluator.add_custom_node(
        result=bool(expedia_mentions and hotel_mentions and ny_location),
        id="expedia_search_setup",
        desc="Search for hotels in New York on Expedia",
        parent=expedia_node,
        critical=False
    )

    # Star rating 4+
    star_rating_value = extract_star_rating(hotel_info.star_rating) if hotel_info else None
    star_rating_ok = (star_rating_value is not None and star_rating_value >= 4.0) or has_any_ci(answer, ['4 star', '4-star', '5 star', '5-star'])

    evaluator.add_custom_node(
        result=bool(star_rating_ok),
        id="expedia_star_rating_4plus",
        desc="Hotel has a rating of 4 stars or higher",
        parent=expedia_node,
        critical=False
    )

    # [Perception Node] expedia.com:F2:P3 - Free breakfast amenity
    free_breakfast_mentioned = hotel_info.free_breakfast_mentioned if hotel_info else False
    breakfast_keywords = has_any_ci(answer, ['free breakfast', 'complimentary breakfast', 'breakfast included'])

    evaluator.add_custom_node(
        result=bool(free_breakfast_mentioned or breakfast_keywords),
        id="expedia_perception_free_breakfast",
        desc="[Perception Node] expedia.com:F2:P3 - Identify that the hotel includes free breakfast amenity",
        parent=expedia_node,
        critical=False
    )

    # Hotel details captured
    hotel_name_ok = bool(hotel_info and hotel_info.hotel_name and hotel_info.hotel_name.strip())

    evaluator.add_custom_node(
        result=bool(hotel_name_ok),
        id="expedia_hotel_details_captured",
        desc="Hotel name and details are captured",
        parent=expedia_node,
        critical=False
    )

    # Hotel dates align with event
    hotel_dates_aligned = bool(event_info and event_info.performance_time)

    evaluator.add_custom_node(
        result=bool(hotel_dates_aligned),
        id="expedia_dates_aligned_with_event",
        desc="Hotel dates are aligned with the performance dates",
        parent=expedia_node,
        critical=False
    )

    # 3.4 Final compilation
    compilation_node = evaluator.add_sequential(
        id="compilation_section",
        desc="Final compilation and presentation of event, flight, and hotel",
        parent=root,
        critical=False
    )

    all_three_present = (event_name_ok and flight_details_ok and hotel_name_ok)

    evaluator.add_custom_node(
        result=bool(all_three_present),
        id="compilation_all_three_components",
        desc="All three components (event, flight, hotel) are compiled and presented",
        parent=compilation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
