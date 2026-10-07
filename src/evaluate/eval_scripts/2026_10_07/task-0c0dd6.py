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
TASK_ID = "task-0c0dd6"
TASK_DESCRIPTION = 'I’m finalizing my itinerary for the next CES (Consumer Electronics Show). First, please check Wikipedia to confirm the exact dates, host city, and main venue name for the next CES. After obtaining the dates and city, search Eventbrite for “Tech Networking” or “After Party” events taking place in the same city during that period. Please identify 3 events that meet the following criteria: priced at $50 or less (or free), and held in the evening during CES dates (after 18:00). If fewer than 3 results are available during CES, you may extend the search to one day before and one day after CES, and clearly note this in the output. Once identified, use Google Maps to calculate the driving distance and estimated travel time from the CES main venue to each of the three event locations. Output the CES start and end dates, host city, and main venue name; and for each of the 3 events: name, exact time, price, Eventbrite link, and driving distance and time from the main venue.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CESInfo(BaseModel):
    """CES dates, host city, and main venue extracted from the answer"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    host_city: Optional[str] = None
    main_venue: Optional[str] = None


class EventInfo(BaseModel):
    """Single event details extracted from the answer"""
    name: Optional[str] = None
    time: Optional[str] = None
    price: Optional[str] = None
    eventbrite_link: Optional[str] = None
    driving_distance: Optional[str] = None
    driving_time: Optional[str] = None


class EventsCollection(BaseModel):
    """Collection of events extracted from the answer"""
    events: List[EventInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ces_info() -> str:
    return """
Extract the CES (Consumer Electronics Show) information from the answer:

- start_date: the start date of CES as stated in the answer
- end_date: the end date of CES as stated in the answer
- host_city: the host city name
- main_venue: the main venue name

If any field is missing, set it to null.
"""


def prompt_extract_events() -> str:
    return """
Extract all the events from the answer. For each event, extract:

- name: the event name
- time: the exact time of the event
- price: the price (may be free or a dollar amount)
- eventbrite_link: the Eventbrite URL/link for the event
- driving_distance: the driving distance from the CES main venue
- driving_time: the estimated driving time from the CES main venue

Return a list of events under the "events" key. If no events are found, return an empty list.
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


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various formats: "January 7, 2026", "2026-01-07", "Jan 7", etc.
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        r'\d{1,2}'
    ]
    return any(re.search(p, text.lower()) for p in date_patterns)


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept free, $0, $X, or numeric values
    if ci_contains(text, 'free'):
        return True
    return bool(re.search(r'\$?\d+', text))


def extract_price_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    if ci_contains(text, 'free'):
        return 0.0
    m = re.search(r'\$?(\d+(?:\.\d+)?)', text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def looks_like_evening_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept times after 18:00 (6 PM)
    # Patterns: "7 PM", "19:00", "8:30 PM", etc.
    time_patterns = [
        r'\b(1[8-9]|2[0-3]):[0-5]\d\b',  # 18:00-23:59
        r'\b(6|7|8|9|10|11)\s*(pm|p\.m\.)\b',  # 6 PM - 11 PM
        r'\b(1[8-9]|2[0-3])\b'  # 18-23
    ]
    return any(re.search(p, text.lower()) for p in time_patterns)


def looks_like_eventbrite_link(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'eventbrite')


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various distance formats: "5 miles", "8 km", "3.2 mi", etc.
    return bool(re.search(r'\d+(\.\d+)?\s*(mi|km|mile|kilometer)', text.lower()))


def looks_like_time_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept time durations: "15 min", "20 minutes", "1 hour", etc.
    return bool(re.search(r'\d+\s*(min|minute|hour|hr)', text.lower()))


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
    ces_info = await evaluator.extract(
        prompt=prompt_extract_ces_info(),
        template_class=CESInfo,
        extraction_name="ces_info"
    )

    events_collection = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsCollection,
        extraction_name="events_collection"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Wikipedia section - CES information lookup
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia lookup for CES dates, city, and main venue",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Search and locate CES entry
    wikipedia_action_ok = (
        has_any_ci(answer, ['wikipedia', 'wiki']) and
        has_any_ci(answer, ['ces', 'consumer electronics show'])
    )
    evaluator.add_custom_node(
        result=bool(wikipedia_action_ok),
        id="wikipedia_action_search",
        desc="[Action Node] wikipedia.org:F1:A20 - Search and locate the CES entry on Wikipedia",
        parent=wikipedia_node,
        critical=False
    )

    # Verify extracted CES information
    dates_ok = (
        looks_like_date(ces_info.start_date) and
        looks_like_date(ces_info.end_date)
    )
    city_ok = bool(ces_info.host_city and ces_info.host_city.strip())
    venue_ok = bool(ces_info.main_venue and ces_info.main_venue.strip())

    evaluator.add_custom_node(
        result=bool(dates_ok),
        id="wikipedia_ces_dates",
        desc="Extracted valid CES start and end dates",
        parent=wikipedia_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(city_ok),
        id="wikipedia_ces_city",
        desc="Extracted CES host city",
        parent=wikipedia_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(venue_ok),
        id="wikipedia_ces_venue",
        desc="Extracted CES main venue name",
        parent=wikipedia_node,
        critical=False
    )

    # 3.2 Eventbrite section - Event search and filtering
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite search for Tech Networking or After Party events",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date filtering
    eventbrite_date_action_ok = (
        has_any_ci(answer, ['eventbrite']) and
        (dates_ok or has_any_ci(answer, ['ces', 'date', 'during']))
    )
    evaluator.add_custom_node(
        result=bool(eventbrite_date_action_ok),
        id="eventbrite_action_date_filter",
        desc="[Action Node] eventbrite.com:F1:A1 - Apply date filtering to search for events during CES period",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A3 - Price filtering
    price_filtering_mentioned = has_any_ci(answer, ['$50', 'free', 'price'])
    evaluator.add_custom_node(
        result=bool(price_filtering_mentioned),
        id="eventbrite_action_price_filter",
        desc="[Action Node] eventbrite.com:F2:A3 - Apply price filtering ($50 or less, or free)",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A10 - Category expansion for Tech/Party filtering
    category_mentioned = has_any_ci(answer, ['tech', 'networking', 'after party', 'party'])
    evaluator.add_custom_node(
        result=bool(category_mentioned),
        id="eventbrite_action_category",
        desc="[Action Node] eventbrite.com:F2:A10 - Search or filter for Tech Networking or After Party categories",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Avoid sold out events
    events = events_collection.events if events_collection and events_collection.events else []
    has_three_events = len(events) >= 3
    sold_out_mentioned = not has_any_ci(answer, ['sold out'])
    evaluator.add_custom_node(
        result=bool(sold_out_mentioned),
        id="eventbrite_perception_avoid_sold_out",
        desc="[Perception Node] eventbrite.com:F1:P1 - Avoid selecting sold out events (status awareness)",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - Extract evening times (after 18:00)
    evening_times_ok = sum(1 for e in events if looks_like_evening_time(e.time)) >= min(3, len(events))
    evaluator.add_custom_node(
        result=bool(evening_times_ok),
        id="eventbrite_perception_evening_times",
        desc="[Perception Node] eventbrite.com:F1:P3 - Identify events held in the evening (after 18:00)",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - Navigate to event detail pages
    detail_page_mentioned = has_any_ci(answer, ['detail', 'event page', 'more info'])
    evaluator.add_custom_node(
        result=bool(detail_page_mentioned or len(events) >= 3),
        id="eventbrite_action_detail_pages",
        desc="[Action Node] eventbrite.com:F3:A8 - Navigate to event detail pages to extract specific times",
        parent=eventbrite_node,
        critical=False
    )

    # Verify extracted event information
    evaluator.add_custom_node(
        result=bool(has_three_events),
        id="eventbrite_three_events",
        desc="Found and extracted 3 events meeting the criteria",
        parent=eventbrite_node,
        critical=False
    )

    # Verify price criteria for extracted events
    prices_valid = sum(1 for e in events if looks_like_price(e.price)) >= min(3, len(events))
    prices_within_budget = sum(
        1 for e in events
        if extract_price_value(e.price) is not None and extract_price_value(e.price) <= 50
    ) >= min(3, len(events))

    evaluator.add_custom_node(
        result=bool(prices_valid and prices_within_budget),
        id="eventbrite_price_criteria",
        desc="Events meet the price criteria ($50 or less, or free)",
        parent=eventbrite_node,
        critical=False
    )

    # Verify Eventbrite links
    links_valid = sum(1 for e in events if looks_like_eventbrite_link(e.eventbrite_link)) >= min(3, len(events))
    evaluator.add_custom_node(
        result=bool(links_valid),
        id="eventbrite_links_present",
        desc="Eventbrite links provided for events",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Google Maps section - Route planning and distance calculation
    gmaps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning from CES main venue to event locations",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A5 - Input origin and destination for route
    gmaps_route_action_ok = (
        has_any_ci(answer, ['google maps', 'maps']) and
        has_any_ci(answer, ['distance', 'route', 'driving'])
    )
    evaluator.add_custom_node(
        result=bool(gmaps_route_action_ok),
        id="gmaps_action_route_input",
        desc="[Action Node] google.com/maps:F2:A5 - Input CES main venue as origin and event locations as destinations",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A7 - Select driving mode
    driving_mode_mentioned = has_any_ci(answer, ['driv', 'car'])
    evaluator.add_custom_node(
        result=bool(driving_mode_mentioned),
        id="gmaps_action_driving_mode",
        desc="[Action Node] google.com/maps:F2:A7 - Select driving transportation mode",
        parent=gmaps_node,
        critical=False
    )

    # Verify extracted distance and time information
    distances_valid = sum(1 for e in events if looks_like_distance(e.driving_distance)) >= min(3, len(events))
    times_valid = sum(1 for e in events if looks_like_time_duration(e.driving_time)) >= min(3, len(events))

    evaluator.add_custom_node(
        result=bool(distances_valid),
        id="gmaps_distances_extracted",
        desc="Driving distances extracted for events from CES main venue",
        parent=gmaps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(times_valid),
        id="gmaps_times_extracted",
        desc="Driving times extracted for events from CES main venue",
        parent=gmaps_node,
        critical=False
    )

    # Optional: Check for extension note if fewer than 3 events during CES dates
    extension_note_ok = (
        len(events) < 3 or
        has_any_ci(answer, ['extend', 'one day before', 'one day after', 'outside ces'])
    )
    evaluator.add_custom_node(
        result=bool(extension_note_ok),
        id="extension_note_if_needed",
        desc="Notes search extension to adjacent days if fewer than 3 events found during CES",
        parent=eventbrite_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
