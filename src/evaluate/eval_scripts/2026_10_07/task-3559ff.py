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
TASK_ID = "task-3559ff"
TASK_DESCRIPTION = 'I’m planning to attend some art-related activities in Los Angeles this coming weekend and want an itinerary that feels full but not rushed. First, check the Getty website for exhibitions or events happening this weekend (lectures, workshops, guided tours, etc.), choose one that looks interesting, and record it. If there are no suitable options this weekend, then find the nearest upcoming Getty event within the next two weeks and note the reason for this adjustment.\n\nThen go to Eventbrite and search for art-related events in Los Angeles for this Saturday and Sunday. Under the Art, Museums, or Culture categories, find 2–3 well-rated events. If the platform does not provide reliable rating data, use visible indicators such as registration popularity, follower/interest counts, or comments as evidence of being “well-rated,” and specify the basis in the output.\n\nArrange the Getty event and the Eventbrite events in chronological order, and use Google Maps to calculate driving time between each pair of consecutive event locations.\n\nHelp me filter for one feasible combination with these constraints:\n- There must be sufficient time between every two events:  \n  previous event end time + travel time + at least 30 minutes buffer ≤ next event start time.\n- Keep 3–4 events in the final plan with no time conflicts.\n- If fewer than 3 events can satisfy the constraints, keep all feasible events and explain why the count is insufficient.\n\nFor each event, output:\n- Event name\n- Start time\n- End time (or duration)\n- Venue address\n- Brief description\n- Event page link\n\nFor each adjacent event pair, output:\n- Google Maps route link\n- Driving distance\n- Estimated travel time'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GettyEvent(BaseModel):
    """Getty event information extracted from the answer"""
    event_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    venue_address: Optional[str] = None
    description: Optional[str] = None
    event_link: Optional[str] = None


class EventbriteEvents(BaseModel):
    """Eventbrite events extracted from the answer"""
    events_count: Optional[int] = None
    events_list: Optional[List[str]] = Field(default_factory=list)
    rating_basis_mentioned: Optional[bool] = None


class ItineraryInfo(BaseModel):
    """Itinerary arrangement and travel information"""
    total_events_in_plan: Optional[int] = None
    chronological_order_maintained: Optional[bool] = None
    has_route_links: Optional[bool] = None
    has_travel_times: Optional[bool] = None
    has_distances: Optional[bool] = None


class TimeConstraints(BaseModel):
    """Time constraint validation information"""
    constraints_validated: Optional[bool] = None
    buffer_time_mentioned: Optional[bool] = None
    explanation_provided: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_getty_event() -> str:
    return """
Extract the Getty event information from the answer:

Return:
- event_name: the name of the Getty event exactly as written
- start_time: the start time of the event
- end_time: the end time or duration
- venue_address: the venue address
- description: brief description of the event
- event_link: the event page URL

If any field is missing, set it to null.
"""


def prompt_extract_eventbrite_events() -> str:
    return """
Extract Eventbrite events information from the answer:

Return:
- events_count: the number of Eventbrite events found (should be 2-3)
- events_list: a list of event names
- rating_basis_mentioned: whether the answer explains the basis for considering events "well-rated" (popularity, followers, etc.)

If any field is missing, set it to null or empty list.
"""


def prompt_extract_itinerary_info() -> str:
    return """
Extract itinerary arrangement information from the answer:

Return:
- total_events_in_plan: total number of events in the final plan (should be 3-4)
- chronological_order_maintained: whether events are arranged in chronological order
- has_route_links: whether Google Maps route links are provided
- has_travel_times: whether estimated travel times are provided
- has_distances: whether driving distances are provided

If any field cannot be determined, set it to null.
"""


def prompt_extract_time_constraints() -> str:
    return """
Extract time constraint validation information from the answer:

Return:
- constraints_validated: whether the answer validates time constraints between events
- buffer_time_mentioned: whether 30-minute buffer time is mentioned or considered
- explanation_provided: if fewer than 3 events, whether an explanation is provided

If any field cannot be determined, set it to null.
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


def contains_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://[^\s]+', text))


def looks_like_getty_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'getty.edu') or ci_contains(text, 'getty')


def looks_like_maps_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'google.com/maps') or ci_contains(text, 'maps.google')


def mentions_weekend(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['weekend', 'saturday', 'sunday'])


def mentions_eventbrite(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'eventbrite')


def mentions_los_angeles(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['los angeles', 'la'])


def extract_event_count(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    matches = re.findall(r'\b(\d+)\s*event', text.lower())
    if matches:
        try:
            return int(matches[0])
        except Exception:
            pass
    return None


def has_time_format(text: Optional[str]) -> bool:
    if not text:
        return False
    time_patterns = [
        r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)',
        r'\d{1,2}\s*(?:am|pm|AM|PM)',
        r'\d{1,2}:\d{2}'
    ]
    return any(re.search(p, text) for p in time_patterns)


def has_address_info(text: Optional[str]) -> bool:
    if not text:
        return False
    address_indicators = ['street', 'st', 'avenue', 'ave', 'blvd', 'drive', 'dr', 'road', 'rd', 'los angeles', 'ca']
    return has_any_ci(text, address_indicators)


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
    getty_event = await evaluator.extract(
        prompt=prompt_extract_getty_event(),
        template_class=GettyEvent,
        extraction_name="getty_event_info"
    )

    eventbrite_events = await evaluator.extract(
        prompt=prompt_extract_eventbrite_events(),
        template_class=EventbriteEvents,
        extraction_name="eventbrite_events_info"
    )

    itinerary_info = await evaluator.extract(
        prompt=prompt_extract_itinerary_info(),
        template_class=ItineraryInfo,
        extraction_name="itinerary_arrangement"
    )

    time_constraints = await evaluator.extract(
        prompt=prompt_extract_time_constraints(),
        template_class=TimeConstraints,
        extraction_name="time_constraints_validation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Getty website section
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty website event search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F2:A1 - Tab switching to view weekend events
    getty_weekend_mentioned = mentions_weekend(answer) and has_any_ci(answer, ['getty'])
    evaluator.add_custom_node(
        result=bool(getty_weekend_mentioned),
        id="getty_tab_switching",
        desc="[Action Node] getty.edu:F2:A1 - Navigate between Current/Future tabs to check weekend exhibitions and events",
        parent=getty_node,
        critical=False
    )

    # [Action Node] getty.edu:F2:A7 - Click card to view details
    getty_has_link = bool(getty_event and getty_event.event_link and looks_like_getty_url(getty_event.event_link))
    evaluator.add_custom_node(
        result=bool(getty_has_link),
        id="getty_click_details",
        desc="[Action Node] getty.edu:F2:A7 - Click event card to enter detail page and obtain complete information",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F2:P5 - Time awareness for weekend filtering
    getty_has_time = bool(getty_event and getty_event.start_time and has_time_format(getty_event.start_time))
    evaluator.add_custom_node(
        result=bool(getty_has_time and mentions_weekend(answer)),
        id="getty_time_awareness",
        desc="[Perception Node] getty.edu:F2:P5 - Identify time markers on event cards to filter weekend activities",
        parent=getty_node,
        critical=False
    )

    # [Action Node] getty.edu:F3:A14 - Scroll to load more content
    getty_event_found = bool(getty_event and getty_event.event_name)
    evaluator.add_custom_node(
        result=bool(getty_event_found),
        id="getty_scroll_load",
        desc="[Action Node] getty.edu:F3:A14 - Scroll or click Load More to view additional event options",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F3:P9 - Calendar date marking awareness
    getty_has_complete_info = bool(
        getty_event and getty_event.event_name and
        getty_event.start_time and getty_event.venue_address
    )
    evaluator.add_custom_node(
        result=bool(getty_has_complete_info),
        id="getty_calendar_awareness",
        desc="[Perception Node] getty.edu:F3:P9 - Identify dates with event markers in the activity calendar",
        parent=getty_node,
        critical=False
    )

    # Getty event completeness (non-prefixed)
    getty_all_fields = bool(
        getty_event and getty_event.event_name and getty_event.start_time and
        getty_event.venue_address and getty_event.description and getty_event.event_link
    )
    evaluator.add_custom_node(
        result=bool(getty_all_fields),
        id="getty_complete_info",
        desc="Getty event includes all required fields: name, time, address, description, link",
        parent=getty_node,
        critical=False
    )

    # 3.2 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite art event search in Los Angeles",
        parent=root,
        critical=False
    )

    # Eventbrite search performed (non-prefixed)
    eventbrite_searched = mentions_eventbrite(answer) and mentions_los_angeles(answer)
    evaluator.add_custom_node(
        result=bool(eventbrite_searched),
        id="eventbrite_search_performed",
        desc="Searched Eventbrite for art-related events in Los Angeles",
        parent=eventbrite_node,
        critical=False
    )

    # Found 2-3 events (non-prefixed)
    events_count_ok = bool(
        eventbrite_events and eventbrite_events.events_count and
        2 <= eventbrite_events.events_count <= 3
    )
    evaluator.add_custom_node(
        result=bool(events_count_ok),
        id="eventbrite_count_check",
        desc="Found 2-3 well-rated Eventbrite events as requested",
        parent=eventbrite_node,
        critical=False
    )

    # Rating basis explained (non-prefixed)
    rating_basis_ok = bool(
        eventbrite_events and eventbrite_events.rating_basis_mentioned or
        has_any_ci(answer, ['popularity', 'followers', 'interest', 'registration', 'well-rated'])
    )
    evaluator.add_custom_node(
        result=bool(rating_basis_ok),
        id="eventbrite_rating_basis",
        desc="Explains basis for considering events well-rated (popularity, followers, etc.)",
        parent=eventbrite_node,
        critical=False
    )

    # Weekend filtering (non-prefixed)
    weekend_filtering = mentions_weekend(answer) and has_any_ci(answer, ['saturday', 'sunday'])
    evaluator.add_custom_node(
        result=bool(weekend_filtering),
        id="eventbrite_weekend_filter",
        desc="Filters events for Saturday and Sunday specifically",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning and travel time calculation",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input origin and destination
    has_routes = bool(
        itinerary_info and itinerary_info.has_route_links or
        looks_like_maps_url(answer)
    )
    evaluator.add_custom_node(
        result=bool(has_routes),
        id="maps_input_locations",
        desc="[Action Node] maps.google.com:F2:A5 - Input event addresses as origin and destination for route planning",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode
    driving_mentioned = has_any_ci(answer, ['driving', 'drive', 'car']) and has_any_ci(answer, ['time', 'travel time'])
    evaluator.add_custom_node(
        result=bool(driving_mentioned),
        id="maps_driving_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode in Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search
    has_addresses = bool(
        getty_event and getty_event.venue_address and has_address_info(answer)
    )
    evaluator.add_custom_node(
        result=bool(has_addresses),
        id="maps_location_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for event addresses in the search box for positioning",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P3 - Extract distance information
    has_distances = bool(
        itinerary_info and itinerary_info.has_distances or
        has_any_ci(answer, ['mile', 'km', 'distance'])
    )
    evaluator.add_custom_node(
        result=bool(has_distances),
        id="maps_extract_distance",
        desc="[Perception Node] maps.google.com:F5:P3 - Extract distance values from route planning results",
        parent=maps_node,
        critical=False
    )

    # Travel time provided (non-prefixed)
    has_travel_times = bool(
        itinerary_info and itinerary_info.has_travel_times or
        has_any_ci(answer, ['minute', 'hour', 'travel time'])
    )
    evaluator.add_custom_node(
        result=bool(has_travel_times),
        id="maps_travel_times",
        desc="Provides estimated travel times between consecutive events",
        parent=maps_node,
        critical=False
    )

    # 3.4 Itinerary planning section
    itinerary_node = evaluator.add_sequential(
        id="itinerary_planning_section",
        desc="Itinerary arrangement and time constraint validation",
        parent=root,
        critical=False
    )

    # Chronological order (non-prefixed)
    chronological_ok = bool(
        itinerary_info and itinerary_info.chronological_order_maintained or
        has_any_ci(answer, ['chronological', 'order', 'arranged'])
    )
    evaluator.add_custom_node(
        result=bool(chronological_ok),
        id="chronological_arrangement",
        desc="Events arranged in chronological order",
        parent=itinerary_node,
        critical=False
    )

    # Event count (3-4 or explained) (non-prefixed)
    total_events = itinerary_info.total_events_in_plan if itinerary_info else None
    event_count_ok = bool(
        (total_events and 3 <= total_events <= 4) or
        (time_constraints and time_constraints.explanation_provided)
    )
    evaluator.add_custom_node(
        result=bool(event_count_ok),
        id="event_count_validation",
        desc="Final plan contains 3-4 events, or explains why fewer events are included",
        parent=itinerary_node,
        critical=False
    )

    # Time constraints validated (non-prefixed)
    constraints_ok = bool(
        time_constraints and time_constraints.constraints_validated or
        has_any_ci(answer, ['buffer', '30 minute', 'sufficient time', 'time between'])
    )
    evaluator.add_custom_node(
        result=bool(constraints_ok),
        id="time_constraints_check",
        desc="Validates time constraints: end time + travel time + 30 min buffer ≤ next start time",
        parent=itinerary_node,
        critical=False
    )

    # Buffer time mentioned (non-prefixed)
    buffer_mentioned = bool(
        time_constraints and time_constraints.buffer_time_mentioned or
        has_any_ci(answer, ['30 minute', 'buffer', 'at least 30'])
    )
    evaluator.add_custom_node(
        result=bool(buffer_mentioned),
        id="buffer_time_consideration",
        desc="Considers 30-minute buffer time between events",
        parent=itinerary_node,
        critical=False
    )

    # No time conflicts (non-prefixed)
    no_conflicts = has_any_ci(answer, ['no conflict', 'feasible', 'compatible']) or constraints_ok
    evaluator.add_custom_node(
        result=bool(no_conflicts),
        id="no_time_conflicts",
        desc="Ensures no time conflicts between events in the final plan",
        parent=itinerary_node,
        critical=False
    )

    # 3.5 Output completeness section
    output_node = evaluator.add_parallel(
        id="output_completeness_section",
        desc="Completeness of output information for all events",
        parent=root,
        critical=False
    )

    # Event details completeness (non-prefixed)
    has_event_names = has_any_ci(answer, ['event', 'name'])
    has_times = has_time_format(answer)
    has_addresses = has_address_info(answer)
    has_descriptions = has_any_ci(answer, ['description', 'about'])
    has_links = contains_url(answer)

    evaluator.add_custom_node(
        result=bool(has_event_names and has_times and has_addresses),
        id="event_basic_info",
        desc="Provides event name, start/end time, and venue address for each event",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_descriptions),
        id="event_descriptions",
        desc="Provides brief descriptions for events",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_links),
        id="event_links",
        desc="Provides event page links",
        parent=output_node,
        critical=False
    )

    # Route information completeness (non-prefixed)
    evaluator.add_custom_node(
        result=bool(has_routes and has_distances and has_travel_times),
        id="route_info_complete",
        desc="Provides Google Maps route links, distances, and travel times for adjacent event pairs",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
