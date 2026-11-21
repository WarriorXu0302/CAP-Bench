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
TASK_ID = "task-7d27af"
TASK_DESCRIPTION = 'This Saturday I’ll be in Chicago and want to plan a perfect date night. First, on Eventbrite, find an art exhibition or jazz concert near Downtown/The Loop. Prioritize options that end before 7:00 PM on Saturday, and check whether the event details mention a dress code or whether it’s suitable for couples. If the current listings do not show a clear end time, choose an event that starts earlier (e.g., before early evening) and has more complete page information, and note in the result whether the end time is clearly provided.\n\nAfter selecting the event, based on its end time (or your noted timing judgment if the end time is unclear) and exact address, go to OpenTable and find a restaurant within walking distance (or a very short drive). I’d like Italian or French cuisine, rated above 4.5, with a price range of $$ or $$$. Most importantly, check whether a table for two is available around 30 minutes after the event ends.\n\nFinally, tell me the name of the event, what time it ends (or whether the end time is unknown), and the name of the restaurant you chose, along with the most suitable reservation time.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from the answer for Eventbrite event"""
    event_name: Optional[str] = None
    event_type: Optional[str] = None
    event_location: Optional[str] = None
    event_end_time: Optional[str] = None
    end_time_clarity: Optional[str] = None
    dress_code_mentioned: Optional[bool] = None
    suitable_for_couples_mentioned: Optional[bool] = None


class RestaurantInfo(BaseModel):
    """Restaurant details extracted from the answer for OpenTable restaurant"""
    restaurant_name: Optional[str] = None
    cuisine_type: Optional[str] = None
    rating_text: Optional[str] = None
    price_range: Optional[str] = None
    reservation_time: Optional[str] = None
    table_availability_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the user's reported Eventbrite event details for Saturday in Chicago Downtown/The Loop from the answer.

Return:
- event_name: the name of the event exactly as stated. If not present, set null.
- event_type: whether it's described as an art exhibition, jazz concert, or other type. If not present, set null.
- event_location: the address or location mentioned. If not present, set null.
- event_end_time: the end time if stated (e.g., "6:30 PM", "7:00 PM"). If not present or unclear, set null.
- end_time_clarity: whether the answer mentions if the end time is clear or unknown/unclear. If not discussed, set null.
- dress_code_mentioned: true if the answer mentions checking for dress code, false otherwise.
- suitable_for_couples_mentioned: true if the answer mentions checking if suitable for couples, false otherwise.

If any field is missing in the answer, set it to null or false as appropriate.
"""


def prompt_extract_restaurant_from_answer() -> str:
    return """
From the answer, extract the OpenTable restaurant details chosen for the date night:

- restaurant_name: the name of the restaurant exactly as stated. If not present, set null.
- cuisine_type: the cuisine type (Italian, French, etc.) exactly as stated. If not present, set null.
- rating_text: the rating mentioned exactly as written (include numbers if present). If not present, set null.
- price_range: the price range mentioned ($$, $$$, etc.) exactly as written. If not present, set null.
- reservation_time: the suggested reservation time exactly as stated. If not present, set null.
- table_availability_mentioned: true if the answer mentions checking table availability for two people, false otherwise.

If any field is missing, set it to null or false as appropriate.
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


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various time formats: "6:30 PM", "7 PM", "18:30", etc.
    patterns = [
        r'\d+\s*:?\s*\d*\s*(p\.?m\.?|a\.?m\.?)',
        r'\d+:\d+',
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def extract_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_price_range(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept $$ or $$$ patterns
    return bool(re.search(r'\$\$+', text))


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
    Restrict evaluator.verify to at most one usage.
    Favor lenient, fault-tolerant checks and allow partial credit.
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

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant_from_answer(),
        template_class=RestaurantInfo,
        extraction_name="restaurant_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite part
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search for Saturday in Chicago Downtown/The Loop",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date selection for Saturday
    saturday_mentioned = has_any_ci(answer, ['saturday', 'sat'])
    date_context_ok = saturday_mentioned or has_any_ci(answer, ['this saturday', 'this weekend'])
    evaluator.add_custom_node(
        result=bool(date_context_ok),
        id="eventbrite_action_date_selection",
        desc="[Action Node] eventbrite.com:F1:A1 - Date selection operation for Saturday",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location selection for Chicago
    chicago_mentioned = has_any_ci(answer, ['chicago'])
    downtown_loop_mentioned = has_any_ci(answer, ['downtown', 'the loop', 'loop'])
    location_context_ok = chicago_mentioned and downtown_loop_mentioned
    evaluator.add_custom_node(
        result=bool(location_context_ok),
        id="eventbrite_action_location_selection",
        desc="[Action Node] eventbrite.com:F1:A12 - Location selection operation for Chicago Downtown/The Loop",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P7 - Content understanding for dress code and couples suitability
    dress_code_check = event_info.dress_code_mentioned if event_info else False
    couples_check = event_info.suitable_for_couples_mentioned if event_info else False
    event_details_check = dress_code_check or couples_check or has_any_ci(answer, ['dress code', 'suitable for couples', 'couples'])
    evaluator.add_custom_node(
        result=bool(event_details_check),
        id="eventbrite_perception_event_details",
        desc="[Perception Node] eventbrite.com:F3:P7 - Deep reading of event description for dress code or couples suitability",
        parent=eventbrite_node,
        critical=False
    )

    # Additional checks for event type and timing
    event_type_ok = event_info and event_info.event_type and has_any_ci(event_info.event_type, ['art', 'exhibition', 'jazz', 'concert'])
    if not event_type_ok:
        event_type_ok = has_any_ci(answer, ['art exhibition', 'jazz concert', 'exhibition', 'concert'])
    evaluator.add_custom_node(
        result=bool(event_type_ok),
        id="eventbrite_event_type",
        desc="Event type matches art exhibition or jazz concert",
        parent=eventbrite_node,
        critical=False
    )

    # End time consideration
    end_time_ok = event_info and (event_info.event_end_time or event_info.end_time_clarity)
    if not end_time_ok:
        end_time_ok = has_any_ci(answer, ['end time', 'ends at', 'ends before', '7:00 pm', '7 pm'])
    evaluator.add_custom_node(
        result=bool(end_time_ok),
        id="eventbrite_end_time_consideration",
        desc="Event end time is mentioned or clarity about end time is provided",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 OpenTable part
    opentable_node = evaluator.add_sequential(
        id="opentable_section",
        desc="OpenTable restaurant search based on event location and timing",
        parent=root,
        critical=False
    )

    # [Action Node] opentable.com:F1:A1 - Multi-condition filtering
    cuisine_ok = restaurant_info and restaurant_info.cuisine_type and has_any_ci(restaurant_info.cuisine_type, ['italian', 'french'])
    if not cuisine_ok:
        cuisine_ok = has_any_ci(answer, ['italian', 'french'])

    rating_value = extract_rating(restaurant_info.rating_text) if restaurant_info else None
    rating_ok = rating_value and rating_value > 4.5
    if not rating_ok:
        rating_ok = has_any_ci(answer, ['4.5', '4.6', '4.7', '4.8', '4.9', '5.0', 'above 4.5'])

    price_ok = restaurant_info and restaurant_info.price_range and looks_like_price_range(restaurant_info.price_range)
    if not price_ok:
        price_ok = has_any_ci(answer, ['$$', '$$$'])

    multi_filter_ok = cuisine_ok and (rating_ok or has_any_ci(answer, ['rating', 'rated'])) and price_ok
    evaluator.add_custom_node(
        result=bool(multi_filter_ok),
        id="opentable_action_multi_filter",
        desc="[Action Node] opentable.com:F1:A1 - Multi-condition filtering (Italian/French, rating >4.5, price $$/$$)",
        parent=opentable_node,
        critical=False
    )

    # [Perception Node] opentable.com:F4:P9 - Time slot availability awareness
    availability_check = restaurant_info and restaurant_info.table_availability_mentioned
    if not availability_check:
        availability_check = has_any_ci(answer, ['available', 'availability', 'table for two', '2 people', 'two people'])

    timing_check = has_any_ci(answer, ['30 minutes after', 'after the event', 'reservation time'])

    time_slot_ok = availability_check and timing_check
    evaluator.add_custom_node(
        result=bool(time_slot_ok),
        id="opentable_perception_time_availability",
        desc="[Perception Node] opentable.com:F4:P9 - Recognition of specific time slot availability status for two people",
        parent=opentable_node,
        critical=False
    )

    # Additional checks for location proximity and reservation details
    proximity_ok = has_any_ci(answer, ['walking distance', 'nearby', 'near', 'short drive', 'close'])
    evaluator.add_custom_node(
        result=bool(proximity_ok),
        id="opentable_location_proximity",
        desc="Restaurant location proximity to event venue is considered",
        parent=opentable_node,
        critical=False
    )

    reservation_time_ok = restaurant_info and restaurant_info.reservation_time and looks_like_time(restaurant_info.reservation_time)
    if not reservation_time_ok:
        reservation_time_ok = has_any_ci(answer, ['reservation', 'book', 'table'])
    evaluator.add_custom_node(
        result=bool(reservation_time_ok),
        id="opentable_reservation_time",
        desc="Specific reservation time is provided",
        parent=opentable_node,
        critical=False
    )

    # 3.3 Final output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Final answer includes all required information",
        parent=root,
        critical=False
    )

    event_name_ok = event_info and event_info.event_name and len(event_info.event_name.strip()) > 0
    evaluator.add_custom_node(
        result=bool(event_name_ok),
        id="output_event_name",
        desc="Event name is provided in the answer",
        parent=output_node,
        critical=False
    )

    restaurant_name_ok = restaurant_info and restaurant_info.restaurant_name and len(restaurant_info.restaurant_name.strip()) > 0
    evaluator.add_custom_node(
        result=bool(restaurant_name_ok),
        id="output_restaurant_name",
        desc="Restaurant name is provided in the answer",
        parent=output_node,
        critical=False
    )

    final_recommendation_ok = event_name_ok and restaurant_name_ok
    evaluator.add_custom_node(
        result=bool(final_recommendation_ok),
        id="output_complete_recommendation",
        desc="Complete date night plan with both event and restaurant",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
