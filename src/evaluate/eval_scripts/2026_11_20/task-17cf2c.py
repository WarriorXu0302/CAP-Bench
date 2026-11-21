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
TASK_ID = "task-17cf2c"
TASK_DESCRIPTION = "I'd like to plan a team-building event for my team of 20 people in San Francisco for an upcoming weekend.\n\nFirst, on Eventbrite, search for an event in San Francisco, applying the 'This Weekend' filter to specify the date range for an upcoming weekend. Prioritize creative workshops or tours. Avoid events marked 'Sales end soon' or 'Going fast', and ensure the time is suitable.\n\nOnce an activity is selected, note down its specific address.\n\nThen, go to OpenTable and search for restaurants within walking distance (approximately 15 minutes) of the chosen activity's location. I'm looking for a well-rated restaurant (4 stars or higher or 'High Rating') that can accommodate a party of 20 (Party Size: 20). Preferably, the cuisine should be Italian or American.\n\nFinally, please provide me with the name, time, and location of the selected activity, along with the matching restaurant's name, rating, and specific available reservation times (e.g., 6:30 PM)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventbriteActivity(BaseModel):
    """Activity details extracted from Eventbrite"""
    activity_name: Optional[str] = None
    activity_time: Optional[str] = None
    activity_address: Optional[str] = None
    mentions_this_weekend: Optional[bool] = None
    mentions_avoiding_fast_selling: Optional[bool] = None


class OpenTableRestaurant(BaseModel):
    """Restaurant details extracted from OpenTable"""
    restaurant_name: Optional[str] = None
    restaurant_rating: Optional[str] = None
    available_reservation_times: Optional[str] = None
    mentions_party_size_20: Optional[bool] = None
    mentions_cuisine_preference: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eventbrite_activity() -> str:
    return """
Extract the Eventbrite activity details from the answer:

- activity_name: the name of the selected activity/event
- activity_time: the time of the activity
- activity_address: the specific address of the activity location
- mentions_this_weekend: true if the answer mentions using 'This Weekend' filter or searching for weekend events
- mentions_avoiding_fast_selling: true if the answer mentions avoiding 'Sales end soon' or 'Going fast' events

Set any missing field to null or false as appropriate.
"""


def prompt_extract_opentable_restaurant() -> str:
    return """
Extract the OpenTable restaurant details from the answer:

- restaurant_name: the name of the selected restaurant
- restaurant_rating: the rating of the restaurant (e.g., '4.5 stars', 'High Rating')
- available_reservation_times: specific available reservation times mentioned (e.g., '6:30 PM', '7:00 PM')
- mentions_party_size_20: true if the answer mentions searching for party size 20
- mentions_cuisine_preference: true if the answer mentions Italian or American cuisine preference

Set any missing field to null or false as appropriate.
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


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Address should contain street indicators and city or zip
    has_number = contains_digits(text)
    has_street_words = has_any_ci(text, ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd', 'boulevard', 'drive', 'dr', 'lane', 'ln'])
    has_location = has_any_ci(text, ['san francisco', 'sf', 'ca', '94'])
    return has_number and (has_street_words or has_location)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Rating should mention stars, numeric rating, or 'high rating'
    has_rating_words = has_any_ci(text, ['star', 'rating', 'rated'])
    has_number = contains_digits(text)
    return has_rating_words or has_number


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for time patterns like 6:30 PM, 18:30, etc.
    time_patterns = [
        r'\d{1,2}:\d{2}\s*(am|pm|AM|PM)',
        r'\d{1,2}\s*(am|pm|AM|PM)',
        r'\d{1,2}:\d{2}'
    ]
    return any(re.search(p, text) for p in time_patterns)


def mentions_eventbrite(answer: str) -> bool:
    return has_any_ci(answer, ['eventbrite'])


def mentions_opentable(answer: str) -> bool:
    return has_any_ci(answer, ['opentable'])


def mentions_san_francisco(answer: str) -> bool:
    return has_any_ci(answer, ['san francisco', 'sf'])


def mentions_creative_activity(answer: str) -> bool:
    return has_any_ci(answer, ['workshop', 'tour', 'creative', 'class', 'art'])


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
        prompt=prompt_extract_eventbrite_activity(),
        template_class=EventbriteActivity,
        extraction_name="eventbrite_activity"
    )

    opentable_info = await evaluator.extract(
        prompt=prompt_extract_opentable_restaurant(),
        template_class=OpenTableRestaurant,
        extraction_name="opentable_restaurant"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search and selection in San Francisco",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A6 - Use 'This Weekend' quick filter
    this_weekend_filter_ok = (
        mentions_eventbrite(answer) and
        (has_any_ci(answer, ['this weekend', 'weekend']) or
         (eventbrite_info and eventbrite_info.mentions_this_weekend))
    )
    evaluator.add_custom_node(
        result=bool(this_weekend_filter_ok),
        id="eventbrite_action_this_weekend_filter",
        desc="[Action Node] eventbrite.com:F1:A6 - Apply 'This Weekend' filter or quick date tab for upcoming weekend",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Avoid 'Going fast'/'Sales end soon' events
    avoids_fast_selling = (
        has_any_ci(answer, ['avoid', 'sales end soon', 'going fast']) or
        (eventbrite_info and eventbrite_info.mentions_avoiding_fast_selling)
    )
    evaluator.add_custom_node(
        result=bool(avoids_fast_selling),
        id="eventbrite_perception_avoid_fast_selling",
        desc="[Perception Node] eventbrite.com:F1:P1 - Recognize and avoid events marked 'Going fast' or 'Sales end soon'",
        parent=eventbrite_node,
        critical=False
    )

    # Activity details provided
    activity_name_ok = bool(eventbrite_info and eventbrite_info.activity_name and eventbrite_info.activity_name.strip())
    activity_time_ok = looks_like_time(eventbrite_info.activity_time) if eventbrite_info else False
    activity_address_ok = looks_like_address(eventbrite_info.activity_address) if eventbrite_info else False

    evaluator.add_custom_node(
        result=bool(activity_name_ok),
        id="eventbrite_activity_name_provided",
        desc="Activity name is provided",
        parent=eventbrite_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(activity_time_ok),
        id="eventbrite_activity_time_provided",
        desc="Activity time is provided",
        parent=eventbrite_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(activity_address_ok),
        id="eventbrite_activity_address_provided",
        desc="Activity specific address is noted down",
        parent=eventbrite_node,
        critical=False
    )

    # Creative workshop or tour prioritization
    creative_priority = mentions_creative_activity(answer)
    evaluator.add_custom_node(
        result=bool(creative_priority),
        id="eventbrite_creative_priority",
        desc="Prioritizes creative workshops or tours",
        parent=eventbrite_node,
        critical=False
    )

    # San Francisco location
    sf_location = mentions_san_francisco(answer)
    evaluator.add_custom_node(
        result=bool(sf_location),
        id="eventbrite_san_francisco_location",
        desc="Searches for events in San Francisco",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 OpenTable section
    opentable_node = evaluator.add_sequential(
        id="opentable_section",
        desc="OpenTable restaurant search near activity location",
        parent=root,
        critical=False
    )

    # [Action Node] opentable.com:F1:A2 - Complex form combination (date + party size 20 + time)
    party_size_20_ok = (
        has_any_ci(answer, ['20 people', 'party of 20', 'party size 20', 'party size: 20']) or
        (opentable_info and opentable_info.mentions_party_size_20)
    )
    date_context_ok = has_any_ci(answer, ['weekend', 'this weekend', 'upcoming weekend'])

    complex_form_ok = mentions_opentable(answer) and party_size_20_ok and date_context_ok

    evaluator.add_custom_node(
        result=bool(complex_form_ok),
        id="opentable_action_complex_form",
        desc="[Action Node] opentable.com:F1:A2 - Use complex form combination with date, party size 20, and time selection",
        parent=opentable_node,
        critical=False
    )

    # [Perception Node] opentable.com:F4:P9 - Recognize available time slot states
    specific_times_ok = (
        opentable_info and
        opentable_info.available_reservation_times and
        looks_like_time(opentable_info.available_reservation_times)
    )
    evaluator.add_custom_node(
        result=bool(specific_times_ok),
        id="opentable_perception_time_slots",
        desc="[Perception Node] opentable.com:F4:P9 - Identify specific available reservation time slots (e.g., clickable vs non-clickable)",
        parent=opentable_node,
        critical=False
    )

    # Restaurant details provided
    restaurant_name_ok = bool(opentable_info and opentable_info.restaurant_name and opentable_info.restaurant_name.strip())
    restaurant_rating_ok = looks_like_rating(opentable_info.restaurant_rating) if opentable_info else False
    high_rating_mentioned = has_any_ci(answer, ['4 star', '4.', 'high rating', 'well-rated'])

    evaluator.add_custom_node(
        result=bool(restaurant_name_ok),
        id="opentable_restaurant_name_provided",
        desc="Restaurant name is provided",
        parent=opentable_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(restaurant_rating_ok or high_rating_mentioned),
        id="opentable_restaurant_rating_provided",
        desc="Restaurant rating is provided (4 stars or higher / High Rating)",
        parent=opentable_node,
        critical=False
    )

    # Cuisine preference
    cuisine_preference_ok = (
        has_any_ci(answer, ['italian', 'american']) or
        (opentable_info and opentable_info.mentions_cuisine_preference)
    )
    evaluator.add_custom_node(
        result=bool(cuisine_preference_ok),
        id="opentable_cuisine_preference",
        desc="Cuisine preference (Italian or American) is considered",
        parent=opentable_node,
        critical=False
    )

    # Walking distance from activity
    walking_distance_ok = has_any_ci(answer, ['walking distance', 'walking', 'near', 'close to', '15 minute'])
    evaluator.add_custom_node(
        result=bool(walking_distance_ok),
        id="opentable_walking_distance",
        desc="Restaurant is within walking distance (approximately 15 minutes) of activity",
        parent=opentable_node,
        critical=False
    )

    # Uses activity address as search center
    uses_activity_location = (
        activity_address_ok and
        mentions_opentable(answer) and
        (walking_distance_ok or has_any_ci(answer, ['activity location', 'event location', 'from the']))
    )
    evaluator.add_custom_node(
        result=bool(uses_activity_location),
        id="opentable_uses_activity_location",
        desc="Uses the activity's address as the search center for OpenTable",
        parent=opentable_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
