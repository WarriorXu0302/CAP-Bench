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
TASK_ID = "task-391330"
TASK_DESCRIPTION = 'I want to go to an event in Seattle within the next two weeks (preferably next Saturday) related to “AI” or “Technology.” First, search on Eventbrite and pick one event that fits the timing and appears relatively formal, then note the exact venue address. Next, go to TripAdvisor and find a hotel very close to that venue, with a requirement of at least 4 stars and a rating of 4.0 or higher. Finally, to confirm convenience, use Google Maps to plan a walking route from the hotel to the event venue, prioritizing options with a walking time of 15 minutes or less. If no option within 15 minutes meets the hotel criteria, keep the hotel criteria unchanged and choose the feasible hotel with the shortest walking time, then state the actual walking time in the output. Output the event name, hotel name, and the walking minutes shown on Google Maps.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from the answer"""
    event_name: Optional[str] = None
    venue_address: Optional[str] = None
    event_date_description: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    star_rating_text: Optional[str] = None
    review_rating_text: Optional[str] = None


class WalkingRouteInfo(BaseModel):
    """Walking route details extracted from the answer"""
    walking_time_minutes_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the event details from the answer that the user found on Eventbrite:

- event_name: the name of the event exactly as stated
- venue_address: the exact venue address for the event
- event_date_description: any description of when the event is happening (e.g., "next Saturday", specific date, etc.)

If any field is missing in the answer, set it to null.
"""


def prompt_extract_hotel_from_answer() -> str:
    return """
Extract the hotel details from the answer that the user found on TripAdvisor:

- hotel_name: the name of the hotel exactly as stated
- star_rating_text: the star rating (e.g., "4 stars", "5-star", etc.) exactly as written
- review_rating_text: the review rating score (e.g., "4.5", "4.0 or higher", etc.) exactly as written

If any field is missing, set it to null.
"""


def prompt_extract_walking_route_from_answer() -> str:
    return """
Extract the walking route information from the answer from Google Maps:

- walking_time_minutes_text: the walking time in minutes exactly as stated (e.g., "12 minutes", "15 min", etc.)

If not present, set it to null.
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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def looks_like_star_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_int(text)
    if num is None:
        return False
    return num >= 4 and has_any_ci(text, ['star'])


def looks_like_review_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num >= 4.0


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_int(text)
    if num is None:
        return False
    return has_any_ci(text, ['min', 'minute'])


def mentions_saturday_or_next_week(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['saturday', 'sat', 'next week', 'next two weeks', 'within two weeks'])


def mentions_ai_or_technology(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['ai', 'artificial intelligence', 'technology', 'tech'])


def mentions_formal(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['formal', 'professional', 'business'])


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
    Restrict evaluator.verify to at most one usage (not used here).
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

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    walking_info = await evaluator.extract(
        prompt=prompt_extract_walking_route_from_answer(),
        template_class=WalkingRouteInfo,
        extraction_name="walking_route_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search and selection in Seattle",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date selector operation (next Saturday timing)
    eventbrite_date_action_ok = (has_any_ci(answer, ['eventbrite']) and
                                  mentions_saturday_or_next_week(answer))
    evaluator.add_custom_node(
        result=bool(eventbrite_date_action_ok),
        id="eventbrite_action_date_selector",
        desc="[Action Node] eventbrite.com:F1:A1 - Operate date selector to find events for next Saturday or within two weeks",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P6 - Visual feature extraction (formal appearance)
    event_mentions_formal = mentions_formal(answer)
    event_has_visual_context = has_any_ci(answer, ['image', 'cover', 'photo', 'picture', 'appearance', 'formal', 'professional'])
    evaluator.add_custom_node(
        result=bool(event_mentions_formal or event_has_visual_context),
        id="eventbrite_perception_formal_appearance",
        desc="[Perception Node] eventbrite.com:F3:P6 - Judge event formality based on cover image or visual style",
        parent=eventbrite_node,
        critical=False
    )

    # Event name extracted
    event_name_ok = bool(event_info and event_info.event_name and event_info.event_name.strip())
    evaluator.add_custom_node(
        result=bool(event_name_ok),
        id="eventbrite_event_name_extracted",
        desc="Event name successfully extracted from answer",
        parent=eventbrite_node,
        critical=False
    )

    # Venue address extracted
    venue_address_ok = bool(event_info and event_info.venue_address and event_info.venue_address.strip())
    evaluator.add_custom_node(
        result=bool(venue_address_ok),
        id="eventbrite_venue_address_extracted",
        desc="Exact venue address successfully extracted from answer",
        parent=eventbrite_node,
        critical=False
    )

    # AI or Technology topic
    topic_ok = mentions_ai_or_technology(answer)
    evaluator.add_custom_node(
        result=bool(topic_ok),
        id="eventbrite_ai_or_tech_topic",
        desc="Event is related to AI or Technology topic",
        parent=eventbrite_node,
        critical=False
    )

    # Seattle location
    seattle_ok = has_any_ci(answer, ['seattle'])
    evaluator.add_custom_node(
        result=bool(seattle_ok),
        id="eventbrite_seattle_location",
        desc="Event is located in Seattle",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor hotel search near event venue",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-filter operation (4+ stars, 4.0+ rating)
    tripadvisor_action_ok = (has_any_ci(answer, ['tripadvisor']) and
                             (has_any_ci(answer, ['4 star', '4-star', 'four star']) or
                              has_any_ci(answer, ['5 star', '5-star', 'five star'])) and
                             has_any_ci(answer, ['4.0', '4.5', '4.', 'rating']))
    evaluator.add_custom_node(
        result=bool(tripadvisor_action_ok),
        id="tripadvisor_action_filter",
        desc="[Action Node] tripadvisor.com:F1:A4 - Apply multi-filter for 4+ star hotels with 4.0+ rating",
        parent=tripadvisor_node,
        critical=False
    )

    # Hotel name extracted
    hotel_name_ok = bool(hotel_info and hotel_info.hotel_name and hotel_info.hotel_name.strip())
    evaluator.add_custom_node(
        result=bool(hotel_name_ok),
        id="tripadvisor_hotel_name_extracted",
        desc="Hotel name successfully extracted from answer",
        parent=tripadvisor_node,
        critical=False
    )

    # Star rating check (4+ stars)
    star_rating_ok = looks_like_star_rating(hotel_info.star_rating_text) if hotel_info else False
    evaluator.add_custom_node(
        result=bool(star_rating_ok),
        id="tripadvisor_star_rating_check",
        desc="Hotel has at least 4-star rating",
        parent=tripadvisor_node,
        critical=False
    )

    # Review rating check (4.0+)
    review_rating_ok = looks_like_review_rating(hotel_info.review_rating_text) if hotel_info else False
    evaluator.add_custom_node(
        result=bool(review_rating_ok),
        id="tripadvisor_review_rating_check",
        desc="Hotel has review rating of 4.0 or higher",
        parent=tripadvisor_node,
        critical=False
    )

    # Close to venue mention
    close_to_venue_ok = has_any_ci(answer, ['close', 'near', 'nearby', 'walking distance', 'proximity'])
    evaluator.add_custom_node(
        result=bool(close_to_venue_ok),
        id="tripadvisor_proximity_to_venue",
        desc="Hotel is described as close to the event venue",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.3 Google Maps section
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps walking route verification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Switch to walking mode
    googlemaps_action_ok = (has_any_ci(answer, ['google maps', 'maps']) and
                           has_any_ci(answer, ['walking', 'walk', 'on foot']))
    evaluator.add_custom_node(
        result=bool(googlemaps_action_ok),
        id="googlemaps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Switch transportation mode to Walking",
        parent=googlemaps_node,
        critical=False
    )

    # Walking time extracted
    walking_time_ok = looks_like_walking_time(walking_info.walking_time_minutes_text) if walking_info else False
    evaluator.add_custom_node(
        result=bool(walking_time_ok),
        id="googlemaps_walking_time_extracted",
        desc="Walking time in minutes successfully extracted from answer",
        parent=googlemaps_node,
        critical=False
    )

    # Walking time preference (15 minutes or less, or explanation if longer)
    walking_minutes = extract_int(walking_info.walking_time_minutes_text) if walking_info else None
    time_preference_met = False
    if walking_minutes is not None:
        if walking_minutes <= 15:
            time_preference_met = True
        else:
            # If longer than 15 minutes, check if explanation is provided
            if has_any_ci(answer, ['shortest', 'actual', 'exceeded', 'longer', 'more than 15']):
                time_preference_met = True

    evaluator.add_custom_node(
        result=bool(time_preference_met),
        id="googlemaps_time_preference_check",
        desc="Walking time is 15 minutes or less, or explanation provided for longer time",
        parent=googlemaps_node,
        critical=False
    )

    # Route planning from hotel to venue
    route_context_ok = has_any_ci(answer, ['hotel', 'venue', 'event', 'from', 'to', 'route'])
    evaluator.add_custom_node(
        result=bool(route_context_ok),
        id="googlemaps_route_context",
        desc="Route is planned from hotel to event venue",
        parent=googlemaps_node,
        critical=False
    )

    # 3.4 Final output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Final output includes all required information",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(event_name_ok),
        id="output_event_name",
        desc="Event name is present in output",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(hotel_name_ok),
        id="output_hotel_name",
        desc="Hotel name is present in output",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(walking_time_ok),
        id="output_walking_minutes",
        desc="Walking minutes from Google Maps is present in output",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
