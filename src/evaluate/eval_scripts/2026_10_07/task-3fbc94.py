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
TASK_ID = "task-3fbc94"
TASK_DESCRIPTION = 'Next Saturday, we are taking elderly family members and children to visit the Getty Villa in Los Angeles (please note: Villa, not Center). We expect to finish around 6 PM and need to find a place for dinner nearby. On TripAdvisor, starting from the Getty Villa, find 3 restaurants within a 10-minute drive, with a rating of 4 stars or higher. Please consider the preferences of our elderly family members, prioritizing Italian or seafood cuisine. Additionally, check reviews for mentions of "family-friendly" or "quiet." After selecting these 3 restaurants, check OpenTable for availability for 6 people between 6 PM and 7 PM next Saturday evening. Finally, list the names of these 3 restaurants, their ratings, positive aspects mentioned in reviews, and the specific available reservation times on OpenTable.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantInfo(BaseModel):
    """Information about a single restaurant"""
    name: Optional[str] = None
    rating: Optional[str] = None
    cuisine_type: Optional[str] = None
    distance_info: Optional[str] = None
    positive_aspects: Optional[str] = None
    opentable_availability: Optional[str] = None


class RestaurantsAnswer(BaseModel):
    """All restaurants extracted from the answer"""
    restaurants: List[RestaurantInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants_from_answer() -> str:
    return """
Extract all restaurants mentioned in the answer that the user found on TripAdvisor near Getty Villa.

For each restaurant, extract:
- name: the restaurant name exactly as written
- rating: the rating (e.g., "4.5 stars", "4 stars") exactly as stated
- cuisine_type: the cuisine type if mentioned (Italian, seafood, etc.)
- distance_info: any distance or travel time information mentioned
- positive_aspects: any positive aspects from reviews mentioned (family-friendly, quiet, etc.)
- opentable_availability: any OpenTable availability times mentioned for next Saturday 6-7 PM for 6 people

Also extract:
- total_count: the total number of restaurants listed

If any field is missing, set it to null. If no restaurants are found, return an empty list.
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


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def mentions_getty_villa(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['getty villa'])


def mentions_tripadvisor(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['tripadvisor', 'trip advisor'])


def mentions_opentable(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['opentable', 'open table'])


def mentions_italian_or_seafood(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['italian', 'seafood', '海鲜', '意大利'])


def mentions_family_or_quiet(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['family-friendly', 'family friendly', 'quiet', 'families', '安静', '家庭'])


def mentions_distance_or_drive(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['minute', 'drive', 'min', 'distance', 'away', 'km', 'mile', '分钟', '车程'])


def looks_like_ten_minutes_or_less(text: Optional[str]) -> bool:
    if not text:
        return False
    nums = re.findall(r'(\d+)', text)
    if not nums:
        return False
    try:
        val = int(nums[0])
        return val <= 10
    except Exception:
        return False


def mentions_saturday_evening(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['saturday', 'sat', '周六', '星期六'])


def mentions_six_people(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['6 people', '6 person', 'six people', '6人', '六人'])


def mentions_time_6_to_7(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['6', '7', 'pm', 'p.m.', '晚上'])


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
    restaurants_info = await evaluator.extract(
        prompt=prompt_extract_restaurants_from_answer(),
        template_class=RestaurantsAnswer,
        extraction_name="restaurants_answer"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor search and filtering for restaurants near Getty Villa",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F6:A1 - Multi-field search with Getty Villa landmark
    getty_villa_search_ok = mentions_getty_villa(answer) and mentions_tripadvisor(answer)
    evaluator.add_custom_node(
        result=bool(getty_villa_search_ok),
        id="tripadvisor_action_landmark_search",
        desc="[Action Node] tripadvisor.com:F6:A1 - Search for restaurants using Getty Villa as the landmark/location on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F6:A6 - Filter by cuisine (Italian or seafood)
    cuisine_filter_ok = mentions_italian_or_seafood(answer)
    evaluator.add_custom_node(
        result=bool(cuisine_filter_ok),
        id="tripadvisor_action_cuisine_filter",
        desc="[Action Node] tripadvisor.com:F6:A6 - Apply cuisine filter for Italian or seafood restaurants",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P16 - Understand distance information in list
    distance_check_ok = False
    if restaurants_info and restaurants_info.restaurants:
        for r in restaurants_info.restaurants:
            if r.distance_info and mentions_distance_or_drive(r.distance_info):
                distance_check_ok = True
                break

    evaluator.add_custom_node(
        result=bool(distance_check_ok),
        id="tripadvisor_perception_distance_list",
        desc="[Perception Node] tripadvisor.com:F2:P16 - Extract and understand distance information (within 10-minute drive) from the restaurant list",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F2:A10 - Expand reviews to check for specific mentions
    review_mentions_ok = mentions_family_or_quiet(answer)
    evaluator.add_custom_node(
        result=bool(review_mentions_ok),
        id="tripadvisor_action_expand_reviews",
        desc="[Action Node] tripadvisor.com:F2:A10 - Expand and read reviews to find mentions of 'family-friendly' or 'quiet'",
        parent=tripadvisor_node,
        critical=False
    )

    # Check that 3 restaurants were found
    has_three_restaurants = (restaurants_info and
                            restaurants_info.restaurants and
                            len(restaurants_info.restaurants) >= 3)
    evaluator.add_custom_node(
        result=bool(has_three_restaurants),
        id="tripadvisor_three_restaurants",
        desc="Found 3 restaurants meeting the criteria",
        parent=tripadvisor_node,
        critical=False
    )

    # Check that restaurants have ratings 4+ stars
    ratings_ok = False
    if restaurants_info and restaurants_info.restaurants:
        ratings_count = 0
        for r in restaurants_info.restaurants[:3]:
            if r.rating and looks_like_rating(r.rating):
                rating_val = extract_float(r.rating)
                if rating_val and rating_val >= 4.0:
                    ratings_count += 1
        ratings_ok = ratings_count >= 3

    evaluator.add_custom_node(
        result=bool(ratings_ok),
        id="tripadvisor_ratings_check",
        desc="All 3 restaurants have ratings of 4 stars or higher",
        parent=tripadvisor_node,
        critical=False
    )

    # Check that positive aspects from reviews are mentioned
    positive_aspects_ok = False
    if restaurants_info and restaurants_info.restaurants:
        for r in restaurants_info.restaurants[:3]:
            if r.positive_aspects and len(r.positive_aspects.strip()) > 0:
                positive_aspects_ok = True
                break

    evaluator.add_custom_node(
        result=bool(positive_aspects_ok),
        id="tripadvisor_positive_aspects",
        desc="Positive aspects mentioned in reviews are included for the restaurants",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.2 OpenTable section
    opentable_node = evaluator.add_sequential(
        id="opentable_section",
        desc="OpenTable availability check for selected restaurants",
        parent=root,
        critical=False
    )

    # [Action Node] opentable.com:F4:A2 - Date, time, and party size selection
    opentable_search_ok = (mentions_opentable(answer) and
                          mentions_saturday_evening(answer) and
                          mentions_six_people(answer) and
                          mentions_time_6_to_7(answer))
    evaluator.add_custom_node(
        result=bool(opentable_search_ok),
        id="opentable_action_datetime_selection",
        desc="[Action Node] opentable.com:F4:A2 - Search on OpenTable with next Saturday evening, 6-7 PM time range, and 6 people",
        parent=opentable_node,
        critical=False
    )

    # [Perception Node] opentable.com:F4:P9 - Identify available time slots
    availability_ok = False
    if restaurants_info and restaurants_info.restaurants:
        for r in restaurants_info.restaurants[:3]:
            if r.opentable_availability and len(r.opentable_availability.strip()) > 0:
                availability_ok = True
                break

    evaluator.add_custom_node(
        result=bool(availability_ok),
        id="opentable_perception_availability",
        desc="[Perception Node] opentable.com:F4:P9 - Identify and extract specific available reservation time slots for the restaurants",
        parent=opentable_node,
        critical=False
    )

    # Check that OpenTable availability is provided for all 3 restaurants
    all_availability_ok = False
    if restaurants_info and restaurants_info.restaurants and len(restaurants_info.restaurants) >= 3:
        availability_count = 0
        for r in restaurants_info.restaurants[:3]:
            if r.opentable_availability and len(r.opentable_availability.strip()) > 0:
                availability_count += 1
        all_availability_ok = availability_count >= 3

    evaluator.add_custom_node(
        result=bool(all_availability_ok),
        id="opentable_all_restaurants_checked",
        desc="OpenTable availability information provided for all 3 restaurants",
        parent=opentable_node,
        critical=False
    )

    # 3.3 Final output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Final answer completeness check",
        parent=root,
        critical=False
    )

    # Check restaurant names are provided
    names_ok = False
    if restaurants_info and restaurants_info.restaurants and len(restaurants_info.restaurants) >= 3:
        names_count = sum(1 for r in restaurants_info.restaurants[:3] if r.name and len(r.name.strip()) > 0)
        names_ok = names_count >= 3

    evaluator.add_custom_node(
        result=bool(names_ok),
        id="output_restaurant_names",
        desc="All 3 restaurant names are listed in the final answer",
        parent=output_node,
        critical=False
    )

    # Check ratings are provided
    output_ratings_ok = False
    if restaurants_info and restaurants_info.restaurants and len(restaurants_info.restaurants) >= 3:
        ratings_count = sum(1 for r in restaurants_info.restaurants[:3] if r.rating and len(r.rating.strip()) > 0)
        output_ratings_ok = ratings_count >= 3

    evaluator.add_custom_node(
        result=bool(output_ratings_ok),
        id="output_ratings",
        desc="Ratings are provided for all 3 restaurants in the final answer",
        parent=output_node,
        critical=False
    )

    # Check positive aspects are provided
    output_positive_ok = False
    if restaurants_info and restaurants_info.restaurants and len(restaurants_info.restaurants) >= 3:
        positive_count = sum(1 for r in restaurants_info.restaurants[:3] if r.positive_aspects and len(r.positive_aspects.strip()) > 0)
        output_positive_ok = positive_count >= 3

    evaluator.add_custom_node(
        result=bool(output_positive_ok),
        id="output_positive_aspects",
        desc="Positive aspects from reviews are provided for all 3 restaurants",
        parent=output_node,
        critical=False
    )

    # Check OpenTable times are provided
    output_times_ok = False
    if restaurants_info and restaurants_info.restaurants and len(restaurants_info.restaurants) >= 3:
        times_count = sum(1 for r in restaurants_info.restaurants[:3] if r.opentable_availability and len(r.opentable_availability.strip()) > 0)
        output_times_ok = times_count >= 3

    evaluator.add_custom_node(
        result=bool(output_times_ok),
        id="output_opentable_times",
        desc="Specific OpenTable available reservation times are provided for all 3 restaurants",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
