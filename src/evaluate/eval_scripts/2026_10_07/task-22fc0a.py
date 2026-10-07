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
TASK_ID = "task-22fc0a"
TASK_DESCRIPTION = 'I’m going to San Francisco this coming Friday to see **“The Racket”** stand-up show at **Cobb’s Comedy Club**, which starts at **8:00 PM**. Please help me plan what to do after the show: first, confirm approximately what time the show will end, then find **3 restaurants on Yelp** that are within a **10-minute walk** and still open at that post-show time. Prefer places rated **4.0 stars or higher**, and **no fast food**.\n\nPlease provide the **name**, **rating**, **exact walking time**, and **closing time** for each of the 3 places. Also, please check which one has Yelp reviews that mention **“late-night snack”** or **“late night”** most frequently.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowEndTime(BaseModel):
    """Estimated show end time extracted from the answer"""
    end_time_text: Optional[str] = None
    estimation_method: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Single restaurant details"""
    name: Optional[str] = None
    rating: Optional[str] = None
    walking_time: Optional[str] = None
    closing_time: Optional[str] = None


class RestaurantsCollection(BaseModel):
    """Collection of 3 restaurants extracted from the answer"""
    restaurant_1: Optional[RestaurantInfo] = None
    restaurant_2: Optional[RestaurantInfo] = None
    restaurant_3: Optional[RestaurantInfo] = None


class LateNightAnalysis(BaseModel):
    """Late-night review analysis extracted from the answer"""
    restaurant_with_most_mentions: Optional[str] = None
    analysis_details: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_show_end_time() -> str:
    return """
Extract the estimated end time for "The Racket" show at Cobb's Comedy Club from the answer.

Return:
- end_time_text: the estimated end time exactly as stated (e.g., "10:00 PM", "around 10 PM").
- estimation_method: brief mention of how the end time was determined (e.g., "typical comedy show duration", "checked on Eventbrite").

If not present, set fields to null.
"""


def prompt_extract_restaurants() -> str:
    return """
From the answer, extract the details for the 3 restaurants found on Yelp.

For each restaurant (restaurant_1, restaurant_2, restaurant_3), extract:
- name: restaurant name exactly as stated
- rating: the rating exactly as stated (include "stars" or numbers)
- walking_time: the walking time exactly as stated (include "minutes" or "min")
- closing_time: the closing time exactly as stated

If any restaurant or field is missing, set it to null.
"""


def prompt_extract_late_night_analysis() -> str:
    return """
From the answer, extract which restaurant has Yelp reviews mentioning "late-night snack" or "late night" most frequently.

Return:
- restaurant_with_most_mentions: the name of the restaurant with the most mentions
- analysis_details: any additional details about the review analysis

If not present, set fields to null.
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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['pm', 'p.m.', 'am', 'a.m.', ':'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5 and has_any_ci(text, ['star', 'rating', '.'])


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'walk', 'minute'])


def is_valid_restaurant(rest: Optional[RestaurantInfo]) -> bool:
    if not rest:
        return False
    return bool(rest.name and rest.name.strip() and
                rest.rating and rest.rating.strip() and
                rest.walking_time and rest.walking_time.strip() and
                rest.closing_time and rest.closing_time.strip())


def meets_rating_threshold(rating_text: Optional[str], threshold: float = 4.0) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    if num is None:
        return False
    return num >= threshold


def within_walking_time(walking_text: Optional[str], max_minutes: int = 10) -> bool:
    if not walking_text:
        return False
    num = extract_float(walking_text)
    if num is None:
        return False
    return num <= max_minutes


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
    show_end = await evaluator.extract(
        prompt=prompt_extract_show_end_time(),
        template_class=ShowEndTime,
        extraction_name="show_end_time"
    )

    restaurants = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsCollection,
        extraction_name="restaurants_collection"
    )

    late_night = await evaluator.extract(
        prompt=prompt_extract_late_night_analysis(),
        template_class=LateNightAnalysis,
        extraction_name="late_night_analysis"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Show end time estimation section
    show_section = evaluator.add_sequential(
        id="show_end_time_section",
        desc="Determine approximate end time for The Racket show at Cobb's Comedy Club",
        parent=root,
        critical=False
    )

    # Check if answer mentions checking show duration or end time
    mentions_show_duration = has_any_ci(answer, ['show', 'comedy', 'duration', 'end', 'finish', 'eventbrite', 'google'])
    has_end_time = show_end and show_end.end_time_text and looks_like_time(show_end.end_time_text)

    evaluator.add_custom_node(
        result=bool(mentions_show_duration and has_end_time),
        id="show_end_time_confirmed",
        desc="Confirms approximate show end time through research (Eventbrite/Google) or estimation",
        parent=show_section,
        critical=False
    )

    # 3.2 Yelp restaurant search section
    yelp_section = evaluator.add_sequential(
        id="yelp_restaurant_search",
        desc="Find 3 restaurants on Yelp near Cobb's Comedy Club",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-condition filtering (rating, hours, no fast food)
    mentions_yelp = has_any_ci(answer, ['yelp'])
    mentions_filtering = has_any_ci(answer, ['rating', '4.0', 'star', 'open', 'hours', 'fast food'])

    evaluator.add_custom_node(
        result=bool(mentions_yelp and mentions_filtering),
        id="yelp_filtering_action",
        desc="[Action Node] yelp.com:F1:A2 - Apply multi-condition filtering (rating 4.0+, post-show hours, no fast food)",
        parent=yelp_section,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Basic business information perception
    rest_list = [restaurants.restaurant_1, restaurants.restaurant_2, restaurants.restaurant_3]
    valid_restaurants = [r for r in rest_list if is_valid_restaurant(r)]
    has_three_restaurants = len(valid_restaurants) >= 3

    all_have_required_fields = has_three_restaurants
    if has_three_restaurants:
        for r in valid_restaurants[:3]:
            if not (looks_like_rating(r.rating) and
                   looks_like_walking_time(r.walking_time) and
                   looks_like_time(r.closing_time)):
                all_have_required_fields = False
                break

    evaluator.add_custom_node(
        result=bool(all_have_required_fields),
        id="yelp_basic_info_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Extract name, rating, and closing time for each restaurant",
        parent=yelp_section,
        critical=False
    )

    # Check rating threshold (4.0+)
    meets_rating_criteria = False
    if has_three_restaurants:
        meets_rating_criteria = all(meets_rating_threshold(r.rating, 4.0) for r in valid_restaurants[:3])

    evaluator.add_custom_node(
        result=bool(meets_rating_criteria),
        id="rating_threshold_check",
        desc="All 3 restaurants meet the 4.0 star rating threshold",
        parent=yelp_section,
        critical=False
    )

    # 3.3 Google Maps walking time verification section
    maps_section = evaluator.add_sequential(
        id="google_maps_walking_time",
        desc="Verify walking times using Google Maps",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Switch to walking mode
    mentions_maps = has_any_ci(answer, ['google maps', 'maps', 'walking time', 'walk'])
    mentions_walking_mode = has_any_ci(answer, ['walk', 'walking', 'foot', 'pedestrian'])

    evaluator.add_custom_node(
        result=bool(mentions_maps and mentions_walking_mode),
        id="maps_walking_mode_action",
        desc="[Action Node] maps.google.com:F2:A7 - Switch to walking mode to verify distances",
        parent=maps_section,
        critical=False
    )

    # Check 10-minute walking constraint
    within_time_limit = False
    if has_three_restaurants:
        within_time_limit = all(within_walking_time(r.walking_time, 10) for r in valid_restaurants[:3])

    evaluator.add_custom_node(
        result=bool(within_time_limit),
        id="walking_time_constraint",
        desc="All 3 restaurants are within 10-minute walk",
        parent=maps_section,
        critical=False
    )

    # 3.4 Yelp review analysis section
    review_section = evaluator.add_sequential(
        id="yelp_review_analysis",
        desc="Analyze Yelp reviews for late-night mentions",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A9 - Click into detail page to view reviews
    mentions_reviews = has_any_ci(answer, ['review', 'comment', 'mention'])
    mentions_late_night = has_any_ci(answer, ['late night', 'late-night'])

    evaluator.add_custom_node(
        result=bool(mentions_reviews and mentions_late_night),
        id="yelp_review_detail_action",
        desc="[Action Node] yelp.com:F1:A9 - Access restaurant detail pages to read reviews",
        parent=review_section,
        critical=False
    )

    # Check if late-night analysis was performed
    has_late_night_analysis = (late_night and
                               late_night.restaurant_with_most_mentions and
                               late_night.restaurant_with_most_mentions.strip())

    evaluator.add_custom_node(
        result=bool(has_late_night_analysis),
        id="late_night_frequency_analysis",
        desc="Identifies which restaurant has most 'late night' or 'late-night snack' mentions in reviews",
        parent=review_section,
        critical=False
    )

    # 3.5 Overall completeness checks
    completeness_section = evaluator.add_parallel(
        id="completeness_checks",
        desc="Overall task completeness",
        parent=root,
        critical=False
    )

    # Check if answer addresses post-show planning context
    addresses_context = has_any_ci(answer, ['after', 'post', 'show', 'comedy', 'cobb'])
    evaluator.add_custom_node(
        result=bool(addresses_context),
        id="post_show_context",
        desc="Answer addresses post-show dining planning context",
        parent=completeness_section,
        critical=False
    )

    # Check if San Francisco location is considered
    mentions_sf = has_any_ci(answer, ['san francisco', 'sf', 'cobb'])
    evaluator.add_custom_node(
        result=bool(mentions_sf),
        id="location_context",
        desc="Answer considers San Francisco and Cobb's Comedy Club location",
        parent=completeness_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
