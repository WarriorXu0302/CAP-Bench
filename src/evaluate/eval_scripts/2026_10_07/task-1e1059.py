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
TASK_ID = "task-1e1059"
TASK_DESCRIPTION = 'Next Friday, I’m planning a family vacation to Miami and need two contingency plans based on the weather.\n\nFirst, check AccuWeather for the **daytime precipitation probability** in Miami for the travel day.  \n- If the precipitation probability is **above 40%**, go to TripAdvisor and find **3 indoor things to do** that are **Good for Kids**, sorted by **rating (highest to lowest)**.  \n- If the precipitation probability is **40% or below**, find **3 outdoor things to do** that are **Good for Kids**, also sorted by **rating (highest to lowest)**.  \n\nIf AccuWeather cannot provide the precipitation probability for that travel day, use the **daytime precipitation probability for the nearest available date** and apply the same threshold logic.\n\nReturn the precipitation probability value for the day (or nearest available date), along with each recommended activity’s **name, rating, review count, and TripAdvisor detail page link**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PrecipitationInfo(BaseModel):
    """Precipitation probability extracted from the answer for Miami next Friday"""
    precipitation_probability: Optional[str] = None
    date_reference: Optional[str] = None


class Activity(BaseModel):
    """Single activity details"""
    name: Optional[str] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    detail_page_url: Optional[str] = None


class ActivitiesInfo(BaseModel):
    """List of activities extracted from the answer"""
    activities: List[Activity] = Field(default_factory=list)
    category_type: Optional[str] = None  # 'indoor' or 'outdoor'


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_precipitation_from_answer() -> str:
    return """
Extract the precipitation probability information for Miami next Friday (or the nearest available date if exact date not available) from the answer.

Return:
- precipitation_probability: the precipitation probability percentage exactly as stated (include % sign if present).
- date_reference: any mention of the date or 'next Friday' or 'nearest available date'.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_activities_from_answer() -> str:
    return """
From the answer, extract the list of recommended activities in Miami (should be 3 activities).

For each activity, extract:
- name: the activity name
- rating: the rating value
- review_count: the number of reviews
- detail_page_url: the TripAdvisor detail page link

Also extract:
- category_type: whether these are 'indoor' or 'outdoor' activities

Return all activities in the 'activities' list. If information is missing, set fields to null.
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


def looks_like_precipitation_probability(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    has_percent = '%' in text or ci_contains(text, 'percent')
    return has_percent or (0 <= num <= 100)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Ratings typically 0-5 or 0-10
    return 0 <= num <= 10


def looks_like_review_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_tripadvisor_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'tripadvisor')


def activities_sorted_by_rating(activities: List[Activity]) -> bool:
    if not activities or len(activities) < 2:
        return True

    ratings = []
    for act in activities:
        if act.rating:
            num = extract_float(act.rating)
            if num is not None:
                ratings.append(num)

    if len(ratings) < 2:
        return True

    # Check descending order
    for i in range(len(ratings) - 1):
        if ratings[i] < ratings[i + 1]:
            return False
    return True


def determine_category_from_precipitation(precip_text: Optional[str]) -> Optional[str]:
    """Determine if answer should be indoor (>40%) or outdoor (<=40%) based on precipitation"""
    if not precip_text:
        return None

    num = extract_float(precip_text)
    if num is None:
        return None

    if num > 40:
        return 'indoor'
    else:
        return 'outdoor'


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
    precip_info = await evaluator.extract(
        prompt=prompt_extract_precipitation_from_answer(),
        template_class=PrecipitationInfo,
        extraction_name="precipitation_info"
    )

    activities_info = await evaluator.extract(
        prompt=prompt_extract_activities_from_answer(),
        template_class=ActivitiesInfo,
        extraction_name="activities_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 AccuWeather part
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather precipitation probability for Miami next Friday",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F1:A1 - Location search and selection
    miami_mentioned = has_any_ci(answer, ['miami'])
    accuweather_mentioned = has_any_ci(answer, ['accuweather'])
    evaluator.add_custom_node(
        result=bool(miami_mentioned and accuweather_mentioned),
        id="accuweather_location_search",
        desc="[Action Node] accuweather.com:F1:A1 - Search for Miami location on AccuWeather",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F2:A2 - Date dimension switching to Daily view
    daily_or_date_mentioned = has_any_ci(answer, ['daily', 'next friday', 'friday', 'day', 'date'])
    evaluator.add_custom_node(
        result=bool(daily_or_date_mentioned and precip_info.date_reference),
        id="accuweather_date_switching",
        desc="[Action Node] accuweather.com:F2:A2 - Navigate to Daily view for next Friday",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F2:P3 - Extract precipitation probability from daily list
    precip_prob_valid = looks_like_precipitation_probability(precip_info.precipitation_probability)
    evaluator.add_custom_node(
        result=bool(precip_prob_valid),
        id="accuweather_precipitation_extraction",
        desc="[Perception Node] accuweather.com:F2:P3 - Extract daytime precipitation probability percentage",
        parent=accuweather_node,
        critical=False
    )

    # 3.2 TripAdvisor part
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor activities search and filtering in Miami",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A1 - Multi-field search (location + keyword)
    tripadvisor_mentioned = has_any_ci(answer, ['tripadvisor'])
    activities_mentioned = has_any_ci(answer, ['things to do', 'activities', 'attraction'])
    evaluator.add_custom_node(
        result=bool(tripadvisor_mentioned and miami_mentioned and activities_mentioned),
        id="tripadvisor_search",
        desc="[Action Node] tripadvisor.com:F4:A1 - Search for things to do in Miami",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A7 - Multi-condition filtering (Good for Kids + Indoor/Outdoor)
    good_for_kids_mentioned = has_any_ci(answer, ['good for kids', 'kid-friendly', 'kids', 'children', 'family'])

    expected_category = determine_category_from_precipitation(precip_info.precipitation_probability)
    actual_category = activities_info.category_type if activities_info else None

    indoor_outdoor_mentioned = has_any_ci(answer, ['indoor', 'outdoor'])
    category_matches = (expected_category and actual_category and
                       ci_contains(actual_category, expected_category))

    evaluator.add_custom_node(
        result=bool(good_for_kids_mentioned and indoor_outdoor_mentioned and category_matches),
        id="tripadvisor_filtering",
        desc="[Action Node] tripadvisor.com:F4:A7 - Apply 'Good for Kids' and Indoor/Outdoor filters based on precipitation probability",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A5 - Sort by rating (highest to lowest)
    sorted_mentioned = has_any_ci(answer, ['sorted', 'sort', 'rating', 'highest', 'descending'])
    ratings_in_descending_order = activities_sorted_by_rating(activities_info.activities if activities_info else [])
    evaluator.add_custom_node(
        result=bool(sorted_mentioned and ratings_in_descending_order),
        id="tripadvisor_sorting",
        desc="[Action Node] tripadvisor.com:F4:A5 - Sort activities by rating (highest to lowest)",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F4:P14 - Extract review count from list
    has_three_activities = activities_info and len(activities_info.activities) >= 3
    all_have_review_counts = (has_three_activities and
                             all(looks_like_review_count(act.review_count)
                                 for act in activities_info.activities[:3]))
    evaluator.add_custom_node(
        result=bool(all_have_review_counts),
        id="tripadvisor_review_count_extraction",
        desc="[Perception Node] tripadvisor.com:F4:P14 - Extract review counts for each activity",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F4:P15 - Identify Good for Kids tag/attribute
    # Check if activities actually mention kids or family attributes
    activities_mention_kids = (activities_info and
                              any(has_any_ci(act.name, ['kid', 'child', 'family'])
                                  for act in activities_info.activities[:3] if act.name))
    evaluator.add_custom_node(
        result=bool(good_for_kids_mentioned and (activities_mention_kids or has_three_activities)),
        id="tripadvisor_kids_attribute",
        desc="[Perception Node] tripadvisor.com:F4:P15 - Identify and filter by 'Good for Kids' attribute",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F2:A8 - Click into detail pages to get URLs
    all_have_urls = (has_three_activities and
                    all(looks_like_tripadvisor_url(act.detail_page_url)
                        for act in activities_info.activities[:3]))
    evaluator.add_custom_node(
        result=bool(all_have_urls),
        id="tripadvisor_detail_page_urls",
        desc="[Action Node] tripadvisor.com:F2:A8 - Obtain TripAdvisor detail page links for each activity",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.3 Additional output completeness checks (non-prefixed)
    output_completeness = evaluator.add_parallel(
        id="output_completeness",
        desc="Completeness of required output information",
        parent=root,
        critical=False
    )

    # Check all activities have names
    all_have_names = (has_three_activities and
                     all(act.name and act.name.strip() for act in activities_info.activities[:3]))
    evaluator.add_custom_node(
        result=bool(all_have_names),
        id="output_activity_names",
        desc="All 3 activities have names provided",
        parent=output_completeness,
        critical=False
    )

    # Check all activities have ratings
    all_have_ratings = (has_three_activities and
                       all(looks_like_rating(act.rating) for act in activities_info.activities[:3]))
    evaluator.add_custom_node(
        result=bool(all_have_ratings),
        id="output_activity_ratings",
        desc="All 3 activities have valid ratings provided",
        parent=output_completeness,
        critical=False
    )

    # Check correct number of activities (3)
    evaluator.add_custom_node(
        result=bool(has_three_activities),
        id="output_three_activities",
        desc="Exactly 3 activities are provided as requested",
        parent=output_completeness,
        critical=False
    )

    # Check precipitation probability is clearly stated
    precip_clearly_stated = (precip_info and precip_info.precipitation_probability and
                            precip_prob_valid)
    evaluator.add_custom_node(
        result=bool(precip_clearly_stated),
        id="output_precipitation_stated",
        desc="Precipitation probability value is clearly stated in the output",
        parent=output_completeness,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
