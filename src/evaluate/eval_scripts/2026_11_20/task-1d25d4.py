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
TASK_ID = "task-1d25d4"
TASK_DESCRIPTION = "I am currently in Portland, OR, but I suffer from allergic rhinitis and am debating whether or not to visit the Portland Japanese Garden. Please help me make a decision.\n\nFirst, go to AccuWeather and check the allergen situation for Portland for the next three days. Do not just look at the general weather; I need the specific levels (e.g., Low/High) for 'Tree Pollen' and 'Mold'. Please also try to find the dedicated Health or Allergy page on their site.\n\nThen, go to Google Maps, search for Portland Japanese Garden, click into the reviews section, and scroll through recent comments to see if any visitors have complained about pollen allergies or mentioned that plants are currently in full bloom.\n\nSynthesize the information from both sources and tell me which of the next three days would be most suitable for a visit, or if I should just avoid going altogether."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AllergenForecast(BaseModel):
    """Allergen forecast data extracted from the answer for Portland next 3 days"""
    tree_pollen_levels: Optional[List[str]] = Field(default_factory=list)
    mold_levels: Optional[List[str]] = Field(default_factory=list)
    forecast_days: Optional[List[str]] = Field(default_factory=list)


class ReviewsInfo(BaseModel):
    """Information extracted from Google Maps reviews of Portland Japanese Garden"""
    mentions_pollen_allergy: Optional[bool] = None
    mentions_bloom_or_flowering: Optional[bool] = None
    review_excerpts: Optional[List[str]] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Final recommendation extracted from the answer"""
    recommended_day: Optional[str] = None
    avoid_visit: Optional[bool] = None
    reasoning: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_allergen_forecast() -> str:
    return """
Extract the allergen forecast information for Portland from AccuWeather for the next three days.

Return:
- tree_pollen_levels: list of Tree Pollen levels for each of the next 3 days (e.g., ["Low", "Moderate", "High"]). If not present, return empty list.
- mold_levels: list of Mold levels for each of the next 3 days. If not present, return empty list.
- forecast_days: list of day labels mentioned (e.g., ["Today", "Tomorrow", "Wednesday"]). If not present, return empty list.

Extract exactly as stated in the answer. If any field is missing, set it to empty list or null.
"""


def prompt_extract_reviews_info() -> str:
    return """
From the answer, extract information about Google Maps reviews for Portland Japanese Garden:

- mentions_pollen_allergy: true if any reviews mention pollen allergies or allergy complaints, false otherwise.
- mentions_bloom_or_flowering: true if any reviews mention plants in full bloom, flowering, or blooming season, false otherwise.
- review_excerpts: list of any relevant review text excerpts mentioned in the answer (if any). If not present, return empty list.

If information is missing, set booleans to null and list to empty.
"""


def prompt_extract_recommendation() -> str:
    return """
Extract the final recommendation from the answer:

- recommended_day: which specific day is recommended for the visit (e.g., "Tomorrow", "Day 2", "Wednesday"), or null if none recommended.
- avoid_visit: true if the answer recommends avoiding the visit altogether, false otherwise, null if unclear.
- reasoning: brief summary of the reasoning provided for the recommendation. If not present, set to null.
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


def looks_like_allergen_level(text: Optional[str]) -> bool:
    if not text:
        return False
    levels = ['low', 'moderate', 'medium', 'high', 'very high', 'extremely high']
    return any(level in text.lower() for level in levels)


def count_valid_levels(levels: Optional[List[str]]) -> int:
    if not levels:
        return 0
    return sum(1 for level in levels if looks_like_allergen_level(level))


def mentions_accuweather_health_or_allergy(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['health', 'allergy', 'allergen', 'pollen forecast'])


def mentions_tree_pollen(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['tree pollen', 'tree', 'pollen'])


def mentions_mold(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['mold', 'mould'])


def mentions_three_days(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['three days', '3 days', 'next three', 'next 3'])


def mentions_google_maps(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['google maps', 'maps'])


def mentions_portland_japanese_garden(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['portland japanese garden', 'japanese garden'])


def mentions_reviews_section(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['review', 'reviews', 'comment'])


def mentions_scrolling_reviews(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['scroll', 'scrolled', 'scrolling', 'went through', 'looked through', 'checked multiple'])


def has_synthesis_and_recommendation(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    has_recommendation_keywords = has_any_ci(answer_text, ['recommend', 'suggest', 'best day', 'avoid', 'should visit', 'suitable'])
    return has_recommendation_keywords


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
    allergen_forecast = await evaluator.extract(
        prompt=prompt_extract_allergen_forecast(),
        template_class=AllergenForecast,
        extraction_name="allergen_forecast"
    )

    reviews_info = await evaluator.extract(
        prompt=prompt_extract_reviews_info(),
        template_class=ReviewsInfo,
        extraction_name="reviews_info"
    )

    recommendation = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=Recommendation,
        extraction_name="recommendation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 AccuWeather allergen forecast section
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather allergen forecast for Portland - next 3 days",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F7:A2 - Navigate to Health/Allergy page
    accuweather_navigation_ok = (
        has_any_ci(answer, ['accuweather']) and
        mentions_accuweather_health_or_allergy(answer)
    )
    evaluator.add_custom_node(
        result=bool(accuweather_navigation_ok),
        id="accuweather_action_health_page",
        desc="[Action Node] accuweather.com:F7:A2 - Navigate to AccuWeather and find the dedicated Health or Allergy page for Portland",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F7:P10 - Extract specific allergen levels (Tree Pollen and Mold)
    tree_pollen_count = count_valid_levels(allergen_forecast.tree_pollen_levels)
    mold_count = count_valid_levels(allergen_forecast.mold_levels)

    tree_pollen_mentioned = mentions_tree_pollen(answer)
    mold_mentioned = mentions_mold(answer)
    three_days_ok = mentions_three_days(answer) or len(allergen_forecast.forecast_days) >= 3

    allergen_extraction_ok = (
        tree_pollen_count >= 1 and
        mold_count >= 1 and
        tree_pollen_mentioned and
        mold_mentioned and
        three_days_ok
    )

    evaluator.add_custom_node(
        result=bool(allergen_extraction_ok),
        id="accuweather_perception_allergen_levels",
        desc="[Perception Node] accuweather.com:F7:P10 - Extract specific levels for Tree Pollen and Mold for the next 3 days",
        parent=accuweather_node,
        critical=False
    )

    # Additional lenient check: mentions Portland location
    mentions_portland = has_any_ci(answer, ['portland'])
    evaluator.add_custom_node(
        result=bool(mentions_portland),
        id="accuweather_mentions_portland",
        desc="Mentions Portland as the location for allergen forecast",
        parent=accuweather_node,
        critical=False
    )

    # 3.2 Google Maps reviews section
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps reviews for Portland Japanese Garden",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F8:A6 - Navigate to reviews section
    googlemaps_navigation_ok = (
        mentions_google_maps(answer) and
        mentions_portland_japanese_garden(answer) and
        mentions_reviews_section(answer)
    )
    evaluator.add_custom_node(
        result=bool(googlemaps_navigation_ok),
        id="googlemaps_action_reviews_tab",
        desc="[Action Node] maps.google.com:F8:A6 - Navigate to Google Maps, search for Portland Japanese Garden, and access the reviews section",
        parent=googlemaps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F8:P17 - Scroll through and extract review information
    scrolling_mentioned = mentions_scrolling_reviews(answer)
    found_allergy_info = reviews_info.mentions_pollen_allergy if reviews_info.mentions_pollen_allergy is not None else False
    found_bloom_info = reviews_info.mentions_bloom_or_flowering if reviews_info.mentions_bloom_or_flowering is not None else False
    has_review_content = len(reviews_info.review_excerpts) > 0 if reviews_info.review_excerpts else False

    reviews_extraction_ok = scrolling_mentioned and (found_allergy_info or found_bloom_info or has_review_content)

    evaluator.add_custom_node(
        result=bool(reviews_extraction_ok),
        id="googlemaps_perception_review_content",
        desc="[Perception Node] maps.google.com:F8:P17 - Scroll through reviews and extract information about pollen allergies or blooming plants",
        parent=googlemaps_node,
        critical=False
    )

    # Additional check: mentions recent reviews or visitor comments
    mentions_recent = has_any_ci(answer, ['recent', 'visitor', 'comment'])
    evaluator.add_custom_node(
        result=bool(mentions_recent),
        id="googlemaps_mentions_recent_reviews",
        desc="Mentions checking recent visitor reviews or comments",
        parent=googlemaps_node,
        critical=False
    )

    # 3.3 Synthesis and recommendation
    synthesis_node = evaluator.add_sequential(
        id="synthesis_section",
        desc="Synthesis of information and final recommendation",
        parent=root,
        critical=False
    )

    # Check if both sources are mentioned in synthesis
    mentions_both_sources = (
        has_any_ci(answer, ['accuweather']) and
        has_any_ci(answer, ['google maps', 'maps', 'review'])
    )
    evaluator.add_custom_node(
        result=bool(mentions_both_sources),
        id="synthesis_mentions_both_sources",
        desc="Synthesis references information from both AccuWeather and Google Maps",
        parent=synthesis_node,
        critical=False
    )

    # Check if final recommendation is present
    has_recommendation = has_synthesis_and_recommendation(answer)
    recommendation_specific = (
        recommendation.recommended_day is not None or
        recommendation.avoid_visit is not None
    )

    evaluator.add_custom_node(
        result=bool(has_recommendation and recommendation_specific),
        id="synthesis_provides_recommendation",
        desc="Provides a clear recommendation on which day to visit or whether to avoid the visit",
        parent=synthesis_node,
        critical=False
    )

    # Check if reasoning is provided
    has_reasoning = bool(recommendation.reasoning and len(recommendation.reasoning.strip()) > 20)
    evaluator.add_custom_node(
        result=bool(has_reasoning),
        id="synthesis_provides_reasoning",
        desc="Provides reasoning that synthesizes allergen data and review insights",
        parent=synthesis_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
