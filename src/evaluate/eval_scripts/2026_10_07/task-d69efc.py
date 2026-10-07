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
TASK_ID = "task-d69efc"
TASK_DESCRIPTION = "I'm planning a family road trip to Yellowstone National Park for a week, starting next Monday. I need to rent a large vehicle, but my child is prone to motion sickness, so I prioritize suspension comfort.\n\nFirst, go to Skyscanner and search for car rental information for pickup and return at Bozeman (BZN) airport, setting the pickup time to 10:00 AM. Filter for 'Large' or 'Premium' SUVs, and note down three representative car models displayed in the search results (e.g., the 'Jeep' from 'Jeep Grand Cherokee or similar').\n\nNext, visit the Edmunds website. Use its car finder tool to locate the review pages for the latest models (2024 or 2025) of these three cars. Pay close attention to the 'Comfort' or 'Ride Quality' sections of the reviews to identify the model with the softest suspension and least likelihood of causing motion sickness.\n\nFinally, go to the Yellowstone National Park Forum on TripAdvisor. Search for 'road construction' and check if there have been any posts in the last month mentioning major road construction on main roads. Based on this, make an overall judgment on whether the chosen car model would be suitable for potential road conditions."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SkyscannerCarModels(BaseModel):
    """Car models extracted from Skyscanner search results"""
    car_model_1: Optional[str] = None
    car_model_2: Optional[str] = None
    car_model_3: Optional[str] = None


class EdmundsComfortReviews(BaseModel):
    """Comfort/ride quality information from Edmunds reviews"""
    car_1_comfort_info: Optional[str] = None
    car_2_comfort_info: Optional[str] = None
    car_3_comfort_info: Optional[str] = None
    recommended_model: Optional[str] = None


class TripAdvisorRoadInfo(BaseModel):
    """Road construction information from TripAdvisor forum"""
    recent_road_construction_mentioned: Optional[bool] = None
    construction_details: Optional[str] = None
    final_suitability_judgment: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_skyscanner_models() -> str:
    return """
Extract the three representative car models the user noted from Skyscanner's car rental search results for Bozeman (BZN) airport.

Return:
- car_model_1: the first car model name (e.g., "Jeep" from "Jeep Grand Cherokee or similar")
- car_model_2: the second car model name
- car_model_3: the third car model name

If any field is missing in the answer, set it to null.
"""


def prompt_extract_edmunds_comfort() -> str:
    return """
From the answer, extract the comfort/ride quality information the user found on Edmunds for the three car models, and identify which model they recommend for the softest suspension.

Return:
- car_1_comfort_info: comfort or ride quality details for the first model
- car_2_comfort_info: comfort or ride quality details for the second model
- car_3_comfort_info: comfort or ride quality details for the third model
- recommended_model: which model the user identified as having the softest suspension/best for motion sickness

If any field is missing, set it to null.
"""


def prompt_extract_tripadvisor_road_info() -> str:
    return """
From the answer, extract the TripAdvisor Yellowstone forum information about road construction and the final suitability judgment.

Return:
- recent_road_construction_mentioned: true if recent (last month) road construction posts were found, false if not mentioned or explicitly none found, null if unclear
- construction_details: any specific details about road construction mentioned
- final_suitability_judgment: the user's overall judgment on whether the chosen car model is suitable for the road conditions

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


def count_non_none(*values) -> int:
    return sum(1 for v in values if v is not None and str(v).strip())


def looks_like_car_brand(text: Optional[str]) -> bool:
    if not text:
        return False
    common_brands = ['jeep', 'toyota', 'ford', 'chevrolet', 'chevy', 'gmc', 'honda',
                     'nissan', 'mazda', 'subaru', 'hyundai', 'kia', 'dodge', 'ram',
                     'volkswagen', 'vw', 'audi', 'bmw', 'mercedes', 'lexus', 'cadillac',
                     'buick', 'volvo', 'land rover', 'range rover', 'lincoln', 'acura']
    return has_any_ci(text, common_brands) or len(text.strip()) > 2


def mentions_time_10am(answer: str) -> bool:
    patterns = [r'\b10\s*:?\s*00?\s*(a\.?m\.?)\b', r'\b10\s*a\.?m\.?\b', r'\b10:00\b']
    return any(re.search(p, answer.lower()) for p in patterns)


def mentions_comfort_or_ride_quality(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['comfort', 'ride quality', 'suspension', 'smooth', 'soft', 'motion sickness',
                'ride comfort', 'comfortable', 'harsh', 'firm', 'plush', 'absorb']
    return has_any_ci(text, keywords)


def mentions_recent_timeframe(answer: str) -> bool:
    return has_any_ci(answer, ['last month', 'recent', 'recently', 'past month', 'within a month'])


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
        strategy=AggregationStrategy.SEQUENTIAL,
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
    skyscanner_models = await evaluator.extract(
        prompt=prompt_extract_skyscanner_models(),
        template_class=SkyscannerCarModels,
        extraction_name="skyscanner_car_models"
    )

    edmunds_comfort = await evaluator.extract(
        prompt=prompt_extract_edmunds_comfort(),
        template_class=EdmundsComfortReviews,
        extraction_name="edmunds_comfort_reviews"
    )

    tripadvisor_info = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_road_info(),
        template_class=TripAdvisorRoadInfo,
        extraction_name="tripadvisor_road_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner car rental search at Bozeman (BZN) airport",
        parent=root,
        critical=False
    )

    # Check if Skyscanner was mentioned and BZN/Bozeman context
    skyscanner_mentioned = has_any_ci(answer, ['skyscanner'])
    bzn_mentioned = has_any_ci(answer, ['bzn', 'bozeman'])

    evaluator.add_custom_node(
        result=bool(skyscanner_mentioned and bzn_mentioned),
        id="skyscanner_basic_navigation",
        desc="Navigate to Skyscanner and search for car rentals at Bozeman (BZN) airport",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F3:A24 - Time picker operation (10:00 AM)
    time_10am_set = mentions_time_10am(answer)
    evaluator.add_custom_node(
        result=bool(time_10am_set),
        id="skyscanner_time_picker",
        desc="[Action Node] skyscanner.com:F3:A24 - Set pickup time to 10:00 AM using time selector",
        parent=skyscanner_node,
        critical=False
    )

    # Check for Large/Premium SUV filtering
    suv_filter_mentioned = has_any_ci(answer, ['large', 'premium', 'suv'])
    evaluator.add_custom_node(
        result=bool(suv_filter_mentioned),
        id="skyscanner_filter_suv",
        desc="Filter for 'Large' or 'Premium' SUVs in search results",
        parent=skyscanner_node,
        critical=False
    )

    # Check three car models extracted
    models_count = count_non_none(skyscanner_models.car_model_1,
                                   skyscanner_models.car_model_2,
                                   skyscanner_models.car_model_3)

    model_1_ok = looks_like_car_brand(skyscanner_models.car_model_1)
    model_2_ok = looks_like_car_brand(skyscanner_models.car_model_2)
    model_3_ok = looks_like_car_brand(skyscanner_models.car_model_3)

    evaluator.add_custom_node(
        result=bool(models_count >= 3 and model_1_ok and model_2_ok and model_3_ok),
        id="skyscanner_three_models_noted",
        desc="Note down three representative car models from search results",
        parent=skyscanner_node,
        critical=False
    )

    # 3.2 Edmunds section
    edmunds_node = evaluator.add_sequential(
        id="edmunds_section",
        desc="Edmunds website car reviews for comfort and ride quality",
        parent=root,
        critical=False
    )

    # Check if Edmunds was mentioned
    edmunds_mentioned = has_any_ci(answer, ['edmunds'])
    evaluator.add_custom_node(
        result=bool(edmunds_mentioned),
        id="edmunds_basic_navigation",
        desc="Navigate to Edmunds website",
        parent=edmunds_node,
        critical=False
    )

    # [Action Node] edmunds.com:F3:A13 - Cascading selector (Make->Model->Year)
    car_finder_mentioned = has_any_ci(answer, ['car finder', 'find', 'search', 'locate'])
    year_mentioned = has_any_ci(answer, ['2024', '2025'])

    evaluator.add_custom_node(
        result=bool(car_finder_mentioned and year_mentioned),
        id="edmunds_cascading_selector",
        desc="[Action Node] edmunds.com:F3:A13 - Use car finder tool with cascading Make->Model->Year selector to locate 2024/2025 models",
        parent=edmunds_node,
        critical=False
    )

    # Check comfort/ride quality review reading
    comfort_1_ok = mentions_comfort_or_ride_quality(edmunds_comfort.car_1_comfort_info)
    comfort_2_ok = mentions_comfort_or_ride_quality(edmunds_comfort.car_2_comfort_info)
    comfort_3_ok = mentions_comfort_or_ride_quality(edmunds_comfort.car_3_comfort_info)
    comfort_mentioned = has_any_ci(answer, ['comfort', 'ride quality', 'suspension'])

    evaluator.add_custom_node(
        result=bool((comfort_1_ok or comfort_2_ok or comfort_3_ok) and comfort_mentioned),
        id="edmunds_comfort_sections_reviewed",
        desc="Review 'Comfort' or 'Ride Quality' sections for the three car models",
        parent=edmunds_node,
        critical=False
    )

    # Check if a recommendation was made
    recommended_model_present = bool(edmunds_comfort.recommended_model and
                                     edmunds_comfort.recommended_model.strip())
    softest_suspension_mentioned = has_any_ci(answer, ['softest', 'soft suspension', 'best for motion sickness',
                                                        'least motion sickness', 'most comfortable', 'smoothest'])

    evaluator.add_custom_node(
        result=bool(recommended_model_present and softest_suspension_mentioned),
        id="edmunds_recommend_best_model",
        desc="Identify the model with the softest suspension and least likelihood of causing motion sickness",
        parent=edmunds_node,
        critical=False
    )

    # 3.3 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor Yellowstone National Park Forum for road construction information",
        parent=root,
        critical=False
    )

    # Check TripAdvisor and forum navigation
    tripadvisor_mentioned = has_any_ci(answer, ['tripadvisor'])
    yellowstone_forum_mentioned = has_any_ci(answer, ['yellowstone', 'forum'])

    evaluator.add_custom_node(
        result=bool(tripadvisor_mentioned and yellowstone_forum_mentioned),
        id="tripadvisor_forum_navigation",
        desc="Navigate to TripAdvisor Yellowstone National Park Forum",
        parent=tripadvisor_node,
        critical=False
    )

    # Check road construction search
    road_construction_search = has_any_ci(answer, ['road construction'])
    evaluator.add_custom_node(
        result=bool(road_construction_search),
        id="tripadvisor_search_road_construction",
        desc="Search for 'road construction' in the forum",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F7:P12 - List filtering by timestamp (last month)
    recent_timeframe_checked = mentions_recent_timeframe(answer)
    road_info_present = tripadvisor_info.recent_road_construction_mentioned is not None

    evaluator.add_custom_node(
        result=bool(recent_timeframe_checked and road_info_present),
        id="tripadvisor_filter_recent_posts",
        desc="[Perception Node] tripadvisor.com:F7:P12 - Check forum posts from the last month by recognizing and filtering timestamps",
        parent=tripadvisor_node,
        critical=False
    )

    # Check major road construction identification
    major_roads_mentioned = has_any_ci(answer, ['main road', 'major road', 'highway', 'primary road'])
    construction_details_present = bool(tripadvisor_info.construction_details and
                                       tripadvisor_info.construction_details.strip())

    evaluator.add_custom_node(
        result=bool(road_info_present and (major_roads_mentioned or construction_details_present)),
        id="tripadvisor_major_road_construction_check",
        desc="Identify if recent posts mention major road construction on main roads",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.4 Final judgment
    final_judgment_present = bool(tripadvisor_info.final_suitability_judgment and
                                  tripadvisor_info.final_suitability_judgment.strip())
    suitability_mentioned = has_any_ci(answer, ['suitable', 'suitability', 'appropriate',
                                                 'work well', 'good choice', 'handle', 'recommend'])

    evaluator.add_custom_node(
        result=bool(final_judgment_present and suitability_mentioned),
        id="overall_suitability_judgment",
        desc="Make overall judgment on whether the chosen car model is suitable for the road conditions",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
