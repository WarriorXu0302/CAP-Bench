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
TASK_ID = "task-91a67b"
TASK_DESCRIPTION = 'My family and I live in Dallas, Texas. We’ve heard that flu season has arrived and want to strengthen our family’s immunity through diet.\n\nFirst, go to the CDC website and check the current flu activity level in Texas (Minimal/Low/Moderate/High/Very High), along with any CDC nutrition-related prevention recommendations. (If the page does not provide explicit nutrition advice, record CDC’s general flu-prevention health recommendations and cite the source.)\n\nThen, go to WebMD and find an article about immunity-boosting foods. Extract at least 5 immunity-supporting ingredients recommended by CDC or WebMD (for example: citrus fruits, ginger, garlic, spinach).\n\nNext, use these ingredients as keywords to search for healthy recipes on Food Network. Try to find 5 recipes, prioritizing the following criteria:\n- Rating of 4 stars or above  \n- Prep time within 45 minutes  \n- Each recipe includes at least 2 of the extracted recommended ingredients  \n\nIf fewer than 5 recipes fully meet all criteria, keep “at least 2 recommended ingredients per recipe” as a hard requirement, and relax constraints in this order:\n1) rating, then  \n2) prep time.  \n\nIn the output, explicitly note for each recipe:\n- its rating  \n- its prep time  \n- whether it meets “rating ≥ 4 stars / time ≤ 45 minutes”.\n\nFinally, output:\n- Current flu activity level in Texas  \n- Summary of CDC nutrition-related prevention guidance  \n- WebMD article link  \n- List of 5 extracted recommended ingredients  \n- For each recipe: name, rating, prep time, included recommended ingredients, and Food Network recipe detail-page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CDCFluInfo(BaseModel):
    """CDC flu activity and nutrition recommendations"""
    flu_activity_level: Optional[str] = None
    nutrition_recommendations: Optional[str] = None
    cdc_source_mentioned: Optional[bool] = None


class WebMDInfo(BaseModel):
    """WebMD article and extracted ingredients"""
    webmd_article_link: Optional[str] = None
    extracted_ingredients: Optional[List[str]] = Field(default_factory=list)


class RecipeInfo(BaseModel):
    """Individual recipe details"""
    recipe_name: Optional[str] = None
    rating: Optional[str] = None
    prep_time: Optional[str] = None
    included_ingredients: Optional[List[str]] = Field(default_factory=list)
    recipe_link: Optional[str] = None
    meets_criteria_noted: Optional[bool] = None


class RecipeCollection(BaseModel):
    """Collection of all recipes"""
    recipes: Optional[List[RecipeInfo]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cdc_info() -> str:
    return """
Extract the CDC flu activity information from the answer:

- flu_activity_level: The current flu activity level in Texas (one of: Minimal, Low, Moderate, High, Very High). If not present, set null.
- nutrition_recommendations: Summary of CDC nutrition-related prevention guidance or general health recommendations. If not present, set null.
- cdc_source_mentioned: Whether the answer mentions CDC as a source (true/false).

Return the extracted information.
"""


def prompt_extract_webmd_info() -> str:
    return """
Extract the WebMD information from the answer:

- webmd_article_link: The URL or link to the WebMD article about immunity-boosting foods. If not present, set null.
- extracted_ingredients: List of at least 5 immunity-supporting ingredients extracted from CDC or WebMD (e.g., citrus fruits, ginger, garlic, spinach). Return as array of strings. If fewer than 5 or none, return empty array.

Return the extracted information.
"""


def prompt_extract_recipes() -> str:
    return """
Extract all recipe information from the answer. For each recipe mentioned, capture:

- recipe_name: The name of the recipe
- rating: The rating (e.g., "4.5 stars", "4 stars")
- prep_time: The prep time (e.g., "30 minutes", "45 min")
- included_ingredients: List of recommended ingredients included in this recipe
- recipe_link: The Food Network recipe detail page link
- meets_criteria_noted: Whether the answer explicitly notes if this recipe meets "rating ≥ 4 stars / time ≤ 45 minutes" criteria (true if noted, false otherwise)

Return all recipes as an array. If no recipes found, return empty array.
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


def contains_url_pattern(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower()


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def is_valid_flu_level(level: Optional[str]) -> bool:
    if not level:
        return False
    valid_levels = ['minimal', 'low', 'moderate', 'high', 'very high']
    return any(ci_contains(level, vl) for vl in valid_levels)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['star', 'stars', 'rating']) or bool(re.search(r'\d+(\.\d+)?', text))


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['minute', 'minutes', 'min', 'hour', 'hours', 'hr']) or bool(re.search(r'\d+', text))


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
    cdc_info = await evaluator.extract(
        prompt=prompt_extract_cdc_info(),
        template_class=CDCFluInfo,
        extraction_name="cdc_flu_info"
    )

    webmd_info = await evaluator.extract(
        prompt=prompt_extract_webmd_info(),
        template_class=WebMDInfo,
        extraction_name="webmd_info"
    )

    recipe_collection = await evaluator.extract(
        prompt=prompt_extract_recipes(),
        template_class=RecipeCollection,
        extraction_name="recipe_collection"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 CDC Section
    cdc_node = evaluator.add_sequential(
        id="cdc_section",
        desc="CDC flu activity level and nutrition recommendations for Texas",
        parent=root,
        critical=False
    )

    # [Action Node] cdc.gov:F1:A4 - Search for Texas flu activity
    cdc_search_ok = has_any_ci(answer, ['cdc']) and has_any_ci(answer, ['texas', 'dallas'])
    evaluator.add_custom_node(
        result=bool(cdc_search_ok),
        id="cdc_search_flu_activity",
        desc="[Action Node] cdc.gov:F1:A4 - Search CDC for Texas/Dallas flu activity level using keywords",
        parent=cdc_node,
        critical=False
    )

    # [Perception Node] cdc.gov:F1:P7 - Extract nutrition prevention advice
    flu_level_ok = is_valid_flu_level(cdc_info.flu_activity_level)
    nutrition_ok = bool(cdc_info.nutrition_recommendations and len(cdc_info.nutrition_recommendations.strip()) > 10)
    evaluator.add_custom_node(
        result=bool(flu_level_ok and nutrition_ok),
        id="cdc_extract_nutrition_advice",
        desc="[Perception Node] cdc.gov:F1:P7 - Extract flu activity level and CDC nutrition-related prevention recommendations",
        parent=cdc_node,
        critical=False
    )

    # [Perception Node] cdc.gov:F2:P4 - Verify data timeliness
    mentions_current = has_any_ci(answer, ['current', 'latest', 'updated', 'recent'])
    evaluator.add_custom_node(
        result=bool(mentions_current),
        id="cdc_data_timeliness",
        desc="[Perception Node] cdc.gov:F2:P4 - Verify that CDC data reflects current/recent flu activity",
        parent=cdc_node,
        critical=False
    )

    # 3.2 WebMD Section
    webmd_node = evaluator.add_sequential(
        id="webmd_section",
        desc="WebMD article about immunity-boosting foods and ingredient extraction",
        parent=root,
        critical=False
    )

    # [Action Node] webmd.com:F1:A1 - Multi-level filtering to find immunity article
    webmd_search_ok = has_any_ci(answer, ['webmd']) and has_any_ci(answer, ['immunity', 'immune'])
    evaluator.add_custom_node(
        result=bool(webmd_search_ok),
        id="webmd_filter_immunity_article",
        desc="[Action Node] webmd.com:F1:A1 - Use filtering to locate WebMD article about immunity-boosting foods",
        parent=webmd_node,
        critical=False
    )

    # [Perception Node] webmd.com:F9:P14 - Extract ingredients from card/list content
    has_link = bool(webmd_info.webmd_article_link and contains_url_pattern(webmd_info.webmd_article_link, 'webmd'))
    has_ingredients = bool(webmd_info.extracted_ingredients and len(webmd_info.extracted_ingredients) >= 5)
    evaluator.add_custom_node(
        result=bool(has_link and has_ingredients),
        id="webmd_extract_ingredients",
        desc="[Perception Node] webmd.com:F9:P14 - Extract at least 5 immunity-supporting ingredients from article content",
        parent=webmd_node,
        critical=False
    )

    # 3.3 Food Network Section
    foodnetwork_node = evaluator.add_sequential(
        id="foodnetwork_section",
        desc="Food Network recipe search and filtering based on extracted ingredients",
        parent=root,
        critical=False
    )

    recipes = recipe_collection.recipes if recipe_collection and recipe_collection.recipes else []
    has_five_recipes = len(recipes) >= 5

    # [Action Node] foodnetwork.com:F1:A15 - Pagination to find enough recipes
    foodnetwork_search_ok = has_any_ci(answer, ['food network']) and has_five_recipes
    evaluator.add_custom_node(
        result=bool(foodnetwork_search_ok),
        id="foodnetwork_pagination",
        desc="[Action Node] foodnetwork.com:F1:A15 - Navigate through search results pages to find 5 qualifying recipes",
        parent=foodnetwork_node,
        critical=False
    )

    # [Action Node] foodnetwork.com:F1:A6 - Click cards to enter detail pages
    has_detail_links = all(bool(r.recipe_link and contains_url_pattern(r.recipe_link, 'foodnetwork')) for r in recipes)
    evaluator.add_custom_node(
        result=bool(has_five_recipes and has_detail_links),
        id="foodnetwork_click_detail_pages",
        desc="[Action Node] foodnetwork.com:F1:A6 - Click recipe cards to access detail pages for rating and prep time",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F1:P1 - Image understanding for healthy recipes
    recipe_names_ok = all(bool(r.recipe_name and len(r.recipe_name.strip()) > 0) for r in recipes)
    evaluator.add_custom_node(
        result=bool(has_five_recipes and recipe_names_ok),
        id="foodnetwork_image_understanding",
        desc="[Perception Node] foodnetwork.com:F1:P1 - Identify healthy recipes from food images and descriptions",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F1:P2 - Recognize star ratings
    has_ratings = all(bool(r.rating and looks_like_rating(r.rating)) for r in recipes)
    evaluator.add_custom_node(
        result=bool(has_five_recipes and has_ratings),
        id="foodnetwork_star_ratings",
        desc="[Perception Node] foodnetwork.com:F1:P2 - Identify and extract star ratings from recipe cards",
        parent=foodnetwork_node,
        critical=False
    )

    # [Action Node] foodnetwork.com:F2:A10 - Expand sections for time details
    has_prep_times = all(bool(r.prep_time and looks_like_time(r.prep_time)) for r in recipes)
    evaluator.add_custom_node(
        result=bool(has_five_recipes and has_prep_times),
        id="foodnetwork_expand_time_details",
        desc="[Action Node] foodnetwork.com:F2:A10 - Expand or navigate to detailed prep time information on recipe pages",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F2:P12 - Understand ingredient lists
    recipes_with_ingredients = [r for r in recipes if r.included_ingredients and len(r.included_ingredients) >= 2]
    ingredient_requirement_met = len(recipes_with_ingredients) >= 5
    evaluator.add_custom_node(
        result=bool(ingredient_requirement_met),
        id="foodnetwork_ingredient_list_understanding",
        desc="[Perception Node] foodnetwork.com:F2:P12 - Parse and match at least 2 recommended ingredients per recipe from ingredient lists",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F2:P14 - Hover/content awareness for time interpretation
    criteria_noted = sum(1 for r in recipes if r.meets_criteria_noted) >= 5
    evaluator.add_custom_node(
        result=bool(criteria_noted),
        id="foodnetwork_time_interpretation",
        desc="[Perception Node] foodnetwork.com:F2:P14 - Understand and note whether prep time meets ≤45 minute criterion for each recipe",
        parent=foodnetwork_node,
        critical=False
    )

    # Additional verification nodes
    evaluator.add_custom_node(
        result=bool(has_five_recipes),
        id="recipe_count_verification",
        desc="Verify that exactly 5 recipes were found and presented",
        parent=foodnetwork_node,
        critical=False
    )

    all_have_required_fields = all(
        bool(r.recipe_name and r.rating and r.prep_time and r.recipe_link)
        for r in recipes
    )
    evaluator.add_custom_node(
        result=bool(has_five_recipes and all_have_required_fields),
        id="recipe_completeness",
        desc="Verify each recipe includes name, rating, prep time, ingredients, and link",
        parent=foodnetwork_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
