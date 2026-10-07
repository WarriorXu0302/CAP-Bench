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
TASK_ID = "task-be5194"
TASK_DESCRIPTION = "I want to prepare some immunity-boosting winter recipes for my family. First, please go to the CDC website to check California's current influenza activity level (Low/Moderate/High/Very High). Then, search 'boost immune system foods winter' on Healthline or WebMD to identify recommended immunity-boosting ingredients (e.g., foods rich in Vitamin C, zinc, probiotics). List at least 5 recommended ingredients. Next, go to the EWG website and search for each of these ingredients individually (e.g., oranges, ginger, garlic, yogurt). Check their Skin Deep safety scores, retaining only low-risk ingredients with a score of 3 or below. Then, check San Francisco's weather forecast for this week on AccuWeather. If the minimum temperature is below 10°C for more than 3 days, prioritize hot soup recipes; otherwise, consider salads or room-temperature dishes. Finally, go to AllRecipes or Food Network and search for recipes using the filtered safe ingredients as keywords (e.g., search 'ginger+garlic+chicken'). Filter for 3-5 suitable recipes (rating 4 stars or higher, cooking time within 60 minutes). Record the recipe name, rating, cooking time, and main ingredients required.\n\nOutput: California's current influenza activity level, CDC query page link, 5 recommended immunity-boosting ingredients and their EWG scores, EWG page links for each ingredient, San Francisco's weekly count of days with minimum temperatures below 10°C, AccuWeather weather page link, recommended recipe type based on weather (hot soup/salad/room-temperature dish), names, ratings, cooking times, main ingredient lists, and recipe detail page links for 3-5 recipes."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CDCFluInfo(BaseModel):
    """CDC flu activity level for California"""
    activity_level: Optional[str] = None
    cdc_page_url: Optional[str] = None


class ImmunityIngredients(BaseModel):
    """Immunity-boosting ingredients from Healthline/WebMD"""
    ingredients: Optional[List[str]] = Field(default_factory=list)


class EWGScores(BaseModel):
    """EWG safety scores for ingredients"""
    ingredient_scores: Optional[Dict[str, Any]] = Field(default_factory=dict)
    ewg_page_urls: Optional[List[str]] = Field(default_factory=list)


class WeatherInfo(BaseModel):
    """San Francisco weather forecast"""
    days_below_10c: Optional[int] = None
    accuweather_url: Optional[str] = None


class RecipeInfo(BaseModel):
    """Recipe details"""
    recipe_type_recommended: Optional[str] = None
    recipes: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cdc_flu() -> str:
    return """
Extract California's current influenza activity level from the answer. Look for mentions of CDC website and California flu activity.

Return:
- activity_level: the exact activity level stated (Low/Moderate/High/Very High). If not present, set null.
- cdc_page_url: any CDC URL mentioned for the flu data. If not present, set null.
"""


def prompt_extract_immunity_ingredients() -> str:
    return """
Extract the list of immunity-boosting ingredients mentioned in the answer from Healthline or WebMD search results.

Return:
- ingredients: a list of ingredient names (e.g., ["oranges", "ginger", "garlic"]). Should contain at least 5 items if available. If none found, return empty list.
"""


def prompt_extract_ewg_scores() -> str:
    return """
Extract EWG Skin Deep safety scores for the ingredients from the answer.

Return:
- ingredient_scores: a dictionary mapping ingredient names to their safety scores (e.g., {"ginger": 2, "garlic": 1}). Include only ingredients with scores mentioned. If none, return empty dict.
- ewg_page_urls: list of EWG page URLs mentioned for ingredient lookups. If none, return empty list.
"""


def prompt_extract_weather_info() -> str:
    return """
Extract San Francisco weather forecast information from the answer.

Return:
- days_below_10c: the number of days this week with minimum temperature below 10°C. If not mentioned, set null.
- accuweather_url: any AccuWeather URL mentioned for San Francisco weather. If not present, set null.
"""


def prompt_extract_recipe_info() -> str:
    return """
Extract recipe information from the answer.

Return:
- recipe_type_recommended: the type of recipe recommended based on weather (e.g., "hot soup", "salad", "room-temperature dish"). If not mentioned, set null.
- recipes: a list of recipe dictionaries, each containing name, rating, cooking_time, ingredients, and url fields. Should contain 3-5 recipes if available. If none found, return empty list.
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
    valid_levels = ['low', 'moderate', 'high', 'very high']
    return any(ci_contains(level, vl) for vl in valid_levels)


def count_ingredients(ingredients: Optional[List[str]]) -> int:
    if not ingredients:
        return 0
    return len([i for i in ingredients if i and i.strip()])


def has_scores_below_threshold(scores: Optional[Dict[str, Any]], threshold: int = 3) -> bool:
    if not scores:
        return False
    for val in scores.values():
        try:
            score = float(val) if not isinstance(val, (int, float)) else val
            if score <= threshold:
                return True
        except Exception:
            continue
    return False


def count_recipes(recipes: Optional[List[Dict[str, Any]]]) -> int:
    if not recipes:
        return 0
    return len(recipes)


def all_recipes_meet_criteria(recipes: Optional[List[Dict[str, Any]]]) -> bool:
    if not recipes:
        return False
    for recipe in recipes:
        rating = recipe.get('rating')
        cooking_time = recipe.get('cooking_time')

        if rating is not None:
            try:
                rating_val = extract_number(str(rating))
                if rating_val is not None and rating_val < 4:
                    return False
            except Exception:
                pass

        if cooking_time is not None:
            try:
                time_val = extract_number(str(cooking_time))
                if time_val is not None and time_val > 60:
                    return False
            except Exception:
                pass

    return True


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
        prompt=prompt_extract_cdc_flu(),
        template_class=CDCFluInfo,
        extraction_name="cdc_flu_info"
    )

    ingredients_info = await evaluator.extract(
        prompt=prompt_extract_immunity_ingredients(),
        template_class=ImmunityIngredients,
        extraction_name="immunity_ingredients"
    )

    ewg_info = await evaluator.extract(
        prompt=prompt_extract_ewg_scores(),
        template_class=EWGScores,
        extraction_name="ewg_scores"
    )

    weather_info = await evaluator.extract(
        prompt=prompt_extract_weather_info(),
        template_class=WeatherInfo,
        extraction_name="weather_info"
    )

    recipe_info = await evaluator.extract(
        prompt=prompt_extract_recipe_info(),
        template_class=RecipeInfo,
        extraction_name="recipe_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 CDC Section
    cdc_node = evaluator.add_sequential(
        id="cdc_section",
        desc="CDC California influenza activity level",
        parent=root,
        critical=False
    )

    cdc_search_ok = has_any_ci(answer, ['cdc']) and has_any_ci(answer, ['california', 'ca']) and has_any_ci(answer, ['flu', 'influenza'])
    evaluator.add_custom_node(
        result=bool(cdc_search_ok),
        id="cdc_search_action",
        desc="[Action Node] cdc.gov:F1:A4 - Search CDC website for California influenza activity",
        parent=cdc_node,
        critical=False
    )

    flu_level_valid = is_valid_flu_level(cdc_info.activity_level)
    evaluator.add_custom_node(
        result=bool(flu_level_valid),
        id="cdc_flu_level_perception",
        desc="[Perception Node] cdc.gov:F2:P3 - Extract valid flu activity level (Low/Moderate/High/Very High)",
        parent=cdc_node,
        critical=False
    )

    cdc_url_present = bool(cdc_info.cdc_page_url and cdc_info.cdc_page_url.strip())
    evaluator.add_custom_node(
        result=bool(cdc_url_present),
        id="cdc_url_included",
        desc="CDC query page link included in output",
        parent=cdc_node,
        critical=False
    )

    # 3.2 Healthline/WebMD Section
    healthline_node = evaluator.add_sequential(
        id="healthline_section",
        desc="Healthline/WebMD immunity-boosting ingredients search",
        parent=root,
        critical=False
    )

    healthline_search_ok = (has_any_ci(answer, ['healthline', 'webmd']) and
                           has_any_ci(answer, ['immune', 'immunity']) and
                           has_any_ci(answer, ['winter', 'food']))
    evaluator.add_custom_node(
        result=bool(healthline_search_ok),
        id="healthline_navigation_action",
        desc="[Action Node] healthline.com:F1:A1 - Navigate and search on Healthline/WebMD for immunity foods",
        parent=healthline_node,
        critical=False
    )

    ingredient_count = count_ingredients(ingredients_info.ingredients)
    ingredients_sufficient = ingredient_count >= 5
    evaluator.add_custom_node(
        result=bool(ingredients_sufficient),
        id="healthline_ingredient_perception",
        desc="[Perception Node] healthline.com:F1:P1 - Identify at least 5 immunity-boosting ingredients from medical-reviewed articles",
        parent=healthline_node,
        critical=False
    )

    medical_context_ok = has_any_ci(answer, ['vitamin', 'zinc', 'probiotic']) or has_any_ci(answer, ['vitamin c', 'immune system'])
    evaluator.add_custom_node(
        result=bool(medical_context_ok),
        id="healthline_medical_context",
        desc="[Perception Node] healthline.com:F1:P3 - Recognize nutritional/medical context of recommended ingredients",
        parent=healthline_node,
        critical=False
    )

    # 3.3 EWG Section
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG Skin Deep safety score verification",
        parent=root,
        critical=False
    )

    ewg_search_ok = has_any_ci(answer, ['ewg']) and (ingredient_count > 0)
    evaluator.add_custom_node(
        result=bool(ewg_search_ok),
        id="ewg_search_action",
        desc="[Action Node] ewg.org:F1:A1 - Search EWG for each ingredient individually",
        parent=ewg_node,
        critical=False
    )

    ewg_category_ok = has_any_ci(answer, ['skin deep', 'safety score', 'score'])
    evaluator.add_custom_node(
        result=bool(ewg_category_ok),
        id="ewg_category_navigation",
        desc="[Action Node] ewg.org:F2:A2 - Navigate EWG product categories to find ingredients",
        parent=ewg_node,
        critical=False
    )

    has_scores = bool(ewg_info.ingredient_scores and len(ewg_info.ingredient_scores) > 0)
    scores_below_threshold = has_scores_below_threshold(ewg_info.ingredient_scores, 3)
    evaluator.add_custom_node(
        result=bool(has_scores and scores_below_threshold),
        id="ewg_score_perception",
        desc="[Perception Node] ewg.org:F2:P3 - Extract safety scores and filter ingredients with score ≤3",
        parent=ewg_node,
        critical=False
    )

    ewg_urls_present = bool(ewg_info.ewg_page_urls and len(ewg_info.ewg_page_urls) > 0)
    evaluator.add_custom_node(
        result=bool(ewg_urls_present),
        id="ewg_urls_included",
        desc="EWG page links for ingredients included in output",
        parent=ewg_node,
        critical=False
    )

    # 3.4 AccuWeather Section
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather San Francisco weekly forecast",
        parent=root,
        critical=False
    )

    accuweather_location_ok = (has_any_ci(answer, ['accuweather']) and
                              has_any_ci(answer, ['san francisco', 'sf']))
    evaluator.add_custom_node(
        result=bool(accuweather_location_ok),
        id="accuweather_location_action",
        desc="[Action Node] accuweather.com:F1:A1 - Select San Francisco location on AccuWeather",
        parent=accuweather_node,
        critical=False
    )

    accuweather_tab_ok = has_any_ci(answer, ['week', 'weekly', 'daily', 'forecast'])
    evaluator.add_custom_node(
        result=bool(accuweather_tab_ok),
        id="accuweather_tab_action",
        desc="[Action Node] accuweather.com:F2:A2 - Switch to weekly/daily forecast tab",
        parent=accuweather_node,
        critical=False
    )

    days_below_10c_present = weather_info.days_below_10c is not None
    evaluator.add_custom_node(
        result=bool(days_below_10c_present),
        id="accuweather_temp_perception",
        desc="[Perception Node] accuweather.com:F2:P2 - Parse daily minimum temperatures and count days below 10°C",
        parent=accuweather_node,
        critical=False
    )

    weather_trend_ok = (days_below_10c_present and
                       has_any_ci(answer, ['temperature', 'temp', '10', 'degree', '°c', 'celsius']))
    evaluator.add_custom_node(
        result=bool(weather_trend_ok),
        id="accuweather_trend_perception",
        desc="[Perception Node] accuweather.com:F2:P3 - Identify multi-day temperature trend for recipe decision",
        parent=accuweather_node,
        critical=False
    )

    accuweather_url_present = bool(weather_info.accuweather_url and weather_info.accuweather_url.strip())
    evaluator.add_custom_node(
        result=bool(accuweather_url_present),
        id="accuweather_url_included",
        desc="AccuWeather weather page link included in output",
        parent=accuweather_node,
        critical=False
    )

    # Recipe type recommendation logic check
    recipe_type_present = bool(recipe_info.recipe_type_recommended and recipe_info.recipe_type_recommended.strip())
    if days_below_10c_present and weather_info.days_below_10c is not None:
        if weather_info.days_below_10c >= 3:
            recipe_logic_ok = has_any_ci(recipe_info.recipe_type_recommended, ['soup', 'hot'])
        else:
            recipe_logic_ok = has_any_ci(recipe_info.recipe_type_recommended, ['salad', 'room-temperature', 'room temperature'])
    else:
        recipe_logic_ok = recipe_type_present

    evaluator.add_custom_node(
        result=bool(recipe_type_present and recipe_logic_ok),
        id="recipe_type_logic",
        desc="Recipe type recommendation matches weather conditions (≥3 cold days → hot soup, else → salad/room-temp)",
        parent=accuweather_node,
        critical=False
    )

    # 3.5 AllRecipes/Food Network Section
    recipe_node = evaluator.add_sequential(
        id="recipe_section",
        desc="AllRecipes/Food Network recipe search and filtering",
        parent=root,
        critical=False
    )

    recipe_site_ok = has_any_ci(answer, ['allrecipes', 'food network'])
    evaluator.add_custom_node(
        result=bool(recipe_site_ok),
        id="recipe_category_navigation",
        desc="[Action Node] allrecipes.com:F1:A1 - Navigate recipe categories on AllRecipes/Food Network",
        parent=recipe_node,
        critical=False
    )

    recipe_search_ok = recipe_site_ok and (has_scores or ingredient_count > 0)
    evaluator.add_custom_node(
        result=bool(recipe_search_ok),
        id="recipe_keyword_search",
        desc="[Action Node] allrecipes.com:F1:A2 - Search recipes using safe ingredient keywords",
        parent=recipe_node,
        critical=False
    )

    recipe_count = count_recipes(recipe_info.recipes)
    recipe_count_ok = 3 <= recipe_count <= 5
    evaluator.add_custom_node(
        result=bool(recipe_count_ok),
        id="recipe_card_click",
        desc="[Action Node] allrecipes.com:F3:A3 - Click recipe cards to view details for 3-5 recipes",
        parent=recipe_node,
        critical=False
    )

    recipe_click_foodnetwork = recipe_site_ok and recipe_count > 0
    evaluator.add_custom_node(
        result=bool(recipe_click_foodnetwork),
        id="recipe_card_click_foodnetwork",
        desc="[Action Node] foodnetwork.com:F1:A6 - Click recipe cards on Food Network",
        parent=recipe_node,
        critical=False
    )

    recipe_image_ok = recipe_count > 0 and recipe_type_present
    evaluator.add_custom_node(
        result=bool(recipe_image_ok),
        id="recipe_image_perception",
        desc="[Perception Node] foodnetwork.com:F1:P1 - Recognize recipe images match recommended type",
        parent=recipe_node,
        critical=False
    )

    rating_criteria_ok = all_recipes_meet_criteria(recipe_info.recipes) if recipe_count > 0 else False
    evaluator.add_custom_node(
        result=bool(rating_criteria_ok),
        id="recipe_rating_perception",
        desc="[Perception Node] foodnetwork.com:F1:P2 - Identify recipes with rating ≥4 stars",
        parent=recipe_node,
        critical=False
    )

    nutrition_context_ok = recipe_count > 0 and (has_any_ci(answer, ['vitamin', 'nutrient', 'immune']) or has_scores)
    evaluator.add_custom_node(
        result=bool(nutrition_context_ok),
        id="recipe_nutrition_perception",
        desc="[Perception Node] foodnetwork.com:F2:P3 - Understand nutritional value of recipe ingredients",
        parent=recipe_node,
        critical=False
    )

    ingredients_list_ok = recipe_count > 0 and any(
        recipe.get('ingredients') or recipe.get('main_ingredients')
        for recipe in recipe_info.recipes
    )
    evaluator.add_custom_node(
        result=bool(ingredients_list_ok),
        id="recipe_ingredients_perception",
        desc="[Perception Node] foodnetwork.com:F2:P12 - Extract main ingredients list from recipe details",
        parent=recipe_node,
        critical=False
    )

    # Output completeness checks
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Output completeness verification",
        parent=root,
        critical=False
    )

    recipe_details_complete = recipe_count > 0 and all(
        recipe.get('name') and (recipe.get('rating') is not None) and
        (recipe.get('cooking_time') is not None)
        for recipe in recipe_info.recipes
    )
    evaluator.add_custom_node(
        result=bool(recipe_details_complete),
        id="recipe_details_complete",
        desc="Recipe names, ratings, and cooking times recorded for all recipes",
        parent=output_node,
        critical=False
    )

    recipe_urls_present = recipe_count > 0 and all(
        recipe.get('url') for recipe in recipe_info.recipes
    )
    evaluator.add_custom_node(
        result=bool(recipe_urls_present),
        id="recipe_urls_included",
        desc="Recipe detail page links included for all recipes",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
