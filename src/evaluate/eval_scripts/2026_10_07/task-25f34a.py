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
TASK_ID = "task-25f34a"
TASK_DESCRIPTION = 'A family member of mine was recently diagnosed with hypertension. The doctor recommended a low-sodium diet, but he is allergic to shellfish and peanuts. I want to create a one-week dinner plan that meets hypertension dietary guidelines while avoiding allergenic ingredients.\n\nFirst, check Mayo Clinic for dietary recommendations for people with hypertension. Specifically record the daily sodium intake target relevant to blood pressure management. If the page does not provide a “hypertension-specific” value, record the sodium recommendation given for adults/cardiovascular health and clearly note the applicable population. Also capture the recommended food categories.\n\nThen go to WebMD to confirm possible cross-reactive foods for shellfish and peanut allergies (e.g., whether other seafood or legumes are safe).\n\nNext, search AllRecipes for low-sodium dinner recipes. Find 5 recipes rated above 4 stars with no more than 600 mg sodium per serving. Review each recipe’s full ingredient list carefully, exclude any recipe containing shellfish, peanuts, or identified cross-reactive ingredients, and keep 3 final safe recipes.\n\nFor the core protein ingredients in these 3 recipes (e.g., chicken, beef), check the EWG database for food safety ratings, ensuring there are no pesticide-residue or additive-risk concerns.\n\nOutput:\n- Summary of Mayo Clinic hypertension diet guidance (daily sodium limit, recommended food types, and the applicable population for the sodium value) + page link  \n- WebMD-confirmed list of cross-reactive allergy foods + page link  \n- For 3 qualifying recipes: recipe name, link, sodium per serving, full ingredient list, and user rating  \n- EWG safety rating + link for each recipe’s core protein ingredient'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MayoClinicGuidance(BaseModel):
    """Mayo Clinic hypertension diet guidance extracted from the answer"""
    daily_sodium_limit: Optional[str] = None
    applicable_population: Optional[str] = None
    recommended_food_types: Optional[str] = None
    mayo_link: Optional[str] = None


class WebMDAllergyInfo(BaseModel):
    """WebMD cross-reactive allergy foods extracted from the answer"""
    cross_reactive_foods: Optional[str] = None
    webmd_link: Optional[str] = None


class RecipeInfo(BaseModel):
    """Individual recipe information"""
    recipe_name: Optional[str] = None
    recipe_link: Optional[str] = None
    sodium_per_serving: Optional[str] = None
    ingredient_list: Optional[str] = None
    user_rating: Optional[str] = None


class AllRecipesInfo(BaseModel):
    """All recipes extracted from the answer"""
    recipes: Optional[List[RecipeInfo]] = Field(default_factory=list)


class ProteinSafetyInfo(BaseModel):
    """EWG safety information for protein ingredients"""
    protein_ingredient: Optional[str] = None
    ewg_rating: Optional[str] = None
    ewg_link: Optional[str] = None


class EWGInfo(BaseModel):
    """All EWG safety ratings extracted from the answer"""
    protein_safety_list: Optional[List[ProteinSafetyInfo]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mayo_clinic() -> str:
    return """
Extract the Mayo Clinic hypertension diet guidance from the answer.

Return:
- daily_sodium_limit: the daily sodium intake target in mg (e.g., "2300 mg", "1500 mg", "<2300mg"). Include units if present.
- applicable_population: which population this sodium limit applies to (e.g., "adults", "hypertension patients", "cardiovascular health").
- recommended_food_types: the food categories recommended (e.g., "fruits, vegetables, whole grains, lean proteins").
- mayo_link: the Mayo Clinic page URL or link reference.

If any field is missing, set it to null.
"""


def prompt_extract_webmd_allergy() -> str:
    return """
Extract the WebMD cross-reactive allergy information from the answer.

Return:
- cross_reactive_foods: list of foods that may cross-react with shellfish or peanut allergies, including safety notes (e.g., "other crustaceans - avoid", "most fish - safe", "tree nuts - caution").
- webmd_link: the WebMD page URL or link reference.

If any field is missing, set it to null.
"""


def prompt_extract_allrecipes() -> str:
    return """
Extract all qualifying recipes from AllRecipes mentioned in the answer.

For each recipe, return:
- recipe_name: the name of the recipe.
- recipe_link: the AllRecipes URL or link reference.
- sodium_per_serving: the sodium content per serving with units (e.g., "450 mg", "580mg").
- ingredient_list: the complete list of ingredients as described.
- user_rating: the user rating (e.g., "4.5 stars", "4.8").

Return a list of recipes. If no recipes are found, return an empty list.
"""


def prompt_extract_ewg() -> str:
    return """
Extract the EWG food safety ratings for protein ingredients from the answer.

For each protein ingredient, return:
- protein_ingredient: the name of the protein ingredient (e.g., "chicken breast", "beef", "salmon").
- ewg_rating: the EWG safety rating or score (e.g., "low risk", "score: 2", "safe").
- ewg_link: the EWG database URL or link reference.

Return a list of protein safety information. If none found, return an empty list.
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


def looks_like_sodium_limit(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_mg = has_any_ci(text, ['mg', 'milligram'])
    return has_number and has_mg


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['star', 'rating']) or contains_digits(text)


def is_valid_sodium_value(sodium_text: Optional[str], max_value: int = 600) -> bool:
    num = extract_number(sodium_text)
    if num is None:
        return False
    return num <= max_value


def is_valid_rating(rating_text: Optional[str], min_value: float = 4.0) -> bool:
    num = extract_number(rating_text)
    if num is None:
        return False
    return num >= min_value


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
    mayo_info = await evaluator.extract(
        prompt=prompt_extract_mayo_clinic(),
        template_class=MayoClinicGuidance,
        extraction_name="mayo_clinic_guidance"
    )

    webmd_info = await evaluator.extract(
        prompt=prompt_extract_webmd_allergy(),
        template_class=WebMDAllergyInfo,
        extraction_name="webmd_allergy_info"
    )

    recipes_info = await evaluator.extract(
        prompt=prompt_extract_allrecipes(),
        template_class=AllRecipesInfo,
        extraction_name="allrecipes_info"
    )

    ewg_info = await evaluator.extract(
        prompt=prompt_extract_ewg(),
        template_class=EWGInfo,
        extraction_name="ewg_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Mayo Clinic section
    mayo_node = evaluator.add_sequential(
        id="mayo_clinic_section",
        desc="Mayo Clinic hypertension dietary guidance",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A2 - Navigate to Patient Care & Health Information menu
    mayo_navigation_ok = has_any_ci(answer, ['mayo clinic', 'mayoclinic']) and has_any_ci(answer, ['hypertension', 'high blood pressure', 'blood pressure'])
    evaluator.add_custom_node(
        result=bool(mayo_navigation_ok),
        id="mayo_menu_navigation",
        desc="[Action Node] mayoclinic.org:F2:A2 - Navigate to Mayo Clinic and access Patient Care & Health Information for hypertension",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A3 - Navigate to Hypertension in disease list
    mayo_hypertension_found = has_any_ci(answer, ['hypertension', 'high blood pressure'])
    evaluator.add_custom_node(
        result=bool(mayo_hypertension_found),
        id="mayo_hypertension_entry",
        desc="[Action Node] mayoclinic.org:F2:A3 - Locate and access the Hypertension entry in disease encyclopedia",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A4 - Switch to Prevention or Diagnosis & treatment tab
    mayo_tab_switch_ok = has_any_ci(answer, ['prevention', 'treatment', 'diet', 'dietary', 'sodium'])
    evaluator.add_custom_node(
        result=bool(mayo_tab_switch_ok),
        id="mayo_tab_switch",
        desc="[Action Node] mayoclinic.org:F2:A4 - Switch to Prevention or Diagnosis & treatment tab to view dietary content",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F2:P2 - Extract specific sodium limit and understand dietary factors
    sodium_limit_present = looks_like_sodium_limit(mayo_info.daily_sodium_limit)
    population_noted = bool(mayo_info.applicable_population and mayo_info.applicable_population.strip())
    food_types_present = bool(mayo_info.recommended_food_types and mayo_info.recommended_food_types.strip())

    evaluator.add_custom_node(
        result=bool(sodium_limit_present and population_noted),
        id="mayo_sodium_limit_extraction",
        desc="[Perception Node] mayoclinic.org:F2:P2 - Extract specific daily sodium intake limit (e.g., <2300mg or <1500mg) and applicable population",
        parent=mayo_node,
        critical=False
    )

    # Check for recommended food categories
    evaluator.add_custom_node(
        result=bool(food_types_present),
        id="mayo_food_categories",
        desc="Extract recommended food categories from Mayo Clinic guidance",
        parent=mayo_node,
        critical=False
    )

    # Check for Mayo Clinic link
    mayo_link_present = bool(mayo_info.mayo_link and mayo_info.mayo_link.strip())
    evaluator.add_custom_node(
        result=bool(mayo_link_present),
        id="mayo_link_provided",
        desc="Provide Mayo Clinic page link",
        parent=mayo_node,
        critical=False
    )

    # 3.2 WebMD section
    webmd_node = evaluator.add_sequential(
        id="webmd_section",
        desc="WebMD cross-reactive allergy information",
        parent=root,
        critical=False
    )

    # [Action Node] webmd.com:F7:A14 - Navigate to allergy entries in disease list
    webmd_navigation_ok = has_any_ci(answer, ['webmd']) and has_any_ci(answer, ['allergy', 'allergies', 'shellfish', 'peanut'])
    evaluator.add_custom_node(
        result=bool(webmd_navigation_ok),
        id="webmd_allergy_navigation",
        desc="[Action Node] webmd.com:F7:A14 - Navigate to WebMD and locate allergy-related entries",
        parent=webmd_node,
        critical=False
    )

    # [Action Node] webmd.com:F7:A10 - Switch to relevant section for cross-reactivity
    webmd_section_switch_ok = has_any_ci(answer, ['cross-react', 'cross react', 'cross-reactive', 'other seafood', 'legume', 'safe'])
    evaluator.add_custom_node(
        result=bool(webmd_section_switch_ok),
        id="webmd_section_switch",
        desc="[Action Node] webmd.com:F7:A10 - Switch to relevant section to view cross-reactivity information",
        parent=webmd_node,
        critical=False
    )

    # [Perception Node] webmd.com:F7:P10 - Extract and understand cross-reactive foods with safety notes
    cross_reactive_present = bool(webmd_info.cross_reactive_foods and webmd_info.cross_reactive_foods.strip())
    safety_judgments = has_any_ci(webmd_info.cross_reactive_foods or '', ['safe', 'avoid', 'caution', 'risk'])

    evaluator.add_custom_node(
        result=bool(cross_reactive_present and safety_judgments),
        id="webmd_cross_reactivity_extraction",
        desc="[Perception Node] webmd.com:F7:P10 - Extract cross-reactive foods with safety judgments (safe/caution/avoid)",
        parent=webmd_node,
        critical=False
    )

    # Check for WebMD link
    webmd_link_present = bool(webmd_info.webmd_link and webmd_info.webmd_link.strip())
    evaluator.add_custom_node(
        result=bool(webmd_link_present),
        id="webmd_link_provided",
        desc="Provide WebMD page link",
        parent=webmd_node,
        critical=False
    )

    # 3.3 AllRecipes section
    allrecipes_node = evaluator.add_sequential(
        id="allrecipes_section",
        desc="AllRecipes low-sodium dinner recipes search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] allrecipes.com:F1:A2 - Search for low-sodium dinner recipes
    allrecipes_search_ok = has_any_ci(answer, ['allrecipes', 'all recipes']) and has_any_ci(answer, ['low-sodium', 'low sodium', 'dinner'])
    evaluator.add_custom_node(
        result=bool(allrecipes_search_ok),
        id="allrecipes_search",
        desc="[Action Node] allrecipes.com:F1:A2 - Search AllRecipes for low-sodium dinner recipes",
        parent=allrecipes_node,
        critical=False
    )

    # [Action Node] allrecipes.com:F2:A9 - Scroll to load more results to find 5 candidates
    recipes_list = recipes_info.recipes if recipes_info and recipes_info.recipes else []
    found_multiple_recipes = len(recipes_list) >= 3
    evaluator.add_custom_node(
        result=bool(found_multiple_recipes),
        id="allrecipes_scroll_results",
        desc="[Action Node] allrecipes.com:F2:A9 - Browse sufficient search results to identify candidate recipes",
        parent=allrecipes_node,
        critical=False
    )

    # [Action Node] allrecipes.com:F3:A3 - Click recipe cards to view details
    has_recipe_details = any(r.ingredient_list for r in recipes_list)
    evaluator.add_custom_node(
        result=bool(has_recipe_details),
        id="allrecipes_view_details",
        desc="[Action Node] allrecipes.com:F3:A3 - Click recipe cards to access detailed ingredient lists",
        parent=allrecipes_node,
        critical=False
    )

    # [Perception Node] allrecipes.com:F3:P1 - Extract sodium content per serving
    valid_sodium_values = [r for r in recipes_list if is_valid_sodium_value(r.sodium_per_serving, 600)]
    all_sodium_valid = len(recipes_list) > 0 and len(valid_sodium_values) == len(recipes_list)
    evaluator.add_custom_node(
        result=bool(all_sodium_valid),
        id="allrecipes_sodium_extraction",
        desc="[Perception Node] allrecipes.com:F3:P1 - Extract sodium content per serving and verify ≤600mg for all recipes",
        parent=allrecipes_node,
        critical=False
    )

    # [Perception Node] allrecipes.com:F3:P2 - Extract and verify ingredient lists exclude allergens
    has_ingredient_lists = all(r.ingredient_list for r in recipes_list)
    no_allergens_mentioned = not any(
        has_any_ci(r.ingredient_list or '', ['shellfish', 'shrimp', 'crab', 'lobster', 'peanut', 'peanuts'])
        for r in recipes_list
    )
    evaluator.add_custom_node(
        result=bool(has_ingredient_lists and no_allergens_mentioned and len(recipes_list) >= 3),
        id="allrecipes_ingredient_verification",
        desc="[Perception Node] allrecipes.com:F3:P2 - Extract complete ingredient lists and verify no shellfish, peanuts, or cross-reactive ingredients",
        parent=allrecipes_node,
        critical=False
    )

    # [Action Node] allrecipes.com:F4:A4 - Identify user ratings
    valid_ratings = [r for r in recipes_list if is_valid_rating(r.user_rating, 4.0)]
    all_ratings_valid = len(recipes_list) > 0 and len(valid_ratings) == len(recipes_list)
    evaluator.add_custom_node(
        result=bool(all_ratings_valid),
        id="allrecipes_rating_verification",
        desc="[Action Node] allrecipes.com:F4:A4 - Verify user ratings ≥4 stars for all selected recipes",
        parent=allrecipes_node,
        critical=False
    )

    # Check for 3 final recipes with complete information
    complete_recipes = [
        r for r in recipes_list
        if r.recipe_name and r.recipe_link and r.sodium_per_serving and r.ingredient_list and r.user_rating
    ]
    has_three_complete = len(complete_recipes) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_complete),
        id="allrecipes_final_three",
        desc="Provide 3 qualifying recipes with complete information (name, link, sodium, ingredients, rating)",
        parent=allrecipes_node,
        critical=False
    )

    # 3.4 EWG section
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG food safety ratings for protein ingredients",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Search EWG database for protein ingredients
    protein_list = ewg_info.protein_safety_list if ewg_info and ewg_info.protein_safety_list else []
    ewg_search_ok = has_any_ci(answer, ['ewg']) and len(protein_list) > 0
    evaluator.add_custom_node(
        result=bool(ewg_search_ok),
        id="ewg_search",
        desc="[Action Node] ewg.org:F1:A1 - Search EWG database for core protein ingredients from recipes",
        parent=ewg_node,
        critical=False
    )

    # [Action Node] ewg.org:F3:A1 - Query ingredient safety information
    has_safety_ratings = all(p.ewg_rating for p in protein_list)
    evaluator.add_custom_node(
        result=bool(has_safety_ratings and len(protein_list) > 0),
        id="ewg_safety_query",
        desc="[Action Node] ewg.org:F3:A1 - Query safety information to ensure no pesticide-residue or additive risks",
        parent=ewg_node,
        critical=False
    )

    # Check that ratings indicate low risk (lenient check)
    low_risk_indicators = [
        has_any_ci(p.ewg_rating or '', ['low', 'safe', 'good', 'acceptable'])
        or (extract_number(p.ewg_rating) is not None and extract_number(p.ewg_rating) < 5)
        for p in protein_list
    ]
    all_low_risk = len(protein_list) > 0 and all(low_risk_indicators)
    evaluator.add_custom_node(
        result=bool(all_low_risk),
        id="ewg_low_risk_verification",
        desc="Verify EWG ratings indicate low risk (score <5 or 'low risk'/'safe' designation)",
        parent=ewg_node,
        critical=False
    )

    # Check for complete EWG information
    complete_ewg = [p for p in protein_list if p.protein_ingredient and p.ewg_rating and p.ewg_link]
    has_complete_ewg = len(complete_ewg) >= 3
    evaluator.add_custom_node(
        result=bool(has_complete_ewg),
        id="ewg_complete_info",
        desc="Provide complete EWG information (ingredient name, rating, link) for each protein",
        parent=ewg_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
