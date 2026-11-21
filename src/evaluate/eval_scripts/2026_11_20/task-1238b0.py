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
TASK_ID = "task-1238b0"
TASK_DESCRIPTION = 'Our family is having a dinner gathering next Saturday to Sunday. My sister has a severe nut allergy, and my cousin is allergic to shellfish. I need help finding some safe recipes.\n\nFirst, please consult Mayo Clinic or WebMD to research cross-reactive foods and common hidden sources for both nut and shellfish allergies (e.g., nut ingredients in certain sauces or condiments). Compile and list these risky ingredients.\n\nNext, navigate to AllRecipes and search for recipes explicitly labeled as "nut-free" and "shellfish-free". Identify 5 recipes that have a rating of 4 stars or higher and are suitable for a family dinner.\n\nSubsequently, meticulously examine the ingredient list of each of these 5 selected recipes for any potential hidden risks. For example, certain sauces or seasonings might contain nut oils, tahini, or other allergenic components. If the safety of any specific ingredient within a recipe is uncertain, use Wikipedia or Healthline to investigate its detailed composition.\n\nFinally, provide the following structured output: For each recipe, include its name, a list of its primary ingredients, its rating, the direct AllRecipes link, and your comprehensive allergy risk assessment. In your assessment, clearly indicate which ingredients have been verified as safe and which ones will require careful inspection of product labels during grocery shopping.\n\nAdditionally, furnish the complete lists of cross-reactive allergens and hidden sources for nut and shellfish allergies that you retrieved from the medical websites (Mayo Clinic or WebMD), along with the corresponding direct page links to these resources.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AllergyInformation(BaseModel):
    """Allergy information extracted from medical websites"""
    nut_cross_reactive: Optional[List[str]] = Field(default_factory=list)
    nut_hidden_sources: Optional[List[str]] = Field(default_factory=list)
    shellfish_cross_reactive: Optional[List[str]] = Field(default_factory=list)
    shellfish_hidden_sources: Optional[List[str]] = Field(default_factory=list)
    medical_sources_mentioned: Optional[List[str]] = Field(default_factory=list)


class RecipeInfo(BaseModel):
    """Information about a single recipe"""
    recipe_name: Optional[str] = None
    ingredients_list: Optional[List[str]] = Field(default_factory=list)
    rating: Optional[str] = None
    allrecipes_link: Optional[str] = None
    risk_assessment: Optional[str] = None


class RecipesCollection(BaseModel):
    """Collection of all recipes found"""
    recipes: Optional[List[RecipeInfo]] = Field(default_factory=list)
    total_count: int = 0


class IngredientVerification(BaseModel):
    """Verification details for uncertain ingredients"""
    uncertain_ingredients: Optional[List[str]] = Field(default_factory=list)
    verification_sources: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_allergy_info() -> str:
    return """
Extract the allergy information the user gathered from medical websites (Mayo Clinic or WebMD):

Return:
- nut_cross_reactive: list of cross-reactive foods for nut allergies
- nut_hidden_sources: list of hidden sources of nuts in foods
- shellfish_cross_reactive: list of cross-reactive foods for shellfish allergies
- shellfish_hidden_sources: list of hidden sources of shellfish in foods
- medical_sources_mentioned: list of medical website names mentioned (e.g., "Mayo Clinic", "WebMD")

If any field is missing, return an empty list.
"""


def prompt_extract_recipes() -> str:
    return """
Extract all recipes found from AllRecipes in the answer.

For each recipe, extract:
- recipe_name: the name of the recipe
- ingredients_list: list of primary ingredients mentioned
- rating: the rating (e.g., "4.5 stars", "4 stars")
- allrecipes_link: direct link to the recipe on AllRecipes
- risk_assessment: the allergy risk assessment text provided

Return:
- recipes: list of all recipe objects
- total_count: total number of recipes found

If information is missing, use null or empty lists as appropriate.
"""


def prompt_extract_ingredient_verification() -> str:
    return """
Extract information about uncertain ingredients that were verified using Wikipedia or Healthline:

Return:
- uncertain_ingredients: list of ingredients that required additional verification
- verification_sources: list of sources used for verification (e.g., "Wikipedia", "Healthline")

If no verification was done or mentioned, return empty lists.
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


def count_items_in_list(items: Optional[List]) -> int:
    if not items:
        return 0
    return len([item for item in items if item])


def mentions_medical_source(answer: str) -> bool:
    return has_any_ci(answer, ['mayo clinic', 'mayoclinic', 'webmd'])


def mentions_allrecipes(answer: str) -> bool:
    return has_any_ci(answer, ['allrecipes', 'all recipes'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for patterns like "4 stars", "4.5 stars", "4+", etc.
    return bool(re.search(r'[4-5](\.\d+)?\s*(star|★)', text.lower()))


def has_url_pattern(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://|www\.', text))


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
    allergy_info = await evaluator.extract(
        prompt=prompt_extract_allergy_info(),
        template_class=AllergyInformation,
        extraction_name="allergy_information"
    )

    recipes_info = await evaluator.extract(
        prompt=prompt_extract_recipes(),
        template_class=RecipesCollection,
        extraction_name="recipes_collection"
    )

    verification_info = await evaluator.extract(
        prompt=prompt_extract_ingredient_verification(),
        template_class=IngredientVerification,
        extraction_name="ingredient_verification"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Medical website research section
    medical_section = evaluator.add_sequential(
        id="medical_research_section",
        desc="Research allergy information from Mayo Clinic or WebMD",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A3 - Navigate using alphabetical index
    medical_nav_ok = mentions_medical_source(answer) and (
        has_any_ci(answer, ['nut allergy', 'nut allergies']) and
        has_any_ci(answer, ['shellfish allergy', 'shellfish allergies'])
    )
    evaluator.add_custom_node(
        result=bool(medical_nav_ok),
        id="medical_alphabetical_navigation",
        desc="[Action Node] mayoclinic.org:F2:A3 - Navigate to nut and shellfish allergy entries using alphabetical index",
        parent=medical_section,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A4 - Switch between tabs
    has_cross_reactive = count_items_in_list(allergy_info.nut_cross_reactive) > 0 or count_items_in_list(allergy_info.shellfish_cross_reactive) > 0
    has_hidden_sources = count_items_in_list(allergy_info.nut_hidden_sources) > 0 or count_items_in_list(allergy_info.shellfish_hidden_sources) > 0
    tab_switch_ok = has_cross_reactive and has_hidden_sources
    evaluator.add_custom_node(
        result=bool(tab_switch_ok),
        id="medical_tab_switching",
        desc="[Action Node] mayoclinic.org:F2:A4 - Switch between tabs to find risk factors (cross-reactive foods and hidden sources)",
        parent=medical_section,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F2:P2 - Understand risk factors
    nut_info_complete = count_items_in_list(allergy_info.nut_cross_reactive) > 0 and count_items_in_list(allergy_info.nut_hidden_sources) > 0
    shellfish_info_complete = count_items_in_list(allergy_info.shellfish_cross_reactive) > 0 and count_items_in_list(allergy_info.shellfish_hidden_sources) > 0
    risk_factors_ok = nut_info_complete and shellfish_info_complete
    evaluator.add_custom_node(
        result=bool(risk_factors_ok),
        id="medical_risk_factors_understanding",
        desc="[Perception Node] mayoclinic.org:F2:P2 - Extract and understand risk factors for both allergies",
        parent=medical_section,
        critical=False
    )

    # [Action Node] webmd.com:F7:A10 - Switch between chapter tabs
    medical_sources_count = count_items_in_list(allergy_info.medical_sources_mentioned)
    chapter_switching_ok = medical_sources_count > 0 and (has_cross_reactive and has_hidden_sources)
    evaluator.add_custom_node(
        result=bool(chapter_switching_ok),
        id="webmd_chapter_switching",
        desc="[Action Node] webmd.com:F7:A10 - Switch between Overview/Symptoms chapters to gather comprehensive information",
        parent=medical_section,
        critical=False
    )

    # [Perception Node] webmd.com:F7:P10 - Understand chapter content
    info_comprehensive = (count_items_in_list(allergy_info.nut_cross_reactive) +
                          count_items_in_list(allergy_info.nut_hidden_sources) +
                          count_items_in_list(allergy_info.shellfish_cross_reactive) +
                          count_items_in_list(allergy_info.shellfish_hidden_sources)) >= 6
    evaluator.add_custom_node(
        result=bool(info_comprehensive),
        id="webmd_chapter_understanding",
        desc="[Perception Node] webmd.com:F7:P10 - Understand and extract structured medical information from chapters",
        parent=medical_section,
        critical=False
    )

    # 3.2 AllRecipes search section
    allrecipes_section = evaluator.add_sequential(
        id="allrecipes_search_section",
        desc="Search AllRecipes for nut-free and shellfish-free recipes",
        parent=root,
        critical=False
    )

    # [Action Node] allrecipes.com:F1:A2 - Fill search form
    allrecipes_search_ok = mentions_allrecipes(answer) and (
        has_any_ci(answer, ['nut-free', 'nut free']) and
        has_any_ci(answer, ['shellfish-free', 'shellfish free'])
    )
    evaluator.add_custom_node(
        result=bool(allrecipes_search_ok),
        id="allrecipes_search_form",
        desc="[Action Node] allrecipes.com:F1:A2 - Search for recipes labeled nut-free and shellfish-free",
        parent=allrecipes_section,
        critical=False
    )

    # [Action Node] allrecipes.com:F3:A3 - Click recipe cards to view details
    found_five_recipes = recipes_info and recipes_info.total_count >= 5
    has_detailed_ingredients = found_five_recipes and any(
        count_items_in_list(recipe.ingredients_list) > 0
        for recipe in (recipes_info.recipes or [])
    )
    evaluator.add_custom_node(
        result=bool(has_detailed_ingredients),
        id="allrecipes_click_cards",
        desc="[Action Node] allrecipes.com:F3:A3 - Click recipe cards to access detailed ingredient lists",
        parent=allrecipes_section,
        critical=False
    )

    # 3.3 Recipe verification section
    verification_section = evaluator.add_sequential(
        id="recipe_verification_section",
        desc="Verify recipe ingredients and assess allergy risks",
        parent=root,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F2:P12 - Understand ingredient lists
    has_risk_assessments = found_five_recipes and any(
        recipe.risk_assessment and len(recipe.risk_assessment) > 20
        for recipe in (recipes_info.recipes or [])
    )
    evaluator.add_custom_node(
        result=bool(has_risk_assessments),
        id="ingredient_list_understanding",
        desc="[Perception Node] foodnetwork.com:F2:P12 - Extract and understand ingredient lists for risk assessment",
        parent=verification_section,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Click search results for uncertain ingredients
    has_uncertain_ingredients = count_items_in_list(verification_info.uncertain_ingredients) > 0
    has_verification_sources = count_items_in_list(verification_info.verification_sources) > 0
    evaluator.add_custom_node(
        result=bool(has_uncertain_ingredients and has_verification_sources),
        id="wikipedia_search_click",
        desc="[Action Node] wikipedia.org:F1:A20 - Search and click results for uncertain ingredients",
        parent=verification_section,
        critical=False
    )

    # [Perception Node] wikipedia.org:F2:P9 - Understand ingredient composition
    verification_detailed = has_uncertain_ingredients and has_any_ci(answer, ['composition', 'component', 'contains', 'made from'])
    evaluator.add_custom_node(
        result=bool(verification_detailed),
        id="wikipedia_composition_understanding",
        desc="[Perception Node] wikipedia.org:F2:P9 - Extract and understand ingredient composition from encyclopedia entries",
        parent=verification_section,
        critical=False
    )

    # 3.4 Output completeness section
    output_section = evaluator.add_parallel(
        id="output_completeness_section",
        desc="Verify completeness of final output",
        parent=root,
        critical=False
    )

    # Check recipe output format
    recipes_have_names = found_five_recipes and all(
        recipe.recipe_name for recipe in (recipes_info.recipes or [])
    )
    recipes_have_ratings = found_five_recipes and any(
        looks_like_rating(recipe.rating) for recipe in (recipes_info.recipes or [])
    )
    recipes_have_links = found_five_recipes and any(
        has_url_pattern(recipe.allrecipes_link) for recipe in (recipes_info.recipes or [])
    )

    evaluator.add_custom_node(
        result=bool(recipes_have_names and recipes_have_ratings and recipes_have_links),
        id="recipe_output_format",
        desc="Recipes include name, rating (4+ stars), and AllRecipes link",
        parent=output_section,
        critical=False
    )

    # Check allergy lists output
    lists_provided = (count_items_in_list(allergy_info.nut_cross_reactive) > 0 and
                     count_items_in_list(allergy_info.nut_hidden_sources) > 0 and
                     count_items_in_list(allergy_info.shellfish_cross_reactive) > 0 and
                     count_items_in_list(allergy_info.shellfish_hidden_sources) > 0)
    evaluator.add_custom_node(
        result=bool(lists_provided),
        id="allergy_lists_output",
        desc="Complete lists of cross-reactive allergens and hidden sources provided",
        parent=output_section,
        critical=False
    )

    # Check medical source links
    has_source_links = has_any_ci(answer, ['mayo', 'webmd']) and has_url_pattern(answer)
    evaluator.add_custom_node(
        result=bool(has_source_links),
        id="medical_source_links",
        desc="Direct links to medical website resources provided",
        parent=output_section,
        critical=False
    )

    # Check verified vs requires-inspection distinction
    mentions_verification = has_any_ci(answer, ['verified', 'safe', 'check', 'inspect', 'label'])
    evaluator.add_custom_node(
        result=bool(mentions_verification and has_risk_assessments),
        id="safety_verification_clarity",
        desc="Risk assessments clearly distinguish verified-safe ingredients from those requiring label inspection",
        parent=output_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
