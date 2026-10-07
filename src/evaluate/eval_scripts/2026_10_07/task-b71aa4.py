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
TASK_ID = "task-b71aa4"
TASK_DESCRIPTION = "For an upcoming weekend, I'd like to cook a classic Beef Wellington. Please go to Food Network, find the highest-rated recipe (prioritizing the one with the most reviews), and list all its ingredients.\n\nNext, due to a pregnant family member at home, I am very concerned about food safety. Please take the ingredient list and check the EWG website to identify any ingredients that fall under the 'Dirty Dozen' (high pesticide residue risk).\n\nFinally, please search for organic supermarkets near my location (assume I am at Union Square, San Francisco) that have a rating of 4.5 or higher. Ideally, they should clearly state their business hours on their detail page and be currently open. Please provide me with their names and addresses."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RecipeInfo(BaseModel):
    """Recipe information extracted from Food Network"""
    recipe_name: Optional[str] = None
    ingredients: Optional[List[str]] = Field(default_factory=list)


class DirtyDozenInfo(BaseModel):
    """Dirty Dozen ingredients extracted from the answer"""
    dirty_dozen_ingredients: Optional[List[str]] = Field(default_factory=list)


class SupermarketInfo(BaseModel):
    """Organic supermarkets near Union Square, SF"""
    supermarkets: Optional[List[Dict[str, str]]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_recipe_from_answer() -> str:
    return """
Extract the Beef Wellington recipe information from Food Network that the user reported in the answer.

Return:
- recipe_name: the name of the recipe exactly as written. If not present, set null.
- ingredients: a list of all ingredients mentioned. If none are present, return an empty list.

If the information is missing, set fields to null or empty list as appropriate.
"""


def prompt_extract_dirty_dozen_from_answer() -> str:
    return """
From the answer, extract any ingredients that the user identified as falling under the EWG 'Dirty Dozen' (high pesticide residue risk).

Return:
- dirty_dozen_ingredients: a list of ingredient names that are flagged as Dirty Dozen. If none are identified, return an empty list.
"""


def prompt_extract_supermarkets_from_answer() -> str:
    return """
From the answer, extract the organic supermarkets near Union Square, San Francisco with rating 4.5 or higher that are currently open.

Return:
- supermarkets: a list of dictionaries, each containing:
  - name: the supermarket name
  - address: the supermarket address
  - rating: the rating if mentioned
  - status: whether it's currently open

If no supermarkets are listed, return an empty list.
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
    Restrict evaluator.verify to at most one usage.
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
    recipe_info = await evaluator.extract(
        prompt=prompt_extract_recipe_from_answer(),
        template_class=RecipeInfo,
        extraction_name="recipe_info"
    )

    dirty_dozen_info = await evaluator.extract(
        prompt=prompt_extract_dirty_dozen_from_answer(),
        template_class=DirtyDozenInfo,
        extraction_name="dirty_dozen_info"
    )

    supermarket_info = await evaluator.extract(
        prompt=prompt_extract_supermarkets_from_answer(),
        template_class=SupermarketInfo,
        extraction_name="supermarket_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Food Network part
    foodnetwork_node = evaluator.add_sequential(
        id="foodnetwork_section",
        desc="Food Network - Find highest-rated Beef Wellington recipe and list ingredients",
        parent=root,
        critical=False
    )

    # [Action Node] foodnetwork.com:F2:A5 - Navigate to Food Network and switch to Highly Rated tab
    foodnetwork_action_ok = (has_any_ci(answer, ['food network']) and
                             has_any_ci(answer, ['beef wellington']))
    tab_switch_ok = has_any_ci(answer, ['highest', 'highly rated', 'top rated', 'most reviews', 'best'])

    evaluator.add_custom_node(
        result=bool(foodnetwork_action_ok and tab_switch_ok),
        id="foodnetwork_action_tab_switch",
        desc="[Action Node] foodnetwork.com:F2:A5 - Navigate to Food Network and switch to 'Highly Rated' or similar sorting tab to find highest-rated recipe",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F2:P12 - Extract ingredients list
    has_recipe_name = bool(recipe_info and recipe_info.recipe_name and recipe_info.recipe_name.strip())
    has_ingredients = bool(recipe_info and recipe_info.ingredients and len(recipe_info.ingredients) > 0)
    ingredients_keyword_ok = has_any_ci(answer, ['ingredient'])

    evaluator.add_custom_node(
        result=bool(has_recipe_name and has_ingredients and ingredients_keyword_ok),
        id="foodnetwork_perception_ingredients",
        desc="[Perception Node] foodnetwork.com:F2:P12 - Extract and list all ingredients from the recipe",
        parent=foodnetwork_node,
        critical=False
    )

    # Additional lenient check: mentions recipe has most reviews or highest rating
    mentions_reviews = has_any_ci(answer, ['review', 'rating', 'star'])
    evaluator.add_custom_node(
        result=bool(mentions_reviews),
        id="foodnetwork_mentions_reviews",
        desc="Mentions review count or rating information when selecting the recipe",
        parent=foodnetwork_node,
        critical=False
    )

    # 3.2 EWG Dirty Dozen part
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG - Check ingredients against Dirty Dozen list for pesticide residue risk",
        parent=root,
        critical=False
    )

    # Check that EWG website was visited
    ewg_action_ok = has_any_ci(answer, ['ewg'])
    evaluator.add_custom_node(
        result=bool(ewg_action_ok),
        id="ewg_action_visit",
        desc="Navigate to EWG website to check ingredients against Dirty Dozen list",
        parent=ewg_node,
        critical=False
    )

    # Check that Dirty Dozen analysis was performed
    dirty_dozen_keyword_ok = has_any_ci(answer, ['dirty dozen', 'pesticide', 'residue'])
    has_dirty_dozen_results = bool(dirty_dozen_info and dirty_dozen_info.dirty_dozen_ingredients is not None)

    evaluator.add_custom_node(
        result=bool(ewg_action_ok and dirty_dozen_keyword_ok and has_dirty_dozen_results),
        id="ewg_perception_dirty_dozen",
        desc="Identify ingredients that fall under the 'Dirty Dozen' high pesticide residue risk category",
        parent=ewg_node,
        critical=False
    )

    # Check that food safety context is mentioned
    food_safety_ok = has_any_ci(answer, ['food safety', 'pregnant', 'safety'])
    evaluator.add_custom_node(
        result=bool(food_safety_ok),
        id="ewg_mentions_food_safety",
        desc="Mentions food safety concern context for checking ingredients",
        parent=ewg_node,
        critical=False
    )

    # 3.3 Google Maps organic supermarkets part
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps - Find organic supermarkets near Union Square, SF with rating 4.5+ that are currently open",
        parent=root,
        critical=False
    )

    # Check that Google Maps was used and location is correct
    maps_action_ok = has_any_ci(answer, ['google maps', 'map', 'search'])
    location_ok = has_any_ci(answer, ['union square', 'san francisco'])
    organic_ok = has_any_ci(answer, ['organic'])

    evaluator.add_custom_node(
        result=bool(maps_action_ok and location_ok and organic_ok),
        id="maps_action_search",
        desc="Search for organic supermarkets near Union Square, San Francisco on Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Filter by rating 4.5 or higher
    has_supermarkets = bool(supermarket_info and supermarket_info.supermarkets and len(supermarket_info.supermarkets) > 0)
    rating_mention_ok = has_any_ci(answer, ['4.5', 'rating', 'star'])

    evaluator.add_custom_node(
        result=bool(has_supermarkets and rating_mention_ok),
        id="maps_perception_rating",
        desc="[Perception Node] maps.google.com:F1:P1 - Identify and filter supermarkets with rating 4.5 or higher from search results",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P15 - Check business hours and current open status
    hours_ok = has_any_ci(answer, ['hours', 'open', 'closed', 'business hours'])
    currently_open_ok = has_any_ci(answer, ['currently open', 'now open', 'open now'])

    evaluator.add_custom_node(
        result=bool(hours_ok and currently_open_ok),
        id="maps_perception_status",
        desc="[Perception Node] maps.google.com:F5:P15 - Check business hours and verify supermarkets are currently open",
        parent=maps_node,
        critical=False
    )

    # Check that names and addresses are provided
    has_names_addresses = False
    if supermarket_info and supermarket_info.supermarkets:
        has_names_addresses = all(
            s.get('name') and s.get('address')
            for s in supermarket_info.supermarkets
        )

    evaluator.add_custom_node(
        result=bool(has_names_addresses),
        id="maps_provides_details",
        desc="Provides supermarket names and addresses as requested",
        parent=maps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
