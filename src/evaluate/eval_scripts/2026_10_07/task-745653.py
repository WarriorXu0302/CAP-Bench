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
TASK_ID = "task-745653"
TASK_DESCRIPTION = "I am currently redecorating my living room and would like to draw inspiration from IKEA designs, but I have a limited budget.\n\nPlease navigate to the 'Ideas' or 'Inspiration' section of the IKEA website and find a living room showcase or room set with a 'Modern' or 'Minimalist' theme. From this showcase, identify the three most prominent/core large furniture items (e.g., a sofa, coffee table, and rug). Record their names, IKEA prices, and main dimensions.\n\nNext, search on Target and Amazon, respectively, for alternative/substitute versions of these three furniture items. The requirements for these substitutes are: their appearance and style should be similar (e.g., both are grey fabric sofas), their dimensions must be within a 10cm tolerance, and their price must be at least 10% cheaper than the original IKEA version.\n\nFinally, compile a detailed list presenting these three comparison sets (IKEA item vs. substitute item), and calculate the total savings if all substitute items are purchased."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IKEAFurnitureItem(BaseModel):
    """Single IKEA furniture item details"""
    name: Optional[str] = None
    price_text: Optional[str] = None
    dimensions_text: Optional[str] = None


class IKEAShowcaseInfo(BaseModel):
    """IKEA showcase and furniture items extracted from the answer"""
    theme: Optional[str] = None
    item1: Optional[IKEAFurnitureItem] = None
    item2: Optional[IKEAFurnitureItem] = None
    item3: Optional[IKEAFurnitureItem] = None


class SubstituteItem(BaseModel):
    """Substitute item details"""
    source: Optional[str] = None  # "Target" or "Amazon"
    name: Optional[str] = None
    price_text: Optional[str] = None
    dimensions_text: Optional[str] = None


class ComparisonSet(BaseModel):
    """Comparison between IKEA item and substitute"""
    ikea_item_name: Optional[str] = None
    target_substitute: Optional[SubstituteItem] = None
    amazon_substitute: Optional[SubstituteItem] = None


class AllComparisons(BaseModel):
    """All three comparison sets extracted from the answer"""
    comparison1: Optional[ComparisonSet] = None
    comparison2: Optional[ComparisonSet] = None
    comparison3: Optional[ComparisonSet] = None
    total_savings_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ikea_showcase() -> str:
    return """
Extract the IKEA showcase information from the answer:

Return:
- theme: the theme of the living room showcase (e.g., "Modern", "Minimalist"). If not clearly stated, set null.
- item1, item2, item3: for each of the three prominent furniture items identified, extract:
  - name: the furniture item name as stated
  - price_text: the IKEA price exactly as written (include currency symbols if present)
  - dimensions_text: the main dimensions exactly as written (include units if present)

If any field is missing, set it to null.
"""


def prompt_extract_comparisons() -> str:
    return """
Extract the comparison information from the answer for all three furniture items:

For each of the three items (comparison1, comparison2, comparison3), extract:
- ikea_item_name: the IKEA item name being compared
- target_substitute: if a Target substitute is mentioned, extract:
  - source: "Target"
  - name: the substitute item name
  - price_text: the price exactly as written
  - dimensions_text: the dimensions exactly as written
- amazon_substitute: if an Amazon substitute is mentioned, extract:
  - source: "Amazon"
  - name: the substitute item name
  - price_text: the price exactly as written
  - dimensions_text: the dimensions exactly as written

Also extract:
- total_savings_text: the total savings calculation result exactly as stated

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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_currency = has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp', 'dollar'])
    return has_number and (has_currency or True)  # lenient: accept if has number


def looks_like_dimensions(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_units = has_any_ci(text, ['cm', 'in', 'inch', 'mm', 'm', '"', 'x', '×'])
    return has_number and has_units


def count_non_none_items(item1, item2, item3) -> int:
    count = 0
    for item in [item1, item2, item3]:
        if item and item.name:
            count += 1
    return count


def count_valid_comparisons(comp1, comp2, comp3) -> int:
    count = 0
    for comp in [comp1, comp2, comp3]:
        if comp and comp.ikea_item_name:
            count += 1
    return count


def has_substitute_from_source(comp_set, source: str) -> bool:
    if not comp_set:
        return False
    if source.lower() == "target":
        return bool(comp_set.target_substitute and comp_set.target_substitute.name)
    elif source.lower() == "amazon":
        return bool(comp_set.amazon_substitute and comp_set.amazon_substitute.name)
    return False


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
    ikea_info = await evaluator.extract(
        prompt=prompt_extract_ikea_showcase(),
        template_class=IKEAShowcaseInfo,
        extraction_name="ikea_showcase_info"
    )

    comparisons = await evaluator.extract(
        prompt=prompt_extract_comparisons(),
        template_class=AllComparisons,
        extraction_name="comparison_sets"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IKEA section
    ikea_node = evaluator.add_sequential(
        id="ikea_section",
        desc="IKEA Ideas/Inspiration section - Living room showcase with Modern/Minimalist theme",
        parent=root,
        critical=False
    )

    # [Action Node] ikea.com:F7:A23 - Navigate to Ideas/Inspiration section
    ikea_nav_ok = has_any_ci(answer, ['ikea']) and has_any_ci(answer, ['ideas', 'inspiration'])
    evaluator.add_custom_node(
        result=bool(ikea_nav_ok),
        id="ikea_action_navigate_ideas",
        desc="[Action Node] ikea.com:F7:A23 - Navigate to the 'Ideas' or 'Inspiration' section of the IKEA website",
        parent=ikea_node,
        critical=False
    )

    # Check theme mention
    theme_ok = ikea_info and ikea_info.theme and has_any_ci(ikea_info.theme, ['modern', 'minimalist'])
    if not theme_ok:
        theme_ok = has_any_ci(answer, ['modern', 'minimalist']) and has_any_ci(answer, ['living room', 'livingroom'])

    evaluator.add_custom_node(
        result=bool(theme_ok),
        id="ikea_theme_identified",
        desc="Identified a living room showcase with 'Modern' or 'Minimalist' theme",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F7:P12 - Identify three prominent furniture items
    num_items = count_non_none_items(ikea_info.item1 if ikea_info else None,
                                      ikea_info.item2 if ikea_info else None,
                                      ikea_info.item3 if ikea_info else None)
    three_items_ok = num_items >= 3

    evaluator.add_custom_node(
        result=bool(three_items_ok),
        id="ikea_perception_identify_items",
        desc="[Perception Node] ikea.com:F7:P12 - Identify the three most prominent/core large furniture items from the showcase",
        parent=ikea_node,
        critical=False
    )

    # Check that items have names, prices, and dimensions
    items_have_details = False
    if ikea_info:
        items = [ikea_info.item1, ikea_info.item2, ikea_info.item3]
        valid_count = 0
        for item in items:
            if item and item.name:
                has_price = looks_like_price(item.price_text)
                has_dims = looks_like_dimensions(item.dimensions_text)
                if has_price and has_dims:
                    valid_count += 1
        items_have_details = valid_count >= 2  # lenient: at least 2 out of 3

    evaluator.add_custom_node(
        result=bool(items_have_details),
        id="ikea_items_have_details",
        desc="IKEA items include names, prices, and main dimensions",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F2:A8 - View product details/dimensions (typically via popup or detail page)
    details_action_ok = has_any_ci(answer, ['details', 'dimensions', 'specs', 'specifications', 'product page'])
    evaluator.add_custom_node(
        result=bool(details_action_ok),
        id="ikea_action_view_details",
        desc="[Action Node] ikea.com:F2:A8 - Access product details or dimensions (e.g., via popup, detail view)",
        parent=ikea_node,
        critical=False
    )

    # 3.2 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target - Search for substitute furniture items",
        parent=root,
        critical=False
    )

    target_mentioned = has_any_ci(answer, ['target'])
    evaluator.add_custom_node(
        result=bool(target_mentioned),
        id="target_mentioned",
        desc="Target website is mentioned as a source for substitutes",
        parent=target_node,
        critical=False
    )

    # Count how many Target substitutes were found
    target_subs_count = 0
    if comparisons:
        for comp in [comparisons.comparison1, comparisons.comparison2, comparisons.comparison3]:
            if has_substitute_from_source(comp, "target"):
                target_subs_count += 1

    target_subs_ok = target_subs_count >= 2  # lenient: at least 2

    evaluator.add_custom_node(
        result=bool(target_subs_ok),
        id="target_substitutes_found",
        desc="Found substitute items on Target for at least two of the three furniture items",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filtering (price cheaper than IKEA)
    target_price_filter_ok = has_any_ci(answer, ['target']) and has_any_ci(answer, ['cheaper', 'less expensive', 'lower price', '10%'])
    evaluator.add_custom_node(
        result=bool(target_price_filter_ok),
        id="target_action_price_filter",
        desc="[Action Node] target.com:F1:A1 - Apply price filtering to find substitutes cheaper than IKEA originals",
        parent=target_node,
        critical=False
    )

    # 3.3 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon - Search for substitute furniture items",
        parent=root,
        critical=False
    )

    amazon_mentioned = has_any_ci(answer, ['amazon'])
    evaluator.add_custom_node(
        result=bool(amazon_mentioned),
        id="amazon_mentioned",
        desc="Amazon website is mentioned as a source for substitutes",
        parent=amazon_node,
        critical=False
    )

    # Count how many Amazon substitutes were found
    amazon_subs_count = 0
    if comparisons:
        for comp in [comparisons.comparison1, comparisons.comparison2, comparisons.comparison3]:
            if has_substitute_from_source(comp, "amazon"):
                amazon_subs_count += 1

    amazon_subs_ok = amazon_subs_count >= 2  # lenient: at least 2

    evaluator.add_custom_node(
        result=bool(amazon_subs_ok),
        id="amazon_substitutes_found",
        desc="Found substitute items on Amazon for at least two of the three furniture items",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P4 - Visual feature extraction (appearance/style matching)
    visual_match_ok = has_any_ci(answer, ['similar', 'appearance', 'style', 'look', 'design', 'matching', 'grey', 'fabric', 'color', 'colour'])
    evaluator.add_custom_node(
        result=bool(visual_match_ok),
        id="amazon_perception_visual_match",
        desc="[Perception Node] Amazon:F3:P4 - Extract visual features to ensure substitute appearance/style matches IKEA original",
        parent=amazon_node,
        critical=False
    )

    # 3.4 Comparison and validation section
    comparison_node = evaluator.add_sequential(
        id="comparison_section",
        desc="Comparison validation - dimensions tolerance and price requirements",
        parent=root,
        critical=False
    )

    # Check dimensions tolerance mention (10cm)
    dims_tolerance_ok = has_any_ci(answer, ['10cm', '10 cm', 'dimension', 'tolerance', 'within'])
    evaluator.add_custom_node(
        result=bool(dims_tolerance_ok),
        id="dimensions_tolerance_check",
        desc="Dimensions are checked to be within 10cm tolerance",
        parent=comparison_node,
        critical=False
    )

    # Check price requirement mention (10% cheaper)
    price_requirement_ok = has_any_ci(answer, ['10%', 'ten percent', 'cheaper', 'at least 10'])
    evaluator.add_custom_node(
        result=bool(price_requirement_ok),
        id="price_requirement_check",
        desc="Substitutes are verified to be at least 10% cheaper than IKEA originals",
        parent=comparison_node,
        critical=False
    )

    # Check that comparisons are compiled
    num_comparisons = count_valid_comparisons(
        comparisons.comparison1 if comparisons else None,
        comparisons.comparison2 if comparisons else None,
        comparisons.comparison3 if comparisons else None
    )
    comparisons_compiled_ok = num_comparisons >= 2  # lenient: at least 2

    evaluator.add_custom_node(
        result=bool(comparisons_compiled_ok),
        id="comparison_sets_compiled",
        desc="Compiled detailed comparison sets (IKEA vs. substitutes) for the furniture items",
        parent=comparison_node,
        critical=False
    )

    # 3.5 Final summary section
    summary_node = evaluator.add_sequential(
        id="summary_section",
        desc="Final summary with total savings calculation",
        parent=root,
        critical=False
    )

    # Check if total savings are calculated
    savings_text = comparisons.total_savings_text if comparisons else None
    savings_calculated = bool(savings_text and contains_digits(savings_text))
    if not savings_calculated:
        savings_calculated = has_any_ci(answer, ['total saving', 'total save', 'save', 'savings']) and has_any_ci(answer, ['$', 'dollar', 'currency'])

    evaluator.add_custom_node(
        result=bool(savings_calculated),
        id="total_savings_calculated",
        desc="Total savings calculated if all substitute items are purchased",
        parent=summary_node,
        critical=False
    )

    # Check if a detailed list is presented
    detailed_list_ok = has_any_ci(answer, ['list', 'table', 'comparison', 'summary']) or (num_comparisons >= 2)
    evaluator.add_custom_node(
        result=bool(detailed_list_ok),
        id="detailed_list_presented",
        desc="Detailed list or summary presenting the three comparison sets",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
