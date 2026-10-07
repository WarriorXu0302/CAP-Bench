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
TASK_ID = "task-32aa99"
TASK_DESCRIPTION = 'I want to refresh my bedroom in a Mid-Century Modern style this month. Please create a soft-furnishing shopping list with **three items**, with a **total budget of no more than $400**.\n\nFirst, go to **IKEA**, search for **“Chest of drawers,”** filter for **Brown/Wood** finishes, and sort by **price from low to high**. Choose one **3-drawer or 4-drawer** chest that is currently marked **“In stock”** and priced **under $200**.\n\nNext, go to **Etsy** and find a **“Bauhaus exhibition”** art poster that matches this style. It must be a **physical item** (not a digital download), from a **Star Seller** shop, with a size of **11x14 inches or larger**.\n\nFinally, go to **Target** and find a **table lamp** with a **Brass/Gold** finish, rated **4 stars or higher**, and available for **Shipping**.\n\nFor each item, provide: **product name, price, key verification info** (IKEA stock status, Etsy shop badge, Target rating), **size/specs**, and the **product detail page link**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IKEAChestInfo(BaseModel):
    """IKEA chest of drawers information extracted from the answer"""
    product_name: Optional[str] = None
    price_text: Optional[str] = None
    stock_status: Optional[str] = None
    color_finish: Optional[str] = None
    drawer_count: Optional[str] = None
    product_url: Optional[str] = None


class EtsyPosterInfo(BaseModel):
    """Etsy poster information extracted from the answer"""
    product_name: Optional[str] = None
    price_text: Optional[str] = None
    shop_badge: Optional[str] = None
    item_type: Optional[str] = None
    size_text: Optional[str] = None
    product_url: Optional[str] = None


class TargetLampInfo(BaseModel):
    """Target lamp information extracted from the answer"""
    product_name: Optional[str] = None
    price_text: Optional[str] = None
    rating_text: Optional[str] = None
    finish_color: Optional[str] = None
    shipping_info: Optional[str] = None
    product_url: Optional[str] = None


class BudgetInfo(BaseModel):
    """Total budget information"""
    total_budget_text: Optional[str] = None
    item_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ikea_chest() -> str:
    return """
Extract the IKEA chest of drawers information from the answer.

Return:
- product_name: the exact product name as stated
- price_text: the price exactly as written (include currency symbols)
- stock_status: the stock status text (e.g., "In stock", "Available", etc.)
- color_finish: the color/finish description (e.g., "Brown", "Wood", etc.)
- drawer_count: mention of drawer count (e.g., "3-drawer", "4-drawer")
- product_url: the product detail page URL

If any field is missing, set it to null.
"""


def prompt_extract_etsy_poster() -> str:
    return """
Extract the Etsy poster information from the answer.

Return:
- product_name: the exact product name as stated
- price_text: the price exactly as written (include currency symbols)
- shop_badge: any mention of shop status (e.g., "Star Seller")
- item_type: mention of whether it's physical or digital (e.g., "Physical", "Print", "Digital Download")
- size_text: the size information (e.g., "11x14", "16x20", etc.)
- product_url: the product detail page URL

If any field is missing, set it to null.
"""


def prompt_extract_target_lamp() -> str:
    return """
Extract the Target lamp information from the answer.

Return:
- product_name: the exact product name as stated
- price_text: the price exactly as written (include currency symbols)
- rating_text: the rating information (e.g., "4.5 stars", "4.2/5")
- finish_color: the finish/color description (e.g., "Brass", "Gold", etc.)
- shipping_info: mention of shipping availability
- product_url: the product detail page URL

If any field is missing, set it to null.
"""


def prompt_extract_budget() -> str:
    return """
Extract budget information from the answer.

Return:
- total_budget_text: any mention of total budget or sum
- item_count: the number of items in the list (should be 3)

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


def looks_like_price_under_200(price_text: Optional[str]) -> bool:
    if not price_text:
        return False
    price = extract_float(price_text)
    if price is None:
        return False
    return price < 200


def looks_like_in_stock(stock_text: Optional[str]) -> bool:
    if not stock_text:
        return False
    return has_any_ci(stock_text, ['in stock', 'available', 'in-stock'])


def looks_like_brown_or_wood(color_text: Optional[str], answer: str) -> bool:
    if color_text:
        if has_any_ci(color_text, ['brown', 'wood', 'wooden', 'walnut', 'oak', 'pine']):
            return True
    return has_any_ci(answer, ['brown', 'wood', 'wooden', 'walnut', 'oak', 'pine'])


def looks_like_3_or_4_drawer(drawer_text: Optional[str], answer: str) -> bool:
    if drawer_text:
        if has_any_ci(drawer_text, ['3-drawer', '3 drawer', '4-drawer', '4 drawer', 'three drawer', 'four drawer']):
            return True
    return has_any_ci(answer, ['3-drawer', '3 drawer', '4-drawer', '4 drawer', 'three drawer', 'four drawer'])


def looks_like_physical_item(item_type: Optional[str], answer: str) -> bool:
    if item_type and has_any_ci(item_type, ['physical', 'print', 'poster', 'shipped']):
        return True
    no_digital = not has_any_ci(answer, ['digital download', 'instant download', 'printable', 'pdf'])
    has_physical = has_any_ci(answer, ['physical', 'print', 'poster', 'shipped', 'shipping'])
    return no_digital and has_physical


def looks_like_star_seller(badge_text: Optional[str], answer: str) -> bool:
    if badge_text and has_any_ci(badge_text, ['star seller']):
        return True
    return has_any_ci(answer, ['star seller'])


def looks_like_size_11x14_or_larger(size_text: Optional[str]) -> bool:
    if not size_text:
        return False
    numbers = re.findall(r'(\d+)\s*x\s*(\d+)', size_text.lower())
    if not numbers:
        return False
    for w, h in numbers:
        try:
            width = int(w)
            height = int(h)
            if (width >= 11 and height >= 14) or (width >= 14 and height >= 11):
                return True
        except Exception:
            continue
    return False


def looks_like_brass_or_gold(finish_text: Optional[str], answer: str) -> bool:
    if finish_text:
        if has_any_ci(finish_text, ['brass', 'gold', 'golden']):
            return True
    return has_any_ci(answer, ['brass', 'gold', 'golden'])


def looks_like_rating_4_or_higher(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    rating = extract_float(rating_text)
    if rating is None:
        return False
    return rating >= 4.0


def looks_like_shipping_available(shipping_text: Optional[str], answer: str) -> bool:
    if shipping_text and has_any_ci(shipping_text, ['shipping', 'ships', 'delivery', 'deliver']):
        return True
    return has_any_ci(answer, ['shipping', 'available for shipping', 'ships'])


def has_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return url.startswith('http://') or url.startswith('https://')


def count_items_in_answer(answer: str) -> int:
    item_markers = ['1.', '2.', '3.', '1)', '2)', '3)', 'first item', 'second item', 'third item']
    count = sum(1 for marker in item_markers if marker.lower() in answer.lower())
    return min(count // 3, 3) if count > 0 else 0


def total_budget_under_400(answer: str, ikea_price: Optional[str], etsy_price: Optional[str], target_price: Optional[str]) -> bool:
    ikea_val = extract_float(ikea_price) if ikea_price else None
    etsy_val = extract_float(etsy_price) if etsy_price else None
    target_val = extract_float(target_price) if target_price else None

    if ikea_val is not None and etsy_val is not None and target_val is not None:
        total = ikea_val + etsy_val + target_val
        return total <= 400
    return has_any_ci(answer, ['under $400', 'within budget', 'total', '$400'])


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
        prompt=prompt_extract_ikea_chest(),
        template_class=IKEAChestInfo,
        extraction_name="ikea_chest_info"
    )

    etsy_info = await evaluator.extract(
        prompt=prompt_extract_etsy_poster(),
        template_class=EtsyPosterInfo,
        extraction_name="etsy_poster_info"
    )

    target_info = await evaluator.extract(
        prompt=prompt_extract_target_lamp(),
        template_class=TargetLampInfo,
        extraction_name="target_lamp_info"
    )

    budget_info = await evaluator.extract(
        prompt=prompt_extract_budget(),
        template_class=BudgetInfo,
        extraction_name="budget_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IKEA section
    ikea_node = evaluator.add_sequential(
        id="ikea_section",
        desc="IKEA chest of drawers - Brown/Wood finish, sorted by price, in stock, under $200",
        parent=root,
        critical=False
    )

    # [Action Node] ikea.com:F1:A3 - Multi-select filter for Brown/Wood
    brown_wood_ok = looks_like_brown_or_wood(ikea_info.color_finish, answer)
    evaluator.add_custom_node(
        result=bool(brown_wood_ok),
        id="ikea_filter_brown_wood",
        desc="[Action Node] ikea.com:F1:A3 - Filter for Brown/Wood finishes",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F1:A4 - Sort by price low to high
    price_under_200 = looks_like_price_under_200(ikea_info.price_text)
    sort_ok = price_under_200 or has_any_ci(answer, ['price from low to high', 'sorted by price', 'low to high', 'cheapest'])
    evaluator.add_custom_node(
        result=bool(sort_ok),
        id="ikea_sort_price",
        desc="[Action Node] ikea.com:F1:A4 - Sort by price from low to high",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F9:P6 - Stock status awareness
    stock_ok = looks_like_in_stock(ikea_info.stock_status)
    evaluator.add_custom_node(
        result=bool(stock_ok),
        id="ikea_stock_status",
        desc="[Perception Node] ikea.com:F9:P6 - Verify item is In stock",
        parent=ikea_node,
        critical=False
    )

    # Additional checks for IKEA
    drawer_ok = looks_like_3_or_4_drawer(ikea_info.drawer_count, answer)
    evaluator.add_custom_node(
        result=bool(drawer_ok),
        id="ikea_drawer_count",
        desc="Verify chest is 3-drawer or 4-drawer",
        parent=ikea_node,
        critical=False
    )

    ikea_url_ok = has_valid_url(ikea_info.product_url)
    evaluator.add_custom_node(
        result=bool(ikea_url_ok),
        id="ikea_url_provided",
        desc="Product detail page URL is provided",
        parent=ikea_node,
        critical=False
    )

    ikea_mentions_chest = has_any_ci(answer, ['chest of drawers', 'chest', 'drawer'])
    evaluator.add_custom_node(
        result=bool(ikea_mentions_chest),
        id="ikea_mentions_chest",
        desc="Mentions chest of drawers product type",
        parent=ikea_node,
        critical=False
    )

    # 3.2 Etsy section
    etsy_node = evaluator.add_sequential(
        id="etsy_section",
        desc="Etsy Bauhaus exhibition poster - Physical item, Star Seller, 11x14+ inches",
        parent=root,
        critical=False
    )

    # [Action Node] Etsy:F3:A8 - Checkbox filter for Physical item
    physical_ok = looks_like_physical_item(etsy_info.item_type, answer)
    evaluator.add_custom_node(
        result=bool(physical_ok),
        id="etsy_filter_physical",
        desc="[Action Node] Etsy:F3:A8 - Filter for Physical item (not digital download)",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] Etsy:F1:P3 - Star Seller status awareness
    star_seller_ok = looks_like_star_seller(etsy_info.shop_badge, answer)
    evaluator.add_custom_node(
        result=bool(star_seller_ok),
        id="etsy_star_seller",
        desc="[Perception Node] Etsy:F1:P3 - Verify shop has Star Seller badge",
        parent=etsy_node,
        critical=False
    )

    # [Action Node] Etsy:F1:A17 - Click into details to verify size
    size_ok = looks_like_size_11x14_or_larger(etsy_info.size_text)
    evaluator.add_custom_node(
        result=bool(size_ok),
        id="etsy_size_verification",
        desc="[Action Node] Etsy:F1:A17 - Verify size is 11x14 inches or larger",
        parent=etsy_node,
        critical=False
    )

    etsy_bauhaus_ok = has_any_ci(answer, ['bauhaus'])
    evaluator.add_custom_node(
        result=bool(etsy_bauhaus_ok),
        id="etsy_bauhaus_mention",
        desc="Mentions Bauhaus exhibition theme",
        parent=etsy_node,
        critical=False
    )

    etsy_url_ok = has_valid_url(etsy_info.product_url)
    evaluator.add_custom_node(
        result=bool(etsy_url_ok),
        id="etsy_url_provided",
        desc="Product detail page URL is provided",
        parent=etsy_node,
        critical=False
    )

    # 3.3 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target table lamp - Brass/Gold finish, 4+ stars, Shipping available",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filter (rating + shipping)
    rating_ok = looks_like_rating_4_or_higher(target_info.rating_text)
    shipping_ok = looks_like_shipping_available(target_info.shipping_info, answer)
    filter_ok = rating_ok and shipping_ok
    evaluator.add_custom_node(
        result=bool(filter_ok),
        id="target_filter_rating_shipping",
        desc="[Action Node] target.com:F1:A1 - Filter for 4+ star rating and Shipping availability",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P4 - Rating data awareness
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="target_rating_perception",
        desc="[Perception Node] target.com:F2:P4 - Extract and verify rating is 4 stars or higher",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P7 - Stock/shipping status awareness
    evaluator.add_custom_node(
        result=bool(shipping_ok),
        id="target_shipping_perception",
        desc="[Perception Node] target.com:F2:P7 - Verify Shipping availability",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F2:A4 - Variant selection for Brass/Gold finish
    brass_gold_ok = looks_like_brass_or_gold(target_info.finish_color, answer)
    evaluator.add_custom_node(
        result=bool(brass_gold_ok),
        id="target_finish_selection",
        desc="[Action Node] target.com:F2:A4 - Select Brass/Gold finish variant",
        parent=target_node,
        critical=False
    )

    target_lamp_ok = has_any_ci(answer, ['lamp', 'table lamp'])
    evaluator.add_custom_node(
        result=bool(target_lamp_ok),
        id="target_lamp_mention",
        desc="Mentions table lamp product type",
        parent=target_node,
        critical=False
    )

    target_url_ok = has_valid_url(target_info.product_url)
    evaluator.add_custom_node(
        result=bool(target_url_ok),
        id="target_url_provided",
        desc="Product detail page URL is provided",
        parent=target_node,
        critical=False
    )

    # 3.4 Overall requirements
    overall_node = evaluator.add_parallel(
        id="overall_requirements",
        desc="Overall task requirements - 3 items, budget under $400, Mid-Century Modern style",
        parent=root,
        critical=False
    )

    item_count_ok = (budget_info.item_count == 3) or count_items_in_answer(answer) == 3
    evaluator.add_custom_node(
        result=bool(item_count_ok),
        id="three_items_provided",
        desc="Provides exactly three items in the shopping list",
        parent=overall_node,
        critical=False
    )

    budget_ok = total_budget_under_400(answer, ikea_info.price_text, etsy_info.price_text, target_info.price_text)
    evaluator.add_custom_node(
        result=bool(budget_ok),
        id="budget_under_400",
        desc="Total budget is $400 or less",
        parent=overall_node,
        critical=False
    )

    style_mention_ok = has_any_ci(answer, ['mid-century modern', 'mid century modern', 'midcentury'])
    evaluator.add_custom_node(
        result=bool(style_mention_ok),
        id="style_context",
        desc="References Mid-Century Modern style context",
        parent=overall_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
