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
TASK_ID = "task-55d8ab"
TASK_DESCRIPTION = "I plan to assemble a 4K 144Hz gaming PC with a budget of $2500. The core hardware list includes an RTX 4080 graphics card, i9-14900K CPU, Z790 motherboard, 32GB DDR5 RAM, and a 2TB NVMe SSD.\n\nFirst, search for these five hardware components separately on Best Buy, Amazon, and eBay. Record the price and stock status of each component on all three platforms, then filter for available options.\n\nNext, go to the Reddit Gaming community and search for keywords related to this build (e.g., 'RTX 4080 i9-14900K build'). Find 2-3 recent discussion threads to evaluate opinions on the compatibility and price-performance ratio of this configuration, paying special attention to any recommendations to avoid potential issues (pitfalls).\n\nFinally, find a suitable desk at IKEA for the PC setup. Requirements: width at least 120cm, weight capacity at least 50kg, and price not exceeding $300.\n\nOutput:\nFor each hardware component: Name, Best Buy price and stock, Amazon price and stock, eBay price and stock, and product page links for each platform.\nFor Reddit posts: Title, link, and a summary of core evaluations (compatibility/price-performance/pitfall advice).\nFor the IKEA desk: Product name, dimensions, weight capacity, price, and product page link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HardwareComponent(BaseModel):
    """Information for a single hardware component across platforms"""
    component_name: Optional[str] = None
    bestbuy_price: Optional[str] = None
    bestbuy_stock: Optional[str] = None
    bestbuy_link: Optional[str] = None
    amazon_price: Optional[str] = None
    amazon_stock: Optional[str] = None
    amazon_link: Optional[str] = None
    ebay_price: Optional[str] = None
    ebay_stock: Optional[str] = None
    ebay_link: Optional[str] = None


class AllHardwareInfo(BaseModel):
    """All five hardware components extracted from the answer"""
    rtx_4080: Optional[HardwareComponent] = None
    i9_14900k: Optional[HardwareComponent] = None
    z790_motherboard: Optional[HardwareComponent] = None
    ddr5_32gb: Optional[HardwareComponent] = None
    nvme_2tb: Optional[HardwareComponent] = None


class RedditThread(BaseModel):
    """Single Reddit discussion thread"""
    title: Optional[str] = None
    link: Optional[str] = None
    summary: Optional[str] = None


class RedditInfo(BaseModel):
    """Reddit threads extracted from the answer"""
    threads: Optional[List[RedditThread]] = []


class IkeaDesk(BaseModel):
    """IKEA desk information"""
    product_name: Optional[str] = None
    dimensions: Optional[str] = None
    weight_capacity: Optional[str] = None
    price: Optional[str] = None
    product_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hardware_info() -> str:
    return """
Extract information for all five hardware components from the answer:
- RTX 4080 graphics card
- i9-14900K CPU
- Z790 motherboard
- 32GB DDR5 RAM
- 2TB NVMe SSD

For each component, extract:
- component_name: the full product name
- bestbuy_price: price on Best Buy (with currency if present)
- bestbuy_stock: stock status on Best Buy
- bestbuy_link: product page link on Best Buy
- amazon_price: price on Amazon
- amazon_stock: stock status on Amazon
- amazon_link: product page link on Amazon
- ebay_price: price on eBay
- ebay_stock: stock status on eBay
- ebay_link: product page link on eBay

Return all five components in the respective fields. Set missing fields to null.
"""


def prompt_extract_reddit_info() -> str:
    return """
Extract Reddit Gaming community discussion threads from the answer related to the RTX 4080 i9-14900K build.

For each thread (2-3 threads), extract:
- title: the thread title
- link: the thread URL
- summary: summary of opinions on compatibility, price-performance ratio, and pitfall advice

Return the list of threads. If no threads found, return empty list.
"""


def prompt_extract_ikea_desk() -> str:
    return """
Extract IKEA desk information from the answer that meets the requirements (width ≥120cm, weight capacity ≥50kg, price ≤$300).

Extract:
- product_name: the desk product name
- dimensions: full dimensions (length x width x height)
- weight_capacity: weight capacity
- price: price with currency
- product_link: product page URL

Set missing fields to null.
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
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar'])


def looks_like_stock_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['in stock', 'out of stock', 'available', 'unavailable',
                              'limited', 'only', 'left', 'currently', 'out-of-stock'])


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def count_hardware_names_mentioned(answer: str) -> int:
    hardware_keywords = [
        ['rtx 4080', 'rtx4080', '4080'],
        ['i9-14900k', 'i9 14900k', '14900k'],
        ['z790', 'z-790'],
        ['32gb', '32 gb', 'ddr5'],
        ['2tb', '2 tb', 'nvme']
    ]
    count = 0
    for keywords in hardware_keywords:
        if has_any_ci(answer, keywords):
            count += 1
    return count


def check_component_info(comp: Optional[HardwareComponent]) -> Dict[str, bool]:
    if not comp:
        return {'has_name': False, 'has_prices': False, 'has_stock': False}

    has_name = bool(comp.component_name and comp.component_name.strip())

    prices_count = sum([
        looks_like_price(comp.bestbuy_price),
        looks_like_price(comp.amazon_price),
        looks_like_price(comp.ebay_price)
    ])
    has_prices = prices_count >= 2  # At least 2 out of 3 platforms

    stock_count = sum([
        looks_like_stock_status(comp.bestbuy_stock),
        looks_like_stock_status(comp.amazon_stock),
        looks_like_stock_status(comp.ebay_stock)
    ])
    has_stock = stock_count >= 2

    return {'has_name': has_name, 'has_prices': has_prices, 'has_stock': has_stock}


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
    hardware_info = await evaluator.extract(
        prompt=prompt_extract_hardware_info(),
        template_class=AllHardwareInfo,
        extraction_name="hardware_components"
    )

    reddit_info = await evaluator.extract(
        prompt=prompt_extract_reddit_info(),
        template_class=RedditInfo,
        extraction_name="reddit_threads"
    )

    ikea_desk = await evaluator.extract(
        prompt=prompt_extract_ikea_desk(),
        template_class=IkeaDesk,
        extraction_name="ikea_desk"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Best Buy section
    bestbuy_node = evaluator.add_sequential(
        id="bestbuy_section",
        desc="Best Buy hardware search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A1 - Search input for hardware
    hardware_count = count_hardware_names_mentioned(answer)
    bestbuy_search_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and hardware_count >= 3
    evaluator.add_custom_node(
        result=bool(bestbuy_search_ok),
        id="bestbuy_search_input",
        desc="[Action Node] bestbuy.com:F1:A1 - Search for hardware components on Best Buy",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A5 - Click product cards to view details
    bestbuy_has_prices = any([
        looks_like_price(getattr(hardware_info.rtx_4080, 'bestbuy_price', None)),
        looks_like_price(getattr(hardware_info.i9_14900k, 'bestbuy_price', None)),
        looks_like_price(getattr(hardware_info.z790_motherboard, 'bestbuy_price', None))
    ])
    evaluator.add_custom_node(
        result=bool(bestbuy_has_prices),
        id="bestbuy_click_details",
        desc="[Action Node] bestbuy.com:F1:A5 - Click product cards to view details and prices",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F2:P5 - Extract stock status
    bestbuy_has_stock = any([
        looks_like_stock_status(getattr(hardware_info.rtx_4080, 'bestbuy_stock', None)),
        looks_like_stock_status(getattr(hardware_info.i9_14900k, 'bestbuy_stock', None)),
        looks_like_stock_status(getattr(hardware_info.z790_motherboard, 'bestbuy_stock', None))
    ])
    evaluator.add_custom_node(
        result=bool(bestbuy_has_stock),
        id="bestbuy_stock_extraction",
        desc="[Perception Node] bestbuy.com:F2:P5 - Extract stock status from product pages",
        parent=bestbuy_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon hardware search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search with autocomplete
    amazon_search_ok = has_any_ci(answer, ['amazon']) and hardware_count >= 3
    evaluator.add_custom_node(
        result=bool(amazon_search_ok),
        id="amazon_search_input",
        desc="[Action Node] Amazon:F1:A13 - Search for hardware components on Amazon",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click product cards
    amazon_has_prices = any([
        looks_like_price(getattr(hardware_info.rtx_4080, 'amazon_price', None)),
        looks_like_price(getattr(hardware_info.i9_14900k, 'amazon_price', None)),
        looks_like_price(getattr(hardware_info.z790_motherboard, 'amazon_price', None))
    ])
    evaluator.add_custom_node(
        result=bool(amazon_has_prices),
        id="amazon_click_details",
        desc="[Action Node] Amazon:F5:A20 - Click product cards to view details",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P24 - Extract stock status
    amazon_has_stock = any([
        looks_like_stock_status(getattr(hardware_info.rtx_4080, 'amazon_stock', None)),
        looks_like_stock_status(getattr(hardware_info.i9_14900k, 'amazon_stock', None)),
        looks_like_stock_status(getattr(hardware_info.z790_motherboard, 'amazon_stock', None))
    ])
    evaluator.add_custom_node(
        result=bool(amazon_has_stock),
        id="amazon_stock_extraction",
        desc="[Perception Node] Amazon:F5:P24 - Extract stock status information",
        parent=amazon_node,
        critical=False
    )

    # 3.3 eBay section
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="eBay hardware search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] eBay:F1:A5 - Category selection
    ebay_search_ok = has_any_ci(answer, ['ebay']) and hardware_count >= 3
    evaluator.add_custom_node(
        result=bool(ebay_search_ok),
        id="ebay_category_search",
        desc="[Action Node] eBay:F1:A5 - Search with Electronics category selection",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F1:A27 - Click product cards
    ebay_has_prices = any([
        looks_like_price(getattr(hardware_info.rtx_4080, 'ebay_price', None)),
        looks_like_price(getattr(hardware_info.i9_14900k, 'ebay_price', None)),
        looks_like_price(getattr(hardware_info.z790_motherboard, 'ebay_price', None))
    ])
    evaluator.add_custom_node(
        result=bool(ebay_has_prices),
        id="ebay_click_details",
        desc="[Action Node] eBay:F1:A27 - Click product cards to view details",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F5:P1 - Image recognition for hardware verification
    ebay_has_components = hardware_count >= 3
    evaluator.add_custom_node(
        result=bool(ebay_has_components),
        id="ebay_image_recognition",
        desc="[Perception Node] eBay:F5:P1 - Verify hardware model through product images",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Filter for new condition
    ebay_filter_ok = has_any_ci(answer, ['new', 'condition']) or (
        getattr(hardware_info.rtx_4080, 'ebay_stock', None) and
        not has_any_ci(getattr(hardware_info.rtx_4080, 'ebay_stock', ''), ['used', 'refurbished'])
    )
    evaluator.add_custom_node(
        result=bool(ebay_filter_ok),
        id="ebay_condition_filter",
        desc="[Action Node] eBay:F3:A11 - Filter for new condition items",
        parent=ebay_node,
        critical=False
    )

    # 3.4 Reddit section (not covered in points, but part of task)
    reddit_node = evaluator.add_sequential(
        id="reddit_section",
        desc="Reddit Gaming community research",
        parent=root,
        critical=False
    )

    reddit_search_ok = has_any_ci(answer, ['reddit']) and has_any_ci(answer, ['rtx 4080', 'i9-14900k', 'i9 14900k'])
    evaluator.add_custom_node(
        result=bool(reddit_search_ok),
        id="reddit_search",
        desc="Search Reddit Gaming for build discussions",
        parent=reddit_node,
        critical=False
    )

    reddit_threads_ok = reddit_info and reddit_info.threads and len(reddit_info.threads) >= 2
    evaluator.add_custom_node(
        result=bool(reddit_threads_ok),
        id="reddit_threads_found",
        desc="Found 2-3 discussion threads with compatibility and pitfall advice",
        parent=reddit_node,
        critical=False
    )

    # 3.5 IKEA section
    ikea_node = evaluator.add_sequential(
        id="ikea_section",
        desc="IKEA desk search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] ikea.com:F1:A3 - Multi-select filters for size and price
    ikea_search_ok = has_any_ci(answer, ['ikea']) and has_any_ci(answer, ['desk', 'table'])
    evaluator.add_custom_node(
        result=bool(ikea_search_ok),
        id="ikea_filter_dimensions_price",
        desc="[Action Node] ikea.com:F1:A3 - Apply filters for dimensions and price range",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F1:A4 - Sort by price
    ikea_price_sort_ok = (ikea_desk and ikea_desk.price) or has_any_ci(answer, ['price', 'sort', '$300'])
    evaluator.add_custom_node(
        result=bool(ikea_price_sort_ok),
        id="ikea_price_sort",
        desc="[Action Node] ikea.com:F1:A4 - Sort results by price",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F2:A8 - Expand product details for weight capacity
    ikea_details_ok = ikea_desk and ikea_desk.weight_capacity and contains_digits(ikea_desk.weight_capacity)
    evaluator.add_custom_node(
        result=bool(ikea_details_ok),
        id="ikea_expand_details",
        desc="[Action Node] ikea.com:F2:A8 - Expand product details to check weight capacity",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P4 - Extract dimensions
    width_val = None
    if ikea_desk and ikea_desk.dimensions:
        width_val = extract_float(ikea_desk.dimensions)

    ikea_dimensions_ok = width_val and width_val >= 120
    evaluator.add_custom_node(
        result=bool(ikea_dimensions_ok),
        id="ikea_dimensions_perception",
        desc="[Perception Node] ikea.com:F2:P4 - Verify desk width ≥120cm from dimensions",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P8 - Extract price
    price_val = None
    if ikea_desk and ikea_desk.price:
        price_val = extract_float(ikea_desk.price)

    ikea_price_ok = price_val and price_val <= 300
    evaluator.add_custom_node(
        result=bool(ikea_price_ok),
        id="ikea_price_perception",
        desc="[Perception Node] ikea.com:F2:P8 - Verify desk price ≤$300",
        parent=ikea_node,
        critical=False
    )

    # Additional weight capacity check (from ikea.com:F2:A8)
    weight_val = None
    if ikea_desk and ikea_desk.weight_capacity:
        weight_val = extract_float(ikea_desk.weight_capacity)

    ikea_weight_ok = weight_val and weight_val >= 50
    evaluator.add_custom_node(
        result=bool(ikea_weight_ok),
        id="ikea_weight_capacity_check",
        desc="Verify desk weight capacity ≥50kg from product details",
        parent=ikea_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
