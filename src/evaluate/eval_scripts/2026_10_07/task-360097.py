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
TASK_ID = "task-360097"
TASK_DESCRIPTION = 'I’m a beauty influencer looking for a few recently trending foundations on Sephora to recommend to my followers. First, search for “foundation” on Sephora, sort by best sellers, and identify 5 popular foundations with ratings above 4.5 and prices between $30 and $60. Record the brand name, exact product name, shade range (at least how many shades are available), price, and rating.\n\nThen, search on Target for foundations from the same brands as these 5 Sephora products, prioritizing the exact same item when possible (matching name/line preferred), and check whether Target offers a lower price. Record Target’s price and stock status.\n\nFinally, visit each of the 5 brands’ official websites (e.g., Fenty Beauty, NARS, etc.). Prioritize locating the product page corresponding to the same item selected on Sephora, and check whether the official site has bundle deals or promotions not available on Sephora/Target (such as a free mini with full-size purchase, member-exclusive pricing, etc.). Capture the official product page URL and promotion details.\n\nIf an exact match for any item cannot be found on Target or the brand’s official site, keep that item and mark the corresponding field as “Not found,” then continue with the remaining items.\n\nOutput fields for each foundation:\n- Brand name  \n- Full product name  \n- Sephora price  \n- Sephora rating  \n- Number of shades  \n- Target price  \n- Target stock status (In stock / Out of stock / Not found)  \n- Brand official website price  \n- Official website promotion details (if any)  \n- Sephora product URL  \n- Target product URL  \n- Brand official website product URL'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FoundationItem(BaseModel):
    """Single foundation product extracted from the answer"""
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    sephora_price: Optional[str] = None
    sephora_rating: Optional[str] = None
    num_shades: Optional[str] = None
    target_price: Optional[str] = None
    target_stock_status: Optional[str] = None
    official_website_price: Optional[str] = None
    official_promotion_details: Optional[str] = None
    sephora_url: Optional[str] = None
    target_url: Optional[str] = None
    official_url: Optional[str] = None


class FoundationCollection(BaseModel):
    """Collection of foundation products extracted from the answer"""
    foundations: List[FoundationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_foundations_from_answer() -> str:
    return """
Extract all foundation products mentioned in the answer. For each foundation, extract:

- brand_name: the brand name
- product_name: the full product name
- sephora_price: Sephora price as stated (include $ symbol if present)
- sephora_rating: Sephora rating as stated
- num_shades: number of shades available as stated
- target_price: Target price as stated, or "Not found" if mentioned
- target_stock_status: Target stock status (In stock / Out of stock / Not found)
- official_website_price: official website price as stated, or "Not found" if mentioned
- official_promotion_details: promotion details from official website, or "None" if no promotion
- sephora_url: Sephora product URL if provided
- target_url: Target product URL if provided, or "Not found" if mentioned
- official_url: Official website product URL if provided

Return a list of all foundations found. If any field is missing, set it to null.
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
    if ci_contains(text, 'not found'):
        return True
    return contains_digits(text) and (ci_contains(text, '$') or ci_contains(text, 'dollar'))


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 1.0 <= num <= 5.0


def looks_like_shade_count(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num >= 1


def looks_like_stock_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['in stock', 'out of stock', 'not found', 'unavailable', 'available'])


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    if ci_contains(text, 'not found'):
        return True
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.net'])


def price_in_range(price_text: Optional[str], min_val: float, max_val: float) -> bool:
    if not price_text:
        return False
    num = extract_float(price_text)
    if num is None:
        return False
    return min_val <= num <= max_val


def rating_above_threshold(rating_text: Optional[str], threshold: float) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    if num is None:
        return False
    return num >= threshold


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
    foundation_collection = await evaluator.extract(
        prompt=prompt_extract_foundations_from_answer(),
        template_class=FoundationCollection,
        extraction_name="foundation_collection"
    )

    foundations = foundation_collection.foundations if foundation_collection else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Sephora section
    sephora_node = evaluator.add_sequential(
        id="sephora_section",
        desc="Sephora foundation search and data collection",
        parent=root,
        critical=False
    )

    # [Action Node] sephora.com:F1:A2 - Sort by best sellers
    bestseller_mention = has_any_ci(answer, ['best seller', 'bestseller', 'best-seller', 'sort', 'sorted'])
    evaluator.add_custom_node(
        result=bool(bestseller_mention),
        id="sephora_sort_bestsellers",
        desc="[Action Node] sephora.com:F1:A2 - Sort foundation search results by best sellers",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F1:A1 - Multi-dimensional filtering (price $30-$60, rating 4.5+)
    price_filter_mention = has_any_ci(answer, ['30', '60', 'price', 'filter'])
    rating_filter_mention = has_any_ci(answer, ['4.5', 'rating', 'filter'])

    # Check extracted data for compliance
    valid_prices = sum(1 for f in foundations if price_in_range(f.sephora_price, 30, 60))
    valid_ratings = sum(1 for f in foundations if rating_above_threshold(f.sephora_rating, 4.5))

    filter_applied = (price_filter_mention or valid_prices >= 3) and (rating_filter_mention or valid_ratings >= 3)

    evaluator.add_custom_node(
        result=bool(filter_applied),
        id="sephora_multi_filter",
        desc="[Action Node] sephora.com:F1:A1 - Apply multi-dimensional filters (price $30-$60, rating 4.5+)",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F1:A3 - Click product cards to access details
    product_detail_mention = has_any_ci(answer, ['detail', 'product page', 'clicked', 'visited'])
    has_detailed_info = sum(1 for f in foundations if f.brand_name and f.product_name and f.num_shades) >= 3

    evaluator.add_custom_node(
        result=bool(product_detail_mention or has_detailed_info),
        id="sephora_click_product_cards",
        desc="[Action Node] sephora.com:F1:A3 - Click product cards to access detailed information",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P2 - Extract rating data
    has_ratings = sum(1 for f in foundations if looks_like_rating(f.sephora_rating)) >= 3
    evaluator.add_custom_node(
        result=bool(has_ratings),
        id="sephora_perceive_ratings",
        desc="[Perception Node] sephora.com:F2:P2 - Extract rating information from product pages",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P7 - Extract price information
    has_prices = sum(1 for f in foundations if looks_like_price(f.sephora_price)) >= 3
    evaluator.add_custom_node(
        result=bool(has_prices),
        id="sephora_perceive_prices",
        desc="[Perception Node] sephora.com:F2:P7 - Extract price information from product pages",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P8 - Extract shade count information
    has_shade_counts = sum(1 for f in foundations if looks_like_shade_count(f.num_shades)) >= 3
    evaluator.add_custom_node(
        result=bool(has_shade_counts),
        id="sephora_perceive_shade_options",
        desc="[Perception Node] sephora.com:F2:P8 - Extract shade range/count from product specifications",
        parent=sephora_node,
        critical=False
    )

    # Check for 5 products collected
    has_five_products = len(foundations) >= 5
    evaluator.add_custom_node(
        result=bool(has_five_products),
        id="sephora_five_products",
        desc="Collected information for 5 foundation products from Sephora",
        parent=sephora_node,
        critical=False
    )

    # 3.2 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target price comparison and stock check",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filtering (brand-based search)
    target_search_mention = has_any_ci(answer, ['target', 'search'])
    brand_filter_mention = has_any_ci(answer, ['brand', 'same brand'])
    has_target_data = sum(1 for f in foundations if f.target_price) >= 3

    evaluator.add_custom_node(
        result=bool((target_search_mention and brand_filter_mention) or has_target_data),
        id="target_brand_filter",
        desc="[Action Node] target.com:F1:A1 - Search Target using brand filters to locate matching products",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product cards to access details
    target_detail_mention = has_any_ci(answer, ['target', 'detail', 'product page'])
    has_target_details = sum(1 for f in foundations if f.target_price and f.target_stock_status) >= 3

    evaluator.add_custom_node(
        result=bool(target_detail_mention or has_target_details),
        id="target_click_product_cards",
        desc="[Action Node] target.com:F1:A21 - Click product cards to access price and stock information",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P6 - Extract price information
    has_target_prices = sum(1 for f in foundations if looks_like_price(f.target_price)) >= 3
    evaluator.add_custom_node(
        result=bool(has_target_prices),
        id="target_perceive_prices",
        desc="[Perception Node] target.com:F2:P6 - Extract price information from Target product pages",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P7 - Extract stock status
    has_stock_status = sum(1 for f in foundations if looks_like_stock_status(f.target_stock_status)) >= 3
    evaluator.add_custom_node(
        result=bool(has_stock_status),
        id="target_perceive_stock",
        desc="[Perception Node] target.com:F2:P7 - Extract stock availability status from Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Perceive product list data
    target_list_mention = has_any_ci(answer, ['target', 'search result', 'list'])
    evaluator.add_custom_node(
        result=bool((target_list_mention and has_target_data) or has_target_details),
        id="target_perceive_list",
        desc="[Perception Node] target.com:F1:P1 - Extract product information from Target search result list",
        parent=target_node,
        critical=False
    )

    # 3.3 Brand official website section
    official_node = evaluator.add_sequential(
        id="official_website_section",
        desc="Brand official website price and promotion check",
        parent=root,
        critical=False
    )

    # Check for official website visits
    official_mention = has_any_ci(answer, ['official', 'brand website', 'official site', 'official website'])
    has_official_urls = sum(1 for f in foundations if looks_like_url(f.official_url)) >= 3
    has_official_prices = sum(1 for f in foundations if looks_like_price(f.official_website_price)) >= 3
    has_promotion_data = sum(1 for f in foundations if f.official_promotion_details) >= 3

    evaluator.add_custom_node(
        result=bool(official_mention or has_official_urls),
        id="official_website_visit",
        desc="Visit brand official websites to locate matching products",
        parent=official_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_official_prices),
        id="official_perceive_prices",
        desc="Extract pricing information from brand official websites",
        parent=official_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_promotion_data),
        id="official_perceive_promotions",
        desc="Identify and extract exclusive promotions or bundle deals from official websites",
        parent=official_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    has_all_urls = sum(1 for f in foundations if looks_like_url(f.sephora_url)) >= 3
    evaluator.add_custom_node(
        result=bool(has_all_urls),
        id="url_completeness",
        desc="Collected product URLs from various sources",
        parent=root,
        critical=False
    )

    has_complete_records = sum(
        1 for f in foundations
        if (f.brand_name and f.product_name and
            looks_like_price(f.sephora_price) and
            looks_like_rating(f.sephora_rating) and
            looks_like_shade_count(f.num_shades))
    ) >= 4

    evaluator.add_custom_node(
        result=bool(has_complete_records),
        id="data_completeness",
        desc="Collected complete records with all required fields for multiple products",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
