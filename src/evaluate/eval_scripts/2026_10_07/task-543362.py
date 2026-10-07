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
TASK_ID = "task-543362"
TASK_DESCRIPTION = 'I am currently conducting cross-border product selection research for beauty products.\n\nFirst, navigate to the official Sephora US website. Under the \'Skincare\' category, identify products marked as \'Best Sellers\'. Select 10 products with a rating of 4.3 or higher. For each selected product, record its product name, brand, Sephora price, and rating.\n\nNext, use the format "Brand + Product Name" as keywords to search for the corresponding products one by one on Amazon. Record the Amazon price, rating, and number of reviews for each product.\n\nThen, search for these same products on Target using the identical "Brand + Product Name" keywords. Record the Target price and stock status (in stock/out of stock).\n\nFinally, visit the official brand website for each product (e.g., Drunk Elephant, La Mer, CeraVe). Locate the official product page. Record the official retail price from the brand\'s website and note whether there are any website-exclusive sets or special offers.\n\nFor each product, output the following information:\nProduct Name, Brand, Sephora Price, Sephora Rating, Sephora Product Link, Amazon Price, Amazon Rating, Amazon Number of Reviews, Amazon Product Link, Target Price, Target Stock Status, Target Product Link, Brand Official Website Price, Availability of Website-Exclusive Sets, Brand Official Website Product Link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    """Single product information extracted from answer"""
    product_name: Optional[str] = None
    brand: Optional[str] = None
    sephora_price: Optional[str] = None
    sephora_rating: Optional[str] = None
    sephora_link: Optional[str] = None
    amazon_price: Optional[str] = None
    amazon_rating: Optional[str] = None
    amazon_reviews: Optional[str] = None
    amazon_link: Optional[str] = None
    target_price: Optional[str] = None
    target_stock: Optional[str] = None
    target_link: Optional[str] = None
    brand_website_price: Optional[str] = None
    brand_website_exclusive_sets: Optional[str] = None
    brand_website_link: Optional[str] = None


class ProductsList(BaseModel):
    """List of products extracted from answer"""
    products: List[ProductInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_products_from_answer() -> str:
    return """
Extract all product information from the answer. For each product mentioned, extract:
- product_name: the product name
- brand: the brand name
- sephora_price: Sephora price as stated
- sephora_rating: Sephora rating as stated
- sephora_link: Sephora product link URL
- amazon_price: Amazon price as stated
- amazon_rating: Amazon rating as stated
- amazon_reviews: Amazon number of reviews as stated
- amazon_link: Amazon product link URL
- target_price: Target price as stated
- target_stock: Target stock status (in stock/out of stock)
- target_link: Target product link URL
- brand_website_price: Brand official website price as stated
- brand_website_exclusive_sets: Whether website-exclusive sets are available
- brand_website_link: Brand official website product link URL

Also provide:
- total_count: the total number of products listed

If any field is missing for a product, set it to null.
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


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and 0 <= val <= 5


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_sephora_url(text: Optional[str]) -> bool:
    return looks_like_url(text) and ci_contains(text, 'sephora')


def looks_like_amazon_url(text: Optional[str]) -> bool:
    return looks_like_url(text) and ci_contains(text, 'amazon')


def looks_like_target_url(text: Optional[str]) -> bool:
    return looks_like_url(text) and ci_contains(text, 'target')


def looks_like_review_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_stock_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['in stock', 'out of stock', 'available', 'unavailable'])


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
    products_info = await evaluator.extract(
        prompt=prompt_extract_products_from_answer(),
        template_class=ProductsList,
        extraction_name="products_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Sephora section
    sephora_node = evaluator.add_sequential(
        id="sephora_section",
        desc="Sephora Best Sellers skincare products extraction",
        parent=root,
        critical=False
    )

    # [Perception Node] sephora.com:F1:P1 - Identify Best Sellers label
    best_sellers_mentioned = has_any_ci(answer, ['best seller', 'bestseller'])
    evaluator.add_custom_node(
        result=bool(best_sellers_mentioned),
        id="sephora_best_sellers_label",
        desc="[Perception Node] sephora.com:F1:P1 - Identify products marked as 'Best Sellers'",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F1:A1 - Multi-dimensional filtering
    skincare_mentioned = has_any_ci(answer, ['skincare', 'skin care'])
    products_count_ok = products_info and products_info.total_count and products_info.total_count >= 10
    evaluator.add_custom_node(
        result=bool(skincare_mentioned and products_count_ok),
        id="sephora_filter_skincare_count",
        desc="[Action Node] sephora.com:F1:A1 - Filter Skincare category and select 10 products",
        parent=sephora_node,
        critical=False
    )

    # Check ratings >= 4.3
    if products_info and products_info.products:
        ratings_valid = []
        for p in products_info.products:
            rating_val = extract_float(p.sephora_rating)
            if rating_val is not None:
                ratings_valid.append(rating_val >= 4.3)

        ratings_ok = len(ratings_valid) > 0 and all(ratings_valid)
    else:
        ratings_ok = False

    evaluator.add_custom_node(
        result=bool(ratings_ok),
        id="sephora_rating_filter",
        desc="[Action Node] sephora.com:F1:A1 - Verify all products have rating >= 4.3",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P2 - Extract rating data
    sephora_ratings_present = False
    if products_info and products_info.products:
        sephora_ratings_present = any(
            looks_like_rating(p.sephora_rating) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(sephora_ratings_present),
        id="sephora_rating_perception",
        desc="[Perception Node] sephora.com:F2:P2 - Extract Sephora rating values",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P7 - Extract price information
    sephora_prices_present = False
    if products_info and products_info.products:
        sephora_prices_present = any(
            p.sephora_price and contains_digits(p.sephora_price) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(sephora_prices_present),
        id="sephora_price_perception",
        desc="[Perception Node] sephora.com:F2:P7 - Extract Sephora price information",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F2:A3 - Click product cards to get details
    sephora_links_present = False
    if products_info and products_info.products:
        sephora_links_present = any(
            looks_like_sephora_url(p.sephora_link) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(sephora_links_present),
        id="sephora_product_card_click",
        desc="[Action Node] sephora.com:F2:A3 - Access product detail pages (verified by valid Sephora URLs)",
        parent=sephora_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon product search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search keyword input
    amazon_mentioned = has_any_ci(answer, ['amazon'])
    brand_product_search = False
    if products_info and products_info.products:
        brand_product_search = any(
            p.brand and p.product_name for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(amazon_mentioned and brand_product_search),
        id="amazon_search_keyword",
        desc="[Action Node] Amazon:F1:A13 - Input search keywords using 'Brand + Product Name' format",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F1:P3 - Product card image recognition
    amazon_products_found = False
    if products_info and products_info.products:
        amazon_products_found = any(
            p.amazon_price or p.amazon_rating or p.amazon_link for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(amazon_products_found),
        id="amazon_product_recognition",
        desc="[Perception Node] Amazon:F1:P3 - Recognize matching products in search results",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click product card to enter details
    amazon_links_present = False
    if products_info and products_info.products:
        amazon_links_present = any(
            looks_like_amazon_url(p.amazon_link) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(amazon_links_present),
        id="amazon_product_card_click",
        desc="[Action Node] Amazon:F5:A20 - Access Amazon product detail pages (verified by valid URLs)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P8 - Object recognition
    amazon_prices_present = False
    if products_info and products_info.products:
        amazon_prices_present = any(
            p.amazon_price and contains_digits(p.amazon_price) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(amazon_prices_present),
        id="amazon_object_recognition",
        desc="[Perception Node] Amazon:F5:P8 - Recognize correct product (verified by extracted price)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F10:P15 - Review content understanding
    amazon_reviews_present = False
    if products_info and products_info.products:
        amazon_reviews_present = any(
            looks_like_review_count(p.amazon_reviews) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(amazon_reviews_present),
        id="amazon_review_count",
        desc="[Perception Node] Amazon:F10:P15 - Extract review count from review section",
        parent=amazon_node,
        critical=False
    )

    # 3.3 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target product search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filtering
    target_mentioned = has_any_ci(answer, ['target'])
    target_products_found = False
    if products_info and products_info.products:
        target_products_found = any(
            p.target_price or p.target_stock or p.target_link for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(target_mentioned and target_products_found),
        id="target_search_filter",
        desc="[Action Node] target.com:F1:A1 - Search and filter products on Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Product list data perception
    target_prices_present = False
    if products_info and products_info.products:
        target_prices_present = any(
            p.target_price and contains_digits(p.target_price) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(target_prices_present),
        id="target_product_list_perception",
        desc="[Perception Node] target.com:F1:P1 - Perceive product data in search results (verified by price extraction)",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product card
    target_links_present = False
    if products_info and products_info.products:
        target_links_present = any(
            looks_like_target_url(p.target_link) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(target_links_present),
        id="target_product_card_click",
        desc="[Action Node] target.com:F1:A21 - Click product cards to access detail pages",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F2:A21 - Access product detail page
    evaluator.add_custom_node(
        result=bool(target_links_present),
        id="target_detail_page_access",
        desc="[Action Node] target.com:F2:A21 - Access Target product detail pages",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P6 - Price information perception
    evaluator.add_custom_node(
        result=bool(target_prices_present),
        id="target_price_perception",
        desc="[Perception Node] target.com:F2:P6 - Extract Target price information",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P7 - Stock status perception
    target_stock_present = False
    if products_info and products_info.products:
        target_stock_present = any(
            looks_like_stock_status(p.target_stock) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(target_stock_present),
        id="target_stock_perception",
        desc="[Perception Node] target.com:F2:P7 - Extract stock status information (in stock/out of stock)",
        parent=target_node,
        critical=False
    )

    # 3.4 Brand official website section (non-prefixed nodes)
    brand_website_node = evaluator.add_sequential(
        id="brand_website_section",
        desc="Brand official website information extraction",
        parent=root,
        critical=False
    )

    brand_websites_visited = False
    if products_info and products_info.products:
        brand_websites_visited = any(
            p.brand_website_link and looks_like_url(p.brand_website_link) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(brand_websites_visited),
        id="brand_website_visit",
        desc="Visit brand official websites for products",
        parent=brand_website_node,
        critical=False
    )

    brand_prices_present = False
    if products_info and products_info.products:
        brand_prices_present = any(
            p.brand_website_price and contains_digits(p.brand_website_price) for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(brand_prices_present),
        id="brand_price_extraction",
        desc="Extract official retail prices from brand websites",
        parent=brand_website_node,
        critical=False
    )

    exclusive_sets_checked = False
    if products_info and products_info.products:
        exclusive_sets_checked = any(
            p.brand_website_exclusive_sets is not None for p in products_info.products
        )

    evaluator.add_custom_node(
        result=bool(exclusive_sets_checked),
        id="brand_exclusive_sets",
        desc="Check for website-exclusive sets or special offers",
        parent=brand_website_node,
        critical=False
    )

    # 3.5 Overall completeness check
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall data completeness across all platforms",
        parent=root,
        critical=False
    )

    # Check if at least 10 products reported
    ten_products_ok = products_info and products_info.total_count and products_info.total_count >= 10
    evaluator.add_custom_node(
        result=bool(ten_products_ok),
        id="ten_products_count",
        desc="Verify at least 10 products are reported",
        parent=completeness_node,
        critical=False
    )

    # Check if products have data from all four platforms
    all_platforms_covered = False
    if products_info and products_info.products:
        for p in products_info.products:
            has_sephora = bool(p.sephora_price and p.sephora_rating and p.sephora_link)
            has_amazon = bool(p.amazon_price or p.amazon_rating or p.amazon_link)
            has_target = bool(p.target_price or p.target_stock or p.target_link)
            has_brand = bool(p.brand_website_price or p.brand_website_link)

            if has_sephora and has_amazon and has_target and has_brand:
                all_platforms_covered = True
                break

    evaluator.add_custom_node(
        result=bool(all_platforms_covered),
        id="all_platforms_data",
        desc="Verify at least one product has complete data from all four platforms",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
