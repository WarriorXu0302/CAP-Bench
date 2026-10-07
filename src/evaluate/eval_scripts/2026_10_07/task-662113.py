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
TASK_ID = "task-662113"
TASK_DESCRIPTION = "I am a beauty blogger and I want to purchase a batch of foundations for a review video. I need to find products with high cost-effectiveness.\n\nFirst, go to Sephora and search for 'foundation'. Sort the results by 'Best Selling' and identify the top 5 foundations from different brands. Record the brand, full product name, Sephora price, and rating for each.\n\nThen, take the full names of these 5 products (brand + product name) and search for them on Amazon and Target, respectively. Extract the price and inventory status from each platform. For Amazon, note whether there is a 'Prime' badge and an 'Official Store' indicator. For Target, check if there are any 'Target Circle' member discounts.\n\nFinally, visit the official websites of these 5 brands one by one (e.g., Fenty Beauty, NARS). On each official website, search for the corresponding product or navigate to the 'foundation' category. Find the official website price for the product, check for any current promotional activities (e.g., 'Buy 2 Get 1 Free', 'Gift with Purchase'), and look for links to bundle deals.\n\nFor each foundation, output the following:\n*   Brand\n*   Full Product Name\n*   Sephora Price and Rating\n*   Sephora Product Page Link\n*   Amazon Price and Prime Eligibility (yes/no)\n*   Amazon Product Page Link\n*   Target Price and Circle Offer Information\n*   Target Product Page Link\n*   Brand Official Website Price\n*   Description of Current Promotions on the Official Website\n*   Official Website Product Page Link"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FoundationProduct(BaseModel):
    """Information for a single foundation product"""
    brand: Optional[str] = None
    product_name: Optional[str] = None
    sephora_price: Optional[str] = None
    sephora_rating: Optional[str] = None
    sephora_link: Optional[str] = None
    amazon_price: Optional[str] = None
    amazon_prime: Optional[str] = None
    amazon_link: Optional[str] = None
    target_price: Optional[str] = None
    target_circle_offer: Optional[str] = None
    target_link: Optional[str] = None
    official_price: Optional[str] = None
    official_promotions: Optional[str] = None
    official_link: Optional[str] = None


class FoundationsList(BaseModel):
    """List of foundation products extracted from the answer"""
    products: List[FoundationProduct] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_foundations_from_answer() -> str:
    return """
Extract all foundation products mentioned in the answer. For each product, extract:
- brand: the brand name
- product_name: the full product name
- sephora_price: the price at Sephora (with currency symbol if present)
- sephora_rating: the rating at Sephora
- sephora_link: the Sephora product page URL
- amazon_price: the price at Amazon (with currency symbol if present)
- amazon_prime: Prime eligibility (yes/no or similar text)
- amazon_link: the Amazon product page URL
- target_price: the price at Target (with currency symbol if present)
- target_circle_offer: Target Circle offer information
- target_link: the Target product page URL
- official_price: the price at the brand's official website (with currency symbol if present)
- official_promotions: description of promotions on the official website
- official_link: the official website product page URL

If any field is missing for a product, set it to null. Return all products found.
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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar', '€', '£'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    num = extract_float(text)
    return num is not None and 0 <= num <= 5


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


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.net', '.org'])


def count_valid_products(products: List[FoundationProduct]) -> int:
    count = 0
    for p in products:
        if p.brand and p.product_name:
            count += 1
    return count


def count_different_brands(products: List[FoundationProduct]) -> int:
    brands = set()
    for p in products:
        if p.brand:
            brands.add(p.brand.lower().strip())
    return len(brands)


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
    foundations_data = await evaluator.extract(
        prompt=prompt_extract_foundations_from_answer(),
        template_class=FoundationsList,
        extraction_name="foundations_list"
    )

    products = foundations_data.products if foundations_data else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Sephora section
    sephora_node = evaluator.add_sequential(
        id="sephora_section",
        desc="Sephora foundation search and product information gathering",
        parent=root,
        critical=False
    )

    # [Action Node] sephora.com:F1:A2 - Sort by Best Selling
    sort_mentioned = has_any_ci(answer, ['best selling', 'bestselling', 'best-selling', 'sort'])
    evaluator.add_custom_node(
        result=bool(sort_mentioned),
        id="sephora_sort_bestselling",
        desc="[Action Node] sephora.com:F1:A2 - Sort foundation search results by 'Best Selling'",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F1:A3 - Click product cards to get detailed information
    has_detailed_info = any(p.brand and p.product_name and p.sephora_price for p in products)
    evaluator.add_custom_node(
        result=bool(has_detailed_info),
        id="sephora_product_card_click",
        desc="[Action Node] sephora.com:F1:A3 - Click product cards to access detailed product pages",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F1:P1 - Identify Bestseller labels
    num_products = count_valid_products(products)
    num_brands = count_different_brands(products)
    top5_from_different_brands = num_products >= 5 and num_brands >= 5
    evaluator.add_custom_node(
        result=bool(top5_from_different_brands),
        id="sephora_bestseller_labels",
        desc="[Perception Node] sephora.com:F1:P1 - Identify top 5 bestselling foundations from different brands",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P2 - Extract rating data
    has_ratings = sum(1 for p in products if looks_like_rating(p.sephora_rating)) >= 3
    evaluator.add_custom_node(
        result=bool(has_ratings),
        id="sephora_rating_perception",
        desc="[Perception Node] sephora.com:F2:P2 - Extract star ratings for foundation products",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P7 - Extract price information
    has_prices = sum(1 for p in products if looks_like_price(p.sephora_price)) >= 3
    evaluator.add_custom_node(
        result=bool(has_prices),
        id="sephora_price_perception",
        desc="[Perception Node] sephora.com:F2:P7 - Extract price information for foundation products",
        parent=sephora_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon product search and information gathering",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Use search suggestions
    amazon_search_mentioned = has_any_ci(answer, ['amazon'])
    evaluator.add_custom_node(
        result=bool(amazon_search_mentioned),
        id="amazon_search_suggestions",
        desc="[Action Node] Amazon:F1:A13 - Search for foundation products on Amazon using full product names",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click product cards
    has_amazon_info = sum(1 for p in products if p.amazon_price or p.amazon_prime) >= 3
    evaluator.add_custom_node(
        result=bool(has_amazon_info),
        id="amazon_product_card_click",
        desc="[Action Node] Amazon:F5:A20 - Click product cards to access detailed product pages on Amazon",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P19 - Identify Prime shipping information
    has_prime_info = sum(1 for p in products if p.amazon_prime and p.amazon_prime.strip()) >= 3
    evaluator.add_custom_node(
        result=bool(has_prime_info),
        id="amazon_prime_perception",
        desc="[Perception Node] Amazon:F5:P19 - Identify Prime shipping eligibility for products",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Identify product status labels (official store, etc.)
    official_store_mentioned = has_any_ci(answer, ['official', 'store', 'seller'])
    evaluator.add_custom_node(
        result=bool(official_store_mentioned or has_amazon_info),
        id="amazon_status_labels",
        desc="[Perception Node] Amazon:F3:P1 - Identify product status labels like Official Store indicators",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P24 - Extract inventory status
    has_amazon_prices = sum(1 for p in products if looks_like_price(p.amazon_price)) >= 3
    evaluator.add_custom_node(
        result=bool(has_amazon_prices),
        id="amazon_inventory_status",
        desc="[Perception Node] Amazon:F5:P24 - Extract inventory status for products on Amazon",
        parent=amazon_node,
        critical=False
    )

    # 3.3 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target product search and information gathering",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product cards
    has_target_info = sum(1 for p in products if p.target_price or p.target_circle_offer) >= 3
    evaluator.add_custom_node(
        result=bool(has_target_info),
        id="target_product_card_click",
        desc="[Action Node] target.com:F1:A21 - Click product cards to access detailed product pages on Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P5 - Identify promotion labels
    circle_mentioned = has_any_ci(answer, ['circle', 'target circle'])
    evaluator.add_custom_node(
        result=bool(circle_mentioned),
        id="target_promotion_labels",
        desc="[Perception Node] target.com:F2:P5 - Identify Target Circle promotion labels",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F4:P14 - Extract Circle offer details
    has_circle_info = sum(1 for p in products if p.target_circle_offer and p.target_circle_offer.strip()) >= 3
    evaluator.add_custom_node(
        result=bool(has_circle_info),
        id="target_circle_offers",
        desc="[Perception Node] target.com:F4:P14 - Extract Target Circle member discount information",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P6 - Extract price information
    has_target_prices = sum(1 for p in products if looks_like_price(p.target_price)) >= 3
    evaluator.add_custom_node(
        result=bool(has_target_prices),
        id="target_price_perception",
        desc="[Perception Node] target.com:F2:P6 - Extract price information for products on Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P7 - Extract inventory status
    evaluator.add_custom_node(
        result=bool(has_target_info),
        id="target_inventory_status",
        desc="[Perception Node] target.com:F2:P7 - Extract inventory status for products on Target",
        parent=target_node,
        critical=False
    )

    # 3.4 Official brand websites section
    official_node = evaluator.add_sequential(
        id="official_websites_section",
        desc="Brand official website product search and information gathering",
        parent=root,
        critical=False
    )

    # Check for official website information
    has_official_prices = sum(1 for p in products if looks_like_price(p.official_price)) >= 3
    has_official_promotions = sum(1 for p in products if p.official_promotions and p.official_promotions.strip()) >= 2

    evaluator.add_custom_node(
        result=bool(has_official_prices),
        id="official_price_extraction",
        desc="Extract official website prices for foundation products",
        parent=official_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_official_promotions),
        id="official_promotions_extraction",
        desc="Extract current promotional activities from brand official websites",
        parent=official_node,
        critical=False
    )

    # Check for comprehensive coverage
    has_links = sum(1 for p in products if (
        looks_like_url(p.sephora_link) and
        looks_like_url(p.amazon_link) and
        looks_like_url(p.target_link) and
        looks_like_url(p.official_link)
    )) >= 3

    evaluator.add_custom_node(
        result=bool(has_links),
        id="comprehensive_links",
        desc="Provide product page links for all platforms (Sephora, Amazon, Target, official websites)",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
