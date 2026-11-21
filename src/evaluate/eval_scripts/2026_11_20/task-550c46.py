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
TASK_ID = "task-550c46"
TASK_DESCRIPTION = "I'm a photography enthusiast planning to purchase a used Sony Alpha a6400 camera body by the end of 2025.\n\nFirst, go to BestBuy.com and search for the Sony Alpha a6400 camera body. Use the filtering options to show only 'Open-Box Excellent' condition items, with a price under $800. Sort the results by 'Price: Low to High' and identify the cheapest one to establish a benchmark price.\n\nUsing this benchmark price, go to eBay.com for price comparison. Search for the same camera body ('Body Only'). Filter for 'Buy It Now' listings and 'Top Rated' sellers to ensure reliability. Find three options that are cheaper than the BestBuy benchmark price.\n\nFinally, considering that used batteries often don't last long, go to Amazon.com and search for a third-party 'dual battery charger kit' compatible with the a6400 (NP-FW50 model). The kit must have a rating of 4 stars or higher. Sort the results by 'Avg. Customer Review' and select the one with the highest rating and over 1000 reviews.\n\nOutput: The BestBuy benchmark price; the prices and seller names for the three cheaper eBay options; the full title, price, and number of reviews for the selected Amazon kit; and the direct product page links for the chosen items from all three websites."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BestBuyInfo(BaseModel):
    """BestBuy benchmark price and link for Sony Alpha a6400"""
    benchmark_price_text: Optional[str] = None
    bestbuy_link: Optional[str] = None
    condition_mention: Optional[str] = None


class EBayOption(BaseModel):
    """Single eBay option with price and seller"""
    price_text: Optional[str] = None
    seller_name: Optional[str] = None
    ebay_link: Optional[str] = None


class EBayInfo(BaseModel):
    """Three eBay options cheaper than BestBuy"""
    option1: Optional[EBayOption] = None
    option2: Optional[EBayOption] = None
    option3: Optional[EBayOption] = None


class AmazonKitInfo(BaseModel):
    """Amazon battery kit details"""
    full_title: Optional[str] = None
    price_text: Optional[str] = None
    review_count_text: Optional[str] = None
    rating_text: Optional[str] = None
    amazon_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bestbuy_from_answer() -> str:
    return """
Extract the BestBuy benchmark price information for the Sony Alpha a6400 camera body from the answer.

Return:
- benchmark_price_text: the benchmark price exactly as stated (include currency symbol if present).
- bestbuy_link: the direct BestBuy product page URL if provided.
- condition_mention: any mention of 'Open-Box Excellent' or similar condition text.

If any field is missing, set it to null.
"""


def prompt_extract_ebay_from_answer() -> str:
    return """
Extract the three eBay options for Sony Alpha a6400 camera body from the answer.

For each of the three options, extract:
- price_text: the price exactly as stated (include currency symbol if present).
- seller_name: the seller's name or username.
- ebay_link: the direct eBay product page URL if provided.

Return option1, option2, and option3. If any option or field is missing, set it to null.
"""


def prompt_extract_amazon_from_answer() -> str:
    return """
Extract the selected Amazon dual battery charger kit details from the answer.

Return:
- full_title: the complete product title exactly as stated.
- price_text: the price exactly as stated (include currency symbol if present).
- review_count_text: the number of reviews exactly as stated.
- rating_text: the star rating exactly as stated.
- amazon_link: the direct Amazon product page URL if provided.

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
    # Remove currency symbols and commas
    cleaned = re.sub(r'[$,]', '', text)
    m = re.search(r'(\d+(\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove commas and extract number
    cleaned = re.sub(r',', '', text)
    m = re.search(r'(\d+)', cleaned)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar'])


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def looks_like_review_count(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_int(text)
    return num is not None and num > 0


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    return num is not None and 0 <= num <= 5


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
    Restrict evaluator.verify to at most one usage (not used here).
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
    bestbuy_info = await evaluator.extract(
        prompt=prompt_extract_bestbuy_from_answer(),
        template_class=BestBuyInfo,
        extraction_name="bestbuy_benchmark"
    )

    ebay_info = await evaluator.extract(
        prompt=prompt_extract_ebay_from_answer(),
        template_class=EBayInfo,
        extraction_name="ebay_options"
    )

    amazon_info = await evaluator.extract(
        prompt=prompt_extract_amazon_from_answer(),
        template_class=AmazonKitInfo,
        extraction_name="amazon_kit"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 BestBuy section
    bestbuy_node = evaluator.add_sequential(
        id="bestbuy_section",
        desc="BestBuy.com - Sony Alpha a6400 Open-Box Excellent benchmark",
        parent=root,
        critical=False
    )

    # [Action Node] bestbuy.com:F8:A16 - Filter for Open-Box Excellent condition
    openbox_mention = has_any_ci(bestbuy_info.condition_mention, ['open-box', 'open box', 'excellent']) or has_any_ci(answer, ['open-box excellent', 'open box excellent'])
    evaluator.add_custom_node(
        result=bool(openbox_mention),
        id="bestbuy_filter_openbox",
        desc="[Action Node] bestbuy.com:F8:A16 - Filter to show only 'Open-Box Excellent' condition items",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A3 - Sort by Price: Low to High
    sort_mention = has_any_ci(answer, ['price: low to high', 'low to high', 'lowest price', 'cheapest'])
    benchmark_price = extract_float(bestbuy_info.benchmark_price_text)
    benchmark_price_ok = benchmark_price is not None and benchmark_price < 800
    evaluator.add_custom_node(
        result=bool(sort_mention and benchmark_price_ok),
        id="bestbuy_sort_price",
        desc="[Action Node] bestbuy.com:F1:A3 - Sort results by 'Price: Low to High' and identify cheapest option under $800",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F8:P11 - Verify Open-Box Excellent status from product page
    has_bestbuy_link = looks_like_url(bestbuy_info.bestbuy_link, 'bestbuy')
    evaluator.add_custom_node(
        result=bool(has_bestbuy_link and openbox_mention),
        id="bestbuy_verify_condition",
        desc="[Perception Node] bestbuy.com:F8:P11 - Verify the product page shows 'Open-Box Excellent' condition",
        parent=bestbuy_node,
        critical=False
    )

    # Lenient check: BestBuy link and price provided
    bestbuy_output_ok = looks_like_price(bestbuy_info.benchmark_price_text) and has_bestbuy_link
    evaluator.add_custom_node(
        result=bool(bestbuy_output_ok),
        id="bestbuy_output_complete",
        desc="BestBuy benchmark price and product link provided in output",
        parent=bestbuy_node,
        critical=False
    )

    # 3.2 eBay section
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="eBay.com - Three cheaper options with Top Rated sellers",
        parent=root,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Filter for Buy It Now listings
    buy_it_now_mention = has_any_ci(answer, ['buy it now', 'buy-it-now'])
    evaluator.add_custom_node(
        result=bool(buy_it_now_mention),
        id="ebay_filter_buy_it_now",
        desc="[Action Node] eBay:F3:A11 - Filter for 'Buy It Now' listings (not auction)",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F1:P5 - Verify Top Rated seller status
    top_rated_mention = has_any_ci(answer, ['top rated', 'top-rated'])
    sellers_provided = sum([
        1 for opt in [ebay_info.option1, ebay_info.option2, ebay_info.option3]
        if opt and opt.seller_name
    ])
    evaluator.add_custom_node(
        result=bool(top_rated_mention and sellers_provided >= 3),
        id="ebay_verify_top_rated",
        desc="[Perception Node] eBay:F1:P5 - Verify sellers have 'Top Rated' status and seller names provided",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A18 - Set price upper limit based on BestBuy benchmark
    ebay_prices = []
    for opt in [ebay_info.option1, ebay_info.option2, ebay_info.option3]:
        if opt and opt.price_text:
            p = extract_float(opt.price_text)
            if p is not None:
                ebay_prices.append(p)

    all_cheaper = len(ebay_prices) >= 3 and benchmark_price is not None and all(p < benchmark_price for p in ebay_prices)
    evaluator.add_custom_node(
        result=bool(all_cheaper),
        id="ebay_price_comparison",
        desc="[Action Node] eBay:F3:A18 - All three eBay options are cheaper than BestBuy benchmark",
        parent=ebay_node,
        critical=False
    )

    # Lenient check: Three eBay options with prices, sellers, and links
    ebay_output_complete = len(ebay_prices) >= 3 and sellers_provided >= 3
    ebay_links_count = sum([
        1 for opt in [ebay_info.option1, ebay_info.option2, ebay_info.option3]
        if opt and looks_like_url(opt.ebay_link, 'ebay')
    ])
    evaluator.add_custom_node(
        result=bool(ebay_output_complete and ebay_links_count >= 3),
        id="ebay_output_complete",
        desc="Three eBay options with prices, seller names, and product links provided",
        parent=ebay_node,
        critical=False
    )

    # 3.3 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon.com - Dual battery charger kit (NP-FW50) with 4+ stars and 1000+ reviews",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Filter for 4 stars or higher rating
    rating_value = extract_float(amazon_info.rating_text)
    rating_ok = rating_value is not None and rating_value >= 4.0
    rating_filter_mention = has_any_ci(answer, ['4 star', '4-star', 'rating'])
    evaluator.add_custom_node(
        result=bool(rating_ok and rating_filter_mention),
        id="amazon_filter_rating",
        desc="[Action Node] Amazon:F3:A15 - Filter for products with 4 stars or higher rating",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F4:A4 - Sort by Avg. Customer Review and select high-review item
    review_count = extract_int(amazon_info.review_count_text)
    review_count_ok = review_count is not None and review_count > 1000
    sort_review_mention = has_any_ci(answer, ['avg. customer review', 'customer review', 'sort'])
    evaluator.add_custom_node(
        result=bool(review_count_ok and sort_review_mention),
        id="amazon_sort_review",
        desc="[Action Node] Amazon:F4:A4 - Sort by 'Avg. Customer Review' and select item with 1000+ reviews",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F1:P3 - Verify dual battery kit from title/image
    dual_battery_mention = has_any_ci(amazon_info.full_title, ['dual', '2-pack', 'two', '2 pack', '2x']) or has_any_ci(answer, ['dual battery', 'two batteries'])
    np_fw50_mention = has_any_ci(amazon_info.full_title, ['np-fw50', 'npfw50', 'np fw50']) or has_any_ci(answer, ['np-fw50'])
    charger_mention = has_any_ci(amazon_info.full_title, ['charger']) or has_any_ci(answer, ['charger kit'])
    evaluator.add_custom_node(
        result=bool(dual_battery_mention and np_fw50_mention and charger_mention),
        id="amazon_verify_dual_kit",
        desc="[Perception Node] Amazon:F1:P3 - Verify product is a dual battery charger kit compatible with NP-FW50",
        parent=amazon_node,
        critical=False
    )

    # Lenient check: Amazon complete output
    amazon_output_ok = (
        amazon_info.full_title and len(amazon_info.full_title.strip()) > 10 and
        looks_like_price(amazon_info.price_text) and
        looks_like_review_count(amazon_info.review_count_text) and
        looks_like_rating(amazon_info.rating_text) and
        looks_like_url(amazon_info.amazon_link, 'amazon')
    )
    evaluator.add_custom_node(
        result=bool(amazon_output_ok),
        id="amazon_output_complete",
        desc="Amazon kit full title, price, review count, rating, and product link provided",
        parent=amazon_node,
        critical=False
    )

    # Compatibility mention
    a6400_compat_mention = has_any_ci(answer, ['a6400', 'alpha a6400', 'sony a6400'])
    evaluator.add_custom_node(
        result=bool(a6400_compat_mention),
        id="amazon_a6400_compatibility",
        desc="Answer mentions compatibility with Sony Alpha a6400",
        parent=amazon_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
