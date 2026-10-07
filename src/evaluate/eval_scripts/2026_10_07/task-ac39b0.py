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
TASK_ID = "task-ac39b0"
TASK_DESCRIPTION = "I am a movie disc collector interested in purchasing some recently released 4K restorations from the Criterion Collection.\n\nFirst, visit the official Criterion website to identify 4K Blu-ray discs that have been newly released in recent months (e.g., the last 3-6 months). From these, select 5 titles that interest me (prioritizing drama or art house films). For each selected title, record the film title, Spine Number, official website price, and special features content (e.g., audio commentaries, documentaries, interviews).\n\nNext, use the Spine Number or film title to search for these 5 discs on Amazon. Check the new item price, if used copies are available, and whether Prime members qualify for free shipping.\n\nFinally, search for the same 5 discs on eBay. Filter for 'Buy It Now' listings (exclude auctions). For each, determine the prices for 'New' and 'Like New' conditions, assess the seller's rating, and note the shipping cost.\n\nThe final output should include, for each disc: Film Title, Spine Number, Criterion official website price, list of special features from the official website, Amazon new item price, Amazon used item price (if available), Prime free shipping eligibility, eBay lowest price for 'New' condition, eBay lowest price for 'Like New' condition, seller rating for the corresponding eBay listing, shipping cost details, and the direct links to the Criterion product page, Amazon product page, and eBay product page."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CriterionDisc(BaseModel):
    """Information for a single Criterion disc"""
    film_title: Optional[str] = None
    spine_number: Optional[str] = None
    criterion_price: Optional[str] = None
    special_features: Optional[List[str]] = Field(default_factory=list)
    release_date: Optional[str] = None
    genre: Optional[str] = None
    criterion_link: Optional[str] = None


class AmazonInfo(BaseModel):
    """Amazon information for a disc"""
    film_title: Optional[str] = None
    new_price: Optional[str] = None
    used_price: Optional[str] = None
    prime_free_shipping: Optional[bool] = None
    amazon_link: Optional[str] = None


class EbayInfo(BaseModel):
    """eBay information for a disc"""
    film_title: Optional[str] = None
    new_price: Optional[str] = None
    like_new_price: Optional[str] = None
    seller_rating: Optional[str] = None
    shipping_cost: Optional[str] = None
    ebay_link: Optional[str] = None


class ExtractedData(BaseModel):
    """All extracted data from the answer"""
    criterion_discs: List[CriterionDisc] = Field(default_factory=list)
    amazon_info: List[AmazonInfo] = Field(default_factory=list)
    ebay_info: List[EbayInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_criterion_discs() -> str:
    return """
Extract all Criterion Collection 4K disc information from the answer.

For each disc, return:
- film_title: the movie title
- spine_number: the Criterion spine number
- criterion_price: the official Criterion website price
- special_features: list of special features mentioned (audio commentaries, documentaries, interviews, etc.)
- release_date: the release date if mentioned
- genre: the genre/type if mentioned (drama, art house, etc.)
- criterion_link: the Criterion product page URL if provided

If any field is missing, set it to null or empty list for special_features.
"""


def prompt_extract_amazon_info() -> str:
    return """
Extract Amazon information for the discs from the answer.

For each disc, return:
- film_title: the movie title
- new_price: the new item price on Amazon
- used_price: the used item price if available (null if not available)
- prime_free_shipping: true if Prime free shipping is available, false otherwise, null if not mentioned
- amazon_link: the Amazon product page URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_ebay_info() -> str:
    return """
Extract eBay information for the discs from the answer.

For each disc, return:
- film_title: the movie title
- new_price: the 'New' condition price
- like_new_price: the 'Like New' condition price
- seller_rating: the seller rating percentage or score
- shipping_cost: the shipping cost details (free or amount)
- ebay_link: the eBay product page URL if provided

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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if contains currency symbol or number with decimal
    return bool(re.search(r'[\$£€¥]|\d+\.\d{2}|\d+', text))


def looks_like_spine_number(text: Optional[str]) -> bool:
    if not text:
        return False
    # Spine numbers are typically numbers, possibly with # prefix
    return bool(re.search(r'#?\d+', text))


def looks_like_percentage(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept percentages like "98.5%", "99%"
    return bool(re.search(r'\d+(\.\d+)?%', text))


def is_recent_release(date_text: Optional[str]) -> bool:
    if not date_text:
        return False
    # Check for 2025 and months Oct-Dec (10-12)
    if '2025' not in date_text:
        return False
    month_patterns = [
        r'\b(10|11|12)\b',  # numeric months
        r'\b(oct|nov|dec|october|november|december)\b'  # month names
    ]
    return any(re.search(p, date_text.lower()) for p in month_patterns)


def mentions_criterion(answer: str) -> bool:
    return has_any_ci(answer, ['criterion'])


def mentions_4k(answer: str) -> bool:
    return has_any_ci(answer, ['4k', '4k blu-ray', '4k disc'])


def mentions_drama_or_arthouse(answer: str) -> bool:
    return has_any_ci(answer, ['drama', 'art house', 'arthouse', 'art-house'])


def mentions_special_features(answer: str) -> bool:
    return has_any_ci(answer, ['special features', 'commentary', 'commentaries', 'documentary', 'interview', 'bonus'])


def mentions_amazon(answer: str) -> bool:
    return has_any_ci(answer, ['amazon'])


def mentions_ebay(answer: str) -> bool:
    return has_any_ci(answer, ['ebay'])


def mentions_prime(answer: str) -> bool:
    return has_any_ci(answer, ['prime', 'prime shipping', 'prime free shipping'])


def mentions_buy_it_now(answer: str) -> bool:
    return has_any_ci(answer, ['buy it now'])


def has_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text))


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_criterion_discs() + "\n\n" + prompt_extract_amazon_info() + "\n\n" + prompt_extract_ebay_info(),
        template_class=ExtractedData,
        extraction_name="all_disc_data"
    )

    criterion_discs = extracted.criterion_discs if extracted.criterion_discs else []
    amazon_info = extracted.amazon_info if extracted.amazon_info else []
    ebay_info = extracted.ebay_info if extracted.ebay_info else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Criterion.com section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion Collection website - 4K disc identification and details",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A3 - Filter by Recent Releases
    recent_releases_ok = any(is_recent_release(disc.release_date) for disc in criterion_discs if disc.release_date)
    evaluator.add_custom_node(
        result=bool(recent_releases_ok or has_any_ci(answer, ['recent', 'recently released', 'new release'])),
        id="criterion_filter_recent",
        desc="[Action Node] criterion.com:F1:A3 - Filter by Recent Releases category for discs released in last 3-6 months",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A1 - Filter by 4K format
    format_4k_ok = mentions_4k(answer) or any(has_any_ci(disc.criterion_link, ['4k']) for disc in criterion_discs if disc.criterion_link)
    evaluator.add_custom_node(
        result=bool(format_4k_ok),
        id="criterion_filter_4k",
        desc="[Action Node] criterion.com:F1:A1 - Filter by 4K Discs format",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A2 - Filter by Drama/Art House genres
    genre_ok = mentions_drama_or_arthouse(answer) or any(has_any_ci(disc.genre, ['drama', 'art']) for disc in criterion_discs if disc.genre)
    evaluator.add_custom_node(
        result=bool(genre_ok),
        id="criterion_filter_genre",
        desc="[Action Node] criterion.com:F1:A2 - Filter by Drama or Art House genres",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A5 - Expand filter panels
    evaluator.add_custom_node(
        result=bool(genre_ok or has_any_ci(answer, ['genre', 'category', 'filter'])),
        id="criterion_expand_filters",
        desc="[Action Node] criterion.com:F1:A5 - Expand Genres/Categories filter panels",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A9 - Click card to enter detail page
    has_details = len(criterion_discs) >= 5 and any(disc.special_features for disc in criterion_discs)
    evaluator.add_custom_node(
        result=bool(has_details),
        id="criterion_click_card",
        desc="[Action Node] criterion.com:F1:A9 - Click movie card to enter detail page for each disc",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F1:P2 - Identify release status
    has_release_info = any(disc.release_date for disc in criterion_discs)
    evaluator.add_custom_node(
        result=bool(has_release_info or recent_releases_ok),
        id="criterion_perception_release",
        desc="[Perception Node] criterion.com:F1:P2 - Identify New Releases or Released date labels",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A12 - Expand special features panel
    has_special_features = any(disc.special_features for disc in criterion_discs)
    evaluator.add_custom_node(
        result=bool(has_special_features),
        id="criterion_expand_special_features",
        desc="[Action Node] criterion.com:F2:A12 - Expand Special Features panel in detail page",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P3 - Extract special features content
    has_detailed_features = any(len(disc.special_features) > 0 for disc in criterion_discs)
    evaluator.add_custom_node(
        result=bool(has_detailed_features and mentions_special_features(answer)),
        id="criterion_perception_features",
        desc="[Perception Node] criterion.com:F2:P3 - Extract special features list (commentaries, documentaries, interviews)",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P5 - Extract format price and stock status
    has_prices = any(disc.criterion_price and looks_like_price(disc.criterion_price) for disc in criterion_discs)
    evaluator.add_custom_node(
        result=bool(has_prices),
        id="criterion_perception_price",
        desc="[Perception Node] criterion.com:F2:P5 - Extract 4K format price from detail page",
        parent=criterion_node,
        critical=False
    )

    # Check for 5 titles
    has_five_titles = len(criterion_discs) >= 5
    evaluator.add_custom_node(
        result=bool(has_five_titles),
        id="criterion_five_titles",
        desc="Selected 5 titles as requested",
        parent=criterion_node,
        critical=False
    )

    # Check for spine numbers
    has_spine_numbers = sum(1 for disc in criterion_discs if disc.spine_number and looks_like_spine_number(disc.spine_number)) >= 5
    evaluator.add_custom_node(
        result=bool(has_spine_numbers),
        id="criterion_spine_numbers",
        desc="Recorded Spine Numbers for selected titles",
        parent=criterion_node,
        critical=False
    )

    # Check for Criterion links
    has_criterion_links = sum(1 for disc in criterion_discs if disc.criterion_link and has_url(disc.criterion_link)) >= 5
    evaluator.add_custom_node(
        result=bool(has_criterion_links),
        id="criterion_product_links",
        desc="Provided direct links to Criterion product pages",
        parent=criterion_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon - Search and price check for the 5 discs",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search using spine number or film title
    amazon_search_ok = mentions_amazon(answer) and len(amazon_info) >= 5
    evaluator.add_custom_node(
        result=bool(amazon_search_ok),
        id="amazon_search",
        desc="[Action Node] Amazon:F1:A13 - Search for discs using Spine Number or film title with autocomplete",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F1:A5 - Select Movies & TV category
    evaluator.add_custom_node(
        result=bool(mentions_amazon(answer) and has_any_ci(answer, ['movie', 'blu-ray', 'disc'])),
        id="amazon_category",
        desc="[Action Node] Amazon:F1:A5 - Limit search to Movies & TV category",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click card to enter detail page
    has_amazon_details = any(info.new_price or info.used_price for info in amazon_info)
    evaluator.add_custom_node(
        result=bool(has_amazon_details),
        id="amazon_click_card",
        desc="[Action Node] Amazon:F5:A20 - Click search result card to enter product detail page",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A7 - Switch tabs to view different options
    has_new_and_used = any(info.new_price and info.used_price for info in amazon_info)
    evaluator.add_custom_node(
        result=bool(has_new_and_used or any(info.used_price is not None for info in amazon_info)),
        id="amazon_switch_tabs",
        desc="[Action Node] Amazon:F5:A7 - Switch between tabs to view new and used options",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P24 - Identify used availability
    has_used_info = any(info.used_price is not None for info in amazon_info)
    evaluator.add_custom_node(
        result=bool(has_used_info),
        id="amazon_perception_used",
        desc="[Perception Node] Amazon:F5:P24 - Identify whether used copies are available",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A30 - Hover to view Prime shipping details
    has_prime_info = any(info.prime_free_shipping is not None for info in amazon_info)
    evaluator.add_custom_node(
        result=bool(has_prime_info and mentions_prime(answer)),
        id="amazon_prime_hover",
        desc="[Action Node] Amazon:F5:A30 - Hover Prime badge to view shipping details",
        parent=amazon_node,
        critical=False
    )

    # Check for Amazon new prices
    has_amazon_new_prices = sum(1 for info in amazon_info if info.new_price and looks_like_price(info.new_price)) >= 5
    evaluator.add_custom_node(
        result=bool(has_amazon_new_prices),
        id="amazon_new_prices",
        desc="Recorded Amazon new item prices for the 5 discs",
        parent=amazon_node,
        critical=False
    )

    # Check for Amazon links
    has_amazon_links = sum(1 for info in amazon_info if info.amazon_link and has_url(info.amazon_link)) >= 5
    evaluator.add_custom_node(
        result=bool(has_amazon_links),
        id="amazon_product_links",
        desc="Provided direct links to Amazon product pages",
        parent=amazon_node,
        critical=False
    )

    # 3.3 eBay section
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="eBay - Search and price check with condition filtering",
        parent=root,
        critical=False
    )

    # [Action Node] eBay:F1:A5 - Select Movies & TV category
    ebay_search_ok = mentions_ebay(answer) and len(ebay_info) >= 5
    evaluator.add_custom_node(
        result=bool(ebay_search_ok and has_any_ci(answer, ['movie', 'blu-ray', 'disc'])),
        id="ebay_category",
        desc="[Action Node] eBay:F1:A5 - Limit search to Movies & TV category",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Filter by New and Like New condition
    has_both_conditions = any(info.new_price and info.like_new_price for info in ebay_info)
    evaluator.add_custom_node(
        result=bool(has_both_conditions or (any(info.new_price for info in ebay_info) and any(info.like_new_price for info in ebay_info))),
        id="ebay_filter_condition",
        desc="[Action Node] eBay:F3:A11 - Filter by New and Like New condition checkboxes",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A34 - Expand filter panels
    evaluator.add_custom_node(
        result=bool(has_both_conditions or has_any_ci(answer, ['condition', 'new', 'like new'])),
        id="ebay_expand_filters",
        desc="[Action Node] eBay:F3:A34 - Expand Condition filter panel",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F5:A27 - Click card to enter detail page
    has_ebay_details = any(info.seller_rating or info.shipping_cost for info in ebay_info)
    evaluator.add_custom_node(
        result=bool(has_ebay_details),
        id="ebay_click_card",
        desc="[Action Node] eBay:F5:A27 - Click product card to enter detail page",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F5:A38 - Hover to view seller info bubble
    has_seller_ratings = any(info.seller_rating and looks_like_percentage(info.seller_rating) for info in ebay_info)
    evaluator.add_custom_node(
        result=bool(has_seller_ratings),
        id="ebay_hover_seller",
        desc="[Action Node] eBay:F5:A38 - Hover seller name to view rating bubble card",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F5:P23 - Extract seller rating from bubble
    evaluator.add_custom_node(
        result=bool(has_seller_ratings),
        id="ebay_perception_rating",
        desc="[Perception Node] eBay:F5:P23 - Extract seller rating percentage from bubble card",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F3:P5 - Identify free shipping or shipping cost
    has_shipping_info = any(info.shipping_cost for info in ebay_info)
    evaluator.add_custom_node(
        result=bool(has_shipping_info),
        id="ebay_perception_shipping",
        desc="[Perception Node] eBay:F3:P5 - Identify Free Shipping tag or shipping cost",
        parent=ebay_node,
        critical=False
    )

    # Check for Buy It Now filtering
    buy_it_now_ok = mentions_buy_it_now(answer) or has_any_ci(answer, ['buy it now', 'exclude auction'])
    evaluator.add_custom_node(
        result=bool(buy_it_now_ok),
        id="ebay_buy_it_now",
        desc="Filtered for Buy It Now listings (excluded auctions)",
        parent=ebay_node,
        critical=False
    )

    # Check for eBay new prices
    has_ebay_new_prices = sum(1 for info in ebay_info if info.new_price and looks_like_price(info.new_price)) >= 5
    evaluator.add_custom_node(
        result=bool(has_ebay_new_prices),
        id="ebay_new_prices",
        desc="Recorded eBay 'New' condition prices for the 5 discs",
        parent=ebay_node,
        critical=False
    )

    # Check for eBay Like New prices
    has_ebay_like_new_prices = sum(1 for info in ebay_info if info.like_new_price and looks_like_price(info.like_new_price)) >= 5
    evaluator.add_custom_node(
        result=bool(has_ebay_like_new_prices),
        id="ebay_like_new_prices",
        desc="Recorded eBay 'Like New' condition prices for the 5 discs",
        parent=ebay_node,
        critical=False
    )

    # Check for eBay links
    has_ebay_links = sum(1 for info in ebay_info if info.ebay_link and has_url(info.ebay_link)) >= 5
    evaluator.add_custom_node(
        result=bool(has_ebay_links),
        id="ebay_product_links",
        desc="Provided direct links to eBay product pages",
        parent=ebay_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall task completeness and data integrity",
        parent=root,
        critical=False
    )

    # Check for cross-platform consistency
    all_five_complete = (len(criterion_discs) >= 5 and len(amazon_info) >= 5 and len(ebay_info) >= 5)
    evaluator.add_custom_node(
        result=bool(all_five_complete),
        id="completeness_all_platforms",
        desc="All 5 discs tracked across all three platforms (Criterion, Amazon, eBay)",
        parent=completeness_node,
        critical=False
    )

    # Check for comprehensive output structure
    has_comprehensive_output = all_five_complete and has_prices and has_amazon_new_prices and has_ebay_new_prices
    evaluator.add_custom_node(
        result=bool(has_comprehensive_output),
        id="completeness_output_structure",
        desc="Output includes all required fields for cross-platform price comparison",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
