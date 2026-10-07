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
TASK_ID = "task-5648f1"
TASK_DESCRIPTION = "**Task Description:**\n\nI want to upgrade my live streaming audio equipment, aiming for the sound quality level of streamer **shroud**.\n\nFirst, navigate to his Twitch channel homepage. By consulting the 'Panels' or 'About' section (which might be located below the chat or in a dedicated tab), identify the specific model of microphone he currently uses.\n\nOnce the model is identified, go to Amazon and search for this microphone, specifically looking for a **new**, in-stock listing with a rating of 4.5 stars or higher. Record its price.\n\nNext, to save money, search for the same microphone on eBay. Use the filtering options to display only 'Used' or 'Refurbished' condition items, and ensure the seller is a 'Top Rated Seller' to ensure a secure transaction. Identify the 3 lowest-priced options (including shipping) on eBay that meet the above criteria.\n\nFinally, output: The complete microphone model, the new price on Amazon, the Amazon product link, the respective prices, condition, seller rating, and corresponding links for the 3 eBay options, and calculate how much money would be saved by purchasing the lowest-priced eBay option compared to the new price on Amazon."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MicrophoneModel(BaseModel):
    """Microphone model extracted from Twitch channel"""
    model_name: Optional[str] = None


class AmazonListing(BaseModel):
    """Amazon listing details extracted from answer"""
    price_text: Optional[str] = None
    product_link: Optional[str] = None
    rating_text: Optional[str] = None
    stock_status: Optional[str] = None


class EbayListing(BaseModel):
    """Single eBay listing details"""
    price_text: Optional[str] = None
    condition: Optional[str] = None
    seller_rating: Optional[str] = None
    is_top_rated: Optional[bool] = None
    listing_link: Optional[str] = None


class EbayListings(BaseModel):
    """Three lowest-priced eBay listings"""
    listing_1: Optional[EbayListing] = None
    listing_2: Optional[EbayListing] = None
    listing_3: Optional[EbayListing] = None


class SavingsCalculation(BaseModel):
    """Savings calculation from answer"""
    savings_amount: Optional[str] = None
    calculation_present: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_microphone_model() -> str:
    return """
Extract the microphone model that shroud uses from the answer. This should be the complete model name/number found on his Twitch channel in the Panels or About section.

Return:
- model_name: the complete microphone model name exactly as stated. If not present, set null.
"""


def prompt_extract_amazon_listing() -> str:
    return """
From the answer, extract the Amazon listing details for the microphone:

- price_text: the price exactly as stated (include currency symbols and units).
- product_link: the Amazon product URL.
- rating_text: the product rating exactly as stated.
- stock_status: any indication of stock availability (e.g., "in stock", "available").

If any field is missing, set it to null.
"""


def prompt_extract_ebay_listings() -> str:
    return """
From the answer, extract the 3 lowest-priced eBay listings that meet the criteria (Used/Refurbished, Top Rated Seller).

For each of the 3 listings, return:
- price_text: the total price including shipping exactly as stated.
- condition: the item condition (Used, Refurbished, etc.).
- seller_rating: the seller rating information.
- is_top_rated: true if confirmed as Top Rated Seller, false otherwise.
- listing_link: the eBay listing URL.

If fewer than 3 listings are present, set the missing ones to null. If any field within a listing is missing, set it to null.
"""


def prompt_extract_savings() -> str:
    return """
From the answer, extract the savings calculation comparing the lowest eBay price to the Amazon new price.

Return:
- savings_amount: the calculated savings amount exactly as stated.
- calculation_present: true if a savings calculation is shown, false otherwise.

If missing, set fields to null.
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
    # Extract first number sequence
    m = re.search(r'(\d+(?:[.,]\d+)?)', text.replace(',', ''))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_currency = has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp'])
    return has_number and (has_currency or extract_float(text) is not None)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.org'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for number followed by "star" or just a number in plausible rating range
    num = extract_float(text)
    if num is not None and 0 <= num <= 5:
        return True
    return has_any_ci(text, ['star', 'rating', '/5'])


def is_used_or_refurbished(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['used', 'refurbished', 'pre-owned', 'preowned'])


def count_ebay_listings(listings: Optional[EbayListings]) -> int:
    if not listings:
        return 0
    count = 0
    if listings.listing_1 and listings.listing_1.price_text:
        count += 1
    if listings.listing_2 and listings.listing_2.price_text:
        count += 1
    if listings.listing_3 and listings.listing_3.price_text:
        count += 1
    return count


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
    mic_model = await evaluator.extract(
        prompt=prompt_extract_microphone_model(),
        template_class=MicrophoneModel,
        extraction_name="microphone_model"
    )

    amazon_info = await evaluator.extract(
        prompt=prompt_extract_amazon_listing(),
        template_class=AmazonListing,
        extraction_name="amazon_listing"
    )

    ebay_info = await evaluator.extract(
        prompt=prompt_extract_ebay_listings(),
        template_class=EbayListings,
        extraction_name="ebay_listings"
    )

    savings_info = await evaluator.extract(
        prompt=prompt_extract_savings(),
        template_class=SavingsCalculation,
        extraction_name="savings_calculation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Twitch section
    twitch_node = evaluator.add_sequential(
        id="twitch_section",
        desc="Navigate Twitch channel to identify shroud's microphone model",
        parent=root,
        critical=False
    )

    # [Action Node] twitch.tv:F8:A6 - Tab switching to Panels/About section
    twitch_tab_action = (has_any_ci(answer, ['twitch', 'shroud']) and
                         has_any_ci(answer, ['panel', 'about', 'tab']))
    evaluator.add_custom_node(
        result=bool(twitch_tab_action),
        id="twitch_tab_switching",
        desc="[Action Node] twitch.tv:F8:A6 - Navigate to Panels or About section on shroud's Twitch channel",
        parent=twitch_node,
        critical=False
    )

    # Check microphone model identified (o1)
    mic_identified = bool(mic_model and mic_model.model_name and len(mic_model.model_name.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(mic_identified),
        id="microphone_model_identified",
        desc="Microphone model successfully identified from Twitch channel",
        parent=twitch_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Search Amazon for new microphone listing with high rating",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search suggestion selection
    amazon_search_action = has_any_ci(answer, ['amazon']) and mic_identified
    evaluator.add_custom_node(
        result=bool(amazon_search_action),
        id="amazon_search_suggestion",
        desc="[Action Node] Amazon:F1:A13 - Search for the microphone using search suggestions",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P16 - Tooltip content awareness for price
    amazon_price_ok = looks_like_price(amazon_info.price_text)
    evaluator.add_custom_node(
        result=bool(amazon_price_ok),
        id="amazon_price_perception",
        desc="[Perception Node] Amazon:F5:P16 - Extract accurate price information (may require tooltip/detail inspection)",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A30 - Hover to display stock information
    amazon_stock_ok = bool(amazon_info.stock_status) or has_any_ci(answer, ['in stock', 'available', 'stock'])
    evaluator.add_custom_node(
        result=bool(amazon_stock_ok),
        id="amazon_stock_hover",
        desc="[Action Node] Amazon:F5:A30 - Verify in-stock status (may require hover interaction)",
        parent=amazon_node,
        critical=False
    )

    # Check Amazon product link (o4)
    amazon_link_ok = looks_like_url(amazon_info.product_link) and ci_contains(amazon_info.product_link, 'amazon')
    evaluator.add_custom_node(
        result=bool(amazon_link_ok),
        id="amazon_product_link",
        desc="Valid Amazon product link provided",
        parent=amazon_node,
        critical=False
    )

    # Check rating 4.5+ (o2)
    rating_num = extract_float(amazon_info.rating_text) if amazon_info.rating_text else None
    rating_ok = rating_num is not None and rating_num >= 4.5
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="amazon_rating_check",
        desc="Amazon listing has rating of 4.5 stars or higher",
        parent=amazon_node,
        critical=False
    )

    # 3.3 eBay section
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="Search eBay for used/refurbished options from Top Rated Sellers",
        parent=root,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Multi-checkbox filtering for condition
    ebay_condition_filter = has_any_ci(answer, ['ebay', 'used', 'refurbished'])
    evaluator.add_custom_node(
        result=bool(ebay_condition_filter),
        id="ebay_condition_filter",
        desc="[Action Node] eBay:F3:A11 - Apply multi-checkbox filter for Used or Refurbished condition",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A34 - Collapsible panel for seller filters
    ebay_seller_filter = has_any_ci(answer, ['top rated', 'top-rated', 'seller'])
    evaluator.add_custom_node(
        result=bool(ebay_seller_filter),
        id="ebay_seller_panel",
        desc="[Action Node] eBay:F3:A34 - Expand collapsible seller filter panel and select Top Rated Seller",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F1:P5 - Status badge recognition for Top Rated
    top_rated_badge_found = False
    if ebay_info:
        for listing in [ebay_info.listing_1, ebay_info.listing_2, ebay_info.listing_3]:
            if listing and listing.is_top_rated:
                top_rated_badge_found = True
                break
    evaluator.add_custom_node(
        result=bool(top_rated_badge_found),
        id="ebay_top_rated_badge",
        desc="[Perception Node] eBay:F1:P5 - Identify Top Rated Seller badge/status in listings",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F5:A38 - Hover bubble card for seller ratings
    seller_rating_present = False
    if ebay_info:
        for listing in [ebay_info.listing_1, ebay_info.listing_2, ebay_info.listing_3]:
            if listing and listing.seller_rating and len(listing.seller_rating.strip()) > 0:
                seller_rating_present = True
                break
    evaluator.add_custom_node(
        result=bool(seller_rating_present),
        id="ebay_seller_rating_hover",
        desc="[Action Node] eBay:F5:A38 - Obtain detailed seller rating (may require hover interaction)",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A18 - Price range/sorting to find lowest prices
    ebay_sorting_action = has_any_ci(answer, ['lowest', 'price', 'sort', 'cheapest'])
    evaluator.add_custom_node(
        result=bool(ebay_sorting_action),
        id="ebay_price_sorting",
        desc="[Action Node] eBay:F3:A18 - Apply price sorting or range filter to identify 3 lowest-priced options",
        parent=ebay_node,
        critical=False
    )

    # Check 3 eBay listings present (o5)
    ebay_count = count_ebay_listings(ebay_info)
    three_listings_ok = ebay_count >= 3
    evaluator.add_custom_node(
        result=bool(three_listings_ok),
        id="ebay_three_listings",
        desc="Three lowest-priced eBay listings identified and reported",
        parent=ebay_node,
        critical=False
    )

    # Check conditions are Used/Refurbished (o6)
    all_used_refurb = True
    if ebay_info:
        for listing in [ebay_info.listing_1, ebay_info.listing_2, ebay_info.listing_3]:
            if listing and listing.condition:
                if not is_used_or_refurbished(listing.condition):
                    all_used_refurb = False
                    break
    evaluator.add_custom_node(
        result=bool(all_used_refurb and ebay_count > 0),
        id="ebay_condition_check",
        desc="All eBay listings are Used or Refurbished condition",
        parent=ebay_node,
        critical=False
    )

    # Check Top Rated Seller confirmation (o7)
    all_top_rated = True
    if ebay_info:
        for listing in [ebay_info.listing_1, ebay_info.listing_2, ebay_info.listing_3]:
            if listing and listing.is_top_rated is not None:
                if not listing.is_top_rated:
                    all_top_rated = False
                    break
    evaluator.add_custom_node(
        result=bool(all_top_rated and ebay_count > 0),
        id="ebay_top_rated_confirmation",
        desc="All eBay listings are from Top Rated Sellers",
        parent=ebay_node,
        critical=False
    )

    # Check eBay links present
    ebay_links_ok = False
    if ebay_info:
        for listing in [ebay_info.listing_1, ebay_info.listing_2, ebay_info.listing_3]:
            if listing and looks_like_url(listing.listing_link):
                ebay_links_ok = True
                break
    evaluator.add_custom_node(
        result=bool(ebay_links_ok),
        id="ebay_listing_links",
        desc="eBay listing links provided",
        parent=ebay_node,
        critical=False
    )

    # 3.4 Calculation section
    calc_node = evaluator.add_sequential(
        id="calculation_section",
        desc="Calculate savings between Amazon new price and lowest eBay option",
        parent=root,
        critical=False
    )

    # Check savings calculation present
    savings_calc_ok = bool(savings_info and savings_info.calculation_present)
    evaluator.add_custom_node(
        result=bool(savings_calc_ok),
        id="savings_calculation_present",
        desc="Savings calculation comparing Amazon and eBay prices provided",
        parent=calc_node,
        critical=False
    )

    # Check savings amount extracted
    savings_amount_ok = looks_like_price(savings_info.savings_amount) if savings_info else False
    evaluator.add_custom_node(
        result=bool(savings_amount_ok),
        id="savings_amount",
        desc="Specific savings amount calculated and reported",
        parent=calc_node,
        critical=False
    )

    # 3.5 Completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_checks",
        desc="Overall output completeness verification",
        parent=root,
        critical=False
    )

    # All required information present
    has_mic_model = mic_identified
    has_amazon_price = amazon_price_ok
    has_amazon_link = amazon_link_ok
    has_ebay_listings = three_listings_ok
    has_savings = savings_calc_ok

    all_required_info = has_mic_model and has_amazon_price and has_amazon_link and has_ebay_listings and has_savings
    evaluator.add_custom_node(
        result=bool(all_required_info),
        id="all_required_info",
        desc="All required output elements present (model, Amazon price/link, 3 eBay listings, savings)",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
