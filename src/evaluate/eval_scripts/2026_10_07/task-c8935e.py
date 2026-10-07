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
TASK_ID = "task-c8935e"
TASK_DESCRIPTION = 'I’m a programmer planning to move to Austin, TX next month, and I want to do tech livestreaming in my spare time. I need to benchmark against top streamers’ setups and estimate my housing budget. Please complete the following steps:\n\n1. Go to **Twitch** under the **“Software and Game Development”** category and find the stream with the highest current **Viewers** count. Enter the channel and open its **“About”** page (you may need to switch tabs or scroll down). In the panels, find the specific models listed for **Microphone** and **Monitor**. If that streamer does not list both items, move to the next channel with the second-highest viewers, and continue until both are found.\n2. Go to **Best Buy** and search for those two devices. Requirements: filter for products with **Customer Rating 4.5 stars or above**, and that are **in stock in the Austin, TX area** (either **Pick Up** or **Shipping** available).\n3. Finally, go to **Zillow** and find rental listings in downtown Austin (**ZIP 78701**). Requirements: rent **$2500–$4500/month**, at least **2 bedrooms (Beds: 2+)**, and prioritize listings that explicitly mention an **“Office”** or **“Study”** space in **Facts and features** or the listing description for equipment placement. Try to provide 3 matching listings; if fewer than 3 are available, return all that can be found and state the total count.\n\nOutput: the reference streamer ID, microphone model, monitor model, current Best Buy prices for both devices, and for 3 listings: address, monthly rent, original text describing Office/Study, plus the detail-page links corresponding to each step.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TwitchStreamerInfo(BaseModel):
    """Twitch streamer information extracted from the answer"""
    streamer_id: Optional[str] = None
    microphone_model: Optional[str] = None
    monitor_model: Optional[str] = None
    about_page_url: Optional[str] = None


class BestBuyDeviceInfo(BaseModel):
    """Best Buy device information extracted from the answer"""
    microphone_price: Optional[str] = None
    microphone_rating: Optional[str] = None
    microphone_url: Optional[str] = None
    monitor_price: Optional[str] = None
    monitor_rating: Optional[str] = None
    monitor_url: Optional[str] = None


class ZillowListing(BaseModel):
    """Single Zillow listing information"""
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    bedrooms: Optional[str] = None
    office_study_text: Optional[str] = None
    listing_url: Optional[str] = None


class ZillowListings(BaseModel):
    """All Zillow listings extracted from the answer"""
    listings: List[ZillowListing] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_twitch_info() -> str:
    return """
Extract the Twitch streamer information from the answer:

- streamer_id: the streamer's username or channel name
- microphone_model: the specific microphone model listed in the About page panels
- monitor_model: the specific monitor model listed in the About page panels
- about_page_url: the URL to the streamer's About page if provided

If any field is missing, set it to null.
"""


def prompt_extract_bestbuy_info() -> str:
    return """
Extract the Best Buy product information from the answer for both the microphone and monitor:

- microphone_price: the current price for the microphone (include currency symbol if present)
- microphone_rating: the customer rating for the microphone
- microphone_url: the Best Buy product page URL for the microphone
- monitor_price: the current price for the monitor (include currency symbol if present)
- monitor_rating: the customer rating for the monitor
- monitor_url: the Best Buy product page URL for the monitor

If any field is missing, set it to null.
"""


def prompt_extract_zillow_listings() -> str:
    return """
Extract all Zillow rental listings from the answer. For each listing, extract:

- address: the full street address
- monthly_rent: the monthly rent amount (include currency symbol if present)
- bedrooms: the number of bedrooms
- office_study_text: the exact text from the listing that mentions Office or Study space
- listing_url: the Zillow detail page URL for this listing

Also extract:
- total_count: the total number of listings found (if explicitly stated)

Return all listings in the listings array. If no listings are found, return an empty array.
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


def extract_price(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = re.sub(r'[,$]', '', text)
    return extract_float(cleaned)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def rating_meets_threshold(text: Optional[str], threshold: float = 4.5) -> bool:
    num = extract_float(text)
    if num is None:
        return False
    return num >= threshold


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.match(r'https?://', text, re.IGNORECASE))


def is_twitch_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'twitch.tv' in text.lower()


def is_bestbuy_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'bestbuy.com' in text.lower()


def is_zillow_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'zillow.com' in text.lower()


def looks_like_rent(text: Optional[str], min_val: float = 2500, max_val: float = 4500) -> bool:
    price = extract_price(text)
    if price is None:
        return False
    return min_val <= price <= max_val


def extract_bedroom_count(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def has_office_or_study(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['office', 'study'])


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
    Restrict evaluator.verify to at most one usage (we'll not use it here).
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
    twitch_info = await evaluator.extract(
        prompt=prompt_extract_twitch_info(),
        template_class=TwitchStreamerInfo,
        extraction_name="twitch_streamer_info"
    )

    bestbuy_info = await evaluator.extract(
        prompt=prompt_extract_bestbuy_info(),
        template_class=BestBuyDeviceInfo,
        extraction_name="bestbuy_device_info"
    )

    zillow_info = await evaluator.extract(
        prompt=prompt_extract_zillow_listings(),
        template_class=ZillowListings,
        extraction_name="zillow_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Twitch section
    twitch_node = evaluator.add_sequential(
        id="twitch_section",
        desc="Twitch - Find top Software and Game Development streamer and extract equipment info",
        parent=root,
        critical=False
    )

    # [Action Node] twitch.tv:F8:A6 - Tab switching to About page
    about_context = has_any_ci(answer, ['about', 'about page'])
    evaluator.add_custom_node(
        result=bool(about_context),
        id="twitch_tab_switch",
        desc="[Action Node] twitch.tv:F8:A6 - Switch to the About page/tab to view panels",
        parent=twitch_node,
        critical=False
    )

    # [Perception Node] twitch.tv:F8:P11 - Content recognition for hardware list
    has_mic = bool(twitch_info.microphone_model and twitch_info.microphone_model.strip())
    has_monitor = bool(twitch_info.monitor_model and twitch_info.monitor_model.strip())
    evaluator.add_custom_node(
        result=bool(has_mic and has_monitor),
        id="twitch_hardware_recognition",
        desc="[Perception Node] twitch.tv:F8:P11 - Recognize and extract microphone and monitor models from About page panels",
        parent=twitch_node,
        critical=False
    )

    # Additional checks
    has_streamer_id = bool(twitch_info.streamer_id and twitch_info.streamer_id.strip())
    evaluator.add_custom_node(
        result=bool(has_streamer_id),
        id="twitch_streamer_identified",
        desc="Streamer ID/username is provided",
        parent=twitch_node,
        critical=False
    )

    twitch_category_context = has_any_ci(answer, ['software', 'game development', 'software and game development'])
    evaluator.add_custom_node(
        result=bool(twitch_category_context),
        id="twitch_category_context",
        desc="Answer mentions the Software and Game Development category context",
        parent=twitch_node,
        critical=False
    )

    # 3.2 Best Buy section
    bestbuy_node = evaluator.add_sequential(
        id="bestbuy_section",
        desc="Best Buy - Search for devices with rating and stock filters",
        parent=root,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A2 - Multi-condition filtering (rating 4.5+)
    mic_rating_ok = rating_meets_threshold(bestbuy_info.microphone_rating, 4.5)
    mon_rating_ok = rating_meets_threshold(bestbuy_info.monitor_rating, 4.5)
    evaluator.add_custom_node(
        result=bool(mic_rating_ok and mon_rating_ok),
        id="bestbuy_rating_filter",
        desc="[Action Node] bestbuy.com:F1:A2 - Filter products with customer rating 4.5 stars or above",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F1:P1 - Stock/status awareness (Austin, TX)
    austin_stock_context = has_any_ci(answer, ['austin', 'stock', 'pick up', 'shipping', 'available', 'in stock'])
    has_bestbuy_urls = is_bestbuy_url(bestbuy_info.microphone_url) and is_bestbuy_url(bestbuy_info.monitor_url)
    evaluator.add_custom_node(
        result=bool(austin_stock_context and has_bestbuy_urls),
        id="bestbuy_stock_awareness",
        desc="[Perception Node] bestbuy.com:F1:P1 - Verify stock availability in Austin, TX area (Pick Up or Shipping)",
        parent=bestbuy_node,
        critical=False
    )

    # Additional checks
    has_mic_price = bool(bestbuy_info.microphone_price and contains_digits(bestbuy_info.microphone_price))
    has_mon_price = bool(bestbuy_info.monitor_price and contains_digits(bestbuy_info.monitor_price))
    evaluator.add_custom_node(
        result=bool(has_mic_price and has_mon_price),
        id="bestbuy_prices_extracted",
        desc="Prices for both microphone and monitor are extracted",
        parent=bestbuy_node,
        critical=False
    )

    # 3.3 Zillow section
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow - Find rental listings in downtown Austin (78701) with Office/Study space",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Location search suggestion (ZIP 78701)
    has_zip_78701 = False
    for listing in zillow_info.listings:
        if listing.address and '78701' in listing.address:
            has_zip_78701 = True
            break
    zip_context = has_any_ci(answer, ['78701', 'downtown austin'])
    evaluator.add_custom_node(
        result=bool(has_zip_78701 or zip_context),
        id="zillow_location_search",
        desc="[Action Node] zillow.com:F1:A35 - Search for downtown Austin location (ZIP 78701)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A12 - Slider/range filter (rent $2500-$4500)
    rent_filter_ok = False
    for listing in zillow_info.listings:
        if listing.monthly_rent and looks_like_rent(listing.monthly_rent, 2500, 4500):
            rent_filter_ok = True
            break
    evaluator.add_custom_node(
        result=bool(rent_filter_ok),
        id="zillow_rent_range_filter",
        desc="[Action Node] zillow.com:F2:A12 - Apply rent range filter ($2500-$4500/month)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A10 - Dropdown selection filter (2+ bedrooms)
    bedroom_filter_ok = False
    for listing in zillow_info.listings:
        bed_count = extract_bedroom_count(listing.bedrooms)
        if bed_count and bed_count >= 2:
            bedroom_filter_ok = True
            break
    evaluator.add_custom_node(
        result=bool(bedroom_filter_ok),
        id="zillow_bedroom_filter",
        desc="[Action Node] zillow.com:F2:A10 - Filter for at least 2 bedrooms (Beds: 2+)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A23 - Expand/collapse panel (Facts and features)
    facts_context = has_any_ci(answer, ['facts', 'features', 'facts and features'])
    evaluator.add_custom_node(
        result=bool(facts_context),
        id="zillow_expand_facts_panel",
        desc="[Action Node] zillow.com:F4:A23 - Expand Facts and features panel to view details",
        parent=zillow_node,
        critical=False
    )

    # [Perception Node] zillow.com:F4:P18 - Text semantic understanding (Office/Study)
    office_study_ok = False
    for listing in zillow_info.listings:
        if listing.office_study_text and has_office_or_study(listing.office_study_text):
            office_study_ok = True
            break
    evaluator.add_custom_node(
        result=bool(office_study_ok),
        id="zillow_office_study_understanding",
        desc="[Perception Node] zillow.com:F4:P18 - Understand and identify Office or Study space in listing features",
        parent=zillow_node,
        critical=False
    )

    # Additional checks
    has_at_least_one_listing = len(zillow_info.listings) >= 1
    evaluator.add_custom_node(
        result=bool(has_at_least_one_listing),
        id="zillow_has_listings",
        desc="At least one rental listing is provided",
        parent=zillow_node,
        critical=False
    )

    has_three_listings = len(zillow_info.listings) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_listings),
        id="zillow_three_listings",
        desc="Three rental listings are provided as requested",
        parent=zillow_node,
        critical=False
    )

    all_listings_have_urls = all(is_zillow_url(listing.listing_url) for listing in zillow_info.listings if listing.listing_url)
    has_any_listing_urls = any(is_zillow_url(listing.listing_url) for listing in zillow_info.listings)
    evaluator.add_custom_node(
        result=bool(has_any_listing_urls),
        id="zillow_listing_urls",
        desc="Zillow listing detail page URLs are provided",
        parent=zillow_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
