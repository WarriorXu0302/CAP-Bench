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
TASK_ID = "task-ba6fa0"
TASK_DESCRIPTION = 'I’m moving to Chicago, IL (ZIP 60601) around the year-end holiday period and need to set up my studio. Start on **Apartments.com**: search for that ZIP code and filter for a **Studio** apartment with move-in availability in the **next available move-in window after today** (if no listings are available in that window, use the earliest move-in date shown). Set the maximum rent to **$2,000**. Select the first listing in the results and note its address and monthly rent.\n\nNext, I need a space-saving bed for this apartment. Go to **IKEA** and search for a **Sleeper sofa**. Apply filters to find one in the **Grey** color family priced between **$600 and $1,200**. Open the product page, verify it is currently **In stock** for delivery, and note the exact product name and price.\n\nFinally, I want to find a real review of this specific sofa. Search the sofa’s full product name on **YouTube**. Find a review video with at least **5,000 views**. Open the video and enable subtitles (**CC**) so I can watch it silently later.\n\n**Output:** Apartment address, monthly rent, and link; IKEA sofa name, price, stock status (In stock/Out of stock), and link; YouTube video title, view count, whether subtitles were enabled (Yes/No), and video link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApartmentInfo(BaseModel):
    """Apartment details extracted from the answer"""
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    link: Optional[str] = None


class IkeaSofaInfo(BaseModel):
    """IKEA sofa details extracted from the answer"""
    product_name: Optional[str] = None
    price: Optional[str] = None
    stock_status: Optional[str] = None
    link: Optional[str] = None


class YouTubeVideoInfo(BaseModel):
    """YouTube video details extracted from the answer"""
    video_title: Optional[str] = None
    view_count: Optional[str] = None
    subtitles_enabled: Optional[str] = None
    video_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartment_from_answer() -> str:
    return """
Extract the apartment details that the user reported from Apartments.com for ZIP 60601.

Return:
- address: the full apartment address exactly as stated
- monthly_rent: the monthly rent amount exactly as written (include currency symbols if present)
- link: the URL/link to the apartment listing if provided

If any field is missing in the answer, set it to null.
"""


def prompt_extract_ikea_sofa_from_answer() -> str:
    return """
Extract the IKEA sleeper sofa details that the user reported.

Return:
- product_name: the exact product name of the sofa
- price: the price exactly as written (include currency symbols if present)
- stock_status: the stock status text (e.g., "In stock", "Out of stock", "Available")
- link: the URL/link to the IKEA product page if provided

If any field is missing, set it to null.
"""


def prompt_extract_youtube_video_from_answer() -> str:
    return """
Extract the YouTube video details that the user reported.

Return:
- video_title: the title of the YouTube video
- view_count: the view count exactly as written (include any formatting like "5,000 views" or "5K views")
- subtitles_enabled: whether subtitles were enabled (e.g., "Yes", "No", "Enabled")
- video_link: the URL to the YouTube video

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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove commas and extract first number
    cleaned = text.replace(',', '')
    m = re.search(r'(\d+(\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_zip_60601(text: Optional[str]) -> bool:
    if not text:
        return False
    return '60601' in text


def looks_like_rent_amount(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain digits and commonly $ or "rent" or dollar amounts
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['$', 'dollar', 'usd']) or re.search(r'\d{3,}', text)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.org'])


def looks_like_apartments_com_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'apartments.com')


def looks_like_ikea_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'ikea.com')


def looks_like_youtube_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['youtube.com', 'youtu.be'])


def is_in_stock(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['in stock', 'available', 'in-stock'])


def extract_view_count_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Handle formats like "5,000 views", "5K views", "5000"
    text_lower = text.lower()
    # Remove "views" suffix
    text_clean = re.sub(r'\s*views?\s*', '', text_lower)

    # Handle K suffix
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k', text_clean)
    if k_match:
        try:
            return int(float(k_match.group(1)) * 1000)
        except Exception:
            pass

    # Extract plain number
    num = extract_number(text)
    if num is not None:
        return int(num)

    return None


def subtitles_enabled(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['yes', 'enabled', 'turned on', 'on', 'cc on', 'subtitles on'])


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
    apartment_info = await evaluator.extract(
        prompt=prompt_extract_apartment_from_answer(),
        template_class=ApartmentInfo,
        extraction_name="apartment_info"
    )

    ikea_info = await evaluator.extract(
        prompt=prompt_extract_ikea_sofa_from_answer(),
        template_class=IkeaSofaInfo,
        extraction_name="ikea_sofa_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_video_from_answer(),
        template_class=YouTubeVideoInfo,
        extraction_name="youtube_video_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Apartments.com section
    apartments_node = evaluator.add_sequential(
        id="apartments_section",
        desc="Apartments.com - Find Studio apartment in ZIP 60601 under $2,000",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A30 - Search for ZIP 60601
    zip_search_ok = (
        has_any_ci(answer, ['apartments.com', 'apartments com']) and
        has_any_ci(answer, ['60601'])
    )
    evaluator.add_custom_node(
        result=bool(zip_search_ok),
        id="apartments_search_zip",
        desc="[Action Node] apartments.com:F1:A30 - Search for ZIP code 60601 on Apartments.com",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Filter for Studio
    studio_filter_ok = has_any_ci(answer, ['studio'])
    evaluator.add_custom_node(
        result=bool(studio_filter_ok),
        id="apartments_filter_studio",
        desc="[Action Node] apartments.com:F1:A2 - Apply filter for Studio apartment type",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A6 - Set move-in date
    move_in_ok = has_any_ci(answer, ['move-in', 'move in', 'available', 'availability'])
    evaluator.add_custom_node(
        result=bool(move_in_ok),
        id="apartments_movein_date",
        desc="[Action Node] apartments.com:F1:A6 - Select next available move-in window",
        parent=apartments_node,
        critical=False
    )

    # Check rent is under $2,000
    rent_num = extract_number(apartment_info.monthly_rent)
    rent_under_2000 = rent_num is not None and rent_num <= 2000
    evaluator.add_custom_node(
        result=bool(rent_under_2000),
        id="apartments_rent_under_2000",
        desc="Apartment rent is under $2,000",
        parent=apartments_node,
        critical=False
    )

    # Verify outputs
    address_ok = bool(apartment_info.address and apartment_info.address.strip() and looks_like_zip_60601(apartment_info.address))
    rent_ok = looks_like_rent_amount(apartment_info.monthly_rent)
    link_ok = looks_like_apartments_com_url(apartment_info.link)

    evaluator.add_custom_node(
        result=bool(address_ok),
        id="apartments_address_valid",
        desc="Apartment address is provided and in ZIP 60601 area",
        parent=apartments_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(rent_ok),
        id="apartments_rent_valid",
        desc="Monthly rent amount is provided",
        parent=apartments_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(link_ok),
        id="apartments_link_valid",
        desc="Apartments.com listing link is provided",
        parent=apartments_node,
        critical=False
    )

    # 3.2 IKEA section
    ikea_node = evaluator.add_sequential(
        id="ikea_section",
        desc="IKEA - Find Grey sleeper sofa priced $600-$1,200",
        parent=root,
        critical=False
    )

    # [Action Node] ikea.com:F1:A3 - Filter for Grey color
    grey_filter_ok = (
        has_any_ci(answer, ['ikea']) and
        has_any_ci(answer, ['grey', 'gray']) and
        has_any_ci(answer, ['sleeper sofa', 'sofa'])
    )
    evaluator.add_custom_node(
        result=bool(grey_filter_ok),
        id="ikea_filter_grey",
        desc="[Action Node] ikea.com:F1:A3 - Apply Grey color filter for sleeper sofa search",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P8 - Price in range $600-$1,200
    price_num = extract_number(ikea_info.price)
    price_in_range = price_num is not None and 600 <= price_num <= 1200
    evaluator.add_custom_node(
        result=bool(price_in_range),
        id="ikea_price_range",
        desc="[Perception Node] ikea.com:F2:P8 - Sofa price is between $600 and $1,200",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P6 - Check stock status
    stock_ok = is_in_stock(ikea_info.stock_status)
    evaluator.add_custom_node(
        result=bool(stock_ok),
        id="ikea_stock_status",
        desc="[Perception Node] ikea.com:F2:P6 - Verify sofa is currently In stock for delivery",
        parent=ikea_node,
        critical=False
    )

    # Verify outputs
    product_name_ok = bool(ikea_info.product_name and ikea_info.product_name.strip())
    price_ok = bool(ikea_info.price and contains_digits(ikea_info.price))
    ikea_link_ok = looks_like_ikea_url(ikea_info.link)

    evaluator.add_custom_node(
        result=bool(product_name_ok),
        id="ikea_product_name_valid",
        desc="IKEA sofa product name is provided",
        parent=ikea_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(price_ok),
        id="ikea_price_valid",
        desc="IKEA sofa price is provided",
        parent=ikea_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ikea_link_ok),
        id="ikea_link_valid",
        desc="IKEA product page link is provided",
        parent=ikea_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube - Find review video with 5,000+ views and enable subtitles",
        parent=root,
        critical=False
    )

    # Check that search term relates to IKEA sofa
    sofa_search_ok = (
        has_any_ci(answer, ['youtube']) and
        (product_name_ok or has_any_ci(answer, ['sofa', 'sleeper']))
    )
    evaluator.add_custom_node(
        result=bool(sofa_search_ok),
        id="youtube_search_sofa",
        desc="Search YouTube for the IKEA sofa product",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P11 - Extract view count >= 5,000
    view_count_num = extract_view_count_number(youtube_info.view_count)
    views_ok = view_count_num is not None and view_count_num >= 5000
    evaluator.add_custom_node(
        result=bool(views_ok),
        id="youtube_view_count",
        desc="[Perception Node] youtube.com:F1:P11 - Video has at least 5,000 views",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Open video
    video_opened = looks_like_youtube_url(youtube_info.video_link)
    evaluator.add_custom_node(
        result=bool(video_opened),
        id="youtube_open_video",
        desc="[Action Node] youtube.com:F1:A22 - Open the YouTube video",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A11 - Enable subtitles
    subtitles_ok = subtitles_enabled(youtube_info.subtitles_enabled)
    evaluator.add_custom_node(
        result=bool(subtitles_ok),
        id="youtube_enable_subtitles",
        desc="[Action Node] youtube.com:F2:A11 - Enable subtitles (CC) on the video",
        parent=youtube_node,
        critical=False
    )

    # Verify outputs
    title_ok = bool(youtube_info.video_title and youtube_info.video_title.strip())
    view_count_ok = bool(youtube_info.view_count and contains_digits(youtube_info.view_count))
    youtube_link_ok = looks_like_youtube_url(youtube_info.video_link)

    evaluator.add_custom_node(
        result=bool(title_ok),
        id="youtube_title_valid",
        desc="YouTube video title is provided",
        parent=youtube_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(view_count_ok),
        id="youtube_view_count_valid",
        desc="YouTube view count is provided",
        parent=youtube_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(youtube_link_ok),
        id="youtube_link_valid",
        desc="YouTube video link is provided",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
