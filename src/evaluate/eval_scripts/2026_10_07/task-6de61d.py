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
TASK_ID = "task-6de61d"
TASK_DESCRIPTION = 'I want to visit that famous cyberpunk-style residential building in Chongqing, but I don’t know its exact name. First, go to Bilibili and search for “Chongqing cyberpunk residential building” (重庆 赛博朋克 居民楼). Find a video with relatively high views, then check the comments or video description to identify the building’s exact name (it should be three Chinese characters).\n\nAfter obtaining the name, go to Airbnb and locate bookable listings around that area. I plan to stay next weekend (Saturday to Sunday), for 2 people, with a budget of no more than RMB 400 per night. Be sure to use map view to find 3 listings within walking distance of that building, preferably with river views.\n\nProvide the names, prices, and ratings of these 3 listings.\n\nIf there are no bookable Airbnb listings in that area, or if there are access/listing restrictions, record the reason for the limitation, then search on Airbnb in a nearby alternative district (e.g., Jiefangbei) and complete the same filtering process.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BilibiliInfo(BaseModel):
    """Information extracted from Bilibili search and video investigation"""
    search_keyword: Optional[str] = None
    video_title: Optional[str] = None
    building_name: Optional[str] = None
    source_type: Optional[str] = None  # "comment" or "description"


class AirbnbListingInfo(BaseModel):
    """Single Airbnb listing details"""
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None


class AirbnbSearchInfo(BaseModel):
    """Airbnb search and filtering details"""
    search_location: Optional[str] = None
    date_range: Optional[str] = None
    guest_count: Optional[str] = None
    budget_filter: Optional[str] = None
    used_map_view: Optional[bool] = None
    listings: Optional[List[AirbnbListingInfo]] = Field(default_factory=list)
    alternative_search: Optional[bool] = None
    alternative_location: Optional[str] = None
    restriction_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bilibili_info() -> str:
    return """
Extract the Bilibili search and building identification process from the answer:

- search_keyword: the search term used on Bilibili (should be related to "重庆 赛博朋克 居民楼" or "Chongqing cyberpunk residential building")
- video_title: the title of the video selected (should have relatively high views)
- building_name: the exact name of the building identified (should be three Chinese characters)
- source_type: where the building name was found - set to "comment" if from comments section, "description" if from video description, or null if unclear

If any field is missing, set it to null.
"""


def prompt_extract_airbnb_info() -> str:
    return """
Extract the Airbnb search and listing information from the answer:

- search_location: the location searched on Airbnb (should be the building name or nearby area)
- date_range: the dates selected (should be next weekend, Saturday to Sunday)
- guest_count: number of guests (should be 2)
- budget_filter: the price filter applied (should be no more than RMB 400 per night)
- used_map_view: whether map view was mentioned or used (true/false)
- listings: array of up to 3 listings, each with name, price, and rating
- alternative_search: whether an alternative search was performed due to restrictions (true/false)
- alternative_location: if alternative search was done, the alternative location (e.g., Jiefangbei)
- restriction_reason: if applicable, the reason why the original search had limitations

If any field is missing, set it to null or empty array for listings.
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


def contains_chinese(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_three_chinese_chars(text: Optional[str]) -> bool:
    if not text:
        return False
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) == 3


def mentions_baixiangju(text: Optional[str]) -> bool:
    """Check if the answer mentions 白象居 (the expected building name)"""
    if not text:
        return False
    return '白象居' in text


def contains_digits(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\d', text))


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_currency = has_any_ci(text, ['rmb', '¥', 'yuan', '元', 'cny'])
    return has_number and (has_currency or extract_number(text) is not None)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_number(text)
    if num is None:
        return False
    # Ratings are typically 0-5 or 0-10
    return 0 <= num <= 10


def check_budget_compliance(price_text: Optional[str], max_budget: float = 400) -> bool:
    """Check if the price is within budget (≤400 RMB)"""
    if not price_text:
        return False
    num = extract_number(price_text)
    if num is None:
        return False
    return num <= max_budget


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
    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_info(),
        template_class=BilibiliInfo,
        extraction_name="bilibili_info"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_info(),
        template_class=AirbnbSearchInfo,
        extraction_name="airbnb_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili search for Chongqing cyberpunk residential building",
        parent=root,
        critical=False
    )

    # Check if search was performed with correct keywords
    search_ok = (has_any_ci(answer, ['bilibili', 'b站']) and
                 (has_any_ci(answer, ['重庆', 'chongqing']) and
                  has_any_ci(answer, ['赛博朋克', 'cyberpunk', '居民楼', 'residential'])))

    evaluator.add_custom_node(
        result=bool(search_ok),
        id="bilibili_search_performed",
        desc="Search performed on Bilibili with keywords related to Chongqing cyberpunk residential building",
        parent=bilibili_node,
        critical=False
    )

    # Check if a video with high views was selected
    video_selected = bool(bilibili_info and bilibili_info.video_title and bilibili_info.video_title.strip())
    mentions_views = has_any_ci(answer, ['views', '播放', '观看', '点击', 'popular', 'high'])

    evaluator.add_custom_node(
        result=bool(video_selected and mentions_views),
        id="bilibili_video_selection",
        desc="Selected a video with relatively high views",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F3:A42 - Scroll to load comments
    comments_checked = (has_any_ci(answer, ['comment', '评论', 'scroll', '滚动']) or
                       (bilibili_info and bilibili_info.source_type == 'comment'))

    evaluator.add_custom_node(
        result=bool(comments_checked),
        id="bilibili_scroll_comments",
        desc="[Action Node] bilibili.com:F3:A42 - Scroll to load and check comments section",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A48 - Expand video description
    description_checked = (has_any_ci(answer, ['description', '简介', '描述', 'expand', '展开']) or
                          (bilibili_info and bilibili_info.source_type == 'description'))

    evaluator.add_custom_node(
        result=bool(description_checked),
        id="bilibili_expand_description",
        desc="[Action Node] bilibili.com:F1:A48 - Expand and check video description",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] - Extract building name (three Chinese characters)
    building_name_extracted = bool(bilibili_info and bilibili_info.building_name)
    building_name_correct_format = is_three_chinese_chars(bilibili_info.building_name if bilibili_info else None)
    # The expected answer is 白象居
    building_name_is_baixiangju = mentions_baixiangju(answer)

    evaluator.add_custom_node(
        result=bool(building_name_extracted and building_name_correct_format),
        id="bilibili_building_name_extracted",
        desc="[Perception Node] Extracted building name as three Chinese characters from comments or description",
        parent=bilibili_node,
        critical=False
    )

    # 3.2 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb search and listing selection",
        parent=root,
        critical=False
    )

    # Check if searched on Airbnb with the building name
    airbnb_search_ok = (has_any_ci(answer, ['airbnb', '爱彼迎']) and
                       (building_name_is_baixiangju or
                        has_any_ci(answer, ['白象居']) or
                        (airbnb_info and airbnb_info.search_location and contains_chinese(airbnb_info.search_location))))

    evaluator.add_custom_node(
        result=bool(airbnb_search_ok),
        id="airbnb_search_performed",
        desc="Searched on Airbnb using the identified building name or nearby area",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date selection (next weekend, Saturday to Sunday)
    date_selected = (airbnb_info and airbnb_info.date_range is not None) or has_any_ci(answer, ['weekend', '周末', 'saturday', 'sunday', '星期六', '星期日', '周六', '周日'])

    evaluator.add_custom_node(
        result=bool(date_selected),
        id="airbnb_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Selected dates for next weekend (Saturday to Sunday)",
        parent=airbnb_node,
        critical=False
    )

    # Check guest count filter (2 people)
    guest_filter_ok = (airbnb_info and airbnb_info.guest_count is not None) or has_any_ci(answer, ['2 people', '2人', 'two people', '两人'])

    evaluator.add_custom_node(
        result=bool(guest_filter_ok),
        id="airbnb_guest_filter",
        desc="Applied filter for 2 guests",
        parent=airbnb_node,
        critical=False
    )

    # Check budget filter (≤400 RMB per night)
    budget_mentioned = has_any_ci(answer, ['400', 'budget', '预算', 'price', '价格'])

    evaluator.add_custom_node(
        result=bool(budget_mentioned),
        id="airbnb_budget_filter",
        desc="Applied budget filter (no more than RMB 400 per night)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A37 - Map view interaction
    map_view_used = (airbnb_info and airbnb_info.used_map_view) or has_any_ci(answer, ['map', '地图', 'map view', '地图模式'])

    evaluator.add_custom_node(
        result=bool(map_view_used),
        id="airbnb_map_interaction",
        desc="[Action Node] airbnb.com:F4:A37 - Used map view to find listings within walking distance",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] - Extract 3 listings with names, prices, and ratings
    listings = airbnb_info.listings if airbnb_info and airbnb_info.listings else []
    has_three_listings = len(listings) >= 3

    # Check if listings have required fields
    listings_complete = True
    if listings:
        for listing in listings[:3]:
            if not (listing.name and listing.price and listing.rating):
                listings_complete = False
                break
    else:
        listings_complete = False

    evaluator.add_custom_node(
        result=bool(has_three_listings and listings_complete),
        id="airbnb_listings_extracted",
        desc="[Perception Node] Extracted 3 listings with names, prices, and ratings",
        parent=airbnb_node,
        critical=False
    )

    # Check if prices are within budget
    prices_within_budget = True
    if listings:
        for listing in listings[:3]:
            if listing.price and not check_budget_compliance(listing.price, 400):
                prices_within_budget = False
                break

    evaluator.add_custom_node(
        result=bool(prices_within_budget and listings),
        id="airbnb_budget_compliance",
        desc="All selected listings are within the RMB 400 per night budget",
        parent=airbnb_node,
        critical=False
    )

    # Check for river view preference mention
    river_view_mentioned = has_any_ci(answer, ['river', '江', '江景', 'river view'])

    evaluator.add_custom_node(
        result=bool(river_view_mentioned),
        id="airbnb_river_view_preference",
        desc="Mentioned preference for river views in listing selection",
        parent=airbnb_node,
        critical=False
    )

    # Check if alternative search was needed
    alternative_search_performed = (airbnb_info and airbnb_info.alternative_search) or has_any_ci(answer, ['alternative', 'jiefangbei', '解放碑', 'restriction', '限制', 'no listing', '没有房源'])

    if alternative_search_performed:
        evaluator.add_custom_node(
            result=True,
            id="airbnb_alternative_search",
            desc="Performed alternative search in nearby district due to restrictions or lack of listings",
            parent=airbnb_node,
            critical=False
        )

        # Check if restriction reason was documented
        reason_documented = (airbnb_info and airbnb_info.restriction_reason is not None)
        evaluator.add_custom_node(
            result=bool(reason_documented),
            id="airbnb_restriction_reason",
            desc="Documented the reason for limitation or restriction",
            parent=airbnb_node,
            critical=False
        )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
