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
TASK_ID = "task-bc96e7"
TASK_DESCRIPTION = "I'm going to Seattle next week to look for apartments, and I've identified a few promising ones on Apartments.com that I'd like to do some preliminary research on.\n\nPlease search for apartments in Seattle on Apartments.com with a budget of $1500-$2500 per month, requiring 1 or 2 bedrooms. Identify 3-5 highly-rated options and record their names, addresses, rent, and ratings.\n\nNext, search on YouTube and Bilibili for any real-footage or tenant review videos for these specific apartments. Prioritize videos published within the last year and note any complaints regarding poor management, slow maintenance, or excessive noise. If relevant videos are found, record their titles and links.\n\nFinally, arrange these apartments by geographical location to facilitate a logical viewing route on the day of my visit, avoiding excessive travel back and forth."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApartmentInfo(BaseModel):
    """Apartment details extracted from the answer"""
    apartment_names: Optional[List[str]] = Field(default_factory=list)
    addresses: Optional[List[str]] = Field(default_factory=list)
    rent_amounts: Optional[List[str]] = Field(default_factory=list)
    ratings: Optional[List[str]] = Field(default_factory=list)
    count: Optional[int] = None


class VideoInfo(BaseModel):
    """Video information extracted from the answer"""
    youtube_videos: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    bilibili_videos: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    mentions_recent: Optional[bool] = None
    mentions_complaints: Optional[bool] = None


class RouteInfo(BaseModel):
    """Route planning information extracted from the answer"""
    has_geographical_order: Optional[bool] = None
    mentions_route_optimization: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartments_from_answer() -> str:
    return """
Extract the apartment information from the answer that the user found on Apartments.com in Seattle.

Return:
- apartment_names: list of apartment names mentioned
- addresses: list of addresses mentioned
- rent_amounts: list of rent/price amounts mentioned (with dollar signs or units if present)
- ratings: list of rating values mentioned
- count: total number of apartments identified (should be 3-5)

If any field is missing or empty, set it to an empty list or null.
"""


def prompt_extract_videos_from_answer() -> str:
    return """
Extract video search information from the answer.

Return:
- youtube_videos: list of YouTube video objects with title and link if mentioned
- bilibili_videos: list of Bilibili video objects with title and link if mentioned
- mentions_recent: true if answer mentions searching for recent/last year videos
- mentions_complaints: true if answer mentions looking for complaints about management, maintenance, or noise

If any field is missing, set it to empty list, null, or false as appropriate.
"""


def prompt_extract_route_from_answer() -> str:
    return """
Extract route planning information from the answer.

Return:
- has_geographical_order: true if the answer presents apartments in some geographical/location-based order
- mentions_route_optimization: true if the answer mentions optimizing viewing route or avoiding back-and-forth travel

If any field is missing, set it to null or false.
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
    has_dollar = '$' in text or ci_contains(text, 'dollar')
    has_number = contains_digits(text)
    return has_dollar and has_number


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept star ratings, numeric ratings, or mentions of "rating"
    has_star = '★' in text or ci_contains(text, 'star')
    has_number = contains_digits(text)
    has_rating_word = ci_contains(text, 'rating')
    return (has_number and (has_star or has_rating_word)) or (has_number and '/' in text)


def extract_numbers_in_range(text: Optional[str], min_val: int, max_val: int) -> List[float]:
    if not text:
        return []
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    result = []
    for n in numbers:
        try:
            val = float(n)
            if min_val <= val <= max_val:
                result.append(val)
        except:
            pass
    return result


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
        prompt=prompt_extract_apartments_from_answer(),
        template_class=ApartmentInfo,
        extraction_name="apartment_info"
    )

    video_info = await evaluator.extract(
        prompt=prompt_extract_videos_from_answer(),
        template_class=VideoInfo,
        extraction_name="video_info"
    )

    route_info = await evaluator.extract(
        prompt=prompt_extract_route_from_answer(),
        template_class=RouteInfo,
        extraction_name="route_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Apartments.com section
    apartments_node = evaluator.add_sequential(
        id="apartments_section",
        desc="Apartments.com search and filtering for Seattle apartments",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A1 - Price range filtering
    price_range_ok = (has_any_ci(answer, ['1500', '2500']) or
                      (extract_numbers_in_range(answer, 1500, 2500)))
    evaluator.add_custom_node(
        result=bool(price_range_ok),
        id="apartments_price_filter",
        desc="[Action Node] apartments.com:F1:A1 - Filter apartments by price range $1500-$2500",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Bedroom count filtering
    bedroom_filter_ok = has_any_ci(answer, ['1 bedroom', '2 bedroom', '1-bedroom', '2-bedroom', '1 bed', '2 bed'])
    evaluator.add_custom_node(
        result=bool(bedroom_filter_ok),
        id="apartments_bedroom_filter",
        desc="[Action Node] apartments.com:F1:A2 - Filter apartments by bedroom count (1 or 2 bedrooms)",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A3 - Sort by rating
    rating_sort_ok = has_any_ci(answer, ['rating', 'highly-rated', 'highly rated', 'high rating', 'top rated'])
    evaluator.add_custom_node(
        result=bool(rating_sort_ok),
        id="apartments_rating_sort",
        desc="[Action Node] apartments.com:F1:A3 - Sort apartments by rating to find highly-rated options",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A7 - Map zoom and drag
    map_interaction_ok = has_any_ci(answer, ['map', 'location', 'geographical', 'geography', 'area'])
    evaluator.add_custom_node(
        result=bool(map_interaction_ok),
        id="apartments_map_interaction",
        desc="[Action Node] apartments.com:F1:A7 - Use map to view apartment location distribution",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A8 - Map marker interaction
    map_marker_ok = map_interaction_ok and has_any_ci(answer, ['location', 'position', 'address'])
    evaluator.add_custom_node(
        result=bool(map_marker_ok),
        id="apartments_map_markers",
        desc="[Action Node] apartments.com:F1:A8 - Click map markers to preview apartment locations",
        parent=apartments_node,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P4 - Map location understanding
    location_understanding_ok = (route_info and route_info.has_geographical_order) or has_any_ci(answer, ['geographical', 'location', 'area', 'neighborhood'])
    evaluator.add_custom_node(
        result=bool(location_understanding_ok),
        id="apartments_location_understanding",
        desc="[Perception Node] apartments.com:F1:P4 - Understand apartment spatial distribution on map",
        parent=apartments_node,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P5 - Map distribution awareness
    distribution_ok = (route_info and route_info.mentions_route_optimization) or has_any_ci(answer, ['route', 'travel', 'distance', 'close', 'nearby', 'cluster'])
    evaluator.add_custom_node(
        result=bool(distribution_ok),
        id="apartments_distribution_awareness",
        desc="[Perception Node] apartments.com:F1:P5 - Recognize apartment density and relative positions on map",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F2:A12 - Click card to view details
    apt_count_ok = apartment_info and apartment_info.count and 3 <= apartment_info.count <= 5
    has_names = apartment_info and apartment_info.apartment_names and len(apartment_info.apartment_names) >= 3
    has_addresses = apartment_info and apartment_info.addresses and len(apartment_info.addresses) >= 3
    details_click_ok = has_names or has_addresses
    evaluator.add_custom_node(
        result=bool(details_click_ok),
        id="apartments_view_details",
        desc="[Action Node] apartments.com:F2:A12 - Click apartment cards to view details (name, address, rent, rating)",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F11:A35 - Click review list
    review_check_ok = has_any_ci(answer, ['review', 'rating', 'tenant', 'resident'])
    evaluator.add_custom_node(
        result=bool(review_check_ok),
        id="apartments_view_reviews",
        desc="[Action Node] apartments.com:F11:A35 - View resident reviews to understand reputation",
        parent=apartments_node,
        critical=False
    )

    # [Perception Node] apartments.com:F11:P15 - Review content understanding
    has_rent_info = apartment_info and apartment_info.rent_amounts and len(apartment_info.rent_amounts) >= 3
    has_rating_info = apartment_info and apartment_info.ratings and len(apartment_info.ratings) >= 3
    review_understanding_ok = has_rent_info and has_rating_info
    evaluator.add_custom_node(
        result=bool(review_understanding_ok),
        id="apartments_review_understanding",
        desc="[Perception Node] apartments.com:F11:P15 - Extract and understand apartment ratings and key details",
        parent=apartments_node,
        critical=False
    )

    # 3.2 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube search for apartment review videos",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video card
    yt_has_videos = video_info and video_info.youtube_videos and len(video_info.youtube_videos) > 0
    evaluator.add_custom_node(
        result=bool(yt_has_videos or has_any_ci(answer, ['youtube'])),
        id="youtube_video_click",
        desc="[Action Node] youtube.com:F1:A22 - Click video cards from search results",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more
    scroll_mention_ok = has_any_ci(answer, ['scroll', 'more', 'additional', 'several', 'multiple'])
    evaluator.add_custom_node(
        result=bool(scroll_mention_ok),
        id="youtube_scroll_results",
        desc="[Action Node] youtube.com:F1:A69 - Scroll to load more search results",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Video thumbnail content recognition
    thumbnail_recognition_ok = yt_has_videos or has_any_ci(answer, ['video', 'footage', 'tour', 'review'])
    evaluator.add_custom_node(
        result=bool(thumbnail_recognition_ok),
        id="youtube_thumbnail_recognition",
        desc="[Perception Node] youtube.com:F1:P4 - Recognize relevant apartment content from video thumbnails",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P30 - Cross-video similarity comparison
    relevance_check_ok = yt_has_videos and has_any_ci(answer, ['relevant', 'specific', 'apartment', 'building'])
    evaluator.add_custom_node(
        result=bool(relevance_check_ok),
        id="youtube_relevance_comparison",
        desc="[Perception Node] youtube.com:F1:P30 - Compare search results to identify most relevant apartment videos",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Filter by upload date
    date_filter_ok = (video_info and video_info.mentions_recent) or has_any_ci(answer, ['recent', 'last year', 'past year', 'within', 'year'])
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="youtube_date_filter",
        desc="[Action Node] youtube.com:F9:A4 - Filter videos by upload date (last year)",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A43 - Video playback control
    playback_ok = has_any_ci(answer, ['watch', 'view', 'play', 'check'])
    evaluator.add_custom_node(
        result=bool(playback_ok),
        id="youtube_playback",
        desc="[Action Node] youtube.com:F2:A43 - Play videos to review content",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A44 - Progress seeking
    seeking_ok = has_any_ci(answer, ['skip', 'fast', 'quickly', 'scan', 'browse'])
    evaluator.add_custom_node(
        result=bool(seeking_ok),
        id="youtube_progress_seek",
        desc="[Action Node] youtube.com:F2:A44 - Skip through video to find relevant complaints",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F2:P21 - Video content understanding
    complaints_mentioned = (video_info and video_info.mentions_complaints) or has_any_ci(answer, ['management', 'maintenance', 'noise', 'complaint', 'issue', 'problem'])
    evaluator.add_custom_node(
        result=bool(complaints_mentioned),
        id="youtube_content_understanding",
        desc="[Perception Node] youtube.com:F2:P21 - Understand video content about management, maintenance, or noise issues",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F5:A28 - Scroll comments
    comments_ok = has_any_ci(answer, ['comment', 'feedback', 'user', 'viewer'])
    evaluator.add_custom_node(
        result=bool(comments_ok),
        id="youtube_scroll_comments",
        desc="[Action Node] youtube.com:F5:A28 - Scroll through comment section to find tenant feedback",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F5:P18 - Comment content understanding
    comment_understanding_ok = comments_ok and complaints_mentioned
    evaluator.add_custom_node(
        result=bool(comment_understanding_ok),
        id="youtube_comment_understanding",
        desc="[Perception Node] youtube.com:F5:P18 - Extract specific complaints from comments about management/maintenance/noise",
        parent=youtube_node,
        critical=False
    )

    # 3.3 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili search for apartment review videos",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by date
    bb_has_videos = video_info and video_info.bilibili_videos and len(video_info.bilibili_videos) > 0
    bb_date_sort_ok = has_any_ci(answer, ['bilibili']) and date_filter_ok
    evaluator.add_custom_node(
        result=bool(bb_date_sort_ok),
        id="bilibili_date_sort",
        desc="[Action Node] bilibili.com:F1:A5 - Sort Bilibili search results by newest first",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Video content relevance recognition
    bb_relevance_ok = bb_has_videos or (has_any_ci(answer, ['bilibili']) and has_any_ci(answer, ['video', 'apartment']))
    evaluator.add_custom_node(
        result=bool(bb_relevance_ok),
        id="bilibili_relevance_recognition",
        desc="[Perception Node] bilibili.com:F1:P30 - Identify relevant apartment review videos from thumbnails and titles",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F2:A67 - Playback control
    bb_playback_ok = has_any_ci(answer, ['bilibili']) and playback_ok
    evaluator.add_custom_node(
        result=bool(bb_playback_ok),
        id="bilibili_playback",
        desc="[Action Node] bilibili.com:F2:A67 - Play Bilibili videos to check content",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F2:P23 - Video content understanding
    bb_content_ok = bb_has_videos and complaints_mentioned
    evaluator.add_custom_node(
        result=bool(bb_content_ok),
        id="bilibili_content_understanding",
        desc="[Perception Node] bilibili.com:F2:P23 - Understand Bilibili video content to verify if it's real apartment footage",
        parent=bilibili_node,
        critical=False
    )

    # 3.4 Route planning section
    route_node = evaluator.add_sequential(
        id="route_section",
        desc="Geographical route planning for apartment visits",
        parent=root,
        critical=False
    )

    # Check if apartments are arranged by location
    route_arranged_ok = route_info and route_info.has_geographical_order
    evaluator.add_custom_node(
        result=bool(route_arranged_ok),
        id="route_geographical_order",
        desc="Apartments arranged in geographical order for efficient viewing route",
        parent=route_node,
        critical=False
    )

    # Check if route optimization is mentioned
    route_optimized_ok = route_info and route_info.mentions_route_optimization
    evaluator.add_custom_node(
        result=bool(route_optimized_ok),
        id="route_optimization_mentioned",
        desc="Route optimization mentioned to minimize back-and-forth travel",
        parent=route_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
