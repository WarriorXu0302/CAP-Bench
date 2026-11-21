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
TASK_ID = "task-4fe5fb"
TASK_DESCRIPTION = "I'll be in New York this upcoming weekend and would like to find an affordable music performance to attend (under $30).\n\nPlease search Eventbrite for events scheduled for this upcoming weekend. Select a band or singer that seems interesting.\n\nTo ensure the quality of the live act, after identifying a performer, search for their live performance videos uploaded within the last year on YouTube (exclude official music videos; focus on actual live footage).\n\nIdentify the live video with the highest view count. Then, provide me with the name of the performance, the Eventbrite link, and the title and view count of that specific YouTube video."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventbriteEventInfo(BaseModel):
    """Event details extracted from the answer"""
    performance_name: Optional[str] = None
    eventbrite_link: Optional[str] = None
    performer_name: Optional[str] = None


class YouTubeVideoInfo(BaseModel):
    """YouTube video details extracted from the answer"""
    video_title: Optional[str] = None
    view_count_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eventbrite_info() -> str:
    return """
Extract the Eventbrite event information from the answer:

- performance_name: the name of the music performance/event found on Eventbrite
- eventbrite_link: the Eventbrite URL for the event
- performer_name: the name of the band or singer performing

If any field is missing in the answer, set it to null.
"""


def prompt_extract_youtube_info() -> str:
    return """
Extract the YouTube video information from the answer:

- video_title: the title of the live performance video with the highest view count
- view_count_text: the view count for that video exactly as stated (include units if present)

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


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or 'www.' in text.lower())


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove commas and extract numbers
    text_clean = text.replace(',', '')
    m = re.findall(r'(\d+(\.\d+)?)', text_clean)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def mentions_price_under_30(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    # Check for price mentions under $30
    if has_any_ci(answer_text, ['under $30', 'under 30', 'less than $30', 'below $30', 'affordable']):
        return True
    # Look for actual price values under 30
    prices = re.findall(r'\$\s*(\d+(?:\.\d+)?)', answer_text)
    if prices:
        try:
            return any(float(p) < 30 for p in prices)
        except Exception:
            pass
    return False


def mentions_this_weekend(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['this weekend', 'upcoming weekend', 'weekend'])


def mentions_new_york(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['new york', 'ny', 'nyc'])


def mentions_live_performance(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['live', 'live performance', 'live footage', 'live video', 'concert', 'session'])


def mentions_exclude_official(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['not official', 'exclude official', 'not music video', 'actual live', 'live footage'])


def mentions_last_year(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['last year', 'within the last year', 'uploaded within', 'past year', 'recent'])


def mentions_highest_views(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['highest view', 'most views', 'most viewed', 'highest views'])


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
    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_info(),
        template_class=EventbriteEventInfo,
        extraction_name="eventbrite_event_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_info(),
        template_class=YouTubeVideoInfo,
        extraction_name="youtube_video_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search and selection in New York for this weekend under $30",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A6 - Navigate using 'This weekend' quick filter
    weekend_filter_ok = mentions_this_weekend(answer) and has_any_ci(answer, ['eventbrite'])
    evaluator.add_custom_node(
        result=bool(weekend_filter_ok),
        id="eventbrite_action_weekend_filter",
        desc="[Action Node] eventbrite.com:F1:A6 - Click 'This weekend' quick filter tag on Eventbrite",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A3 - Apply price filter (under $30)
    price_filter_ok = mentions_price_under_30(answer) and has_any_ci(answer, ['eventbrite'])
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="eventbrite_action_price_filter",
        desc="[Action Node] eventbrite.com:F2:A3 - Apply price filter to show events under $30",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Select New York location
    location_filter_ok = mentions_new_york(answer) and has_any_ci(answer, ['eventbrite'])
    evaluator.add_custom_node(
        result=bool(location_filter_ok),
        id="eventbrite_action_location_filter",
        desc="[Action Node] eventbrite.com:F1:A12 - Select New York as the location using the location selector",
        parent=eventbrite_node,
        critical=False
    )

    # Check if event details are provided
    has_event_name = bool(eventbrite_info and eventbrite_info.performance_name and eventbrite_info.performance_name.strip())
    has_eventbrite_link = bool(eventbrite_info and eventbrite_info.eventbrite_link and looks_like_url(eventbrite_info.eventbrite_link, 'eventbrite'))
    has_performer = bool(eventbrite_info and eventbrite_info.performer_name and eventbrite_info.performer_name.strip())

    evaluator.add_custom_node(
        result=bool(has_event_name and has_eventbrite_link and has_performer),
        id="eventbrite_event_details_complete",
        desc="Provides complete event details: performance name, Eventbrite link, and performer name",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube live performance video search and selection",
        parent=root,
        critical=False
    )

    # Check if performer name from Eventbrite is used in YouTube search context
    performer_mentioned_in_youtube_context = False
    if has_performer and eventbrite_info.performer_name:
        performer_mentioned_in_youtube_context = has_any_ci(answer, [eventbrite_info.performer_name]) and has_any_ci(answer, ['youtube'])

    evaluator.add_custom_node(
        result=bool(performer_mentioned_in_youtube_context),
        id="youtube_search_with_performer",
        desc="Searches YouTube for the performer identified from Eventbrite",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Filter out official music videos, focus on live footage
    live_filter_ok = mentions_live_performance(answer) or mentions_exclude_official(answer)
    evaluator.add_custom_node(
        result=bool(live_filter_ok),
        id="youtube_perception_live_filter",
        desc="[Perception Node] youtube.com:F1:P4 - Distinguish between official music videos and actual live performance footage based on title/description semantics",
        parent=youtube_node,
        critical=False
    )

    # Check if time filter (last year) is mentioned
    time_filter_ok = mentions_last_year(answer)
    evaluator.add_custom_node(
        result=bool(time_filter_ok),
        id="youtube_time_filter",
        desc="Filters videos uploaded within the last year",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P30 - Compare and identify highest view count video
    has_video_title = bool(youtube_info and youtube_info.video_title and youtube_info.video_title.strip())
    has_view_count = bool(youtube_info and youtube_info.view_count_text and contains_digits(youtube_info.view_count_text))
    mentions_comparison = mentions_highest_views(answer)

    evaluator.add_custom_node(
        result=bool(has_video_title and has_view_count and mentions_comparison),
        id="youtube_perception_highest_views",
        desc="[Perception Node] youtube.com:F1:P30 - Compare multiple videos and identify the one with the highest view count",
        parent=youtube_node,
        critical=False
    )

    # Check if complete YouTube video information is provided
    youtube_info_complete = has_video_title and has_view_count
    evaluator.add_custom_node(
        result=bool(youtube_info_complete),
        id="youtube_video_details_complete",
        desc="Provides complete YouTube video details: title and view count",
        parent=youtube_node,
        critical=False
    )

    # 3.3 Overall completeness check
    all_required_info = (has_event_name and has_eventbrite_link and
                        has_video_title and has_view_count)

    evaluator.add_custom_node(
        result=bool(all_required_info),
        id="task_output_complete",
        desc="Provides all required outputs: performance name, Eventbrite link, video title, and view count",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
