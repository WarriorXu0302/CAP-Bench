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
TASK_ID = "task-ed7343"
TASK_DESCRIPTION = "Yesterday, I saw a deep learning lecture on Eventbrite. The speaker was Yann LeCun, and the topic was the development history of convolutional neural networks. It was scheduled for last Wednesday at 8 PM. Unfortunately, I had an engagement and missed the live broadcast. Could you help me find a recording of it?\n\nFirst, please search on YouTube for a full recording of this lecture. For any relevant videos, note down the video title, duration, viewership, and publication date.\n\nNext, check Bilibili (B站) to see if anyone has uploaded a version with Chinese subtitles or created an explanatory video about it. Please also note down the relevant information for those.\n\nFinally, visit MIT OpenCourseWare (OCW) to see if there are any courses or lecture resources related to Yann LeCun.\n\nI want to compare these platforms to determine the best learning approach – whether it's by watching the original recording, a Chinese explanatory video, or a systematic course."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class YouTubeVideoInfo(BaseModel):
    """YouTube video information extracted from the answer"""
    video_title: Optional[str] = None
    duration: Optional[str] = None
    viewership: Optional[str] = None
    publication_date: Optional[str] = None


class BilibiliVideoInfo(BaseModel):
    """Bilibili video information extracted from the answer"""
    video_title: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None
    upload_date: Optional[str] = None
    video_type: Optional[str] = None  # e.g., "Chinese subtitles", "explanation", "original"


class MITOCWInfo(BaseModel):
    """MIT OCW course/resource information extracted from the answer"""
    course_or_resource_title: Optional[str] = None
    instructor_mentioned: Optional[str] = None
    has_video_lectures: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_youtube_info() -> str:
    return """
Extract YouTube video information from the answer about Yann LeCun's lecture on convolutional neural networks.

Return:
- video_title: the title of any relevant video mentioned (if multiple, pick the most relevant one)
- duration: the video duration exactly as stated
- viewership: the view count exactly as stated
- publication_date: the publication/upload date exactly as stated

If any field is missing, set it to null.
"""


def prompt_extract_bilibili_info() -> str:
    return """
Extract Bilibili (B站) video information from the answer about Yann LeCun's lecture.

Return:
- video_title: the title of any relevant video mentioned
- duration: the video duration if stated
- view_count: the play/view count (播放量) if stated
- upload_date: the upload date if stated
- video_type: indicate if it's described as having "Chinese subtitles", being an "explanation/讲解", or "original/搬运"

If any field is missing, set it to null.
"""


def prompt_extract_mitocw_info() -> str:
    return """
Extract MIT OpenCourseWare (OCW) information from the answer related to Yann LeCun.

Return:
- course_or_resource_title: the title of any course or resource mentioned
- instructor_mentioned: whether Yann LeCun is mentioned as related to the course/resource
- has_video_lectures: whether the answer indicates the resource includes video lectures (true/false/null)

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


def mentions_platform(answer: str, platform_names: List[str]) -> bool:
    return has_any_ci(answer, platform_names)


def mentions_yann_lecun(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['yann lecun', 'lecun', 'yan lecun', '杨立昆', 'Yann LeCun'])


def mentions_cnn_topic(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['convolutional neural network', 'cnn', 'convnet', '卷积神经网络'])


def has_video_metadata(title: Optional[str], duration: Optional[str], views: Optional[str], date: Optional[str]) -> bool:
    """Check if at least 3 out of 4 metadata fields are present"""
    fields_present = sum([
        bool(title and title.strip()),
        bool(duration and duration.strip()),
        bool(views and views.strip()),
        bool(date and date.strip())
    ])
    return fields_present >= 3


def mentions_chinese_subtitles_or_explanation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['chinese subtitle', '中文字幕', 'explanation', '讲解', 'explained', '解说'])


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
    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_info(),
        template_class=YouTubeVideoInfo,
        extraction_name="youtube_video_info"
    )

    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_info(),
        template_class=BilibiliVideoInfo,
        extraction_name="bilibili_video_info"
    )

    mitocw_info = await evaluator.extract(
        prompt=prompt_extract_mitocw_info(),
        template_class=MITOCWInfo,
        extraction_name="mitocw_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube search for Yann LeCun CNN lecture recording",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video card to enter details
    youtube_mentions_platform = mentions_platform(answer, ['youtube', 'YouTube'])
    youtube_mentions_lecun = mentions_yann_lecun(answer)
    youtube_mentions_topic = mentions_cnn_topic(answer)

    evaluator.add_custom_node(
        result=bool(youtube_mentions_platform and (youtube_mentions_lecun or youtube_mentions_topic)),
        id="youtube_action_search_and_click",
        desc="[Action Node] youtube.com:F1:A22 - Search YouTube and click into video card to view details",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more search results
    youtube_multiple_results = (
        youtube_info and youtube_info.video_title and
        (has_any_ci(answer, ['videos', 'results', 'found', 'several', 'multiple']) or
         contains_digits(answer))
    )
    evaluator.add_custom_node(
        result=bool(youtube_multiple_results),
        id="youtube_action_scroll_results",
        desc="[Action Node] youtube.com:F1:A69 - Scroll to load more search results to find complete recording",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify thumbnail content relevance
    youtube_has_metadata = has_video_metadata(
        youtube_info.video_title,
        youtube_info.duration,
        youtube_info.viewership,
        youtube_info.publication_date
    )
    evaluator.add_custom_node(
        result=bool(youtube_has_metadata),
        id="youtube_perception_thumbnail_relevance",
        desc="[Perception Node] youtube.com:F1:P4 - Identify thumbnail content relevance and extract video metadata",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P30 - Compare across results to find most relevant video
    youtube_comparison_indicators = has_any_ci(answer, ['relevant', 'complete', 'full', 'recording', 'lecture', 'most'])
    evaluator.add_custom_node(
        result=bool(youtube_comparison_indicators and youtube_has_metadata),
        id="youtube_perception_compare_relevance",
        desc="[Perception Node] youtube.com:F1:P30 - Compare across search results to find most relevant complete recording",
        parent=youtube_node,
        critical=False
    )

    # 3.2 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili (B站) search for Chinese subtitles or explanation videos",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Switch sorting method
    bilibili_mentions_platform = mentions_platform(answer, ['bilibili', 'b站', 'B站'])
    bilibili_mentions_sorting = has_any_ci(answer, ['sort', '排序', 'view', 'play', '播放量'])

    evaluator.add_custom_node(
        result=bool(bilibili_mentions_platform and bilibili_mentions_sorting),
        id="bilibili_action_sort",
        desc="[Action Node] bilibili.com:F1:A5 - Switch sorting method (e.g., by view count) to find quality content",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Identify search result content relevance
    bilibili_mentions_video_type = mentions_chinese_subtitles_or_explanation(answer)
    evaluator.add_custom_node(
        result=bool(bilibili_mentions_platform and bilibili_mentions_video_type),
        id="bilibili_perception_content_relevance",
        desc="[Perception Node] bilibili.com:F1:P30 - Identify whether results are Chinese subtitle versions or explanation videos",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P11 - Identify specific content in thumbnails
    bilibili_has_metadata = has_video_metadata(
        bilibili_info.video_title,
        bilibili_info.duration,
        bilibili_info.view_count,
        bilibili_info.upload_date
    )
    evaluator.add_custom_node(
        result=bool(bilibili_has_metadata and bilibili_mentions_video_type),
        id="bilibili_perception_thumbnail_content",
        desc="[Perception Node] bilibili.com:F1:P11 - Identify specific content in thumbnails to distinguish video type (subtitle vs explanation)",
        parent=bilibili_node,
        critical=False
    )

    # 3.3 MIT OCW section
    mitocw_node = evaluator.add_sequential(
        id="mitocw_section",
        desc="MIT OpenCourseWare search for Yann LeCun related courses",
        parent=root,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F1:A7 - Multi-condition combined filtering
    mitocw_mentions_platform = mentions_platform(answer, ['mit ocw', 'opencourseware', 'mit opencourseware'])
    mitocw_mentions_filtering = has_any_ci(answer, ['filter', 'search', 'instructor', 'topic', 'course'])

    evaluator.add_custom_node(
        result=bool(mitocw_mentions_platform and mitocw_mentions_filtering),
        id="mitocw_action_filter",
        desc="[Action Node] ocw.mit.edu:F1:A7 - Use multi-condition filtering (by instructor or topic) to find relevant courses",
        parent=mitocw_node,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F1:A8 - Filter by resource type
    mitocw_mentions_video = has_any_ci(answer, ['video', 'lecture', 'recording', '视频'])
    evaluator.add_custom_node(
        result=bool(mitocw_mentions_platform and mitocw_mentions_video),
        id="mitocw_action_filter_video",
        desc="[Action Node] ocw.mit.edu:F1:A8 - Filter by resource type to find courses with video lectures",
        parent=mitocw_node,
        critical=False
    )

    # [Perception Node] ocw.mit.edu:F1:P2 - Identify course resource type markers
    mitocw_has_results = bool(mitocw_info and mitocw_info.course_or_resource_title)
    mitocw_video_indicated = mitocw_info.has_video_lectures if mitocw_info else False

    evaluator.add_custom_node(
        result=bool(mitocw_has_results and (mitocw_video_indicated or mitocw_mentions_video)),
        id="mitocw_perception_resource_type",
        desc="[Perception Node] ocw.mit.edu:F1:P2 - Identify whether courses include video lecture resources",
        parent=mitocw_node,
        critical=False
    )

    # 3.4 Cross-platform comparison
    comparison_node = evaluator.add_custom_node(
        result=bool(
            youtube_mentions_platform and
            bilibili_mentions_platform and
            mitocw_mentions_platform
        ),
        id="cross_platform_comparison",
        desc="Compare resources across all three platforms for learning approach selection",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
