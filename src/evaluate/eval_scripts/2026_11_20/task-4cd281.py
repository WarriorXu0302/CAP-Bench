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
TASK_ID = "task-4cd281"
TASK_DESCRIPTION = 'I want to systematically learn MLOps (Machine Learning Operations) and plan to earn a certificate on Coursera. Please help me find relevant courses, but I have two strict requirements: the difficulty level must be **"Intermediate"**, and the duration must be **"1–3 Months."**  \n\nTo ensure quality, please prioritize sorting by **Highest Rated**. If that sort option is unavailable on the current page or only visible after login, then, among the results that already match the two filters above, choose the course with the highest rating from the top few visible results and state the basis for your selection. After selecting a course, note the instructor’s name.  \n\nSince I’m concerned the teaching style might be too dry, I’d like to verify it. Please search YouTube for the instructor’s name and find a publicly available talk or long-form teaching video with the **highest view count** and a duration of **more than 20 minutes** (no Shorts or brief promo clips).  \n\nFinally, provide the **course title**, **instructor name**, and the **YouTube video title and link**. If you encounter login/paywall restrictions on Coursera that prevent viewing full course details, record where the restriction occurs and continue the task using all visible information.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CourseraInfo(BaseModel):
    """Course information extracted from the answer for Coursera MLOps search"""
    course_title: Optional[str] = None
    instructor_name: Optional[str] = None
    difficulty_mentioned: Optional[str] = None
    duration_mentioned: Optional[str] = None
    rating_mentioned: Optional[str] = None


class YouTubeInfo(BaseModel):
    """YouTube video information extracted from the answer"""
    video_title: Optional[str] = None
    video_link: Optional[str] = None
    view_count_mentioned: Optional[str] = None
    duration_mentioned: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_coursera_from_answer() -> str:
    return """
Extract the Coursera course details reported in the answer for MLOps learning:

Return:
- course_title: the exact title of the course selected. If not present, set null.
- instructor_name: the instructor's name for the course. If not present, set null.
- difficulty_mentioned: any mention of the difficulty level (e.g., "Intermediate"). If not present, set null.
- duration_mentioned: any mention of the course duration (e.g., "1-3 Months", "2 months"). If not present, set null.
- rating_mentioned: any mention of rating or review score. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_youtube_from_answer() -> str:
    return """
Extract the YouTube video details reported in the answer for the instructor:

Return:
- video_title: the title of the YouTube video found. If not present, set null.
- video_link: the URL or link to the YouTube video. If not present, set null.
- view_count_mentioned: any mention of view count or views. If not present, set null.
- duration_mentioned: any mention of video duration (e.g., "30 minutes", ">20 min"). If not present, set null.

If any field is missing in the answer, set it to null.
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


def looks_like_intermediate_difficulty(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'intermediate')


def looks_like_duration_1_3_months(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'1\s*[-–—]\s*3\s*month',
        r'1\s*to\s*3\s*month',
        r'2\s*month',
        r'3\s*month'
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept mentions of rating, stars, score, or numeric values suggesting ratings
    keywords = ['rating', 'rated', 'star', 'score', 'review']
    return has_any_ci(text, keywords) or bool(re.search(r'\d+(\.\d+)?/\d+', text))


def looks_like_youtube_link(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'youtube\.com/watch',
        r'youtu\.be/',
        r'youtube\.com/.*v='
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def looks_like_view_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['view', 'views', 'watched']) and contains_digits(text)


def looks_like_duration_over_20min(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept mentions like "30 minutes", "25 min", "1 hour", ">20", "more than 20"
    over_20 = re.search(r'(>|more\s+than|over)\s*20', text.lower())
    specific_duration = re.search(r'(\d+)\s*(min|minute|hour|hr)', text.lower())
    if over_20:
        return True
    if specific_duration:
        num = int(specific_duration.group(1))
        unit = specific_duration.group(2).lower()
        if 'hour' in unit or 'hr' in unit:
            return True
        if ('min' in unit) and num >= 20:
            return True
    return False


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
    coursera_info = await evaluator.extract(
        prompt=prompt_extract_coursera_from_answer(),
        template_class=CourseraInfo,
        extraction_name="coursera_course_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_from_answer(),
        template_class=YouTubeInfo,
        extraction_name="youtube_video_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Coursera section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera MLOps course search with filtering and sorting",
        parent=root,
        critical=False
    )

    # [Action Node] coursera.org:F2:A2 - Apply difficulty filter (Intermediate)
    difficulty_filter_ok = (
        has_any_ci(answer, ['coursera', 'mlops']) and
        looks_like_intermediate_difficulty(coursera_info.difficulty_mentioned or answer)
    )
    evaluator.add_custom_node(
        result=bool(difficulty_filter_ok),
        id="coursera_action_difficulty_filter",
        desc="[Action Node] coursera.org:F2:A2 - Apply difficulty filter to select 'Intermediate' level courses",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A10 - Apply duration filter (1-3 Months)
    duration_filter_ok = (
        has_any_ci(answer, ['coursera']) and
        looks_like_duration_1_3_months(coursera_info.duration_mentioned or answer)
    )
    evaluator.add_custom_node(
        result=bool(duration_filter_ok),
        id="coursera_action_duration_filter",
        desc="[Action Node] coursera.org:F2:A10 - Apply duration filter to select '1-3 Months' courses",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Apply sort by highest rated
    sort_by_rating_ok = (
        has_any_ci(answer, ['coursera']) and
        (has_any_ci(answer, ['highest rated', 'sort', 'rating', 'top rated']) or
         looks_like_rating(coursera_info.rating_mentioned or answer))
    )
    evaluator.add_custom_node(
        result=bool(sort_by_rating_ok),
        id="coursera_action_sort_rating",
        desc="[Action Node] coursera.org:F1:A12 - Sort courses by 'Highest Rated' or select based on rating",
        parent=coursera_node,
        critical=False
    )

    # Extract course and instructor information
    course_title_ok = bool(coursera_info.course_title and coursera_info.course_title.strip())
    instructor_name_ok = bool(coursera_info.instructor_name and coursera_info.instructor_name.strip())

    evaluator.add_custom_node(
        result=bool(course_title_ok and instructor_name_ok),
        id="coursera_extract_course_instructor",
        desc="Extract course title and instructor name from the selected course",
        parent=coursera_node,
        critical=False
    )

    # Check mentions of MLOps context
    mlops_context_ok = has_any_ci(answer, ['mlops', 'machine learning operations', 'ml ops'])
    evaluator.add_custom_node(
        result=bool(mlops_context_ok),
        id="coursera_mentions_mlops",
        desc="Mentions MLOps or Machine Learning Operations context in the search",
        parent=coursera_node,
        critical=False
    )

    # 3.2 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube search for instructor's video with view count and duration criteria",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Search and scan for video with highest views and >20 min duration
    youtube_search_ok = (
        has_any_ci(answer, ['youtube']) and
        instructor_name_ok and
        has_any_ci(answer, ['search', 'find', 'video'])
    )
    evaluator.add_custom_node(
        result=bool(youtube_search_ok),
        id="youtube_action_search_instructor",
        desc="[Action Node] youtube.com:F1:A69 - Search YouTube for the instructor's name and scan video list",
        parent=youtube_node,
        critical=False
    )

    # Check view count criteria (highest views)
    view_count_ok = looks_like_view_count(youtube_info.view_count_mentioned or answer)
    evaluator.add_custom_node(
        result=bool(view_count_ok),
        id="youtube_check_view_count",
        desc="Identify or mention the video with highest view count",
        parent=youtube_node,
        critical=False
    )

    # Check duration criteria (>20 minutes)
    duration_over_20_ok = looks_like_duration_over_20min(youtube_info.duration_mentioned or answer)
    evaluator.add_custom_node(
        result=bool(duration_over_20_ok),
        id="youtube_check_duration",
        desc="Verify video duration is more than 20 minutes (no Shorts or brief clips)",
        parent=youtube_node,
        critical=False
    )

    # Extract video details
    video_title_ok = bool(youtube_info.video_title and youtube_info.video_title.strip())
    video_link_ok = looks_like_youtube_link(youtube_info.video_link or answer)

    evaluator.add_custom_node(
        result=bool(video_title_ok and video_link_ok),
        id="youtube_extract_video_details",
        desc="Extract YouTube video title and link",
        parent=youtube_node,
        critical=False
    )

    # 3.3 Final output completeness check
    final_output_node = evaluator.add_parallel(
        id="final_output_completeness",
        desc="Verify all required information is provided in the final answer",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(course_title_ok),
        id="output_course_title",
        desc="Final answer includes the course title",
        parent=final_output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(instructor_name_ok),
        id="output_instructor_name",
        desc="Final answer includes the instructor name",
        parent=final_output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(video_title_ok),
        id="output_video_title",
        desc="Final answer includes the YouTube video title",
        parent=final_output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(video_link_ok),
        id="output_video_link",
        desc="Final answer includes the YouTube video link",
        parent=final_output_node,
        critical=False
    )

    # Optional: Check for mention of login/paywall restrictions
    restriction_mentioned = has_any_ci(answer, ['login', 'paywall', 'restriction', 'sign in', 'account required'])
    evaluator.add_custom_node(
        result=bool(restriction_mentioned),
        id="mentions_restrictions",
        desc="Acknowledges any login or paywall restrictions encountered (if applicable)",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
