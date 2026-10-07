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
TASK_ID = "task-5ea2c0"
TASK_DESCRIPTION = "I am a full-stack developer, and I want to learn a new AI tool towards the end of the year. Please go to the GitHub Trending page and filter for the most popular Python projects for 'this month'. Carefully examine the top-ranked projects. Identify the first project whose description explicitly contains either 'Agent' or 'LLM' as keywords, and which also has a `requirements.txt` file present in its repository's root directory. Once found, search on YouTube for practical tutorials for this project. Please ensure to use the filter function to only show videos released 'this year' and with a duration exceeding 4 minutes (to ensure they are not short trailers). Identify the one with the highest view count. Output: GitHub project name, GitHub project link, Star count, direct link to the `requirements.txt` file, YouTube video title, video duration, video publication date, video view count, YouTube video link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GitHubProjectInfo(BaseModel):
    """GitHub project information extracted from the answer"""
    project_name: Optional[str] = None
    project_link: Optional[str] = None
    star_count: Optional[str] = None
    requirements_txt_link: Optional[str] = None


class YouTubeVideoInfo(BaseModel):
    """YouTube video information extracted from the answer"""
    video_title: Optional[str] = None
    video_duration: Optional[str] = None
    video_publication_date: Optional[str] = None
    video_view_count: Optional[str] = None
    video_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_github_info() -> str:
    return """
Extract the GitHub project information from the answer:

Return:
- project_name: the name of the GitHub project
- project_link: the URL to the GitHub project repository
- star_count: the star count as stated (include any formatting like "10k", "1.2k", etc.)
- requirements_txt_link: the direct URL to the requirements.txt file

If any field is missing in the answer, set it to null.
"""


def prompt_extract_youtube_info() -> str:
    return """
Extract the YouTube video information from the answer:

Return:
- video_title: the title of the YouTube video
- video_duration: the duration of the video as stated (e.g., "5:30", "10 minutes", etc.)
- video_publication_date: the publication date as stated
- video_view_count: the view count as stated (include any formatting like "10K views", "1.2M", etc.)
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


def looks_like_github_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'github.com')


def looks_like_youtube_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return has_any_ci(url, ['youtube.com', 'youtu.be'])


def looks_like_requirements_txt_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return looks_like_github_url(url) and ci_contains(url, 'requirements.txt')


def is_python_language_mentioned(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return ci_contains(answer, 'python')


def duration_exceeds_4_minutes(duration: Optional[str]) -> bool:
    if not duration:
        return False
    # Try to parse duration in various formats
    # Format: "5:30", "10:00", "1:30:00", "5 minutes", "300 seconds", etc.
    duration_lower = duration.lower()

    # Extract numbers
    numbers = re.findall(r'\d+', duration)
    if not numbers:
        return False

    try:
        # Check for HH:MM:SS or MM:SS format
        if ':' in duration:
            parts = [int(x) for x in numbers]
            if len(parts) == 2:  # MM:SS
                minutes = parts[0]
                return minutes > 4 or (minutes == 4 and parts[1] > 0)
            elif len(parts) == 3:  # HH:MM:SS
                total_minutes = parts[0] * 60 + parts[1]
                return total_minutes > 4

        # Check for "X minutes" format
        if 'minute' in duration_lower:
            minutes = int(numbers[0])
            return minutes > 4

        # Check for "X seconds" format
        if 'second' in duration_lower:
            seconds = int(numbers[0])
            return seconds > 240  # 4 minutes = 240 seconds

        # If just a number with colon, assume MM:SS
        if len(numbers) >= 2:
            return int(numbers[0]) > 4 or (int(numbers[0]) == 4 and int(numbers[1]) > 0)
    except Exception:
        pass

    return False


def looks_like_year_2025(date: Optional[str]) -> bool:
    if not date:
        return False
    return '2025' in date


def has_view_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['view', 'views', 'k', 'm'])


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
    github_info = await evaluator.extract(
        prompt=prompt_extract_github_info(),
        template_class=GitHubProjectInfo,
        extraction_name="github_project_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_info(),
        template_class=YouTubeVideoInfo,
        extraction_name="youtube_video_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 GitHub Trending section
    github_node = evaluator.add_sequential(
        id="github_trending_section",
        desc="GitHub Trending: Filter and identify the target Python project",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F8:A13 - Language filter (Python)
    python_language_ok = is_python_language_mentioned(answer) and looks_like_github_url(github_info.project_link)
    evaluator.add_custom_node(
        result=bool(python_language_ok),
        id="github_language_filter",
        desc="[Action Node] github.com:F8:A13 - Filter for Python projects on GitHub Trending",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F8:A14 - Date range filter (this month)
    this_month_ok = has_any_ci(answer, ['this month', 'monthly', 'month'])
    star_count_ok = github_info.star_count and contains_digits(github_info.star_count)
    evaluator.add_custom_node(
        result=bool(this_month_ok and star_count_ok),
        id="github_date_filter",
        desc="[Action Node] github.com:F8:A14 - Filter for 'this month' trending projects and verify high star count",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F8:P5 - List content understanding (Agent or LLM keywords)
    has_agent_or_llm = has_any_ci(answer, ['agent', 'llm'])
    project_name_ok = github_info.project_name and len(github_info.project_name.strip()) > 0
    evaluator.add_custom_node(
        result=bool(has_agent_or_llm and project_name_ok),
        id="github_keyword_identification",
        desc="[Perception Node] github.com:F8:P5 - Identify first project with 'Agent' or 'LLM' in description",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F8:A23 - Click into project details
    project_link_ok = looks_like_github_url(github_info.project_link)
    evaluator.add_custom_node(
        result=bool(project_link_ok),
        id="github_click_project",
        desc="[Action Node] github.com:F8:A23 - Click into the project repository to view details",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P13 - File tree state recognition (requirements.txt)
    requirements_file_ok = looks_like_requirements_txt_url(github_info.requirements_txt_link)
    evaluator.add_custom_node(
        result=bool(requirements_file_ok),
        id="github_requirements_file",
        desc="[Perception Node] github.com:F3:P13 - Verify requirements.txt exists in root directory",
        parent=github_node,
        critical=False
    )

    # 3.2 YouTube search section
    youtube_node = evaluator.add_sequential(
        id="youtube_search_section",
        desc="YouTube: Search for tutorial videos with filters",
        parent=root,
        critical=False
    )

    # Check if project name is used as search query
    search_query_ok = (github_info.project_name and
                      ci_contains(answer, github_info.project_name)) or has_any_ci(answer, ['youtube', 'tutorial'])
    evaluator.add_custom_node(
        result=bool(search_query_ok),
        id="youtube_search_query",
        desc="Search YouTube using the identified GitHub project name as keyword",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Multiple filter operations (year and duration)
    duration_filter_ok = youtube_info.video_duration and duration_exceeds_4_minutes(youtube_info.video_duration)
    year_filter_ok = youtube_info.video_publication_date and looks_like_year_2025(youtube_info.video_publication_date)
    filter_mentioned = has_any_ci(answer, ['filter', 'this year', '2025', 'duration', 'minute'])
    evaluator.add_custom_node(
        result=bool(duration_filter_ok and year_filter_ok and filter_mentioned),
        id="youtube_filters",
        desc="[Action Node] youtube.com:F9:A4 - Apply filters: 'this year' release date and duration > 4 minutes",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Card content understanding (view count)
    view_count_ok = youtube_info.video_view_count and has_view_count(youtube_info.video_view_count)
    highest_views_mentioned = has_any_ci(answer, ['highest', 'most views', 'most viewed', 'top'])
    evaluator.add_custom_node(
        result=bool(view_count_ok and highest_views_mentioned),
        id="youtube_view_count",
        desc="[Perception Node] youtube.com:F1:P4 - Identify video with highest view count",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more results
    scroll_context = has_any_ci(answer, ['scroll', 'load', 'browse', 'compare', 'examine'])
    evaluator.add_custom_node(
        result=bool(scroll_context or view_count_ok),
        id="youtube_scroll_load",
        desc="[Action Node] youtube.com:F1:A69 - Scroll through results to compare view counts",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click to video details
    video_link_ok = looks_like_youtube_url(youtube_info.video_link)
    video_title_ok = youtube_info.video_title and len(youtube_info.video_title.strip()) > 0
    evaluator.add_custom_node(
        result=bool(video_link_ok and video_title_ok),
        id="youtube_click_video",
        desc="[Action Node] youtube.com:F1:A22 - Click into video details to get full information",
        parent=youtube_node,
        critical=False
    )

    # 3.3 Output completeness check (non-prefixed)
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required outputs are provided",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(github_info.project_name and github_info.project_link and
                   github_info.star_count and github_info.requirements_txt_link),
        id="github_outputs_complete",
        desc="All GitHub outputs present: project name, link, star count, requirements.txt link",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(youtube_info.video_title and youtube_info.video_duration and
                   youtube_info.video_publication_date and youtube_info.video_view_count and
                   youtube_info.video_link),
        id="youtube_outputs_complete",
        desc="All YouTube outputs present: title, duration, date, view count, link",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
