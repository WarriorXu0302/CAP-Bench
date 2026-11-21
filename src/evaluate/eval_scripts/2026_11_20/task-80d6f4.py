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
TASK_ID = "task-80d6f4"
TASK_DESCRIPTION = 'I’m very interested in Samsung’s foldable-phone hinge technology and want to find some in-depth technical material. Please search Google Patents for patents filed by Samsung Electronics related to “foldable hinge.” Make sure to use filters so that only **granted patents (Grant)** are shown, and the **filing date is after January 1, 2021**. After finding one that appears to be a core patent, record its patent number and title.  \n\nIf there are no results under these strict conditions, keep **Grant** unchanged and gradually relax the date condition to **after January 1, 2019**, and specify the actual date range used in your results.  \n\nThen go to YouTube and search for a “Samsung Galaxy Z Fold 5 teardown” video that uses this type of technology. Find a popular video with **over 500,000 views**. Open it and verify whether it supports subtitles (CC). If there are no results over 500,000 views, select the video with the highest view count in the current results and note the actual view count.  \n\nFinally, tell me the patent number, the video title, and whether subtitles are available.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PatentInfo(BaseModel):
    """Patent information extracted from the answer"""
    patent_number: Optional[str] = None
    patent_title: Optional[str] = None
    filing_date_condition: Optional[str] = None


class VideoInfo(BaseModel):
    """YouTube video information extracted from the answer"""
    video_title: Optional[str] = None
    view_count_text: Optional[str] = None
    subtitles_available: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_patent_from_answer() -> str:
    return """
Extract the patent information from the answer that the user found on Google Patents:

- patent_number: the patent number exactly as stated (e.g., "US1234567B2", "US 1234567 B2")
- patent_title: the patent title exactly as stated
- filing_date_condition: the actual date condition used (e.g., "after January 1, 2021", "after January 1, 2019")

If any field is missing, set it to null.
"""


def prompt_extract_video_from_answer() -> str:
    return """
Extract the YouTube video information from the answer:

- video_title: the exact title of the video as stated
- view_count_text: the view count exactly as written (include numbers and text like "views", "million", etc.)
- subtitles_available: whether subtitles/CC are available (look for "yes", "no", "available", "supported", etc.)

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
    # Try to find numbers, including those with commas or 'k', 'M' notation
    text_lower = text.lower()
    # Handle "1.5M", "500k" style
    m = re.search(r'(\d+(?:\.\d+)?)\s*([km])', text_lower)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if unit == 'k':
            return num * 1000
        elif unit == 'm':
            return num * 1000000
    # Handle regular numbers with commas
    m = re.search(r'([\d,]+(?:\.\d+)?)', text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except Exception:
            return None
    return None


def looks_like_patent_number(text: Optional[str]) -> bool:
    if not text:
        return False
    # Patent numbers typically have format like US1234567B2, US 1234567 B2, etc.
    return bool(re.search(r'(US|EP|WO|CN|JP|KR)\s*\d+', text, re.IGNORECASE))


def mentions_grant_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['grant', 'granted', 'granted patents'])


def mentions_date_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['2021', '2019', 'filing date', 'after'])


def looks_like_subtitle_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['yes', 'no', 'available', 'not available', 'supported', 'not supported', 'cc'])


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
    patent_info = await evaluator.extract(
        prompt=prompt_extract_patent_from_answer(),
        template_class=PatentInfo,
        extraction_name="patent_info"
    )

    video_info = await evaluator.extract(
        prompt=prompt_extract_video_from_answer(),
        template_class=VideoInfo,
        extraction_name="video_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Google Patents section
    google_patents_node = evaluator.add_sequential(
        id="google_patents_section",
        desc="Google Patents search for Samsung foldable hinge patents",
        parent=root,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A1 - Complex filtering configuration
    # Check for grant filter and date filter usage
    grant_filter_ok = mentions_grant_filter(answer)
    date_filter_ok = mentions_date_filter(answer)
    samsung_keyword_ok = has_any_ci(answer, ['samsung'])
    hinge_keyword_ok = has_any_ci(answer, ['hinge', 'foldable'])

    filter_action_ok = grant_filter_ok and date_filter_ok and samsung_keyword_ok and hinge_keyword_ok

    evaluator.add_custom_node(
        result=bool(filter_action_ok),
        id="google_patents_filter_action",
        desc="[Action Node] patents.google.com:F1:A1 - Apply multiple filters: granted patents only, filing date (2021 or 2019), Samsung, foldable hinge",
        parent=google_patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P1 - Metadata identification
    # Check for patent number and title extraction
    patent_number_ok = looks_like_patent_number(patent_info.patent_number)
    patent_title_ok = bool(patent_info.patent_title and patent_info.patent_title.strip())

    evaluator.add_custom_node(
        result=bool(patent_number_ok and patent_title_ok),
        id="google_patents_metadata_perception",
        desc="[Perception Node] patents.google.com:F3:P1 - Extract patent number and title from patent detail page",
        parent=google_patents_node,
        critical=False
    )

    # Additional check: mentions date condition relaxation if needed
    date_relaxation_mentioned = has_any_ci(answer, ['2019']) or has_any_ci(answer, ['relax', 'changed'])
    evaluator.add_custom_node(
        result=bool(patent_info.filing_date_condition and patent_info.filing_date_condition.strip()),
        id="google_patents_date_condition",
        desc="Specifies the actual date range condition used in the search",
        parent=google_patents_node,
        critical=False
    )

    # 3.2 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube search for Samsung Galaxy Z Fold 5 teardown video",
        parent=root,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P12 - View count identification and comparison
    # Check for view count extraction and comparison with 500k threshold
    view_count_num = extract_number(video_info.view_count_text)
    view_count_mentioned = bool(video_info.view_count_text and video_info.view_count_text.strip())

    # Accept if either over 500k or if they noted the actual highest view count
    view_count_ok = view_count_mentioned and (view_count_num is not None)

    evaluator.add_custom_node(
        result=bool(view_count_ok),
        id="youtube_view_count_perception",
        desc="[Perception Node] youtube.com:F1:P12 - Identify video view count and compare with 500,000 threshold or note highest available",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F2:P20 - Subtitle availability check
    # Check for CC/subtitle status
    subtitle_status_ok = looks_like_subtitle_status(video_info.subtitles_available)

    evaluator.add_custom_node(
        result=bool(subtitle_status_ok),
        id="youtube_subtitle_status_perception",
        desc="[Perception Node] youtube.com:F2:P20 - Check CC icon status or settings menu to determine subtitle availability",
        parent=youtube_node,
        critical=False
    )

    # Additional checks
    video_title_ok = bool(video_info.video_title and video_info.video_title.strip())
    z_fold_5_mentioned = has_any_ci(answer, ['z fold 5', 'fold 5', 'galaxy z fold 5'])
    teardown_mentioned = has_any_ci(answer, ['teardown'])

    evaluator.add_custom_node(
        result=bool(video_title_ok),
        id="youtube_video_title",
        desc="Extracts and reports the video title",
        parent=youtube_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(z_fold_5_mentioned and teardown_mentioned),
        id="youtube_search_keywords",
        desc="Uses appropriate search keywords: Samsung Galaxy Z Fold 5 teardown",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
