import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-3a8e81"
TASK_DESCRIPTION = 'I’m self-studying Prof. Patrick Winston’s *Artificial Intelligence* (6.034) on MIT OCW. The course is classic, but I want to connect it with the latest research.\n\nFirst, search for this course on OCW, and make sure to check the **“Video Lectures”** filter in the left sidebar so we use a version that includes videos. After opening the course page, check the **Calendar** or **Syllabus** and find the topic title corresponding to **“Lecture 12.”**\n\nThen, go to arXiv and use **Advanced Search**. Under the **Computer Science** category, search for that topic. Crucially, you must use the date selector to restrict the time range to the **most recent 24 months up to today**.\n\nFrom the results, find the **first paper** whose **Comments** or **Abstract** explicitly mentions **“GitHub”** or **“Code available,”** and tell me its **title** and **arXiv ID**.\n\nIf there is no result under those conditions, keep the same topic and Computer Science category, widen the time range to the **most recent 36 months up to today**, search again, and explicitly note in the output that the relaxed time window was used.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Lecture12Info(BaseModel):
    """Lecture 12 topic extracted from the answer"""
    lecture_12_topic: Optional[str] = None


class ArxivPaperInfo(BaseModel):
    """ArXiv paper information extracted from the answer"""
    paper_title: Optional[str] = None
    arxiv_id: Optional[str] = None
    mentions_github_or_code: Optional[bool] = None
    time_window_relaxed: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_lecture_12_topic() -> str:
    return """
Extract the topic title for Lecture 12 from the MIT OCW 6.034 Artificial Intelligence course that the user reported finding in the Calendar or Syllabus.

Return:
- lecture_12_topic: the exact topic title for Lecture 12 as stated in the answer. If not present, set null.
"""


def prompt_extract_arxiv_paper() -> str:
    return """
From the answer, extract the arXiv paper information that the user reported finding.

Return:
- paper_title: the exact title of the paper as stated in the answer. If not present, set null.
- arxiv_id: the arXiv ID of the paper (e.g., "2301.12345"). If not present, set null.
- mentions_github_or_code: true if the answer indicates the paper's Comments or Abstract mentions "GitHub" or "Code available", otherwise false or null.
- time_window_relaxed: true if the answer explicitly states that the search time window was widened to 36 months, otherwise false or null.
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


def looks_like_arxiv_id(text: Optional[str]) -> bool:
    if not text:
        return False
    # ArXiv IDs typically look like: YYMM.NNNNN or YYMM.NNNNNN
    pattern = r'\b\d{4}\.\d{4,5}\b'
    return bool(re.search(pattern, text))


def mentions_video_lectures_filter(answer: str) -> bool:
    """Check if answer mentions using the Video Lectures filter"""
    keywords = ['video lectures', 'video lecture filter', 'filter', 'checked', 'selected']
    return has_any_ci(answer, keywords) and has_any_ci(answer, ['video'])


def mentions_calendar_or_syllabus(answer: str) -> bool:
    """Check if answer mentions checking Calendar or Syllabus"""
    return has_any_ci(answer, ['calendar', 'syllabus'])


def mentions_advanced_search(answer: str) -> bool:
    """Check if answer mentions using arXiv Advanced Search"""
    return has_any_ci(answer, ['advanced search'])


def mentions_computer_science_category(answer: str) -> bool:
    """Check if answer mentions Computer Science category"""
    return has_any_ci(answer, ['computer science', 'cs category'])


def mentions_24_month_restriction(answer: str) -> bool:
    """Check if answer mentions 24-month date restriction"""
    return has_any_ci(answer, ['24 month', '24-month', 'two year', 'past 24', 'last 24', 'recent 24'])


def mentions_github_or_code(answer: str) -> bool:
    """Check if answer mentions GitHub or Code available"""
    return has_any_ci(answer, ['github', 'code available'])


def mentions_comments_or_abstract(answer: str) -> bool:
    """Check if answer mentions checking Comments or Abstract"""
    return has_any_ci(answer, ['comment', 'abstract'])


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
        strategy=AggregationStrategy.SEQUENTIAL,
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
    lecture_info = await evaluator.extract(
        prompt=prompt_extract_lecture_12_topic(),
        template_class=Lecture12Info,
        extraction_name="lecture_12_info"
    )

    paper_info = await evaluator.extract(
        prompt=prompt_extract_arxiv_paper(),
        template_class=ArxivPaperInfo,
        extraction_name="arxiv_paper_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 MIT OCW section
    ocw_node = evaluator.add_sequential(
        id="ocw_section",
        desc="MIT OCW 6.034 Artificial Intelligence course navigation and topic extraction",
        parent=root,
        critical=False
    )

    # Check for OCW mention and course search
    ocw_mentioned = has_any_ci(answer, ['ocw', 'mit opencourseware', 'opencourseware'])
    course_mentioned = has_any_ci(answer, ['6.034', 'artificial intelligence', 'patrick winston'])

    evaluator.add_custom_node(
        result=bool(ocw_mentioned and course_mentioned),
        id="ocw_course_search",
        desc="Mentions searching for the MIT 6.034 Artificial Intelligence course on OCW",
        parent=ocw_node,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F1:A8 - Video Lectures filter checkbox
    video_filter_ok = mentions_video_lectures_filter(answer)
    evaluator.add_custom_node(
        result=bool(video_filter_ok),
        id="ocw_video_lectures_filter",
        desc="[Action Node] ocw.mit.edu:F1:A8 - Check the Video Lectures filter in the left sidebar",
        parent=ocw_node,
        critical=False
    )

    # Check for Calendar/Syllabus navigation
    calendar_syllabus_ok = mentions_calendar_or_syllabus(answer)
    evaluator.add_custom_node(
        result=bool(calendar_syllabus_ok),
        id="ocw_calendar_syllabus_access",
        desc="Mentions accessing the Calendar or Syllabus section",
        parent=ocw_node,
        critical=False
    )

    # [Perception Node] ocw.mit.edu:F3:P11 - Extract Lecture 12 topic from table
    lecture_12_mentioned = has_any_ci(answer, ['lecture 12', 'lec 12', 'lecture twelve'])
    lecture_12_topic_extracted = bool(lecture_info and lecture_info.lecture_12_topic and lecture_info.lecture_12_topic.strip())

    evaluator.add_custom_node(
        result=bool(lecture_12_mentioned and lecture_12_topic_extracted),
        id="ocw_lecture_12_topic_extraction",
        desc="[Perception Node] ocw.mit.edu:F3:P11 - Extract the topic title for Lecture 12 from the Calendar/Syllabus table",
        parent=ocw_node,
        critical=False
    )

    # 3.2 ArXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="ArXiv advanced search with date and category restrictions",
        parent=root,
        critical=False
    )

    # Check for ArXiv mention and Advanced Search
    arxiv_mentioned = has_any_ci(answer, ['arxiv'])
    advanced_search_ok = mentions_advanced_search(answer)

    evaluator.add_custom_node(
        result=bool(arxiv_mentioned and advanced_search_ok),
        id="arxiv_advanced_search_access",
        desc="Mentions going to arXiv and using Advanced Search",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F6:A28 - Computer Science category selection
    cs_category_ok = mentions_computer_science_category(answer)
    evaluator.add_custom_node(
        result=bool(cs_category_ok),
        id="arxiv_cs_category_selection",
        desc="[Action Node] arxiv.org:F6:A28 - Select Computer Science category in Advanced Search",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F6:A12 - Date range restriction to 24 months
    date_restriction_ok = mentions_24_month_restriction(answer) or has_any_ci(answer, ['date', 'time range', 'past', 'recent'])
    evaluator.add_custom_node(
        result=bool(date_restriction_ok),
        id="arxiv_date_restriction",
        desc="[Action Node] arxiv.org:F6:A12 - Set date range to most recent 24 months in the date selector",
        parent=arxiv_node,
        critical=False
    )

    # Check for topic search (using Lecture 12 topic)
    topic_search_ok = bool(lecture_12_topic_extracted and has_any_ci(answer, ['search']))
    evaluator.add_custom_node(
        result=bool(topic_search_ok),
        id="arxiv_topic_search",
        desc="Searches arXiv for the Lecture 12 topic",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P6 - Scan Comments/Abstract for GitHub mention
    github_code_ok = mentions_github_or_code(answer)
    comments_abstract_ok = mentions_comments_or_abstract(answer)

    evaluator.add_custom_node(
        result=bool(github_code_ok and comments_abstract_ok),
        id="arxiv_github_code_perception",
        desc="[Perception Node] arxiv.org:F3:P6 - Scan Comments or Abstract fields for GitHub or Code available mention",
        parent=arxiv_node,
        critical=False
    )

    # 3.3 Paper information extraction
    paper_node = evaluator.add_parallel(
        id="paper_info_section",
        desc="Extract paper title and arXiv ID",
        parent=root,
        critical=False
    )

    # Check for paper title extraction
    paper_title_ok = bool(paper_info and paper_info.paper_title and paper_info.paper_title.strip())
    evaluator.add_custom_node(
        result=bool(paper_title_ok),
        id="paper_title_extracted",
        desc="Paper title is extracted and provided",
        parent=paper_node,
        critical=False
    )

    # Check for arXiv ID extraction
    arxiv_id_ok = bool(paper_info and paper_info.arxiv_id and looks_like_arxiv_id(paper_info.arxiv_id))
    evaluator.add_custom_node(
        result=bool(arxiv_id_ok),
        id="arxiv_id_extracted",
        desc="ArXiv ID is extracted and has valid format",
        parent=paper_node,
        critical=False
    )

    # Check if paper mentions GitHub/Code
    paper_has_github = bool(paper_info and paper_info.mentions_github_or_code)
    evaluator.add_custom_node(
        result=bool(paper_has_github),
        id="paper_has_github_code",
        desc="Paper's Comments or Abstract mentions GitHub or Code available",
        parent=paper_node,
        critical=False
    )

    # 3.4 Optional: Check for relaxed time window
    relaxed_window_node = evaluator.add_leaf(
        id="relaxed_time_window_check",
        desc="Check if answer explicitly notes using relaxed 36-month time window",
        parent=root,
        critical=False
    )

    time_window_relaxed = bool(paper_info and paper_info.time_window_relaxed)
    mentions_36_months = has_any_ci(answer, ['36 month', '36-month', 'three year', 'widened', 'relaxed', 'expanded'])

    evaluator.add_custom_node(
        result=bool(time_window_relaxed or mentions_36_months),
        id="explicit_time_window_relaxation",
        desc="If applicable, explicitly notes that the time window was relaxed to 36 months",
        parent=relaxed_window_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
