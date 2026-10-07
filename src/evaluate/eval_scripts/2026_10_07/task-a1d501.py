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
TASK_ID = "task-a1d501"
TASK_DESCRIPTION = "I am currently preparing my final thesis on global extreme poverty (as of December 2025).\n\nFirst, please visit OurWorldInData and search for 'Share of population living in extreme poverty'. On the chart page, in the table view, identify the country with the highest poverty rate in the most recent year's data, and record its name, the year, and the specific proportion.\n\nNext, to gain a deeper understanding of the underlying economic principles, find a 'Development Economics' course on Coursera. Requirements: suitable for 'Intermediate' level, taught in English, and with a rating of 4.7 or higher. Please check the syllabus to confirm if it includes sections on 'Randomized Control Trials' (RCT) or 'Impact Evaluation'.\n\nFinally, I plan to use R language to process this type of panel data. Go to StackOverflow and search for questions about 'ggplot2 panel data visualization', filtering for the one with the highest number of votes and an accepted answer.\n\nOutput:\n*   OWID: country with the highest poverty rate, year, value;\n*   Coursera: course name, rating, name(s) of chapter(s) containing RCT/Impact Evaluation;\n*   SO: question title, number of votes, summary of the accepted answer;\n*   and direct URLs for all three pages."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OWIDData(BaseModel):
    """Data extracted from OurWorldInData for extreme poverty"""
    country_name: Optional[str] = None
    year: Optional[str] = None
    poverty_rate: Optional[str] = None
    owid_url: Optional[str] = None


class CourseraData(BaseModel):
    """Data extracted from Coursera for Development Economics course"""
    course_name: Optional[str] = None
    rating: Optional[str] = None
    rct_chapters: Optional[str] = None
    coursera_url: Optional[str] = None


class StackOverflowData(BaseModel):
    """Data extracted from StackOverflow for ggplot2 panel data"""
    question_title: Optional[str] = None
    votes: Optional[str] = None
    answer_summary: Optional[str] = None
    stackoverflow_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_owid_from_answer() -> str:
    return """
Extract the OurWorldInData information reported in the answer about the country with the highest extreme poverty rate:

Return:
- country_name: the name of the country with the highest poverty rate
- year: the year of the data
- poverty_rate: the specific proportion/percentage value
- owid_url: the direct URL to the OurWorldInData page

If any field is missing, set it to null.
"""


def prompt_extract_coursera_from_answer() -> str:
    return """
Extract the Coursera course information reported in the answer about Development Economics:

Return:
- course_name: the name of the course found
- rating: the course rating (should be 4.7 or higher)
- rct_chapters: the name(s) of chapter(s) containing RCT or Impact Evaluation content
- coursera_url: the direct URL to the Coursera course page

If any field is missing, set it to null.
"""


def prompt_extract_stackoverflow_from_answer() -> str:
    return """
Extract the StackOverflow information reported in the answer about ggplot2 panel data visualization:

Return:
- question_title: the title of the question
- votes: the number of votes
- answer_summary: a summary of the accepted answer
- stackoverflow_url: the direct URL to the StackOverflow question

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


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_year(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check if it contains a 4-digit year between 2000-2030
    return bool(re.search(r'\b(20[0-2]\d|2030)\b', text))


def looks_like_percentage(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '%') or extract_float(text) is not None)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and 0 <= val <= 5


def is_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and (text.startswith('http://') or text.startswith('https://'))


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
    owid_info = await evaluator.extract(
        prompt=prompt_extract_owid_from_answer(),
        template_class=OWIDData,
        extraction_name="owid_data"
    )

    coursera_info = await evaluator.extract(
        prompt=prompt_extract_coursera_from_answer(),
        template_class=CourseraData,
        extraction_name="coursera_data"
    )

    stackoverflow_info = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_from_answer(),
        template_class=StackOverflowData,
        extraction_name="stackoverflow_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 OurWorldInData section
    owid_node = evaluator.add_sequential(
        id="owid_section",
        desc="OurWorldInData extreme poverty data extraction",
        parent=root,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A4 - Switch to table view
    table_view_ok = has_any_ci(answer, ['table', 'table view'])
    evaluator.add_custom_node(
        result=bool(table_view_ok),
        id="owid_action_table_view",
        desc="[Action Node] ourworldindata.org:F1:A4 - Switch to table view on the chart page",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A6 - Sort table to find highest poverty rate
    sort_action_ok = (owid_info.country_name is not None and
                      owid_info.poverty_rate is not None and
                      looks_like_percentage(owid_info.poverty_rate))
    evaluator.add_custom_node(
        result=bool(sort_action_ok),
        id="owid_action_sort",
        desc="[Action Node] ourworldindata.org:F1:A6 - Sort table by poverty rate to identify the highest value",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P6 - Extract table data (country, year, rate)
    country_ok = owid_info.country_name is not None and len(owid_info.country_name.strip()) > 0
    year_ok = looks_like_year(owid_info.year)
    rate_ok = looks_like_percentage(owid_info.poverty_rate)

    evaluator.add_custom_node(
        result=bool(country_ok and year_ok and rate_ok),
        id="owid_perception_data",
        desc="[Perception Node] ourworldindata.org:F1:P6 - Extract country name, year, and poverty rate from table data",
        parent=owid_node,
        critical=False
    )

    # URL verification
    owid_url_ok = is_valid_url(owid_info.owid_url, 'ourworldindata.org')
    evaluator.add_custom_node(
        result=bool(owid_url_ok),
        id="owid_url_provided",
        desc="Provides valid OurWorldInData URL",
        parent=owid_node,
        critical=False
    )

    # 3.2 Coursera section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera Development Economics course search and verification",
        parent=root,
        critical=False
    )

    # [Perception Node] coursera.org:F1:P2 - Content matching for Development Economics
    course_name_ok = (coursera_info.course_name is not None and
                      has_any_ci(coursera_info.course_name, ['development', 'economic']))
    evaluator.add_custom_node(
        result=bool(course_name_ok),
        id="coursera_perception_content",
        desc="[Perception Node] coursera.org:F1:P2 - Identify course related to Development Economics",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A2 - Filter by Intermediate difficulty
    difficulty_ok = has_any_ci(answer, ['intermediate'])
    evaluator.add_custom_node(
        result=bool(difficulty_ok),
        id="coursera_action_difficulty",
        desc="[Action Node] coursera.org:F2:A2 - Filter search results by Intermediate difficulty level",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A12 - Sort by rating to find 4.7+
    rating_val = extract_float(coursera_info.rating)
    rating_ok = rating_val is not None and rating_val >= 4.7
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="coursera_action_rating_sort",
        desc="[Action Node] coursera.org:F2:A12 - Sort by rating to identify courses with 4.7 or higher",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F3:A23 - Expand syllabus to check content
    syllabus_expanded_ok = (coursera_info.rct_chapters is not None and
                            len(coursera_info.rct_chapters.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(syllabus_expanded_ok),
        id="coursera_action_expand_syllabus",
        desc="[Action Node] coursera.org:F3:A23 - Expand syllabus to check for RCT or Impact Evaluation content",
        parent=coursera_node,
        critical=False
    )

    # Verify RCT/Impact Evaluation content
    rct_content_ok = (coursera_info.rct_chapters is not None and
                      (has_any_ci(coursera_info.rct_chapters, ['rct', 'randomized control', 'randomized controlled']) or
                       has_any_ci(coursera_info.rct_chapters, ['impact evaluation'])))
    evaluator.add_custom_node(
        result=bool(rct_content_ok),
        id="coursera_rct_content_verified",
        desc="Confirms syllabus includes RCT or Impact Evaluation sections",
        parent=coursera_node,
        critical=False
    )

    # URL verification
    coursera_url_ok = is_valid_url(coursera_info.coursera_url, 'coursera.org')
    evaluator.add_custom_node(
        result=bool(coursera_url_ok),
        id="coursera_url_provided",
        desc="Provides valid Coursera URL",
        parent=coursera_node,
        critical=False
    )

    # 3.3 StackOverflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="StackOverflow ggplot2 panel data question search",
        parent=root,
        critical=False
    )

    # [Action Node] stackoverflow.com:F2:A2 - Sort by votes
    votes_val = extract_float(stackoverflow_info.votes)
    votes_ok = votes_val is not None and votes_val > 0
    evaluator.add_custom_node(
        result=bool(votes_ok),
        id="stackoverflow_action_sort_votes",
        desc="[Action Node] stackoverflow.com:F2:A2 - Sort search results by votes to find highest voted question",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F2:P20 - Verify accepted answer exists
    accepted_answer_ok = (stackoverflow_info.answer_summary is not None and
                          len(stackoverflow_info.answer_summary.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(accepted_answer_ok),
        id="stackoverflow_perception_accepted",
        desc="[Perception Node] stackoverflow.com:F2:P20 - Identify question with accepted answer",
        parent=stackoverflow_node,
        critical=False
    )

    # Question title verification
    title_ok = (stackoverflow_info.question_title is not None and
                len(stackoverflow_info.question_title.strip()) > 0 and
                (has_any_ci(stackoverflow_info.question_title, ['ggplot', 'panel', 'data']) or
                 has_any_ci(answer, ['ggplot', 'panel'])))
    evaluator.add_custom_node(
        result=bool(title_ok),
        id="stackoverflow_question_title",
        desc="Extracts question title related to ggplot2 and panel data",
        parent=stackoverflow_node,
        critical=False
    )

    # URL verification
    stackoverflow_url_ok = is_valid_url(stackoverflow_info.stackoverflow_url, 'stackoverflow.com')
    evaluator.add_custom_node(
        result=bool(stackoverflow_url_ok),
        id="stackoverflow_url_provided",
        desc="Provides valid StackOverflow URL",
        parent=stackoverflow_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
