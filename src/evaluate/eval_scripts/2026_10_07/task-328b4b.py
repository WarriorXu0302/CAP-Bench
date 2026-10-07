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
TASK_ID = "task-328b4b"
TASK_DESCRIPTION = 'I’ve just finished learning Python fundamentals and want to switch into data analysis, but I’m not sure whether pandas or PyTorch is currently more in demand.  \nFirst, go to Stack Overflow and open the two tag pages ([pandas] and [pytorch]). Record each tag’s **“Total questions”** count. Then, under each tag, sort by **Newest** and count the number of new questions posted in the **past 24 hours** as today’s activity level. Compare the two and identify which library is more active.\n\nNext, for the winning library, go to Udemy and find **3 advanced courses** that meet all of the following criteria:  \n- Rating above **4.5**  \n- **English-language** instruction  \n- Duration over **17 hours**  \n- Level set to **Expert** or **Intermediate**\n\nIf fewer than 3 courses meet all criteria, keep the rating, language, and duration requirements unchanged, first relax the Level filter to **All Levels**, and clearly mark which courses were added after relaxing this condition.\n\nOutput required:  \n- For Stack Overflow: total question count and past-24-hour question count for both libraries, plus the name of the winning library  \n- For Udemy: the 3 course titles, instructors, ratings, durations, prices, and course links'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StackOverflowTagData(BaseModel):
    """Stack Overflow tag statistics extracted from the answer"""
    pandas_total: Optional[str] = None
    pandas_24h: Optional[str] = None
    pytorch_total: Optional[str] = None
    pytorch_24h: Optional[str] = None
    winner: Optional[str] = None


class UdemyCourse(BaseModel):
    """Single Udemy course details"""
    title: Optional[str] = None
    instructor: Optional[str] = None
    rating: Optional[str] = None
    duration: Optional[str] = None
    price: Optional[str] = None
    link: Optional[str] = None
    level_relaxed: Optional[bool] = None


class UdemyCoursesData(BaseModel):
    """All three Udemy courses extracted from the answer"""
    course1: Optional[UdemyCourse] = None
    course2: Optional[UdemyCourse] = None
    course3: Optional[UdemyCourse] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stackoverflow_data() -> str:
    return """
Extract the Stack Overflow tag statistics for pandas and pytorch from the answer.

Return:
- pandas_total: the total questions count for the pandas tag exactly as stated
- pandas_24h: the number of questions posted in the past 24 hours for pandas
- pytorch_total: the total questions count for the pytorch tag exactly as stated
- pytorch_24h: the number of questions posted in the past 24 hours for pytorch
- winner: the name of the winning library (either "pandas" or "pytorch")

If any field is missing in the answer, set it to null.
"""


def prompt_extract_udemy_courses() -> str:
    return """
Extract the 3 Udemy courses from the answer for the winning library.

For each of the three courses, extract:
- title: the course title exactly as stated
- instructor: the instructor name exactly as stated
- rating: the rating value exactly as stated (should be above 4.5)
- duration: the duration exactly as stated (should be over 17 hours)
- price: the price exactly as stated
- link: the course URL/link if provided
- level_relaxed: true if the answer mentions this course was added after relaxing the Level filter, false otherwise

Return course1, course2, and course3 with the above fields. If any course or field is missing, set it to null.
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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove common separators like commas
    cleaned = re.sub(r'[,\s]', '', text)
    m = re.search(r'(\d+)', cleaned)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_question_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and extract_int(text) is not None


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and 0 <= val <= 5


def rating_above_45(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and val > 4.5


def looks_like_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['hour', 'hr', 'h'])


def duration_over_17(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and val > 17


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp']) or re.search(r'\d', text))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http', 'udemy.com', 'www.'])


def mentions_level_relaxed(answer_text: Optional[str], course_title: Optional[str]) -> bool:
    if not answer_text or not course_title:
        return False
    # Check if the answer mentions relaxing level filter near this course
    return has_any_ci(answer_text, ['relax', 'all levels', 'level filter'])


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
    stackoverflow_data = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_data(),
        template_class=StackOverflowTagData,
        extraction_name="stackoverflow_tag_data"
    )

    udemy_data = await evaluator.extract(
        prompt=prompt_extract_udemy_courses(),
        template_class=UdemyCoursesData,
        extraction_name="udemy_courses_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Stack Overflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="Stack Overflow tag comparison for pandas and pytorch",
        parent=root,
        critical=False
    )

    # [Action Node] stackoverflow.com:F3:A19 - Search and locate tags
    mentions_stackoverflow = has_any_ci(answer, ['stack overflow', 'stackoverflow'])
    mentions_pandas = has_any_ci(answer, ['pandas'])
    mentions_pytorch = has_any_ci(answer, ['pytorch'])
    stackoverflow_action_ok = mentions_stackoverflow and mentions_pandas and mentions_pytorch

    evaluator.add_custom_node(
        result=bool(stackoverflow_action_ok),
        id="stackoverflow_action_search_tags",
        desc="[Action Node] stackoverflow.com:F3:A19 - Navigate to Stack Overflow and locate both pandas and pytorch tag pages",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F3:P16 - Extract total questions and 24h activity
    pandas_total_ok = looks_like_question_count(stackoverflow_data.pandas_total)
    pandas_24h_ok = looks_like_question_count(stackoverflow_data.pandas_24h)
    pytorch_total_ok = looks_like_question_count(stackoverflow_data.pytorch_total)
    pytorch_24h_ok = looks_like_question_count(stackoverflow_data.pytorch_24h)

    all_counts_ok = pandas_total_ok and pandas_24h_ok and pytorch_total_ok and pytorch_24h_ok

    evaluator.add_custom_node(
        result=bool(all_counts_ok),
        id="stackoverflow_perception_counts",
        desc="[Perception Node] stackoverflow.com:F3:P16 - Extract Total questions and past 24 hours question counts for both tags",
        parent=stackoverflow_node,
        critical=False
    )

    # Winner identification
    winner_identified = bool(stackoverflow_data.winner and stackoverflow_data.winner.strip())
    winner_valid = winner_identified and has_any_ci(stackoverflow_data.winner, ['pandas', 'pytorch'])

    evaluator.add_custom_node(
        result=bool(winner_valid),
        id="stackoverflow_winner_identified",
        desc="Identify and report which library is the winner (more active)",
        parent=stackoverflow_node,
        critical=False
    )

    # Check for "Newest" sorting mention
    mentions_newest = has_any_ci(answer, ['newest', 'sort', 'sorted by'])
    evaluator.add_custom_node(
        result=bool(mentions_newest),
        id="stackoverflow_mentions_newest_sort",
        desc="Mentions sorting by Newest to count 24-hour activity",
        parent=stackoverflow_node,
        critical=False
    )

    # 3.2 Udemy section
    udemy_node = evaluator.add_sequential(
        id="udemy_section",
        desc="Udemy course search for the winning library",
        parent=root,
        critical=False
    )

    # [Action Node] udemy.com:F1:A6 - Search for courses on Udemy
    mentions_udemy = has_any_ci(answer, ['udemy'])
    winner_in_udemy_context = False
    if stackoverflow_data.winner:
        winner_in_udemy_context = has_any_ci(answer, [stackoverflow_data.winner])

    udemy_search_ok = mentions_udemy and winner_in_udemy_context

    evaluator.add_custom_node(
        result=bool(udemy_search_ok),
        id="udemy_action_search",
        desc="[Action Node] udemy.com:F1:A6 - Navigate to Udemy and search for courses related to the winning library",
        parent=udemy_node,
        critical=False
    )

    # Collect all three courses
    courses = []
    if udemy_data.course1:
        courses.append(udemy_data.course1)
    if udemy_data.course2:
        courses.append(udemy_data.course2)
    if udemy_data.course3:
        courses.append(udemy_data.course3)

    # [Action Node] udemy.com:F1:A1 - Rating filter above 4.5
    rating_filter_ok = all(rating_above_45(c.rating) for c in courses if c and c.rating)
    mentions_rating_filter = has_any_ci(answer, ['rating', '4.5'])

    evaluator.add_custom_node(
        result=bool(rating_filter_ok and mentions_rating_filter and len(courses) >= 3),
        id="udemy_action_rating_filter",
        desc="[Action Node] udemy.com:F1:A1 - Apply rating filter above 4.5 and extract ratings from results",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A2 - Duration filter over 17 hours
    duration_filter_ok = all(duration_over_17(c.duration) for c in courses if c and c.duration)
    mentions_duration_filter = has_any_ci(answer, ['duration', '17 hours', '17 hour', '17h'])

    evaluator.add_custom_node(
        result=bool(duration_filter_ok and mentions_duration_filter and len(courses) >= 3),
        id="udemy_action_duration_filter",
        desc="[Action Node] udemy.com:F1:A2 - Apply duration filter over 17 hours and extract durations from results",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A3 - Level filter (Expert or Intermediate)
    mentions_level_filter = has_any_ci(answer, ['level', 'expert', 'intermediate'])
    mentions_level_relaxation = has_any_ci(answer, ['relax', 'all levels'])

    evaluator.add_custom_node(
        result=bool(mentions_level_filter),
        id="udemy_action_level_filter",
        desc="[Action Node] udemy.com:F1:A3 - Apply Level filter (Expert or Intermediate) and handle relaxation if needed",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A4 - English language filter
    mentions_english = has_any_ci(answer, ['english', 'language'])

    evaluator.add_custom_node(
        result=bool(mentions_english),
        id="udemy_action_language_filter",
        desc="[Action Node] udemy.com:F1:A4 - Apply English language filter",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P3 - Extract instructor names
    instructors_ok = all(bool(c.instructor and c.instructor.strip()) for c in courses if c)

    evaluator.add_custom_node(
        result=bool(instructors_ok and len(courses) >= 3),
        id="udemy_perception_instructors",
        desc="[Perception Node] udemy.com:F1:P3 - Extract instructor names from course listings",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P2 - Extract prices
    prices_ok = all(looks_like_price(c.price) for c in courses if c and c.price)

    evaluator.add_custom_node(
        result=bool(prices_ok and len(courses) >= 3),
        id="udemy_perception_prices",
        desc="[Perception Node] udemy.com:F1:P2 - Extract course prices (including discount detection)",
        parent=udemy_node,
        critical=False
    )

    # Extract course titles and links
    titles_ok = all(bool(c.title and c.title.strip()) for c in courses if c)
    links_ok = all(looks_like_url(c.link) for c in courses if c and c.link)

    evaluator.add_custom_node(
        result=bool(titles_ok and len(courses) >= 3),
        id="udemy_course_titles_extracted",
        desc="Extract all 3 course titles",
        parent=udemy_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(links_ok and len(courses) >= 3),
        id="udemy_course_links_extracted",
        desc="Extract all 3 course links",
        parent=udemy_node,
        critical=False
    )

    # Check if relaxed courses are marked
    if mentions_level_relaxation:
        relaxed_courses_marked = any(c.level_relaxed for c in courses if c)
        evaluator.add_custom_node(
            result=bool(relaxed_courses_marked),
            id="udemy_relaxed_courses_marked",
            desc="Courses added after level relaxation are clearly marked",
            parent=udemy_node,
            critical=False
        )

    # Overall completeness: 3 courses with all required fields
    three_complete_courses = len(courses) >= 3 and all(
        c and c.title and c.instructor and c.rating and c.duration and c.price
        for c in courses
    )

    evaluator.add_custom_node(
        result=bool(three_complete_courses),
        id="udemy_three_complete_courses",
        desc="All 3 courses have complete information (title, instructor, rating, duration, price)",
        parent=udemy_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
