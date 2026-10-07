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
TASK_ID = "task-f8c8b3"
TASK_DESCRIPTION = "I am designing an occupational health and safety training system for a manufacturing enterprise, and I need to integrate authoritative data, online courses, and learning materials.\n\nFirst, go to the BLS (U.S. Bureau of Labor Statistics) to search for occupational injury statistics in Manufacturing. Filter for annual data from 2020-2024 and record the trends in injury rates and days away from work.\n\nNext, go to Coursera to search for Occupational Health and Safety-related courses. Requirements: rating above 4.5, include video lectures and hands-on projects, duration 4-8 weeks, suitable for Beginner or Intermediate level. Identify the 3 most suitable courses, expand to view each course's syllabus, and record the core module names.\n\nThen, go to Khan Academy to search for educational videos related to ergonomics or occupational safety. Find 2-3 short videos (under 10 minutes) suitable for employees with no prior knowledge.\n\nAfter that, go to MIT OCW to search for courses on industrial hygiene or occupational health engineering. Find one course with complete Lecture Notes and Assignments. View the Syllabus and record the main technical topics covered.\n\nFinally, go to Goodreads to search for books related to occupational health management. Filter for books with a rating above 4.0 and more than 100 reviews. Find 3 recommended books and check the review section to understand what kind of audience readers recommend these books for.\n\nOutput:\n(1) BLS Manufacturing Injury Data: Annual injury rates and days away from work for 2020-2024, and the link to the data page;\n(2) Coursera Courses: Names of the 3 courses, ratings, durations, difficulty levels, list of core modules (at least 3 modules), and course links;\n(3) Khan Academy Videos: Titles of 2-3 videos, durations, topics, and video links;\n(4) MIT OCW Course: Course number, course name, instructor, list of main technical topics covered (at least 5), and course link;\n(5) Goodreads Books: Titles of the 3 books, authors, ratings, number of reviews, target audience recommended by readers, and links to the book detail pages."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BLSData(BaseModel):
    """BLS Manufacturing injury data extracted from the answer"""
    year_2020_injury_rate: Optional[str] = None
    year_2021_injury_rate: Optional[str] = None
    year_2022_injury_rate: Optional[str] = None
    year_2023_injury_rate: Optional[str] = None
    year_2024_injury_rate: Optional[str] = None
    year_2020_days_away: Optional[str] = None
    year_2021_days_away: Optional[str] = None
    year_2022_days_away: Optional[str] = None
    year_2023_days_away: Optional[str] = None
    year_2024_days_away: Optional[str] = None
    data_page_url: Optional[str] = None


class CourseraCourse(BaseModel):
    """Single Coursera course information"""
    course_name: Optional[str] = None
    rating: Optional[str] = None
    duration: Optional[str] = None
    difficulty_level: Optional[str] = None
    core_modules: Optional[List[str]] = Field(default_factory=list)
    course_link: Optional[str] = None


class CourseraData(BaseModel):
    """All three Coursera courses extracted from the answer"""
    course_1: Optional[CourseraCourse] = None
    course_2: Optional[CourseraCourse] = None
    course_3: Optional[CourseraCourse] = None


class KhanAcademyVideo(BaseModel):
    """Single Khan Academy video information"""
    title: Optional[str] = None
    duration: Optional[str] = None
    topic: Optional[str] = None
    video_link: Optional[str] = None


class KhanAcademyData(BaseModel):
    """Khan Academy videos extracted from the answer"""
    videos: Optional[List[KhanAcademyVideo]] = Field(default_factory=list)


class MITOCWData(BaseModel):
    """MIT OCW course extracted from the answer"""
    course_number: Optional[str] = None
    course_name: Optional[str] = None
    instructor: Optional[str] = None
    technical_topics: Optional[List[str]] = Field(default_factory=list)
    course_link: Optional[str] = None


class GoodreadsBook(BaseModel):
    """Single Goodreads book information"""
    title: Optional[str] = None
    author: Optional[str] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    target_audience: Optional[str] = None
    book_link: Optional[str] = None


class GoodreadsData(BaseModel):
    """All three Goodreads books extracted from the answer"""
    book_1: Optional[GoodreadsBook] = None
    book_2: Optional[GoodreadsBook] = None
    book_3: Optional[GoodreadsBook] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bls_data() -> str:
    return """
Extract the BLS Manufacturing injury statistics from the answer for years 2020-2024.

Return:
- year_2020_injury_rate, year_2021_injury_rate, year_2022_injury_rate, year_2023_injury_rate, year_2024_injury_rate: injury rates for each year as written
- year_2020_days_away, year_2021_days_away, year_2022_days_away, year_2023_days_away, year_2024_days_away: days away from work for each year as written
- data_page_url: the BLS data page URL

If any field is missing, set it to null.
"""


def prompt_extract_coursera_data() -> str:
    return """
Extract information about the 3 Coursera courses from the answer.

For each course return:
- course_name: course title
- rating: rating value
- duration: duration in weeks
- difficulty_level: Beginner or Intermediate
- core_modules: list of core module names (at least 3)
- course_link: URL to the course

If any field is missing, set it to null or empty list.
"""


def prompt_extract_khan_academy_data() -> str:
    return """
Extract information about 2-3 Khan Academy videos from the answer.

For each video return:
- title: video title
- duration: video length
- topic: subject area or topic
- video_link: URL to the video

Return a list of videos. If any field is missing, set it to null.
"""


def prompt_extract_mit_ocw_data() -> str:
    return """
Extract information about the MIT OCW course from the answer.

Return:
- course_number: course number
- course_name: course title
- instructor: instructor name
- technical_topics: list of main technical topics covered (at least 5)
- course_link: URL to the course

If any field is missing, set it to null or empty list.
"""


def prompt_extract_goodreads_data() -> str:
    return """
Extract information about the 3 Goodreads books from the answer.

For each book return:
- title: book title
- author: author name
- rating: rating value
- review_count: number of reviews
- target_audience: recommended audience from reviews
- book_link: URL to the book detail page

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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.'])


def count_year_data_points(bls_data: BLSData) -> int:
    """Count how many years have at least one data point"""
    count = 0
    years = [
        (bls_data.year_2020_injury_rate, bls_data.year_2020_days_away),
        (bls_data.year_2021_injury_rate, bls_data.year_2021_days_away),
        (bls_data.year_2022_injury_rate, bls_data.year_2022_days_away),
        (bls_data.year_2023_injury_rate, bls_data.year_2023_days_away),
        (bls_data.year_2024_injury_rate, bls_data.year_2024_days_away),
    ]
    for ir, da in years:
        if ir or da:
            count += 1
    return count


def count_valid_courses(coursera_data: CourseraData) -> int:
    """Count how many courses have basic required info"""
    count = 0
    for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
        if course and course.course_name and course.rating:
            count += 1
    return count


def count_valid_videos(khan_data: KhanAcademyData) -> int:
    """Count valid videos"""
    if not khan_data or not khan_data.videos:
        return 0
    return len([v for v in khan_data.videos if v and v.title])


def count_valid_books(goodreads_data: GoodreadsData) -> int:
    """Count how many books have basic required info"""
    count = 0
    for book in [goodreads_data.book_1, goodreads_data.book_2, goodreads_data.book_3]:
        if book and book.title and book.author:
            count += 1
    return count


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
    bls_data = await evaluator.extract(
        prompt=prompt_extract_bls_data(),
        template_class=BLSData,
        extraction_name="bls_manufacturing_data"
    )

    coursera_data = await evaluator.extract(
        prompt=prompt_extract_coursera_data(),
        template_class=CourseraData,
        extraction_name="coursera_courses"
    )

    khan_data = await evaluator.extract(
        prompt=prompt_extract_khan_academy_data(),
        template_class=KhanAcademyData,
        extraction_name="khan_academy_videos"
    )

    mit_data = await evaluator.extract(
        prompt=prompt_extract_mit_ocw_data(),
        template_class=MITOCWData,
        extraction_name="mit_ocw_course"
    )

    goodreads_data = await evaluator.extract(
        prompt=prompt_extract_goodreads_data(),
        template_class=GoodreadsData,
        extraction_name="goodreads_books"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 BLS Section
    bls_node = evaluator.add_sequential(
        id="bls_section",
        desc="BLS Manufacturing injury statistics 2020-2024",
        parent=root,
        critical=False
    )

    # [Action Node] bls.gov:F1:A21 - Search for Manufacturing injury data
    bls_search_ok = has_any_ci(answer, ['bls', 'bureau of labor statistics']) and has_any_ci(answer, ['manufacturing'])
    evaluator.add_custom_node(
        result=bool(bls_search_ok),
        id="bls_action_search",
        desc="[Action Node] bls.gov:F1:A21 - Search for Manufacturing occupational injury statistics on BLS",
        parent=bls_node,
        critical=False
    )

    # [Action Node] bls.gov:F1:A3 - Filter for 2020-2024 annual data
    year_count = count_year_data_points(bls_data)
    has_time_filter = year_count >= 3  # At least 3 years of data suggests filtering worked
    evaluator.add_custom_node(
        result=bool(has_time_filter),
        id="bls_action_filter_years",
        desc="[Action Node] bls.gov:F1:A3 - Filter for annual data from 2020-2024",
        parent=bls_node,
        critical=False
    )

    # [Perception Node] bls.gov:F1:P16 - Extract injury rates and days away trends
    has_injury_data = year_count >= 4  # At least 4 years to show trend
    has_url = looks_like_url(bls_data.data_page_url) and has_any_ci(bls_data.data_page_url, ['bls.gov'])
    evaluator.add_custom_node(
        result=bool(has_injury_data and has_url),
        id="bls_perception_data_trends",
        desc="[Perception Node] bls.gov:F1:P16 - Record injury rates and days away from work trends with data page URL",
        parent=bls_node,
        critical=False
    )

    # 3.2 Coursera Section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera Occupational Health and Safety courses",
        parent=root,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Sort/rank courses by rating
    course_count = count_valid_courses(coursera_data)
    has_ratings = False
    if course_count >= 3:
        ratings = []
        for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
            if course and course.rating:
                r = extract_float(course.rating)
                if r and r >= 4.5:
                    ratings.append(r)
        has_ratings = len(ratings) >= 3

    evaluator.add_custom_node(
        result=bool(has_ratings),
        id="coursera_action_sort_rating",
        desc="[Action Node] coursera.org:F1:A12 - Find 3 courses with rating >= 4.5",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A8 - Filter for video lectures and hands-on projects
    has_video_project = False
    if course_count >= 3:
        has_video_project = has_any_ci(answer, ['video', 'lecture', 'project'])

    evaluator.add_custom_node(
        result=bool(has_video_project),
        id="coursera_action_filter_learning_product",
        desc="[Action Node] coursera.org:F2:A8 - Filter for courses with video lectures and hands-on projects",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A10 - Filter for duration 4-8 weeks
    has_duration = False
    if course_count >= 3:
        durations_ok = []
        for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
            if course and course.duration:
                d = extract_float(course.duration)
                if d and 4 <= d <= 8:
                    durations_ok.append(True)
        has_duration = len(durations_ok) >= 2  # At least 2 out of 3

    evaluator.add_custom_node(
        result=bool(has_duration),
        id="coursera_action_filter_duration",
        desc="[Action Node] coursera.org:F2:A10 - Filter for duration 4-8 weeks",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F2:A2 - Filter for Beginner or Intermediate level
    has_level = False
    if course_count >= 3:
        levels_ok = []
        for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
            if course and course.difficulty_level:
                if has_any_ci(course.difficulty_level, ['beginner', 'intermediate']):
                    levels_ok.append(True)
        has_level = len(levels_ok) >= 2

    evaluator.add_custom_node(
        result=bool(has_level),
        id="coursera_action_filter_level",
        desc="[Action Node] coursera.org:F2:A2 - Filter for Beginner or Intermediate level",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F3:A23 - Expand to view syllabus
    has_syllabus = False
    if course_count >= 3:
        module_counts = []
        for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
            if course and course.core_modules:
                module_counts.append(len(course.core_modules))
        has_syllabus = len([c for c in module_counts if c >= 3]) >= 2  # At least 2 courses with 3+ modules

    evaluator.add_custom_node(
        result=bool(has_syllabus),
        id="coursera_action_expand_syllabus",
        desc="[Action Node] coursera.org:F3:A23 - Expand course syllabus to view core modules",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F3:P7 - Record core module names
    modules_ok = False
    if course_count >= 3:
        for course in [coursera_data.course_1, coursera_data.course_2, coursera_data.course_3]:
            if course and course.core_modules and len(course.core_modules) >= 3:
                modules_ok = True
                break

    evaluator.add_custom_node(
        result=bool(modules_ok),
        id="coursera_perception_modules",
        desc="[Perception Node] coursera.org:F3:P7 - Record core module names from syllabus",
        parent=coursera_node,
        critical=False
    )

    # 3.3 Khan Academy Section
    khan_node = evaluator.add_sequential(
        id="khan_academy_section",
        desc="Khan Academy ergonomics and occupational safety videos",
        parent=root,
        critical=False
    )

    # [Action Node] khanacademy.org:F1:A11 - Search for ergonomics or occupational safety videos
    khan_search_ok = has_any_ci(answer, ['khan academy']) and has_any_ci(answer, ['ergonomics', 'occupational safety', 'occupational', 'safety'])
    evaluator.add_custom_node(
        result=bool(khan_search_ok),
        id="khan_action_search",
        desc="[Action Node] khanacademy.org:F1:A11 - Search for ergonomics or occupational safety videos",
        parent=khan_node,
        critical=False
    )

    # [Action Node] khanacademy.org:F1:A6 - Filter for short videos (under 10 minutes)
    video_count = count_valid_videos(khan_data)
    videos_under_10min = False
    if video_count >= 2 and khan_data.videos:
        under_10 = []
        for video in khan_data.videos:
            if video and video.duration:
                # Extract minutes
                m = extract_float(video.duration)
                if m and m <= 10:
                    under_10.append(True)
        videos_under_10min = len(under_10) >= 2

    evaluator.add_custom_node(
        result=bool(videos_under_10min),
        id="khan_action_filter_duration",
        desc="[Action Node] khanacademy.org:F1:A6 - Filter for videos under 10 minutes",
        parent=khan_node,
        critical=False
    )

    # [Perception Node] khanacademy.org:F1:P20 - Record video topics
    has_topics = False
    if video_count >= 2 and khan_data.videos:
        topics_present = [v for v in khan_data.videos if v and v.topic]
        has_topics = len(topics_present) >= 2

    evaluator.add_custom_node(
        result=bool(has_topics),
        id="khan_perception_topics",
        desc="[Perception Node] khanacademy.org:F1:P20 - Record video topics and classifications",
        parent=khan_node,
        critical=False
    )

    # 3.4 MIT OCW Section
    mit_node = evaluator.add_sequential(
        id="mit_ocw_section",
        desc="MIT OCW industrial hygiene or occupational health engineering course",
        parent=root,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F1:A7 - Search/filter for industrial hygiene or occupational health courses
    mit_search_ok = has_any_ci(answer, ['mit ocw', 'mit', 'ocw']) and has_any_ci(answer, ['industrial hygiene', 'occupational health', 'hygiene', 'occupational'])
    evaluator.add_custom_node(
        result=bool(mit_search_ok),
        id="mit_action_search",
        desc="[Action Node] ocw.mit.edu:F1:A7 - Search for industrial hygiene or occupational health engineering courses",
        parent=mit_node,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F1:A8 - Filter for courses with Lecture Notes and Assignments
    has_resources = has_any_ci(answer, ['lecture notes', 'assignments', 'lecture', 'assignment'])
    evaluator.add_custom_node(
        result=bool(has_resources),
        id="mit_action_filter_resources",
        desc="[Action Node] ocw.mit.edu:F1:A8 - Filter for courses with complete Lecture Notes and Assignments",
        parent=mit_node,
        critical=False
    )

    # [Action Node] ocw.mit.edu:F3:A3 - Click to view Syllabus tab
    has_syllabus_tab = has_any_ci(answer, ['syllabus'])
    evaluator.add_custom_node(
        result=bool(has_syllabus_tab),
        id="mit_action_view_syllabus",
        desc="[Action Node] ocw.mit.edu:F3:A3 - Switch to Syllabus tab to view course outline",
        parent=mit_node,
        critical=False
    )

    # [Perception Node] ocw.mit.edu:F3:P9 - Extract technical topics from syllabus
    has_technical_topics = False
    if mit_data and mit_data.technical_topics:
        has_technical_topics = len(mit_data.technical_topics) >= 5

    evaluator.add_custom_node(
        result=bool(has_technical_topics),
        id="mit_perception_topics",
        desc="[Perception Node] ocw.mit.edu:F3:P9 - Record main technical topics covered (at least 5)",
        parent=mit_node,
        critical=False
    )

    # [Perception Node] ocw.mit.edu:F1:P3 - Extract course metadata
    has_metadata = False
    if mit_data:
        has_metadata = bool(mit_data.course_number and mit_data.course_name and mit_data.instructor)

    evaluator.add_custom_node(
        result=bool(has_metadata),
        id="mit_perception_metadata",
        desc="[Perception Node] ocw.mit.edu:F1:P3 - Extract course number, name, and instructor",
        parent=mit_node,
        critical=False
    )

    # 3.5 Goodreads Section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads occupational health management books",
        parent=root,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A1 - Search for occupational health management books
    goodreads_search_ok = has_any_ci(answer, ['goodreads']) and has_any_ci(answer, ['occupational health', 'health management', 'occupational'])
    evaluator.add_custom_node(
        result=bool(goodreads_search_ok),
        id="goodreads_action_search",
        desc="[Action Node] goodreads.com:F1:A1 - Search for occupational health management books",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P2 - Identify books with rating > 4.0 and > 100 reviews
    book_count = count_valid_books(goodreads_data)
    books_meet_criteria = False
    if book_count >= 3:
        criteria_met = []
        for book in [goodreads_data.book_1, goodreads_data.book_2, goodreads_data.book_3]:
            if book and book.rating and book.review_count:
                r = extract_float(book.rating)
                rc = extract_float(book.review_count)
                if r and rc and r > 4.0 and rc > 100:
                    criteria_met.append(True)
        books_meet_criteria = len(criteria_met) >= 2

    evaluator.add_custom_node(
        result=bool(books_meet_criteria),
        id="goodreads_perception_filter_criteria",
        desc="[Perception Node] goodreads.com:F2:P2 - Identify books with rating > 4.0 and > 100 reviews",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P6 - Extract precise ratings and review counts
    has_precise_data = False
    if book_count >= 3:
        for book in [goodreads_data.book_1, goodreads_data.book_2, goodreads_data.book_3]:
            if book and book.rating and book.review_count:
                if contains_digits(book.rating) and contains_digits(book.review_count):
                    has_precise_data = True
                    break

    evaluator.add_custom_node(
        result=bool(has_precise_data),
        id="goodreads_perception_precise_data",
        desc="[Perception Node] goodreads.com:F2:P6 - Extract precise rating and review count values",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F2:A2 - Switch to Community Reviews section
    has_reviews_section = has_any_ci(answer, ['review', 'community', 'audience', 'reader'])
    evaluator.add_custom_node(
        result=bool(has_reviews_section),
        id="goodreads_action_view_reviews",
        desc="[Action Node] goodreads.com:F2:A2 - View Community Reviews to understand target audience",
        parent=goodreads_node,
        critical=False
    )

    # Perception: Target audience from reviews
    has_audience = False
    if book_count >= 3:
        audiences = []
        for book in [goodreads_data.book_1, goodreads_data.book_2, goodreads_data.book_3]:
            if book and book.target_audience:
                audiences.append(True)
        has_audience = len(audiences) >= 2

    evaluator.add_custom_node(
        result=bool(has_audience),
        id="goodreads_perception_audience",
        desc="Extract target audience recommendations from reader reviews",
        parent=goodreads_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
