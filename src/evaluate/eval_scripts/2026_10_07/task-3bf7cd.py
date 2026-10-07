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
TASK_ID = "task-3bf7cd"
TASK_DESCRIPTION = "I've recently been watching 'Inception' and was thoroughly impressed by its non-linear narrative and multi-layered dream structure. I'd like to systematically learn about such complex narrative techniques and screenwriting skills.\n\nFirst, please go to the IMDb detail page for this movie. Specifically, focus on the technique-related keywords discussed by professional film critics and screenwriting enthusiasts in the comments section. Identify the screenwriting techniques mentioned, such as narrative structure, timeline management, and character arcs.\n\nThen, using these keywords, search Coursera and edX for relevant film production or screenwriting courses. Ideally, these courses should have a rating of 4.5 or higher and be backed by a university or reputable institution. Please find 3-5 such courses, noting down the course title, platform, institution, rating, and main content focus.\n\nFinally, on Amazon, search for screenwriting textbooks using these technique keywords. Find books with a rating of 4 stars or higher. Please select the 5 most relevant ones, examining the book cover, title, author, rating, and reader reviews to confirm they indeed cover these techniques.\n\nLastly, please compile a complete learning roadmap for me: What core techniques did the film use → Which courses systematically teach these → Which books can provide in-depth learning."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IMDbTechniques(BaseModel):
    """Screenwriting techniques extracted from IMDb comments"""
    techniques_mentioned: Optional[List[str]] = Field(default_factory=list)
    mentions_comments: Optional[bool] = None
    mentions_imdb: Optional[bool] = None


class CourseraEdXCourses(BaseModel):
    """Courses extracted from the answer"""
    courses: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    mentions_coursera: Optional[bool] = None
    mentions_edx: Optional[bool] = None


class AmazonBooks(BaseModel):
    """Books extracted from the answer"""
    books: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    mentions_amazon: Optional[bool] = None


class LearningRoadmap(BaseModel):
    """Learning roadmap structure"""
    has_techniques_section: Optional[bool] = None
    has_courses_section: Optional[bool] = None
    has_books_section: Optional[bool] = None
    has_structured_roadmap: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_imdb_techniques() -> str:
    return """
Extract information about the IMDb visit and technique keywords from the answer.

Return:
- techniques_mentioned: list of screenwriting technique keywords mentioned (e.g., "narrative structure", "non-linear narrative", "timeline management", "character arcs", etc.)
- mentions_comments: true if the answer mentions browsing or reading IMDb comments/reviews
- mentions_imdb: true if the answer mentions visiting IMDb

If any field is missing, set appropriate defaults.
"""


def prompt_extract_courses() -> str:
    return """
Extract the courses information from the answer.

Return:
- courses: list of course objects, each containing available fields like title, platform, institution, rating, content_focus
- mentions_coursera: true if Coursera is mentioned
- mentions_edx: true if edX is mentioned

If no courses are found, return empty list.
"""


def prompt_extract_books() -> str:
    return """
Extract the books information from the answer.

Return:
- books: list of book objects, each containing available fields like title, author, rating, cover_description, reviews_mention
- mentions_amazon: true if Amazon is mentioned

If no books are found, return empty list.
"""


def prompt_extract_roadmap() -> str:
    return """
Extract the learning roadmap structure from the answer.

Return:
- has_techniques_section: true if there's a section describing techniques used in Inception
- has_courses_section: true if there's a section listing courses that teach these techniques
- has_books_section: true if there's a section listing books for in-depth learning
- has_structured_roadmap: true if there's a clear flow from techniques → courses → books

Set to false if sections are missing.
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


def count_items(items: Optional[List]) -> int:
    if not items:
        return 0
    return len(items)


def has_high_ratings(courses: Optional[List[Dict]], min_rating: float = 4.5) -> bool:
    if not courses:
        return False
    count = 0
    for course in courses:
        rating_val = course.get('rating')
        if rating_val:
            if isinstance(rating_val, (int, float)) and rating_val >= min_rating:
                count += 1
            elif isinstance(rating_val, str):
                num = extract_float(rating_val)
                if num and num >= min_rating:
                    count += 1
    return count > 0


def has_institutions(courses: Optional[List[Dict]]) -> bool:
    if not courses:
        return False
    return any(course.get('institution') for course in courses)


def books_have_ratings(books: Optional[List[Dict]], min_rating: float = 4.0) -> bool:
    if not books:
        return False
    count = 0
    for book in books:
        rating_val = book.get('rating')
        if rating_val:
            if isinstance(rating_val, (int, float)) and rating_val >= min_rating:
                count += 1
            elif isinstance(rating_val, str):
                num = extract_float(rating_val)
                if num and num >= min_rating:
                    count += 1
    return count > 0


def mentions_reviews_or_cover(books: Optional[List[Dict]]) -> bool:
    if not books:
        return False
    return any(book.get('reviews_mention') or book.get('cover_description') for book in books)


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
    imdb_info = await evaluator.extract(
        prompt=prompt_extract_imdb_techniques(),
        template_class=IMDbTechniques,
        extraction_name="imdb_techniques"
    )

    courses_info = await evaluator.extract(
        prompt=prompt_extract_courses(),
        template_class=CourseraEdXCourses,
        extraction_name="courses_info"
    )

    books_info = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=AmazonBooks,
        extraction_name="books_info"
    )

    roadmap_info = await evaluator.extract(
        prompt=prompt_extract_roadmap(),
        template_class=LearningRoadmap,
        extraction_name="learning_roadmap"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb Section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb Inception detail page and comment analysis",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F3:A26 - Navigate to detail page
    imdb_detail_action = (
        (imdb_info.mentions_imdb or has_any_ci(answer, ['imdb'])) and
        has_any_ci(answer, ['inception', 'detail'])
    )
    evaluator.add_custom_node(
        result=bool(imdb_detail_action),
        id="imdb_action_detail_page",
        desc="[Action Node] imdb.com:F3:A26 - Navigate to the IMDb detail page for Inception",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F1:A10 - Browse comments with pagination
    imdb_pagination_action = (
        (imdb_info.mentions_comments or has_any_ci(answer, ['comment', 'review'])) and
        (has_any_ci(answer, ['browse', 'read', 'check', 'look through', 'multiple']))
    )
    evaluator.add_custom_node(
        result=bool(imdb_pagination_action),
        id="imdb_action_pagination",
        desc="[Action Node] imdb.com:F1:A10 - Browse through multiple comments using pagination",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F4:P25 - Understand comment content
    techniques_count = count_items(imdb_info.techniques_mentioned)
    has_screenwriting_terms = has_any_ci(answer, [
        'narrative', 'structure', 'timeline', 'character arc', 'plot', 'screenplay', 'non-linear'
    ])
    imdb_understanding = techniques_count >= 2 and has_screenwriting_terms
    evaluator.add_custom_node(
        result=bool(imdb_understanding),
        id="imdb_perception_understand_comments",
        desc="[Perception Node] imdb.com:F4:P25 - Understand and extract screenwriting technique keywords from comments",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Coursera Section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera course search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] coursera.org:F2:A8 - Multi-select filtering
    coursera_filter_action = (
        courses_info.mentions_coursera or has_any_ci(answer, ['coursera'])
    ) and has_any_ci(answer, ['filter', 'select', 'search'])
    evaluator.add_custom_node(
        result=bool(coursera_filter_action),
        id="coursera_action_filter",
        desc="[Action Node] coursera.org:F2:A8 - Use multi-select filters to narrow course search",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A25 - Scroll to load more results
    coursera_scroll_action = (
        courses_info.mentions_coursera and
        count_items(courses_info.courses) >= 3
    )
    evaluator.add_custom_node(
        result=bool(coursera_scroll_action),
        id="coursera_action_scroll",
        desc="[Action Node] coursera.org:F1:A25 - Scroll through course results to find 3-5 courses",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Sort by rating
    coursera_sort_action = (
        has_any_ci(answer, ['rating', 'sort', '4.5']) and
        courses_info.mentions_coursera
    )
    evaluator.add_custom_node(
        result=bool(coursera_sort_action),
        id="coursera_action_sort",
        desc="[Action Node] coursera.org:F1:A12 - Sort courses by rating to find high-rated options",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F3:P2 - Recognize course covers
    coursera_visual_perception = (
        count_items(courses_info.courses) > 0 and
        has_any_ci(answer, ['course', 'cover', 'card', 'image'])
    )
    evaluator.add_custom_node(
        result=bool(coursera_visual_perception),
        id="coursera_perception_covers",
        desc="[Perception Node] coursera.org:F3:P2 - Recognize and assess course cover images for relevance",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F3:P7 - Understand course syllabus
    coursera_content_understanding = (
        count_items(courses_info.courses) > 0 and
        any(course.get('content_focus') for course in (courses_info.courses or []))
    )
    evaluator.add_custom_node(
        result=bool(coursera_content_understanding),
        id="coursera_perception_syllabus",
        desc="[Perception Node] coursera.org:F3:P7 - Understand course syllabus and content focus",
        parent=coursera_node,
        critical=False
    )

    # Coursera results validation
    coursera_results = (
        count_items(courses_info.courses) >= 3 and
        has_high_ratings(courses_info.courses, 4.5) and
        has_institutions(courses_info.courses)
    )
    evaluator.add_custom_node(
        result=bool(coursera_results),
        id="coursera_results_quality",
        desc="Found 3-5 courses with 4.5+ rating from reputable institutions on Coursera",
        parent=coursera_node,
        critical=False
    )

    # 3.3 edX Section
    edx_node = evaluator.add_sequential(
        id="edx_section",
        desc="edX course search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] edx.org:F1:A5 - Pagination through results
    edx_pagination_action = (
        courses_info.mentions_edx or has_any_ci(answer, ['edx'])
    ) and has_any_ci(answer, ['page', 'browse', 'multiple', 'several'])
    evaluator.add_custom_node(
        result=bool(edx_pagination_action),
        id="edx_action_pagination",
        desc="[Action Node] edx.org:F1:A5 - Browse through multiple pages of edX search results",
        parent=edx_node,
        critical=False
    )

    # [Perception Node] edx.org:F3:P6 - Understand learning outcomes
    edx_content_understanding = (
        courses_info.mentions_edx and
        count_items(courses_info.courses) > 0
    )
    evaluator.add_custom_node(
        result=bool(edx_content_understanding),
        id="edx_perception_outcomes",
        desc="[Perception Node] edx.org:F3:P6 - Understand course learning outcomes and content",
        parent=edx_node,
        critical=False
    )

    # 3.4 Amazon Section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon book search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A6 - Pagination through book results
    amazon_pagination_action = (
        books_info.mentions_amazon or has_any_ci(answer, ['amazon'])
    ) and count_items(books_info.books) >= 5
    evaluator.add_custom_node(
        result=bool(amazon_pagination_action),
        id="amazon_action_pagination",
        desc="[Action Node] Amazon:F1:A6 - Browse through multiple pages to find 5 relevant books",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Rating filter
    amazon_rating_filter = (
        books_info.mentions_amazon and
        has_any_ci(answer, ['rating', '4 star', 'filter'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_rating_filter),
        id="amazon_action_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Apply rating filter for 4+ stars",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A26 - View book cover images
    amazon_cover_action = (
        books_info.mentions_amazon and
        has_any_ci(answer, ['cover', 'image', 'preview'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_cover_action),
        id="amazon_action_cover_preview",
        desc="[Action Node] Amazon:F5:A26 - Click to preview book cover images",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P3 - Recognize book covers
    amazon_cover_perception = (
        count_items(books_info.books) > 0 and
        mentions_reviews_or_cover(books_info.books)
    )
    evaluator.add_custom_node(
        result=bool(amazon_cover_perception),
        id="amazon_perception_covers",
        desc="[Perception Node] Amazon:F5:P3 - Recognize and assess book cover design for relevance",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P9 - Extract visual features
    amazon_visual_features = (
        count_items(books_info.books) > 0 and
        any(book.get('cover_description') for book in (books_info.books or []))
    )
    evaluator.add_custom_node(
        result=bool(amazon_visual_features),
        id="amazon_perception_visual_features",
        desc="[Perception Node] Amazon:F5:P9 - Extract visual features from book covers (design, professionalism)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F10:P15 - Understand review content
    amazon_review_understanding = (
        count_items(books_info.books) > 0 and
        has_any_ci(answer, ['review', 'reader', 'confirm', 'validate'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_review_understanding),
        id="amazon_perception_reviews",
        desc="[Perception Node] Amazon:F10:P15 - Understand reader reviews to confirm technique coverage",
        parent=amazon_node,
        critical=False
    )

    # Amazon results validation
    amazon_results = (
        count_items(books_info.books) == 5 and
        books_have_ratings(books_info.books, 4.0)
    )
    evaluator.add_custom_node(
        result=bool(amazon_results),
        id="amazon_results_quality",
        desc="Found 5 books with 4+ star ratings on Amazon",
        parent=amazon_node,
        critical=False
    )

    # 3.5 Learning Roadmap Section
    roadmap_node = evaluator.add_sequential(
        id="roadmap_section",
        desc="Complete learning roadmap compilation",
        parent=root,
        critical=False
    )

    roadmap_structure = (
        roadmap_info.has_techniques_section and
        roadmap_info.has_courses_section and
        roadmap_info.has_books_section and
        roadmap_info.has_structured_roadmap
    )
    evaluator.add_custom_node(
        result=bool(roadmap_structure),
        id="roadmap_complete",
        desc="Compiled complete learning roadmap: techniques → courses → books",
        parent=roadmap_node,
        critical=False
    )

    # Integration check
    integration_ok = (
        count_items(imdb_info.techniques_mentioned) > 0 and
        count_items(courses_info.courses) >= 3 and
        count_items(books_info.books) >= 3 and
        roadmap_info.has_structured_roadmap
    )
    evaluator.add_custom_node(
        result=bool(integration_ok),
        id="integration_complete",
        desc="Successfully integrated IMDb techniques with Coursera/edX courses and Amazon books into roadmap",
        parent=roadmap_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
