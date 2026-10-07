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
TASK_ID = "task-2d3de9"
TASK_DESCRIPTION = "I just watched 'Dune 2' and want to delve deeper into the story behind the film.\n\nFirst, go to IMDb, find the movie, and note down the director's name, the original author from the 'Writers' section, and the title of the original book.\n\nThen, search for the original book title on Goodreads. Once found, provide its rating, number of reviews, and page count. If it's part of a series, list the titles of other books in the same series.\n\nNext, go to YouTube and search for 'Dune 2 behind the scenes' or 'Dune 2 making of'. Filter the results to videos longer than 15 minutes, sort them by view count in descending order, and identify the top 3 videos that are behind-the-scenes/making-of specials.\n\nFinally, visit the official Criterion Collection website and search for the director's name. Check if any of his other works are featured. If so, list the title of the work, available formats (DVD/Blu-ray/4K), and a brief description of the special features.\n\n**Output:**\nMovie Director's Name, Original Book Title, Original Author, Goodreads Rating, Number of Reviews, Page Count, Other Series Titles (if applicable), Goodreads Book Link, Titles of the top 3 YouTube videos, View Counts, Durations, Video Links, Titles of other director's works featured on Criterion, Formats, Brief Description of Special Features, Criterion Collection Detail Page Links."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IMDbInfo(BaseModel):
    """IMDb movie information extracted from the answer"""
    director_name: Optional[str] = None
    original_author: Optional[str] = None
    original_book_title: Optional[str] = None
    imdb_url: Optional[str] = None


class GoodreadsInfo(BaseModel):
    """Goodreads book information extracted from the answer"""
    rating: Optional[str] = None
    review_count: Optional[str] = None
    page_count: Optional[str] = None
    series_titles: Optional[List[str]] = Field(default_factory=list)
    goodreads_url: Optional[str] = None


class YouTubeVideo(BaseModel):
    """Single YouTube video information"""
    title: Optional[str] = None
    view_count: Optional[str] = None
    duration: Optional[str] = None
    url: Optional[str] = None


class YouTubeInfo(BaseModel):
    """YouTube videos extracted from the answer"""
    top_videos: Optional[List[YouTubeVideo]] = Field(default_factory=list)


class CriterionWork(BaseModel):
    """Single Criterion Collection work"""
    title: Optional[str] = None
    formats: Optional[List[str]] = Field(default_factory=list)
    special_features_desc: Optional[str] = None
    detail_url: Optional[str] = None


class CriterionInfo(BaseModel):
    """Criterion Collection information extracted from the answer"""
    works: Optional[List[CriterionWork]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_imdb_from_answer() -> str:
    return """
Extract the IMDb information for 'Dune 2' from the answer:

- director_name: the director's name exactly as stated
- original_author: the original author from the 'Writers' section
- original_book_title: the title of the original book
- imdb_url: the IMDb movie detail page URL if mentioned

If any field is missing, set it to null.
"""


def prompt_extract_goodreads_from_answer() -> str:
    return """
Extract the Goodreads book information from the answer:

- rating: the book rating exactly as stated (include format like "4.25" or "4.25/5")
- review_count: the number of reviews exactly as stated
- page_count: the page count exactly as stated
- series_titles: list of other books in the same series if mentioned (empty list if not a series or not mentioned)
- goodreads_url: the Goodreads book detail page URL if mentioned

If any field is missing, set it to null or empty list as appropriate.
"""


def prompt_extract_youtube_from_answer() -> str:
    return """
Extract the top 3 YouTube behind-the-scenes/making-of videos from the answer:

For each video, extract:
- title: the video title
- view_count: the view count exactly as stated
- duration: the duration exactly as stated
- url: the YouTube video URL

Return as a list of top_videos. If fewer than 3 videos are mentioned, include only what's present.
"""


def prompt_extract_criterion_from_answer() -> str:
    return """
Extract the Criterion Collection works by the director from the answer:

For each work, extract:
- title: the title of the work
- formats: list of available formats (e.g., ["DVD", "Blu-ray", "4K"])
- special_features_desc: brief description of special features
- detail_url: the Criterion Collection detail page URL if mentioned

Return as a list of works. If no works are found, return an empty list.
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


def is_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept formats like "4.25", "4.25/5", "4.2 out of 5", etc.
    return contains_digits(text) and (bool(re.search(r'\d+\.\d+', text)) or bool(re.search(r'\d+/\d+', text)))


def looks_like_count(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept numbers with commas or plain numbers
    return bool(re.search(r'\d+', text))


def looks_like_duration_over_15min(text: Optional[str]) -> bool:
    if not text:
        return False
    # Look for patterns like "20:15", "1:05:30", "25 minutes", etc.
    # Check if it's over 15 minutes
    time_patterns = [
        r'(\d+):(\d+):(\d+)',  # HH:MM:SS
        r'(\d+):(\d+)',         # MM:SS or HH:MM
        r'(\d+)\s*min',         # X minutes
    ]

    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # HH:MM:SS
                hours, mins, secs = map(int, groups)
                total_mins = hours * 60 + mins
                return total_mins >= 15
            elif len(groups) == 2:  # MM:SS or could be HH:MM
                first, second = map(int, groups)
                # If first number is > 15, assume it's minutes
                if first >= 15:
                    return True
                # If first is small, might be hours
                if first > 0:
                    return True
            elif len(groups) == 1:  # X minutes
                mins = int(groups[0])
                return mins >= 15

    return False


def looks_like_behind_the_scenes_title(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['behind the scenes', 'making of', 'featurette', 'bts', 'making-of',
                'documentary', 'production', 'backstage']
    return has_any_ci(text, keywords)


def extract_view_count_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove commas and extract number
    text_clean = text.replace(',', '')
    match = re.search(r'(\d+)', text_clean)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


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
        prompt=prompt_extract_imdb_from_answer(),
        template_class=IMDbInfo,
        extraction_name="imdb_info"
    )

    goodreads_info = await evaluator.extract(
        prompt=prompt_extract_goodreads_from_answer(),
        template_class=GoodreadsInfo,
        extraction_name="goodreads_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_from_answer(),
        template_class=YouTubeInfo,
        extraction_name="youtube_info"
    )

    criterion_info = await evaluator.extract(
        prompt=prompt_extract_criterion_from_answer(),
        template_class=CriterionInfo,
        extraction_name="criterion_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb movie information extraction for Dune 2",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Select search type dropdown
    imdb_search_type_ok = has_any_ci(answer, ['imdb']) and (
        has_any_ci(answer, ['dune 2', 'dune: part two', 'dune part two'])
    )
    evaluator.add_custom_node(
        result=bool(imdb_search_type_ok),
        id="imdb_search_type_selection",
        desc="[Action Node] imdb.com:F1:A1 - Navigate to IMDb and search for the movie (implies selecting search type as Titles)",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F1:A10 - Paginate search results
    # Check if answer indicates finding the correct movie (2024 version)
    correct_movie_found = (
        imdb_info.imdb_url and is_valid_url(imdb_info.imdb_url, 'imdb')
    ) or has_any_ci(answer, ['2024'])
    evaluator.add_custom_node(
        result=bool(correct_movie_found),
        id="imdb_search_pagination",
        desc="[Action Node] imdb.com:F1:A10 - Locate the correct Dune 2 (2024) movie entry (may require pagination)",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A26 - Click movie card to enter details
    detail_page_accessed = bool(imdb_info.imdb_url) or (
        imdb_info.director_name or imdb_info.original_author or imdb_info.original_book_title
    )
    evaluator.add_custom_node(
        result=bool(detail_page_accessed),
        id="imdb_movie_card_click",
        desc="[Action Node] imdb.com:F3:A26 - Click movie card to enter detail page",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P16 - Extract director, author, and book title from Writers section
    director_ok = bool(imdb_info.director_name and imdb_info.director_name.strip())
    author_ok = bool(imdb_info.original_author and imdb_info.original_author.strip())
    book_ok = bool(imdb_info.original_book_title and imdb_info.original_book_title.strip())
    writers_extraction_ok = director_ok and author_ok and book_ok

    evaluator.add_custom_node(
        result=bool(writers_extraction_ok),
        id="imdb_writers_extraction",
        desc="[Perception Node] imdb.com:F3:P16 - Extract director name, original author, and book title from Writers section",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads book information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A5 - Click book card to enter details
    goodreads_accessed = has_any_ci(answer, ['goodreads']) and (
        bool(goodreads_info.goodreads_url) or bool(goodreads_info.rating)
    )
    evaluator.add_custom_node(
        result=bool(goodreads_accessed),
        id="goodreads_book_card_click",
        desc="[Action Node] goodreads.com:F1:A5 - Search for and click book card to enter detail page",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P5 - Extract page count from book details
    page_count_ok = bool(goodreads_info.page_count) and looks_like_count(goodreads_info.page_count)
    evaluator.add_custom_node(
        result=bool(page_count_ok),
        id="goodreads_page_count_extraction",
        desc="[Perception Node] goodreads.com:F2:P5 - Extract page count from Book Details section",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P6 - Hover to get precise rating and review count
    rating_ok = bool(goodreads_info.rating) and looks_like_rating(goodreads_info.rating)
    review_count_ok = bool(goodreads_info.review_count) and looks_like_count(goodreads_info.review_count)
    hover_info_ok = rating_ok and review_count_ok

    evaluator.add_custom_node(
        result=bool(hover_info_ok),
        id="goodreads_rating_hover",
        desc="[Perception Node] goodreads.com:F2:P6 - Hover over rating area to extract precise rating and review count",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F2:A2 - Switch tabs to view series information
    series_info_accessed = (
        len(goodreads_info.series_titles) > 0
    ) or has_any_ci(answer, ['series', 'dune messiah', 'children of dune'])
    evaluator.add_custom_node(
        result=bool(series_info_accessed),
        id="goodreads_series_tab_switch",
        desc="[Action Node] goodreads.com:F2:A2 - Switch to Lists or Book Details tab to view series information",
        parent=goodreads_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube behind-the-scenes videos extraction",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A70 - Switch to Videos tab
    youtube_accessed = has_any_ci(answer, ['youtube']) and (
        len(youtube_info.top_videos) > 0 or has_any_ci(answer, ['behind the scenes', 'making of'])
    )
    evaluator.add_custom_node(
        result=bool(youtube_accessed),
        id="youtube_videos_tab_switch",
        desc="[Action Node] youtube.com:F1:A70 - Search on YouTube and switch to Videos tab to filter results",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more results
    has_multiple_videos = len(youtube_info.top_videos) >= 2
    evaluator.add_custom_node(
        result=bool(has_multiple_videos),
        id="youtube_scroll_load_more",
        desc="[Action Node] youtube.com:F1:A69 - Scroll to load more search results to find top videos by view count",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify behind-the-scenes content relevance
    bts_titles_ok = False
    if len(youtube_info.top_videos) > 0:
        bts_count = sum(1 for v in youtube_info.top_videos if looks_like_behind_the_scenes_title(v.title))
        bts_titles_ok = bts_count >= 2

    evaluator.add_custom_node(
        result=bool(bts_titles_ok),
        id="youtube_bts_content_identification",
        desc="[Perception Node] youtube.com:F1:P4 - Identify videos as behind-the-scenes/making-of content from titles and thumbnails",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P30 - Compare thumbnails to find most relevant videos
    top_3_found = len(youtube_info.top_videos) >= 3
    all_have_urls = all(bool(v.url) for v in youtube_info.top_videos[:3]) if top_3_found else False

    evaluator.add_custom_node(
        result=bool(top_3_found and all_have_urls),
        id="youtube_thumbnail_comparison",
        desc="[Perception Node] youtube.com:F1:P30 - Compare multiple video thumbnails and metadata to identify top 3 behind-the-scenes videos",
        parent=youtube_node,
        critical=False
    )

    # Check if videos are sorted by view count (descending)
    view_counts_sorted = False
    if len(youtube_info.top_videos) >= 2:
        view_counts = [extract_view_count_number(v.view_count) for v in youtube_info.top_videos[:3]]
        if all(vc is not None for vc in view_counts):
            view_counts_sorted = all(view_counts[i] >= view_counts[i+1] for i in range(len(view_counts)-1))

    evaluator.add_custom_node(
        result=bool(view_counts_sorted),
        id="youtube_view_count_sorting",
        desc="Videos are sorted by view count in descending order",
        parent=youtube_node,
        critical=False
    )

    # Check if videos are over 15 minutes
    duration_filter_ok = False
    if len(youtube_info.top_videos) > 0:
        long_videos = sum(1 for v in youtube_info.top_videos if looks_like_duration_over_15min(v.duration))
        duration_filter_ok = long_videos >= 2

    evaluator.add_custom_node(
        result=bool(duration_filter_ok),
        id="youtube_duration_filter",
        desc="Videos are filtered to be longer than 15 minutes",
        parent=youtube_node,
        critical=False
    )

    # 3.4 Criterion Collection section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion Collection director works extraction",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A6 - Select director from Directors filter
    criterion_accessed = has_any_ci(answer, ['criterion']) and (
        len(criterion_info.works) > 0 or has_any_ci(answer, ['director'])
    )
    evaluator.add_custom_node(
        result=bool(criterion_accessed),
        id="criterion_director_filter_selection",
        desc="[Action Node] criterion.com:F1:A6 - Navigate to Criterion and select director from Directors filter",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A5 - Expand Directors filter panel
    director_search_ok = criterion_accessed and (
        len(criterion_info.works) > 0 or has_any_ci(answer, ['villeneuve', imdb_info.director_name if imdb_info.director_name else ''])
    )
    evaluator.add_custom_node(
        result=bool(director_search_ok),
        id="criterion_filter_panel_expansion",
        desc="[Action Node] criterion.com:F1:A5 - Expand Directors filter panel to view full director list",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A9 - Click movie card to enter details
    detail_pages_accessed = len(criterion_info.works) > 0 and any(bool(w.detail_url) for w in criterion_info.works)
    evaluator.add_custom_node(
        result=bool(detail_pages_accessed),
        id="criterion_movie_card_click",
        desc="[Action Node] criterion.com:F1:A9 - Click movie cards to enter detail pages for each work",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F1:P9 - Identify movie card basic information
    works_found = len(criterion_info.works) > 0
    titles_extracted = all(bool(w.title and w.title.strip()) for w in criterion_info.works) if works_found else False

    evaluator.add_custom_node(
        result=bool(works_found and titles_extracted),
        id="criterion_movie_card_identification",
        desc="[Perception Node] criterion.com:F1:P9 - Identify and extract work titles from movie cards",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A13 - Switch format tabs to check availability
    formats_extracted = False
    if works_found:
        formats_count = sum(1 for w in criterion_info.works if len(w.formats) > 0)
        formats_extracted = formats_count > 0

    evaluator.add_custom_node(
        result=bool(formats_extracted),
        id="criterion_format_tab_switch",
        desc="[Action Node] criterion.com:F2:A13 - Switch format tabs (DVD/Blu-ray/4K) to check availability",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A12 - Expand Special Features panel
    special_features_extracted = False
    if works_found:
        features_count = sum(1 for w in criterion_info.works if bool(w.special_features_desc and w.special_features_desc.strip()))
        special_features_extracted = features_count > 0

    evaluator.add_custom_node(
        result=bool(special_features_extracted),
        id="criterion_special_features_expansion",
        desc="[Action Node] criterion.com:F2:A12 - Expand Special Features panel to view complete list",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P3 - Understand special features content
    features_content_ok = False
    if special_features_extracted:
        # Check if descriptions mention typical special features
        features_keywords = ['commentary', 'interview', 'documentary', 'essay', 'trailer', 'featurette']
        features_with_keywords = sum(
            1 for w in criterion_info.works
            if w.special_features_desc and has_any_ci(w.special_features_desc, features_keywords)
        )
        features_content_ok = features_with_keywords > 0

    evaluator.add_custom_node(
        result=bool(features_content_ok),
        id="criterion_special_features_understanding",
        desc="[Perception Node] criterion.com:F2:P3 - Understand and extract special features content types",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P5 - Identify format availability and pricing
    available_formats_ok = False
    if formats_extracted:
        # Check if formats listed are actual formats (not "out of stock" etc.)
        valid_format_keywords = ['dvd', 'blu-ray', 'blu ray', '4k', 'uhd']
        works_with_valid_formats = sum(
            1 for w in criterion_info.works
            if any(has_any_ci(fmt, valid_format_keywords) for fmt in w.formats)
        )
        available_formats_ok = works_with_valid_formats > 0

    evaluator.add_custom_node(
        result=bool(available_formats_ok),
        id="criterion_format_availability_identification",
        desc="[Perception Node] criterion.com:F2:P5 - Identify which formats are available (not out of stock)",
        parent=criterion_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
