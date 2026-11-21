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
TASK_ID = "task-a3fee4"
TASK_DESCRIPTION = "I am an independent musician who recently uploaded several electronic music tracks on SoundCloud, and I'd like to explore their commercial potential. Could you first help me search on SoundCloud for electronic music of a similar style to identify tracks with high play counts and understand what level of engagement is considered popular? Then, on Spotify, find a few independent artists who produce similar music to see their monthly listener numbers and get a sense of the potential market size. Next, help me search on Etsy for music-related merchandise being sold, such as vinyl records, limited edition posters, etc. Examine their price ranges and sales figures to assess the feasibility of creating physical merchandise. Finally, on Coursera, look for courses related to music marketing or copyright, specifically those with high ratings and strong practical applicability. I want to systematically learn how to promote and protect my work."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SoundCloudInfo(BaseModel):
    """Information extracted about SoundCloud electronic music search"""
    search_performed: Optional[bool] = None
    electronic_genre_mentioned: Optional[bool] = None
    play_counts_mentioned: Optional[bool] = None
    popularity_threshold_discussed: Optional[bool] = None
    track_examples: Optional[List[str]] = Field(default_factory=list)


class SpotifyInfo(BaseModel):
    """Information extracted about Spotify artist search"""
    search_performed: Optional[bool] = None
    independent_artists_mentioned: Optional[bool] = None
    monthly_listeners_mentioned: Optional[bool] = None
    artist_examples: Optional[List[str]] = Field(default_factory=list)


class EtsyInfo(BaseModel):
    """Information extracted about Etsy merchandise search"""
    search_performed: Optional[bool] = None
    merchandise_types_mentioned: Optional[List[str]] = Field(default_factory=list)
    price_ranges_mentioned: Optional[bool] = None
    sales_figures_mentioned: Optional[bool] = None


class CourseraInfo(BaseModel):
    """Information extracted about Coursera course search"""
    search_performed: Optional[bool] = None
    music_marketing_courses: Optional[bool] = None
    copyright_courses: Optional[bool] = None
    ratings_mentioned: Optional[bool] = None
    practical_applicability_mentioned: Optional[bool] = None
    course_examples: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_soundcloud_info() -> str:
    return """
Extract information about the SoundCloud search from the answer:

- search_performed: true if the answer indicates a SoundCloud search was conducted
- electronic_genre_mentioned: true if electronic music genre is mentioned
- play_counts_mentioned: true if play counts or play numbers are discussed
- popularity_threshold_discussed: true if there's discussion about what level of plays is considered popular
- track_examples: list of any specific track names or artists mentioned from SoundCloud

Set fields to null/empty if not present.
"""


def prompt_extract_spotify_info() -> str:
    return """
Extract information about the Spotify artist search from the answer:

- search_performed: true if the answer indicates a Spotify search was conducted
- independent_artists_mentioned: true if independent artists are mentioned
- monthly_listeners_mentioned: true if monthly listener numbers are discussed
- artist_examples: list of any specific artist names mentioned from Spotify

Set fields to null/empty if not present.
"""


def prompt_extract_etsy_info() -> str:
    return """
Extract information about the Etsy merchandise search from the answer:

- search_performed: true if the answer indicates an Etsy search was conducted
- merchandise_types_mentioned: list of merchandise types mentioned (vinyl, posters, etc.)
- price_ranges_mentioned: true if price ranges or pricing is discussed
- sales_figures_mentioned: true if sales numbers or sales data is mentioned

Set fields to null/empty if not present.
"""


def prompt_extract_coursera_info() -> str:
    return """
Extract information about the Coursera course search from the answer:

- search_performed: true if the answer indicates a Coursera search was conducted
- music_marketing_courses: true if music marketing courses are mentioned
- copyright_courses: true if copyright-related courses are mentioned
- ratings_mentioned: true if course ratings are discussed
- practical_applicability_mentioned: true if practical applicability is discussed
- course_examples: list of any specific course names mentioned

Set fields to null/empty if not present.
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
    soundcloud_info = await evaluator.extract(
        prompt=prompt_extract_soundcloud_info(),
        template_class=SoundCloudInfo,
        extraction_name="soundcloud_info"
    )

    spotify_info = await evaluator.extract(
        prompt=prompt_extract_spotify_info(),
        template_class=SpotifyInfo,
        extraction_name="spotify_info"
    )

    etsy_info = await evaluator.extract(
        prompt=prompt_extract_etsy_info(),
        template_class=EtsyInfo,
        extraction_name="etsy_info"
    )

    coursera_info = await evaluator.extract(
        prompt=prompt_extract_coursera_info(),
        template_class=CourseraInfo,
        extraction_name="coursera_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 SoundCloud section
    soundcloud_node = evaluator.add_sequential(
        id="soundcloud_section",
        desc="SoundCloud electronic music search and popularity analysis",
        parent=root,
        critical=False
    )

    # [Action Node] soundcloud.com:F1:A9 - Filter by content type (Tracks)
    soundcloud_tracks_filter = (
        has_any_ci(answer, ['soundcloud']) and
        has_any_ci(answer, ['track', 'tracks', 'electronic music'])
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_tracks_filter),
        id="soundcloud_tracks_filter",
        desc="[Action Node] soundcloud.com:F1:A9 - Filter content type to Tracks when searching for electronic music",
        parent=soundcloud_node,
        critical=False
    )

    # [Action Node] soundcloud.com:F1:A10 - Time filter for recent trends
    soundcloud_time_filter = (
        has_any_ci(answer, ['recent', 'recently', 'latest', 'new', 'current', 'trend'])
        if has_any_ci(answer, ['soundcloud']) else False
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_time_filter),
        id="soundcloud_time_filter",
        desc="[Action Node] soundcloud.com:F1:A10 - Apply time filter to find recent popular tracks",
        parent=soundcloud_node,
        critical=False
    )

    # [Action Node] soundcloud.com:F1:A28 - Scroll to load more tracks
    soundcloud_scroll = (
        (soundcloud_info and soundcloud_info.track_examples and len(soundcloud_info.track_examples) > 2) or
        has_any_ci(answer, ['multiple', 'several', 'various', 'many tracks', 'browse'])
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_scroll),
        id="soundcloud_scroll",
        desc="[Action Node] soundcloud.com:F1:A28 - Scroll through multiple tracks to compare play counts",
        parent=soundcloud_node,
        critical=False
    )

    # [Perception Node] soundcloud.com:F1:P7 - Understand play count data
    soundcloud_play_understanding = (
        soundcloud_info and soundcloud_info.play_counts_mentioned and
        (soundcloud_info.popularity_threshold_discussed or contains_digits(answer))
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_play_understanding),
        id="soundcloud_play_understanding",
        desc="[Perception Node] soundcloud.com:F1:P7 - Extract and understand play count data from track listings",
        parent=soundcloud_node,
        critical=False
    )

    # [Action Node] soundcloud.com:F2:A11 - Select Electronic genre
    soundcloud_genre_select = (
        has_any_ci(answer, ['soundcloud']) and
        has_any_ci(answer, ['electronic', 'genre'])
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_genre_select),
        id="soundcloud_genre_select",
        desc="[Action Node] soundcloud.com:F2:A11 - Select Electronic genre using genre filter",
        parent=soundcloud_node,
        critical=False
    )

    # [Perception Node] soundcloud.com:F2:P1 - Recognize track cover art style
    soundcloud_cover_perception = (
        has_any_ci(answer, ['cover', 'artwork', 'image', 'visual', 'style', 'similar style'])
        if has_any_ci(answer, ['soundcloud']) else False
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_cover_perception),
        id="soundcloud_cover_perception",
        desc="[Perception Node] soundcloud.com:F2:P1 - Recognize cover art to identify style similarity",
        parent=soundcloud_node,
        critical=False
    )

    # 3.2 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify independent artist search and audience analysis",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A1 - Navigate to Spotify search
    spotify_navigation = (
        has_any_ci(answer, ['spotify']) and
        has_any_ci(answer, ['search', 'find', 'look for'])
    )
    evaluator.add_custom_node(
        result=bool(spotify_navigation),
        id="spotify_navigation",
        desc="[Action Node] open.spotify.com:F2:A1 - Navigate to Spotify search page",
        parent=spotify_node,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A10 - Click artist cards
    spotify_artist_click = (
        spotify_info and spotify_info.independent_artists_mentioned and
        (spotify_info.artist_examples and len(spotify_info.artist_examples) > 0 or
         has_any_ci(answer, ['artist', 'profile']))
    )
    evaluator.add_custom_node(
        result=bool(spotify_artist_click),
        id="spotify_artist_click",
        desc="[Action Node] open.spotify.com:F2:A10 - Click on artist cards to view details",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F2:P1 - Recognize cover art for style matching
    spotify_cover_perception = (
        has_any_ci(answer, ['cover', 'artwork', 'image', 'similar', 'style'])
        if has_any_ci(answer, ['spotify']) else False
    )
    evaluator.add_custom_node(
        result=bool(spotify_cover_perception),
        id="spotify_cover_perception",
        desc="[Perception Node] open.spotify.com:F2:P1 - Recognize artist/album covers to identify style similarity",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F2:P23 - Extract monthly listener data
    spotify_listeners_data = (
        spotify_info and spotify_info.monthly_listeners_mentioned and
        contains_digits(answer)
    )
    evaluator.add_custom_node(
        result=bool(spotify_listeners_data),
        id="spotify_listeners_data",
        desc="[Perception Node] open.spotify.com:F2:P23 - Extract monthly listener numbers from artist cards",
        parent=spotify_node,
        critical=False
    )

    # 3.3 Etsy section
    etsy_node = evaluator.add_sequential(
        id="etsy_section",
        desc="Etsy music merchandise market research",
        parent=root,
        critical=False
    )

    # [Action Node] Etsy:F1:A17 - Click product cards
    etsy_product_click = (
        has_any_ci(answer, ['etsy']) and
        has_any_ci(answer, ['price', 'sales', 'detail', 'click', 'view'])
    )
    evaluator.add_custom_node(
        result=bool(etsy_product_click),
        id="etsy_product_click",
        desc="[Action Node] Etsy:F1:A17 - Click on product cards to view pricing and sales details",
        parent=etsy_node,
        critical=False
    )

    # [Action Node] Etsy:F1:A26 - Scroll to browse multiple products
    etsy_scroll = (
        has_any_ci(answer, ['etsy']) and
        (has_any_ci(answer, ['multiple', 'several', 'various', 'range', 'browse']) or
         (etsy_info and etsy_info.merchandise_types_mentioned and len(etsy_info.merchandise_types_mentioned) > 1))
    )
    evaluator.add_custom_node(
        result=bool(etsy_scroll),
        id="etsy_scroll",
        desc="[Action Node] Etsy:F1:A26 - Scroll through product listings to compare prices and sales",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] Etsy:F1:P1 - Recognize product images
    etsy_image_recognition = (
        has_any_ci(answer, ['vinyl', 'poster', 'merchandise', 'product'])
        if has_any_ci(answer, ['etsy']) else False
    )
    evaluator.add_custom_node(
        result=bool(etsy_image_recognition),
        id="etsy_image_recognition",
        desc="[Perception Node] Etsy:F1:P1 - Identify product types from product images",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] Etsy:F1:P2 - Extract visual features
    etsy_visual_features = (
        has_any_ci(answer, ['music', 'music-related', 'merchandise'])
        if has_any_ci(answer, ['etsy']) else False
    )
    evaluator.add_custom_node(
        result=bool(etsy_visual_features),
        id="etsy_visual_features",
        desc="[Perception Node] Etsy:F1:P2 - Extract visual features to identify music-related merchandise",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] Etsy:F1:P3 - Recognize status markers (sales, ratings)
    etsy_status_markers = (
        etsy_info and etsy_info.sales_figures_mentioned and
        (has_any_ci(answer, ['rating', 'review', 'sold']) or contains_digits(answer))
    )
    evaluator.add_custom_node(
        result=bool(etsy_status_markers),
        id="etsy_status_markers",
        desc="[Perception Node] Etsy:F1:P3 - Extract sales figures and ratings from product cards",
        parent=etsy_node,
        critical=False
    )

    # [Action Node] Etsy:F3:A4 - Sort by sales or relevance
    etsy_sort = (
        has_any_ci(answer, ['etsy']) and
        has_any_ci(answer, ['popular', 'best selling', 'top', 'sort'])
    )
    evaluator.add_custom_node(
        result=bool(etsy_sort),
        id="etsy_sort",
        desc="[Action Node] Etsy:F3:A4 - Apply sorting to find popular merchandise",
        parent=etsy_node,
        critical=False
    )

    # 3.4 Coursera section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera music marketing and copyright course search",
        parent=root,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Sort by rating
    coursera_sort = (
        has_any_ci(answer, ['coursera']) and
        has_any_ci(answer, ['rating', 'high rated', 'top rated', 'sort'])
    )
    evaluator.add_custom_node(
        result=bool(coursera_sort),
        id="coursera_sort",
        desc="[Action Node] coursera.org:F1:A12 - Sort courses by rating to find highly-rated courses",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A18 - Click course cards
    coursera_click = (
        has_any_ci(answer, ['coursera']) and
        (coursera_info and coursera_info.course_examples and len(coursera_info.course_examples) > 0 or
         has_any_ci(answer, ['course', 'click', 'view', 'detail']))
    )
    evaluator.add_custom_node(
        result=bool(coursera_click),
        id="coursera_click",
        desc="[Action Node] coursera.org:F1:A18 - Click on course cards to view details",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F1:P1 - Recognize rating badges
    coursera_rating_perception = (
        coursera_info and coursera_info.ratings_mentioned and
        contains_digits(answer)
    )
    evaluator.add_custom_node(
        result=bool(coursera_rating_perception),
        id="coursera_rating_perception",
        desc="[Perception Node] coursera.org:F1:P1 - Extract rating information from course cards",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F3:A4 - Switch tabs to view syllabus
    coursera_tab_switch = (
        has_any_ci(answer, ['coursera']) and
        has_any_ci(answer, ['syllabus', 'curriculum', 'content', 'practical', 'applicability'])
    )
    evaluator.add_custom_node(
        result=bool(coursera_tab_switch),
        id="coursera_tab_switch",
        desc="[Action Node] coursera.org:F3:A4 - Switch to syllabus tab to assess practical applicability",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F3:P7 - Understand syllabus content
    coursera_syllabus_understanding = (
        coursera_info and coursera_info.practical_applicability_mentioned and
        has_any_ci(answer, ['content', 'module', 'lesson', 'practical'])
    )
    evaluator.add_custom_node(
        result=bool(coursera_syllabus_understanding),
        id="coursera_syllabus_understanding",
        desc="[Perception Node] coursera.org:F3:P7 - Extract and understand course syllabus for practical applicability",
        parent=coursera_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
