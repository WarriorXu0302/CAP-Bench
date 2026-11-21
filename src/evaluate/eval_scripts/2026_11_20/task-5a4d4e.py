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
TASK_ID = "task-5a4d4e"
TASK_DESCRIPTION = "I recently want to arrange music in the style of Bill Evans, and I need to find some specific references and advanced tutorials.\n\nFirst, please go to Spotify and search for the official playlist 'This Is Bill Evans'. Tell me the current number of 'Saves' for this playlist and the title of the first track on the list. I need to confirm the style.\n\nAfter confirmation, please go to Udemy to find courses for me. Search for 'Jazz Piano Improvisation'. To ensure quality, please filter the courses in the sidebar by a rating of 4.5 or higher, and select 'Intermediate' or 'Expert' for the difficulty level. I need to confirm whether the course outline teaches the specific technique of 'Rootless Voicings'.\n\nPlease find a course with either a 'Bestseller' or 'Highest Rated' label. Expand its 'Course content' section and carefully examine the titles of the lectures/lessons. Once you find a course that includes this technique, send me the course link and excerpt the name of the chapter or lesson where this technique is mentioned."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SpotifyPlaylistInfo(BaseModel):
    """Spotify playlist information extracted from the answer"""
    playlist_name: Optional[str] = None
    saves_count: Optional[str] = None
    first_track_title: Optional[str] = None


class UdemyCourseInfo(BaseModel):
    """Udemy course information extracted from the answer"""
    course_link: Optional[str] = None
    course_title: Optional[str] = None
    has_bestseller_or_highest_rated: Optional[bool] = None
    rootless_voicings_lesson_name: Optional[str] = None


class UdemyFilters(BaseModel):
    """Udemy filter application details extracted from the answer"""
    rating_filter_applied: Optional[bool] = None
    difficulty_filter_applied: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_spotify_info() -> str:
    return """
Extract the Spotify playlist information for 'This Is Bill Evans' from the answer.

Return:
- playlist_name: the name of the playlist as mentioned (should contain 'Bill Evans' or 'This Is Bill Evans')
- saves_count: the number of saves/likes for the playlist exactly as stated (include any formatting like commas)
- first_track_title: the title of the first track on the playlist exactly as stated

If any field is missing in the answer, set it to null.
"""


def prompt_extract_udemy_course() -> str:
    return """
Extract the Udemy course information from the answer, specifically the course that teaches 'Rootless Voicings'.

Return:
- course_link: the URL or link to the course
- course_title: the title of the course
- has_bestseller_or_highest_rated: true if the answer mentions the course has a 'Bestseller' or 'Highest Rated' label, false otherwise
- rootless_voicings_lesson_name: the exact name of the chapter or lesson that mentions 'Rootless Voicings'

If any field is missing, set it to null (or false for the boolean).
"""


def prompt_extract_udemy_filters() -> str:
    return """
Extract information about the Udemy filter operations from the answer.

Return:
- rating_filter_applied: true if the answer mentions filtering by rating 4.5 or higher, false otherwise
- difficulty_filter_applied: true if the answer mentions filtering by 'Intermediate' or 'Expert' difficulty level, false otherwise

Set to false if not mentioned.
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
    # Remove commas and extract first number
    cleaned = text.replace(',', '')
    m = re.search(r'(\d+(\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_saves_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def mentions_bill_evans(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['bill evans'])


def mentions_spotify(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['spotify'])


def mentions_udemy(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['udemy'])


def mentions_jazz_piano_improvisation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['jazz piano improvisation', 'jazz piano'])


def mentions_rootless_voicings(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['rootless voicings', 'rootless voicing'])


def mentions_course_content_expansion(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['course content', 'expand', 'section'])


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://') or 'udemy.com' in text.lower()


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
    spotify_info = await evaluator.extract(
        prompt=prompt_extract_spotify_info(),
        template_class=SpotifyPlaylistInfo,
        extraction_name="spotify_playlist_info"
    )

    udemy_course_info = await evaluator.extract(
        prompt=prompt_extract_udemy_course(),
        template_class=UdemyCourseInfo,
        extraction_name="udemy_course_info"
    )

    udemy_filters = await evaluator.extract(
        prompt=prompt_extract_udemy_filters(),
        template_class=UdemyFilters,
        extraction_name="udemy_filters"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Spotify part
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify - 'This Is Bill Evans' playlist information",
        parent=root,
        critical=False
    )

    # Check if Spotify was visited and playlist searched
    spotify_visited = mentions_spotify(answer) and mentions_bill_evans(answer)
    evaluator.add_custom_node(
        result=bool(spotify_visited),
        id="spotify_navigation",
        desc="Navigate to Spotify and search for 'This Is Bill Evans' playlist",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F6:P23 - Extract saves count
    saves_count_ok = looks_like_saves_count(spotify_info.saves_count)
    saves_mentioned = has_any_ci(answer, ['saves', 'save', 'likes', 'like'])
    evaluator.add_custom_node(
        result=bool(saves_count_ok and saves_mentioned),
        id="spotify_perception_saves",
        desc="[Perception Node] open.spotify.com:F6:P23 - Extract the current number of 'Saves' for the playlist",
        parent=spotify_node,
        critical=False
    )

    # Extract first track title
    first_track_ok = bool(spotify_info.first_track_title and spotify_info.first_track_title.strip())
    evaluator.add_custom_node(
        result=bool(first_track_ok),
        id="spotify_perception_first_track",
        desc="Extract the title of the first track on the playlist",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Udemy part
    udemy_node = evaluator.add_sequential(
        id="udemy_section",
        desc="Udemy - Jazz Piano Improvisation courses with filters and content inspection",
        parent=root,
        critical=False
    )

    # Check if Udemy was visited and search performed
    udemy_visited = mentions_udemy(answer) and mentions_jazz_piano_improvisation(answer)
    evaluator.add_custom_node(
        result=bool(udemy_visited),
        id="udemy_navigation_search",
        desc="Navigate to Udemy and search for 'Jazz Piano Improvisation'",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A1 - Filter by rating 4.5 or higher
    rating_filter_ok = bool(udemy_filters.rating_filter_applied) or has_any_ci(answer, ['4.5', 'rating'])
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),
        id="udemy_action_rating_filter",
        desc="[Action Node] udemy.com:F1:A1 - Filter courses by rating of 4.5 or higher",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A3 - Filter by difficulty level (Intermediate or Expert)
    difficulty_filter_ok = bool(udemy_filters.difficulty_filter_applied) or has_any_ci(answer, ['intermediate', 'expert'])
    evaluator.add_custom_node(
        result=bool(difficulty_filter_ok),
        id="udemy_action_difficulty_filter",
        desc="[Action Node] udemy.com:F1:A3 - Filter courses by difficulty level 'Intermediate' or 'Expert'",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P1 - Identify course with Bestseller or Highest Rated label
    label_ok = bool(udemy_course_info.has_bestseller_or_highest_rated) or has_any_ci(answer, ['bestseller', 'highest rated'])
    evaluator.add_custom_node(
        result=bool(label_ok),
        id="udemy_perception_label",
        desc="[Perception Node] udemy.com:F1:P1 - Identify a course with 'Bestseller' or 'Highest Rated' label",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F2:A10 - Expand 'Course content' section
    expansion_ok = mentions_course_content_expansion(answer)
    evaluator.add_custom_node(
        result=bool(expansion_ok),
        id="udemy_action_expand_content",
        desc="[Action Node] udemy.com:F2:A10 - Expand the 'Course content' section to view lesson titles",
        parent=udemy_node,
        critical=False
    )

    # Verify Rootless Voicings technique is found in course content
    rootless_voicings_found = mentions_rootless_voicings(answer) and bool(udemy_course_info.rootless_voicings_lesson_name and udemy_course_info.rootless_voicings_lesson_name.strip())
    evaluator.add_custom_node(
        result=bool(rootless_voicings_found),
        id="udemy_perception_rootless_voicings",
        desc="Find and identify a lesson/chapter that teaches 'Rootless Voicings' technique",
        parent=udemy_node,
        critical=False
    )

    # Verify course link is provided
    course_link_ok = is_valid_url(udemy_course_info.course_link)
    evaluator.add_custom_node(
        result=bool(course_link_ok),
        id="udemy_course_link_provided",
        desc="Provide the course link/URL",
        parent=udemy_node,
        critical=False
    )

    # Verify lesson name is excerpted
    lesson_name_excerpted = bool(udemy_course_info.rootless_voicings_lesson_name and udemy_course_info.rootless_voicings_lesson_name.strip())
    evaluator.add_custom_node(
        result=bool(lesson_name_excerpted),
        id="udemy_lesson_name_excerpted",
        desc="Excerpt the name of the chapter or lesson where 'Rootless Voicings' is mentioned",
        parent=udemy_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
