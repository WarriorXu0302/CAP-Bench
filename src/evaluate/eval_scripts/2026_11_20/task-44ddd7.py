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
TASK_ID = "task-44ddd7"
TASK_DESCRIPTION = "I recently discovered an electronic musician named 'Ross from Friends' on SoundCloud, and I find his style quite unique. I'd like to learn more about him.\n\nPlease start by navigating to his official SoundCloud page. Confirm if he is from the United Kingdom (UK) and identify the name of his track with the highest number of plays.\n\nThen, search for this artist on the Discogs database. After confirming it's the same individual (by comparing tracks or musical style), please find his earliest physical release (e.g., Vinyl or Cassette). Specify the year of release and the title of this record.\n\nFinally, check Wikipedia for an article about him. If an article exists, please provide his real name.\n\nPlease compile and present these three pieces of information: the SoundCloud track with the highest play count, the year and title of his earliest physical release from Discogs, and his real name (if available)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SoundCloudInfo(BaseModel):
    """Information extracted from SoundCloud about Ross from Friends"""
    is_from_uk: Optional[bool] = None
    uk_confirmation_text: Optional[str] = None
    highest_play_track: Optional[str] = None


class DiscogsInfo(BaseModel):
    """Information extracted from Discogs about Ross from Friends"""
    earliest_physical_release_year: Optional[str] = None
    earliest_physical_release_title: Optional[str] = None
    format_type: Optional[str] = None


class WikipediaInfo(BaseModel):
    """Information extracted from Wikipedia about Ross from Friends"""
    real_name: Optional[str] = None
    wikipedia_exists: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_soundcloud_info() -> str:
    return """
Extract the following information about 'Ross from Friends' from the answer based on SoundCloud:

- is_from_uk: boolean indicating whether the answer confirms he is from the UK/United Kingdom. Set to true if confirmed, false if explicitly denied, null if not mentioned.
- uk_confirmation_text: the exact text or location information that confirms UK origin (e.g., "United Kingdom", "UK", "London", etc.). Set to null if not present.
- highest_play_track: the name of the track with the highest number of plays as stated in the answer. Set to null if not present.

If any field is missing, set it to null or appropriate default.
"""


def prompt_extract_discogs_info() -> str:
    return """
Extract the following information about 'Ross from Friends' from the answer based on Discogs:

- earliest_physical_release_year: the year of the earliest physical release (e.g., "2015", "2016"). Set to null if not present.
- earliest_physical_release_title: the title of the earliest physical release. Set to null if not present.
- format_type: the format of the release (e.g., "Vinyl", "Cassette", "CD"). Set to null if not present.

If any field is missing, set it to null.
"""


def prompt_extract_wikipedia_info() -> str:
    return """
Extract the following information about 'Ross from Friends' from the answer based on Wikipedia:

- real_name: the real name of the artist as stated in the answer. Set to null if not present.
- wikipedia_exists: boolean indicating whether a Wikipedia article was found. Set to true if found, false if not found, null if not mentioned.

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


def looks_like_year(text: Optional[str]) -> bool:
    if not text:
        return False
    # Match 4-digit year (1900-2099)
    return bool(re.search(r'\b(19|20)\d{2}\b', text))


def looks_like_uk_location(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['uk', 'united kingdom', 'britain', 'england', 'london', 'manchester', 'scotland', 'wales'])


def looks_like_physical_format(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['vinyl', 'cassette', 'cd', 'tape', 'lp', 'ep'])


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

    discogs_info = await evaluator.extract(
        prompt=prompt_extract_discogs_info(),
        template_class=DiscogsInfo,
        extraction_name="discogs_info"
    )

    wikipedia_info = await evaluator.extract(
        prompt=prompt_extract_wikipedia_info(),
        template_class=WikipediaInfo,
        extraction_name="wikipedia_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 SoundCloud section
    soundcloud_node = evaluator.add_sequential(
        id="soundcloud_section",
        desc="SoundCloud investigation for Ross from Friends",
        parent=root,
        critical=False
    )

    # [Action Node] soundcloud.com:F6:A13 - Tab switching to find highest play track
    # The task asks to "identify the name of his track with the highest number of plays"
    # This implicitly requires switching to Tracks or Popular Tracks tab
    soundcloud_navigation_ok = (
        has_any_ci(answer, ['soundcloud']) and
        has_any_ci(answer, ['ross from friends'])
    )
    evaluator.add_custom_node(
        result=bool(soundcloud_navigation_ok),
        id="soundcloud_navigate",
        desc="Navigate to Ross from Friends' official SoundCloud page",
        parent=soundcloud_node,
        critical=False
    )

    # Check if answer mentions tracks or popular tracks context (indicating tab switching)
    tab_switching_context = has_any_ci(answer, ['tracks', 'popular', 'all tracks', 'top track'])
    evaluator.add_custom_node(
        result=bool(tab_switching_context),
        id="soundcloud_tab_switching",
        desc="[Action Node] soundcloud.com:F6:A13 - Switch to Tracks or Popular Tracks tab to identify highest play track",
        parent=soundcloud_node,
        critical=False
    )

    # [Perception Node] soundcloud.com:F6:P15 - Confirm UK origin
    uk_confirmed = (
        soundcloud_info.is_from_uk == True or
        looks_like_uk_location(soundcloud_info.uk_confirmation_text) or
        has_any_ci(answer, ['from the uk', 'from united kingdom', 'based in uk', 'british', 'from britain'])
    )
    evaluator.add_custom_node(
        result=bool(uk_confirmed),
        id="soundcloud_uk_confirmation",
        desc="[Perception Node] soundcloud.com:F6:P15 - Confirm artist is from the UK by reading location/bio information",
        parent=soundcloud_node,
        critical=False
    )

    # Identify highest play track
    highest_track_identified = bool(
        soundcloud_info.highest_play_track and
        soundcloud_info.highest_play_track.strip()
    )
    play_count_context = has_any_ci(answer, ['plays', 'play count', 'most played', 'highest'])
    evaluator.add_custom_node(
        result=bool(highest_track_identified and play_count_context),
        id="soundcloud_highest_track",
        desc="Identify the track with the highest number of plays",
        parent=soundcloud_node,
        critical=False
    )

    # 3.2 Discogs section
    discogs_node = evaluator.add_sequential(
        id="discogs_section",
        desc="Discogs database investigation for Ross from Friends",
        parent=root,
        critical=False
    )

    # Navigate to Discogs and search
    discogs_navigation_ok = (
        has_any_ci(answer, ['discogs']) and
        has_any_ci(answer, ['ross from friends'])
    )
    evaluator.add_custom_node(
        result=bool(discogs_navigation_ok),
        id="discogs_navigate",
        desc="Navigate to Discogs and search for Ross from Friends",
        parent=discogs_node,
        critical=False
    )

    # Verify identity (mentioned comparing tracks or style)
    identity_verification = has_any_ci(answer, ['confirm', 'same', 'verified', 'comparing', 'match'])
    evaluator.add_custom_node(
        result=bool(identity_verification),
        id="discogs_identity_verification",
        desc="Verify identity by comparing tracks or musical style",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F1:A1 - Multi-select filtering for physical formats
    # The task asks for "earliest physical release (e.g., Vinyl or Cassette)"
    # This requires filtering Format to include Vinyl, CD, Cassette and exclude File/Digital
    format_filtering_context = (
        has_any_ci(answer, ['format', 'vinyl', 'cassette', 'physical']) or
        has_any_ci(answer, ['filter', 'exclude digital', 'exclude file'])
    )
    evaluator.add_custom_node(
        result=bool(format_filtering_context),
        id="discogs_format_filtering",
        desc="[Action Node] discogs.com:F1:A1 - Apply format filter to show only physical releases (Vinyl, CD, Cassette)",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F1:A6 - Sort by year to find earliest
    # The task asks for "earliest physical release"
    # This requires sorting by Year, Oldest First
    sorting_context = has_any_ci(answer, ['sort', 'earliest', 'oldest', 'first release', 'chronological'])
    evaluator.add_custom_node(
        result=bool(sorting_context),
        id="discogs_year_sorting",
        desc="[Action Node] discogs.com:F1:A6 - Sort releases by year (oldest first) to find earliest physical release",
        parent=discogs_node,
        critical=False
    )

    # Extract earliest physical release details
    year_ok = looks_like_year(discogs_info.earliest_physical_release_year)
    title_ok = bool(discogs_info.earliest_physical_release_title and discogs_info.earliest_physical_release_title.strip())
    format_ok = looks_like_physical_format(discogs_info.format_type)

    evaluator.add_custom_node(
        result=bool(year_ok and title_ok),
        id="discogs_earliest_release_info",
        desc="Extract year and title of earliest physical release",
        parent=discogs_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(format_ok),
        id="discogs_format_type",
        desc="Confirm the format is physical (Vinyl, Cassette, CD)",
        parent=discogs_node,
        critical=False
    )

    # 3.3 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia investigation for Ross from Friends",
        parent=root,
        critical=False
    )

    # Navigate to Wikipedia
    wikipedia_navigation_ok = (
        has_any_ci(answer, ['wikipedia']) and
        has_any_ci(answer, ['ross from friends'])
    )
    evaluator.add_custom_node(
        result=bool(wikipedia_navigation_ok),
        id="wikipedia_navigate",
        desc="Search Wikipedia for Ross from Friends article",
        parent=wikipedia_node,
        critical=False
    )

    # Check if article exists
    article_exists = (
        wikipedia_info.wikipedia_exists == True or
        has_any_ci(answer, ['article exists', 'wikipedia article', 'found on wikipedia'])
    )
    evaluator.add_custom_node(
        result=bool(article_exists),
        id="wikipedia_article_exists",
        desc="Confirm Wikipedia article exists for the artist",
        parent=wikipedia_node,
        critical=False
    )

    # Extract real name
    real_name_extracted = bool(
        wikipedia_info.real_name and
        wikipedia_info.real_name.strip() and
        not ci_contains(wikipedia_info.real_name, 'not found') and
        not ci_contains(wikipedia_info.real_name, 'unavailable')
    )
    evaluator.add_custom_node(
        result=bool(real_name_extracted),
        id="wikipedia_real_name",
        desc="Extract the artist's real name from Wikipedia",
        parent=wikipedia_node,
        critical=False
    )

    # 3.4 Final compilation check
    compilation_node = evaluator.add_parallel(
        id="final_compilation",
        desc="Final compilation of all three pieces of information",
        parent=root,
        critical=False
    )

    # Check if all three pieces are present in answer
    has_soundcloud_info = bool(soundcloud_info.highest_play_track)
    has_discogs_info = bool(discogs_info.earliest_physical_release_year and discogs_info.earliest_physical_release_title)
    has_wikipedia_info = bool(wikipedia_info.real_name)

    evaluator.add_custom_node(
        result=bool(has_soundcloud_info),
        id="compiled_soundcloud_track",
        desc="Compiled information includes SoundCloud track with highest plays",
        parent=compilation_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_discogs_info),
        id="compiled_discogs_release",
        desc="Compiled information includes year and title of earliest Discogs physical release",
        parent=compilation_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_wikipedia_info),
        id="compiled_wikipedia_name",
        desc="Compiled information includes real name from Wikipedia (if available)",
        parent=compilation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
