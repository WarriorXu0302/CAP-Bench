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
TASK_ID = "task-6fd2dc"
TASK_DESCRIPTION = 'I recently watched "Interstellar" and was deeply moved by Hans Zimmer\'s soundtrack, especially the organ piece. I\'d like to find the complete soundtrack album for the movie to listen to, and also explore other classic works by Hans Zimmer. Additionally, I recall that there are musicians on Bilibili (B站) who have specifically analyzed the soundtrack composition techniques for this film. Please help me find analytical videos with high view counts and in-depth discussions, preferably content that is over 10 minutes long and offers deep insights, as I want to systematically learn how film scores enhance emotional atmosphere.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InterstellarSoundtrackInfo(BaseModel):
    """Soundtrack information extracted from the answer"""
    soundtrack_album_mentioned: Optional[bool] = None
    spotify_mentioned: Optional[bool] = None
    album_details: Optional[str] = None


class HansZimmerWorksInfo(BaseModel):
    """Hans Zimmer's other works extracted from the answer"""
    other_works_mentioned: Optional[bool] = None
    works_list: Optional[str] = None
    imdb_mentioned: Optional[bool] = None


class BilibiliVideosInfo(BaseModel):
    """Bilibili video analysis information extracted from the answer"""
    bilibili_mentioned: Optional[bool] = None
    video_analysis_found: Optional[bool] = None
    video_titles: Optional[str] = None
    duration_mentioned: Optional[bool] = None
    view_count_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_soundtrack_info() -> str:
    return """
Extract information about the Interstellar soundtrack album from the answer.

Return:
- soundtrack_album_mentioned: true if the answer mentions finding or discussing the Interstellar soundtrack album
- spotify_mentioned: true if Spotify is mentioned as a platform to find the album
- album_details: any specific details about the album (track list, album name, etc.) exactly as written

If any field is missing, set it to null or false as appropriate.
"""


def prompt_extract_zimmer_works() -> str:
    return """
Extract information about Hans Zimmer's other classic works from the answer.

Return:
- other_works_mentioned: true if the answer mentions other works by Hans Zimmer beyond Interstellar
- works_list: any list or mention of other films/works by Hans Zimmer exactly as written
- imdb_mentioned: true if IMDb is mentioned as a source for Hans Zimmer's filmography

If any field is missing, set it to null or false as appropriate.
"""


def prompt_extract_bilibili_videos() -> str:
    return """
Extract information about Bilibili videos analyzing the Interstellar soundtrack from the answer.

Return:
- bilibili_mentioned: true if Bilibili (B站) is mentioned
- video_analysis_found: true if specific analysis videos are mentioned or found
- video_titles: any video titles or descriptions mentioned exactly as written
- duration_mentioned: true if video duration (especially 10+ minutes) is mentioned
- view_count_mentioned: true if view counts or popularity metrics are mentioned

If any field is missing, set it to null or false as appropriate.
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


def mentions_interstellar(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['interstellar', '星际穿越'])


def mentions_hans_zimmer(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['hans zimmer', 'zimmer'])


def mentions_soundtrack_album(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['soundtrack', 'album', 'ost', '原声', '专辑', '配乐'])


def mentions_bilibili(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['bilibili', 'b站', 'b站'])


def mentions_video_analysis(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['analysis', 'analytical', '解析', '分析', 'technique', '手法', '技巧'])


def mentions_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['10 minute', '10分钟', 'minute', 'duration', '时长', 'long'])


def mentions_view_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['view', 'views', '播放', '播放量', 'popular', 'high view'])


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
    soundtrack_info = await evaluator.extract(
        prompt=prompt_extract_soundtrack_info(),
        template_class=InterstellarSoundtrackInfo,
        extraction_name="interstellar_soundtrack_info"
    )

    zimmer_works_info = await evaluator.extract(
        prompt=prompt_extract_zimmer_works(),
        template_class=HansZimmerWorksInfo,
        extraction_name="hans_zimmer_works_info"
    )

    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_videos(),
        template_class=BilibiliVideosInfo,
        extraction_name="bilibili_videos_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb section - Hans Zimmer's filmography and tab switching
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb - Explore Hans Zimmer's filmography and other classic works",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F3:A7 - Tab switching to view filmography
    imdb_tab_switch_ok = (
        has_any_ci(answer, ['imdb']) and
        mentions_hans_zimmer(answer) and
        has_any_ci(answer, ['filmography', 'works', 'composer', '作品', '电影'])
    )
    evaluator.add_custom_node(
        result=bool(imdb_tab_switch_ok),
        id="imdb_tab_switching",
        desc="[Action Node] imdb.com:F3:A7 - Navigate to Hans Zimmer's IMDb page and switch tabs to view filmography details",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P16 - Understanding structured filmography information
    filmography_understanding_ok = (
        zimmer_works_info.other_works_mentioned and
        zimmer_works_info.works_list and
        len(str(zimmer_works_info.works_list).strip()) > 20
    )
    evaluator.add_custom_node(
        result=bool(filmography_understanding_ok),
        id="imdb_filmography_understanding",
        desc="[Perception Node] imdb.com:F3:P16 - Extract and understand structured filmography data (years, film names) for Hans Zimmer's classic works",
        parent=imdb_node,
        critical=False
    )

    # Additional check: mentions Interstellar verification on IMDb
    imdb_interstellar_ok = (
        has_any_ci(answer, ['imdb']) and
        mentions_interstellar(answer) and
        mentions_hans_zimmer(answer)
    )
    evaluator.add_custom_node(
        result=bool(imdb_interstellar_ok),
        id="imdb_interstellar_verification",
        desc="Verifies Interstellar and Hans Zimmer connection on IMDb",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Spotify section - Finding the complete soundtrack album
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify - Locate the complete Interstellar soundtrack album",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A9 - Click album card to enter details
    spotify_click_ok = (
        soundtrack_info.spotify_mentioned and
        mentions_interstellar(answer) and
        mentions_soundtrack_album(answer)
    )
    evaluator.add_custom_node(
        result=bool(spotify_click_ok),
        id="spotify_album_card_click",
        desc="[Action Node] open.spotify.com:F2:A9 - Click on the Interstellar soundtrack album card to enter the album details page",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F2:P1 - Album cover recognition
    album_cover_recognition_ok = (
        soundtrack_info.soundtrack_album_mentioned and
        soundtrack_info.album_details and
        (has_any_ci(answer, ['space', 'planet', 'star', '太空', '星球', '宇宙']) or
         has_any_ci(str(soundtrack_info.album_details), ['interstellar', '星际']))
    )
    evaluator.add_custom_node(
        result=bool(album_cover_recognition_ok),
        id="spotify_album_cover_recognition",
        desc="[Perception Node] open.spotify.com:F2:P1 - Recognize the Interstellar soundtrack album cover with space/planet visual elements",
        parent=spotify_node,
        critical=False
    )

    # Additional check: mentions complete album or track listing
    spotify_complete_album_ok = (
        soundtrack_info.spotify_mentioned and
        has_any_ci(answer, ['complete', 'full', 'track', 'song', '完整', '曲目'])
    )
    evaluator.add_custom_node(
        result=bool(spotify_complete_album_ok),
        id="spotify_complete_album_found",
        desc="Confirms finding the complete soundtrack album with track details",
        parent=spotify_node,
        critical=False
    )

    # 3.3 Bilibili section - Finding analytical videos
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili - Find in-depth soundtrack analysis videos with high view counts",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sorting by view count
    bilibili_sort_ok = (
        bilibili_info.bilibili_mentioned and
        bilibili_info.view_count_mentioned and
        has_any_ci(answer, ['sort', 'filter', '排序', '筛选', 'view count', '播放量'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_sort_ok),
        id="bilibili_sort_by_views",
        desc="[Action Node] bilibili.com:F1:A5 - Use sorting/filtering to order search results by view count (播放量)",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Video content relevance judgment
    video_relevance_ok = (
        bilibili_info.video_analysis_found and
        mentions_interstellar(answer) and
        mentions_video_analysis(answer) and
        has_any_ci(answer, ['soundtrack', 'music', 'score', '配乐', '音乐'])
    )
    evaluator.add_custom_node(
        result=bool(video_relevance_ok),
        id="bilibili_video_relevance",
        desc="[Perception Node] bilibili.com:F1:P30 - Identify videos specifically analyzing Interstellar's soundtrack composition techniques from thumbnails and titles",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A12 - Pagination to browse more results
    bilibili_pagination_ok = (
        bilibili_info.bilibili_mentioned and
        has_any_ci(answer, ['page', 'more', 'next', 'browse', '翻页', '更多'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_pagination_ok),
        id="bilibili_pagination",
        desc="[Action Node] bilibili.com:F1:A12 - Browse through multiple pages of search results to find suitable analysis videos",
        parent=bilibili_node,
        critical=False
    )

    # Additional checks for Bilibili criteria
    bilibili_duration_ok = (
        bilibili_info.duration_mentioned and
        mentions_duration(answer) and
        has_any_ci(answer, ['10', 'ten', 'long', '长', '深度'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_duration_ok),
        id="bilibili_duration_check",
        desc="Checks for videos over 10 minutes long with in-depth analysis",
        parent=bilibili_node,
        critical=False
    )

    bilibili_high_views_ok = (
        bilibili_info.view_count_mentioned and
        (has_any_ci(answer, ['high view', 'popular', 'many view', '高播放', '热门']) or
         contains_digits(answer))
    )
    evaluator.add_custom_node(
        result=bool(bilibili_high_views_ok),
        id="bilibili_high_view_count",
        desc="Confirms finding videos with high view counts as requested",
        parent=bilibili_node,
        critical=False
    )

    # 3.4 Overall task completion check
    overall_complete = (
        mentions_interstellar(answer) and
        mentions_hans_zimmer(answer) and
        (soundtrack_info.spotify_mentioned or mentions_soundtrack_album(answer)) and
        (bilibili_info.bilibili_mentioned or mentions_bilibili(answer))
    )
    evaluator.add_custom_node(
        result=bool(overall_complete),
        id="overall_task_coverage",
        desc="Confirms the answer addresses all three main components: soundtrack album, Hans Zimmer's works, and Bilibili analysis videos",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
