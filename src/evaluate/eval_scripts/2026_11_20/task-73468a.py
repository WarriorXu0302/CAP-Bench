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
TASK_ID = "task-73468a"
TASK_DESCRIPTION = 'I’m planning to produce an in-depth analysis video on an indie puzzle game, around 20 minutes long. First, help me find a game on Steam tagged **indie + puzzle** that was released within the past 12 months, has an **Overwhelmingly Positive** rating, and has **more than 1,000 reviews**.\n\nThen, check this game on Metacritic and record its **Metascore** and **User Score**, including both the numeric scores and their rating labels.\n\nNext, search Fandom for the game’s Wiki and go to a **Design** or **Gameplay**-related section to review detailed explanations of its mechanics. If there is no Fandom Wiki for this game, or no relevant Design/Gameplay section, use the game’s **Steam Community Guides** page or an **official/developer-published gameplay explanation page** as an alternative source, extract the key mechanics points, and clearly note the replacement source link.\n\nAfter that, go to YouTube and search for **[game name + gameplay]**, filter for videos longer than 10 minutes, sort by view count, and pick the top 2 high-view gameplay videos. Then search **[game name + analysis]** or **[game name + review]**, again filter for videos longer than 10 minutes and sort by view count, and select the top 2 in-depth commentary videos.\n\nFinally, go to Bilibili and search for the game’s Chinese or English name plus **解说 / 评测** (commentary/review), filter for videos longer than 10 minutes, sort by play count, and pick the top 2 Chinese commentary videos.\n\nOutput the following:\n- Game title  \n- Steam review status (e.g., Overwhelmingly Positive)  \n- Steam review count  \n- Steam link  \n- Metacritic Metascore (numeric)  \n- Metascore rating label  \n- Metacritic User Score (numeric)  \n- Metacritic link  \n- Summary of the game’s core mechanics/design highlights (preferably from Fandom; if an alternative source is used, explicitly label it)  \n- Mechanics information source link  \n- Titles and links of 2 YouTube gameplay videos  \n- Titles and links of 2 YouTube analysis/review videos  \n- Titles and links of 2 Bilibili commentary videos'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SteamGameInfo(BaseModel):
    """Steam game information extracted from the answer"""
    game_title: Optional[str] = None
    review_status: Optional[str] = None
    review_count: Optional[int] = None
    steam_link: Optional[str] = None


class MetacriticInfo(BaseModel):
    """Metacritic scores extracted from the answer"""
    metascore: Optional[int] = None
    metascore_label: Optional[str] = None
    user_score: Optional[float] = None
    metacritic_link: Optional[str] = None


class MechanicsInfo(BaseModel):
    """Game mechanics information extracted from the answer"""
    mechanics_summary: Optional[str] = None
    mechanics_source_link: Optional[str] = None
    is_alternative_source: Optional[bool] = None


class YouTubeVideo(BaseModel):
    """Single YouTube video information"""
    title: Optional[str] = None
    link: Optional[str] = None


class YouTubeVideos(BaseModel):
    """YouTube videos extracted from the answer"""
    gameplay_video_1: Optional[YouTubeVideo] = None
    gameplay_video_2: Optional[YouTubeVideo] = None
    analysis_video_1: Optional[YouTubeVideo] = None
    analysis_video_2: Optional[YouTubeVideo] = None


class BilibiliVideo(BaseModel):
    """Single Bilibili video information"""
    title: Optional[str] = None
    link: Optional[str] = None


class BilibiliVideos(BaseModel):
    """Bilibili videos extracted from the answer"""
    commentary_video_1: Optional[BilibiliVideo] = None
    commentary_video_2: Optional[BilibiliVideo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_steam_info() -> str:
    return """
Extract the Steam game information from the answer:

- game_title: the game's title
- review_status: the Steam review status (e.g., "Overwhelmingly Positive", "特别好评")
- review_count: the number of reviews as an integer
- steam_link: the Steam store page URL

If any field is missing, set it to null.
"""


def prompt_extract_metacritic_info() -> str:
    return """
Extract the Metacritic information from the answer:

- metascore: the Metascore numeric value (0-100)
- metascore_label: the Metascore rating label (e.g., "Universal Acclaim", "Generally Favorable")
- user_score: the User Score numeric value (0-10)
- metacritic_link: the Metacritic page URL

If any field is missing, set it to null.
"""


def prompt_extract_mechanics_info() -> str:
    return """
Extract the game mechanics information from the answer:

- mechanics_summary: the summary of core mechanics/design highlights
- mechanics_source_link: the URL of the mechanics information source
- is_alternative_source: true if an alternative source (not Fandom) was used, false otherwise

If any field is missing, set it to null.
"""


def prompt_extract_youtube_videos() -> str:
    return """
Extract the YouTube video information from the answer. For each video, extract title and link.

Return:
- gameplay_video_1: first gameplay video (title and link)
- gameplay_video_2: second gameplay video (title and link)
- analysis_video_1: first analysis/review video (title and link)
- analysis_video_2: second analysis/review video (title and link)

If any video is missing, set it to null.
"""


def prompt_extract_bilibili_videos() -> str:
    return """
Extract the Bilibili video information from the answer. For each video, extract title and link.

Return:
- commentary_video_1: first commentary video (title and link)
- commentary_video_2: second commentary video (title and link)

If any video is missing, set it to null.
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


def is_valid_url(url: Optional[str], domain: str) -> bool:
    if not url:
        return False
    return domain.lower() in url.lower()


def is_overwhelmingly_positive(review_status: Optional[str]) -> bool:
    if not review_status:
        return False
    return has_any_ci(review_status, ['overwhelmingly positive', '特别好评'])


def has_indie_puzzle_tags(answer: str) -> bool:
    indie_ok = has_any_ci(answer, ['indie', '独立'])
    puzzle_ok = has_any_ci(answer, ['puzzle', '解谜'])
    return indie_ok and puzzle_ok


def looks_like_recent_release(answer: str) -> bool:
    # Check for mentions of recent dates (2024, 2025) or "past 12 months", "within the past year", etc.
    return has_any_ci(answer, ['2024', '2025', 'past 12 months', 'within the past', 'recently released'])


def is_metascore_valid(score: Optional[int]) -> bool:
    if score is None:
        return False
    return 0 <= score <= 100


def is_user_score_valid(score: Optional[float]) -> bool:
    if score is None:
        return False
    return 0 <= score <= 10


def has_video_duration_info(answer: str) -> bool:
    # Check for mentions of video duration filtering
    return has_any_ci(answer, ['10 minutes', '10分钟', 'longer than 10', 'duration', 'length'])


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
    steam_info = await evaluator.extract(
        prompt=prompt_extract_steam_info(),
        template_class=SteamGameInfo,
        extraction_name="steam_game_info"
    )

    metacritic_info = await evaluator.extract(
        prompt=prompt_extract_metacritic_info(),
        template_class=MetacriticInfo,
        extraction_name="metacritic_info"
    )

    mechanics_info = await evaluator.extract(
        prompt=prompt_extract_mechanics_info(),
        template_class=MechanicsInfo,
        extraction_name="mechanics_info"
    )

    youtube_videos = await evaluator.extract(
        prompt=prompt_extract_youtube_videos(),
        template_class=YouTubeVideos,
        extraction_name="youtube_videos"
    )

    bilibili_videos = await evaluator.extract(
        prompt=prompt_extract_bilibili_videos(),
        template_class=BilibiliVideos,
        extraction_name="bilibili_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Steam section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam game search and information retrieval",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Multi-tag filtering (indie + puzzle)
    tags_ok = has_indie_puzzle_tags(answer)
    evaluator.add_custom_node(
        result=bool(tags_ok),
        id="steam_action_multi_tag",
        desc="[Action Node] store.steampowered.com:F1:A1 - Filter games by indie + puzzle tags",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A3 - Date range filtering
    date_filter_ok = looks_like_recent_release(answer)
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="steam_action_date_filter",
        desc="[Action Node] store.steampowered.com:F1:A3 - Filter by release date (past 12 months)",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Review status recognition
    review_status_ok = is_overwhelmingly_positive(steam_info.review_status)
    evaluator.add_custom_node(
        result=bool(review_status_ok),
        id="steam_perception_review_status",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Identify Overwhelmingly Positive review status",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P3 - Review count extraction
    review_count_ok = steam_info.review_count is not None and steam_info.review_count > 1000
    evaluator.add_custom_node(
        result=bool(review_count_ok),
        id="steam_perception_review_count",
        desc="[Perception Node] store.steampowered.com:F1:P3 - Extract review count (>1000)",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F2:A9 - Click game card to enter details
    steam_link_ok = is_valid_url(steam_info.steam_link, 'steampowered.com')
    evaluator.add_custom_node(
        result=bool(steam_link_ok),
        id="steam_action_game_details",
        desc="[Action Node] store.steampowered.com:F2:A9 - Navigate to game details page",
        parent=steam_node,
        critical=False
    )

    # 3.2 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic score retrieval",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A6 - Click game card on Metacritic
    metacritic_link_ok = is_valid_url(metacritic_info.metacritic_link, 'metacritic.com')
    evaluator.add_custom_node(
        result=bool(metacritic_link_ok),
        id="metacritic_action_game_card",
        desc="[Action Node] metacritic.comgame:F1:A6 - Navigate to game page on Metacritic",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F2:P1 - Metascore recognition
    metascore_ok = is_metascore_valid(metacritic_info.metascore)
    metascore_label_ok = bool(metacritic_info.metascore_label and metacritic_info.metascore_label.strip())
    evaluator.add_custom_node(
        result=bool(metascore_ok and metascore_label_ok),
        id="metacritic_perception_metascore",
        desc="[Perception Node] metacritic.comgame:F2:P1 - Extract Metascore and rating label",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F2:P2 - User Score recognition
    user_score_ok = is_user_score_valid(metacritic_info.user_score)
    evaluator.add_custom_node(
        result=bool(user_score_ok),
        id="metacritic_perception_user_score",
        desc="[Perception Node] metacritic.comgame:F2:P2 - Extract User Score",
        parent=metacritic_node,
        critical=False
    )

    # 3.3 Fandom/mechanics section
    fandom_node = evaluator.add_sequential(
        id="fandom_section",
        desc="Game mechanics information retrieval from Fandom or alternative source",
        parent=root,
        critical=False
    )

    # [Action Node] fandom.com:F1:A27 - Expand table of contents
    mechanics_summary_ok = bool(mechanics_info.mechanics_summary and len(mechanics_info.mechanics_summary.strip()) > 20)
    evaluator.add_custom_node(
        result=bool(mechanics_summary_ok),
        id="fandom_action_expand_toc",
        desc="[Action Node] fandom.com:F1:A27 - Navigate to Design/Gameplay section",
        parent=fandom_node,
        critical=False
    )

    # [Perception Node] fandom.com:F1:P13 - Extract mechanics information
    mechanics_source_ok = bool(mechanics_info.mechanics_source_link and mechanics_info.mechanics_source_link.strip())
    evaluator.add_custom_node(
        result=bool(mechanics_summary_ok and mechanics_source_ok),
        id="fandom_perception_mechanics",
        desc="[Perception Node] fandom.com:F1:P13 - Extract and summarize game mechanics",
        parent=fandom_node,
        critical=False
    )

    # 3.4 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube video search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video cards
    gameplay_1_ok = bool(youtube_videos.gameplay_video_1 and
                        youtube_videos.gameplay_video_1.title and
                        youtube_videos.gameplay_video_1.link)
    gameplay_2_ok = bool(youtube_videos.gameplay_video_2 and
                        youtube_videos.gameplay_video_2.title and
                        youtube_videos.gameplay_video_2.link)
    analysis_1_ok = bool(youtube_videos.analysis_video_1 and
                        youtube_videos.analysis_video_1.title and
                        youtube_videos.analysis_video_1.link)
    analysis_2_ok = bool(youtube_videos.analysis_video_2 and
                        youtube_videos.analysis_video_2.title and
                        youtube_videos.analysis_video_2.link)

    all_videos_ok = gameplay_1_ok and gameplay_2_ok and analysis_1_ok and analysis_2_ok
    evaluator.add_custom_node(
        result=bool(all_videos_ok),
        id="youtube_action_video_cards",
        desc="[Action Node] youtube.com:F1:A22 - Navigate to video detail pages",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Duration filtering
    duration_filter_ok = has_video_duration_info(answer)
    evaluator.add_custom_node(
        result=bool(duration_filter_ok),
        id="youtube_action_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Filter videos by duration (>10 minutes)",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Thumbnail content recognition
    gameplay_search_ok = has_any_ci(answer, ['gameplay', '游戏实况'])
    analysis_search_ok = has_any_ci(answer, ['analysis', 'review', '分析', '评测'])
    evaluator.add_custom_node(
        result=bool(gameplay_search_ok and analysis_search_ok),
        id="youtube_perception_thumbnail",
        desc="[Perception Node] youtube.com:F1:P4 - Identify appropriate video types from thumbnails",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P9 - Thumbnail object recognition
    evaluator.add_custom_node(
        result=bool(gameplay_1_ok and gameplay_2_ok),
        id="youtube_perception_gameplay_thumbnails",
        desc="[Perception Node] youtube.com:F1:P9 - Verify gameplay footage in thumbnails",
        parent=youtube_node,
        critical=False
    )

    # 3.5 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili video search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by play count
    sort_mention_ok = has_any_ci(answer, ['播放量', 'play count', 'view count', '排序'])
    evaluator.add_custom_node(
        result=bool(sort_mention_ok),
        id="bilibili_action_sort",
        desc="[Action Node] bilibili.com:F1:A5 - Sort videos by play count",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A48 - Duration filter panel
    bilibili_duration_ok = has_video_duration_info(answer)
    evaluator.add_custom_node(
        result=bool(bilibili_duration_ok),
        id="bilibili_action_duration_filter",
        desc="[Action Node] bilibili.com:F1:A48 - Expand and apply duration filter (>10 minutes)",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Thumbnail understanding for commentary
    commentary_1_ok = bool(bilibili_videos.commentary_video_1 and
                          bilibili_videos.commentary_video_1.title and
                          bilibili_videos.commentary_video_1.link)
    commentary_2_ok = bool(bilibili_videos.commentary_video_2 and
                          bilibili_videos.commentary_video_2.title and
                          bilibili_videos.commentary_video_2.link)
    evaluator.add_custom_node(
        result=bool(commentary_1_ok and commentary_2_ok),
        id="bilibili_perception_thumbnail",
        desc="[Perception Node] bilibili.com:F1:P30 - Identify commentary videos from thumbnails",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P11 - Thumbnail object recognition
    commentary_search_ok = has_any_ci(answer, ['解说', '评测', 'commentary', 'review'])
    evaluator.add_custom_node(
        result=bool(commentary_search_ok and commentary_1_ok and commentary_2_ok),
        id="bilibili_perception_objects",
        desc="[Perception Node] bilibili.com:F1:P11 - Verify game content in thumbnails",
        parent=bilibili_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
