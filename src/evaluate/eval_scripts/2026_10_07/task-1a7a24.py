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
TASK_ID = "task-1a7a24"
TASK_DESCRIPTION = 'I am producing an "Annual Indie Game Review" video and need to collect materials for three games: Pacific Drive, Balatro, and Animal Well.  For each game, please perform the following steps in order:\n\n1.  First, on YouTube, search for the game\'s "Official Launch Trailer". Filter for the video with the highest views that is published by the official developer or publisher channel, and ensure it supports 4K or 1080p resolution.\n2.  Next, go to Metacritic and search for the game\'s PC version page. Record its User Score. Then, find a user review with a score of 10 and extract a core statement from it.\n3.  Finally, on Twitch, search for the game\'s category. Navigate to the "Clips" section, set the filter to "All Time", and identify the clip with the highest view count.\n\nPlease output the following for each of the three games: YouTube Trailer Title, Publishing Channel, Metacritic User Score, Selected Review Summary, Twitch Most Viewed Clip Title, View Count, Streamer Name, and all relevant detail page links.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameMaterials(BaseModel):
    """Materials extracted for a single game"""
    game_name: Optional[str] = None
    youtube_trailer_title: Optional[str] = None
    youtube_channel: Optional[str] = None
    youtube_link: Optional[str] = None
    metacritic_user_score: Optional[str] = None
    metacritic_review_summary: Optional[str] = None
    metacritic_link: Optional[str] = None
    twitch_clip_title: Optional[str] = None
    twitch_clip_views: Optional[str] = None
    twitch_streamer_name: Optional[str] = None
    twitch_clip_link: Optional[str] = None


class AllGamesMaterials(BaseModel):
    """Materials for all three games"""
    pacific_drive: Optional[GameMaterials] = None
    balatro: Optional[GameMaterials] = None
    animal_well: Optional[GameMaterials] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_games() -> str:
    return """
Extract the materials collected for each of the three games: Pacific Drive, Balatro, and Animal Well.

For each game, extract:
- game_name: the game's name as mentioned
- youtube_trailer_title: the YouTube trailer title
- youtube_channel: the publishing channel name
- youtube_link: the YouTube video link
- metacritic_user_score: the Metacritic User Score
- metacritic_review_summary: the extracted core statement from a 10-score user review
- metacritic_link: the Metacritic PC version page link
- twitch_clip_title: the title of the most viewed clip
- twitch_clip_views: the view count of that clip
- twitch_streamer_name: the streamer name for that clip
- twitch_clip_link: the Twitch clip link

If any field is missing for any game, set it to null.
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


def looks_like_youtube_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return has_any_ci(url, ['youtube.com', 'youtu.be'])


def looks_like_metacritic_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'metacritic.com')


def looks_like_metacritic_pc_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'metacritic.com') and ci_contains(url, '/pc/')


def looks_like_twitch_clip_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'twitch.tv') and ci_contains(url, 'clip')


def looks_like_official_channel(channel: Optional[str]) -> bool:
    if not channel:
        return False
    official_keywords = ['official', 'developer', 'publisher', 'studio', 'games', 'entertainment']
    return has_any_ci(channel, official_keywords)


def looks_like_user_score(score: Optional[str]) -> bool:
    if not score:
        return False
    num = extract_float(score)
    if num is None:
        return False
    return 0 <= num <= 10


def looks_like_high_view_count(views: Optional[str]) -> bool:
    if not views:
        return False
    return contains_digits(views)


# --------------------------------------------------------------------------- #
# Game-specific evaluation builder                                            #
# --------------------------------------------------------------------------- #
def evaluate_game_materials(
    evaluator: Evaluator,
    parent_node: str,
    game_data: Optional[GameMaterials],
    game_name: str,
    answer: str
) -> None:
    """
    Build evaluation nodes for a single game's materials.
    """
    game_node = evaluator.add_sequential(
        id=f"{game_name.lower().replace(' ', '_')}_section",
        desc=f"Materials collection for {game_name}",
        parent=parent_node,
        critical=False
    )

    # YouTube section
    youtube_node = evaluator.add_sequential(
        id=f"{game_name.lower().replace(' ', '_')}_youtube",
        desc=f"YouTube Official Launch Trailer for {game_name}",
        parent=game_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click to enter detail page
    youtube_link_ok = looks_like_youtube_url(game_data.youtube_link if game_data else None)
    evaluator.add_custom_node(
        result=bool(youtube_link_ok),
        id=f"{game_name.lower().replace(' ', '_')}_youtube_action_click",
        desc=f"[Action Node] youtube.com:F1:A22 - Navigate to and extract the YouTube video detail page link for {game_name}",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify official channel
    channel_official_ok = looks_like_official_channel(game_data.youtube_channel if game_data else None)
    evaluator.add_custom_node(
        result=bool(channel_official_ok),
        id=f"{game_name.lower().replace(' ', '_')}_youtube_perception_official",
        desc=f"[Perception Node] youtube.com:F1:P4 - Identify that the video is from an official developer or publisher channel for {game_name}",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P5 - Filter out ads
    not_ad_link = youtube_link_ok and game_data and game_data.youtube_link and not has_any_ci(game_data.youtube_link, ['/ads/', 'googleads'])
    evaluator.add_custom_node(
        result=bool(not_ad_link),
        id=f"{game_name.lower().replace(' ', '_')}_youtube_perception_not_ad",
        desc=f"[Perception Node] youtube.com:F1:P5 - Ensure the video is not a promoted/ad result for {game_name}",
        parent=youtube_node,
        critical=False
    )

    # Additional check: trailer title exists
    trailer_title_ok = bool(game_data and game_data.youtube_trailer_title and game_data.youtube_trailer_title.strip())
    evaluator.add_custom_node(
        result=bool(trailer_title_ok),
        id=f"{game_name.lower().replace(' ', '_')}_youtube_title_exists",
        desc=f"YouTube trailer title is present for {game_name}",
        parent=youtube_node,
        critical=False
    )

    # Metacritic section
    metacritic_node = evaluator.add_sequential(
        id=f"{game_name.lower().replace(' ', '_')}_metacritic",
        desc=f"Metacritic PC version User Score and review for {game_name}",
        parent=game_node,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A1 - Multi-condition filter for PC platform
    metacritic_pc_url_ok = looks_like_metacritic_pc_url(game_data.metacritic_link if game_data else None)
    evaluator.add_custom_node(
        result=bool(metacritic_pc_url_ok),
        id=f"{game_name.lower().replace(' ', '_')}_metacritic_action_pc_filter",
        desc=f"[Action Node] metacritic.comgame:F1:A1 - Navigate to the PC platform version page for {game_name}",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F1:P1 - Extract User Score
    user_score_ok = looks_like_user_score(game_data.metacritic_user_score if game_data else None)
    evaluator.add_custom_node(
        result=bool(user_score_ok),
        id=f"{game_name.lower().replace(' ', '_')}_metacritic_perception_user_score",
        desc=f"[Perception Node] metacritic.comgame:F1:P1 - Extract the User Score (0-10) for {game_name}",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A6 - Click into user review detail
    review_summary_ok = bool(game_data and game_data.metacritic_review_summary and game_data.metacritic_review_summary.strip())
    evaluator.add_custom_node(
        result=bool(review_summary_ok),
        id=f"{game_name.lower().replace(' ', '_')}_metacritic_action_review_detail",
        desc=f"[Action Node] metacritic.comgame:F1:A6 - Access user review section and extract a core statement from a 10-score review for {game_name}",
        parent=metacritic_node,
        critical=False
    )

    # Twitch section
    twitch_node = evaluator.add_sequential(
        id=f"{game_name.lower().replace(' ', '_')}_twitch",
        desc=f"Twitch most viewed clip (All Time) for {game_name}",
        parent=game_node,
        critical=False
    )

    # [Action Node] twitch.tv:F3:A2 - Set time filter to All Time
    all_time_mention = has_any_ci(answer, ['all time']) or has_any_ci(answer, ['alltime'])
    evaluator.add_custom_node(
        result=bool(all_time_mention),
        id=f"{game_name.lower().replace(' ', '_')}_twitch_action_alltime_filter",
        desc=f"[Action Node] twitch.tv:F3:A2 - Set the Clips filter to 'All Time' for {game_name}",
        parent=twitch_node,
        critical=False
    )

    # [Action Node] twitch.tv:F3:A11 - Scroll/browse list to find highest view count
    clip_views_ok = looks_like_high_view_count(game_data.twitch_clip_views if game_data else None)
    evaluator.add_custom_node(
        result=bool(clip_views_ok),
        id=f"{game_name.lower().replace(' ', '_')}_twitch_action_browse_highest",
        desc=f"[Action Node] twitch.tv:F3:A11 - Browse clips and identify the one with the highest view count for {game_name}",
        parent=twitch_node,
        critical=False
    )

    # [Action Node] twitch.tv:F7:A2 - Enter clip detail page
    twitch_clip_url_ok = looks_like_twitch_clip_url(game_data.twitch_clip_link if game_data else None)
    evaluator.add_custom_node(
        result=bool(twitch_clip_url_ok),
        id=f"{game_name.lower().replace(' ', '_')}_twitch_action_clip_detail",
        desc=f"[Action Node] twitch.tv:F7:A2 - Access the clip detail page to extract full information for {game_name}",
        parent=twitch_node,
        critical=False
    )

    # Additional checks: clip title and streamer name exist
    clip_title_ok = bool(game_data and game_data.twitch_clip_title and game_data.twitch_clip_title.strip())
    evaluator.add_custom_node(
        result=bool(clip_title_ok),
        id=f"{game_name.lower().replace(' ', '_')}_twitch_clip_title_exists",
        desc=f"Twitch clip title is present for {game_name}",
        parent=twitch_node,
        critical=False
    )

    streamer_name_ok = bool(game_data and game_data.twitch_streamer_name and game_data.twitch_streamer_name.strip())
    evaluator.add_custom_node(
        result=bool(streamer_name_ok),
        id=f"{game_name.lower().replace(' ', '_')}_twitch_streamer_name_exists",
        desc=f"Twitch streamer name is present for {game_name}",
        parent=twitch_node,
        critical=False
    )


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
    all_games = await evaluator.extract(
        prompt=prompt_extract_all_games(),
        template_class=AllGamesMaterials,
        extraction_name="all_games_materials"
    )

    # -------- 3. Build evaluation tree for each game --------------------- #
    evaluate_game_materials(
        evaluator=evaluator,
        parent_node=root,
        game_data=all_games.pacific_drive if all_games else None,
        game_name="Pacific Drive",
        answer=answer
    )

    evaluate_game_materials(
        evaluator=evaluator,
        parent_node=root,
        game_data=all_games.balatro if all_games else None,
        game_name="Balatro",
        answer=answer
    )

    evaluate_game_materials(
        evaluator=evaluator,
        parent_node=root,
        game_data=all_games.animal_well if all_games else None,
        game_name="Animal Well",
        answer=answer
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
