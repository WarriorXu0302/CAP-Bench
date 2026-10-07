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
TASK_ID = "task-72d45e"
TASK_DESCRIPTION = 'I\'m looking for an engaging RPG on Steam to pass the time. Please go to the Steam store and search for "RPG". Use the filters on the left to only show games priced under 50 RMB and with "Overwhelmingly Positive" reviews. Browse through several pages and select at least 5 promising indie games that are not DLCs. Then, for each selected game, go to the HowLongToBeat website and look up their "Main Story" completion time. Finally, filter out games with a "Main Story" completion time exceeding 20 hours. List the game title, its current price on Steam, and its "Main Story" completion time from HLTB.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SteamFiltersInfo(BaseModel):
    """Information about Steam filters applied"""
    mentions_rpg_tag: Optional[bool] = False
    mentions_indie_tag: Optional[bool] = False
    mentions_price_filter: Optional[bool] = False
    mentions_review_filter: Optional[bool] = False
    mentions_pagination: Optional[bool] = False


class GameEntry(BaseModel):
    """Single game entry with all required information"""
    title: Optional[str] = None
    price: Optional[str] = None
    main_story_time: Optional[str] = None


class GamesListInfo(BaseModel):
    """List of games extracted from the answer"""
    games: Optional[List[GameEntry]] = Field(default_factory=list)
    total_games_count: Optional[int] = 0


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_steam_filters() -> str:
    return """
From the answer, determine whether the user applied the required Steam filters:

Return:
- mentions_rpg_tag: true if the answer mentions filtering by RPG tag/genre
- mentions_indie_tag: true if the answer mentions filtering by Indie tag/games
- mentions_price_filter: true if the answer mentions applying a price filter (under 50 RMB or similar)
- mentions_review_filter: true if the answer mentions filtering by "Overwhelmingly Positive" reviews
- mentions_pagination: true if the answer mentions browsing through multiple pages

Set each field to false if not mentioned.
"""


def prompt_extract_games_list() -> str:
    return """
Extract all the final game recommendations from the answer. Each game should have:
- title: the game's name
- price: the current price on Steam (with currency if present)
- main_story_time: the "Main Story" completion time from HowLongToBeat (with hours if present)

Return:
- games: a list of game entries
- total_games_count: the number of games in the list

If no games are listed, return an empty games list and total_games_count as 0.
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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    # Accept common currency symbols and text
    return has_any_ci(text, ['rmb', '¥', 'yuan', '元']) or contains_digits(text)


def looks_like_time_hours(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    # Accept hour mentions
    return has_any_ci(text, ['hour', 'hours', 'h', 'hrs']) or contains_digits(text)


def time_within_20_hours(time_text: Optional[str]) -> bool:
    if not time_text:
        return False
    num = extract_number(time_text)
    if num is None:
        return False
    return num <= 20.0


def game_has_complete_info(game: GameEntry) -> bool:
    return (game.title is not None and
            game.title.strip() != "" and
            looks_like_price(game.price) and
            looks_like_time_hours(game.main_story_time))


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
    filters_info = await evaluator.extract(
        prompt=prompt_extract_steam_filters(),
        template_class=SteamFiltersInfo,
        extraction_name="steam_filters_applied"
    )

    games_info = await evaluator.extract(
        prompt=prompt_extract_games_list(),
        template_class=GamesListInfo,
        extraction_name="games_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Steam filtering section
    steam_node = evaluator.add_sequential(
        id="steam_filtering_section",
        desc="Steam store search and filtering for RPG indie games",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Select RPG and Indie tags
    rpg_indie_tags_ok = (
        (filters_info.mentions_rpg_tag or has_any_ci(answer, ['rpg', 'role-playing', 'role playing'])) and
        (filters_info.mentions_indie_tag or has_any_ci(answer, ['indie']))
    )
    evaluator.add_custom_node(
        result=bool(rpg_indie_tags_ok),
        id="steam_action_tags",
        desc="[Action Node] store.steampowered.com:F1:A1 - Select RPG and Indie tags/filters",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A5 - Adjust price slider to under 50 RMB
    price_filter_ok = (
        filters_info.mentions_price_filter or
        has_any_ci(answer, ['price', '50', 'rmb', '元']) and has_any_ci(answer, ['under', 'below', 'less than', 'filter'])
    )
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="steam_action_price",
        desc="[Action Node] store.steampowered.com:F1:A5 - Adjust price range slider to under 50 RMB",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Recognize "Overwhelmingly Positive" review status
    review_filter_ok = (
        filters_info.mentions_review_filter or
        has_any_ci(answer, ['overwhelmingly positive', 'overwhelmingly', 'review'])
    )
    evaluator.add_custom_node(
        result=bool(review_filter_ok),
        id="steam_perception_reviews",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Identify and filter by 'Overwhelmingly Positive' review status",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A8 - Browse through multiple pages
    pagination_ok = (
        filters_info.mentions_pagination or
        has_any_ci(answer, ['page', 'pages', 'browse', 'browsing', 'multiple', 'several'])
    )
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="steam_action_pagination",
        desc="[Action Node] store.steampowered.com:F1:A8 - Browse through multiple pages of search results",
        parent=steam_node,
        critical=False
    )

    # Non-prefixed: Mentions excluding DLCs
    excludes_dlc_ok = has_any_ci(answer, ['dlc', 'downloadable content', 'not dlc', 'exclude dlc'])
    evaluator.add_custom_node(
        result=bool(excludes_dlc_ok),
        id="steam_mentions_exclude_dlc",
        desc="Mentions excluding DLCs from selection",
        parent=steam_node,
        critical=False
    )

    # 3.2 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="howlongtobeat_section",
        desc="HowLongToBeat lookup for Main Story completion times",
        parent=root,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story completion times
    hltb_mention_ok = has_any_ci(answer, ['howlongtobeat', 'hltb', 'how long to beat'])
    main_story_mention_ok = has_any_ci(answer, ['main story', 'main', 'story'])

    evaluator.add_custom_node(
        result=bool(hltb_mention_ok and main_story_mention_ok),
        id="hltb_perception_main_story",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract 'Main Story' completion time data for each game",
        parent=hltb_node,
        critical=False
    )

    # 3.3 Results quality section
    results_node = evaluator.add_parallel(
        id="results_quality_section",
        desc="Quality and completeness of final game recommendations",
        parent=root,
        critical=False
    )

    # Check minimum 5 games (after 20-hour filter)
    games_list = games_info.games if games_info and games_info.games else []
    has_minimum_games = len(games_list) >= 5
    evaluator.add_custom_node(
        result=bool(has_minimum_games),
        id="results_minimum_count",
        desc="At least 5 games in final recommendation list",
        parent=results_node,
        critical=False
    )

    # Check that games have complete information (title, price, time)
    complete_games_count = sum(1 for game in games_list if game_has_complete_info(game))
    has_complete_info = complete_games_count >= 5
    evaluator.add_custom_node(
        result=bool(has_complete_info),
        id="results_complete_info",
        desc="At least 5 games have complete information (title, Steam price, HLTB Main Story time)",
        parent=results_node,
        critical=False
    )

    # Check that recommended games are within 20-hour limit
    games_within_limit_count = sum(1 for game in games_list if time_within_20_hours(game.main_story_time))
    all_within_limit = (len(games_list) > 0 and games_within_limit_count == len(games_list))
    evaluator.add_custom_node(
        result=bool(all_within_limit),
        id="results_time_filter",
        desc="All recommended games have Main Story time <= 20 hours",
        parent=results_node,
        critical=False
    )

    # Non-prefixed: Mentions the 20-hour filtering criterion
    mentions_20h_filter = has_any_ci(answer, ['20 hour', '20 hours', '20h', 'exceed 20', 'over 20', 'more than 20'])
    evaluator.add_custom_node(
        result=bool(mentions_20h_filter),
        id="results_mentions_time_limit",
        desc="Explicitly mentions the 20-hour time limit filtering criterion",
        parent=results_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
