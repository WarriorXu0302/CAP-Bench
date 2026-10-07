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
TASK_ID = "task-2bf0e1"
TASK_DESCRIPTION = 'Lately my commuting time has been pretty fragmented, so I’m looking for a few highly rated, short-play RPGs suitable for Steam Deck.  \nFirst, go to Metacritic and filter for RPG titles with a Metascore above 85. Platform can be either PC or Switch. Sort by user rating (if that sort option is unavailable, use the closest user-related sort available on the site, or keep the default sort and continue). Check multiple pages rather than only the first popular page.\n\nThen pick 3 titles you think are good candidates, and look up their **Main Story** playtime on HowLongToBeat. Prioritize games with a main story of **20 hours or less**. If there are fewer than 3 titles strictly under 20 hours, keep the **Metascore ≥ 85** requirement unchanged and fill the list to 3 from the candidate pool, clearly marking which ones exceed 20 hours.\n\nFinally, go to Steam and confirm whether these 3 games are Steam Deck verified (shown with the green **Deck Verified** icon).\n\nPlease output the following for the 3 games: **title, Metascore, HLTB Main Story length, and Steam Deck compatibility status**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Information about a single game"""
    title: Optional[str] = None
    metascore: Optional[str] = None
    hltb_main_story: Optional[str] = None
    steam_deck_status: Optional[str] = None


class ExtractedGames(BaseModel):
    """All three games extracted from the answer"""
    game1: Optional[GameInfo] = None
    game2: Optional[GameInfo] = None
    game3: Optional[GameInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games_from_answer() -> str:
    return """
Extract information about the 3 RPG games the user reported from the answer.

For each game, extract:
- title: the game title exactly as written
- metascore: the Metascore value (number or text with score)
- hltb_main_story: the Main Story playtime from HowLongToBeat exactly as written (include units if present)
- steam_deck_status: the Steam Deck compatibility status exactly as written (e.g., "Verified", "Playable", "Unsupported", etc.)

Return game1, game2, and game3. If fewer than 3 games are present, set missing games to null.
If any field for a game is missing, set it to null.
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


def looks_like_metascore(text: Optional[str]) -> bool:
    """Check if text looks like a Metascore (number between 0-100)"""
    if not text:
        return False
    num = extract_number(text)
    if num is None:
        return False
    return 0 <= num <= 100


def looks_like_playtime(text: Optional[str]) -> bool:
    """Check if text looks like a playtime duration"""
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['hour', 'hr', 'h', 'minute', 'min'])


def looks_like_deck_status(text: Optional[str]) -> bool:
    """Check if text looks like a Steam Deck status"""
    if not text:
        return False
    return has_any_ci(text, ['verified', 'playable', 'unsupported', 'unknown', 'deck'])


def count_valid_games(games: ExtractedGames) -> int:
    """Count how many games have at least a title"""
    count = 0
    for game in [games.game1, games.game2, games.game3]:
        if game and game.title and game.title.strip():
            count += 1
    return count


def check_metascore_threshold(metascore_text: Optional[str], threshold: float = 85.0) -> bool:
    """Check if metascore meets or exceeds threshold"""
    if not metascore_text:
        return False
    num = extract_number(metascore_text)
    if num is None:
        return False
    return num >= threshold


def check_playtime_hours(playtime_text: Optional[str]) -> Optional[float]:
    """Extract playtime in hours if possible"""
    if not playtime_text:
        return None
    num = extract_number(playtime_text)
    return num


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
    games_info = await evaluator.extract(
        prompt=prompt_extract_games_from_answer(),
        template_class=ExtractedGames,
        extraction_name="extracted_games"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic filtering and sorting for high-rated RPGs",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A1 - Multi-condition filtering
    mentions_metacritic = has_any_ci(answer, ['metacritic'])
    mentions_rpg = has_any_ci(answer, ['rpg'])
    mentions_score_85 = has_any_ci(answer, ['85', 'metascore']) or any(
        check_metascore_threshold(g.metascore) for g in [games_info.game1, games_info.game2, games_info.game3] if g
    )
    mentions_platform = has_any_ci(answer, ['pc', 'switch', 'platform'])

    filtering_ok = mentions_metacritic and mentions_rpg and mentions_score_85
    evaluator.add_custom_node(
        result=bool(filtering_ok),
        id="metacritic_action_filtering",
        desc="[Action Node] metacritic.comgame:F1:A1 - Filter for RPG titles with Metascore ≥ 85 on PC or Switch platform",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A2 - Sorting by user rating
    mentions_sort = has_any_ci(answer, ['sort', 'user rating', 'user score', 'ordered'])
    evaluator.add_custom_node(
        result=bool(mentions_sort),
        id="metacritic_action_sorting",
        desc="[Action Node] metacritic.comgame:F1:A2 - Sort by user rating or closest user-related sort option",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A11 - Multi-page browsing
    mentions_multiple_pages = has_any_ci(answer, ['page', 'next', 'multiple', 'several', 'browse'])
    evaluator.add_custom_node(
        result=bool(mentions_multiple_pages),
        id="metacritic_action_pagination",
        desc="[Action Node] metacritic.comgame:F1:A11 - Check multiple pages rather than only the first page",
        parent=metacritic_node,
        critical=False
    )

    # 3.2 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="hltb_section",
        desc="HowLongToBeat playtime verification for Main Story",
        parent=root,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story playtime
    mentions_hltb = has_any_ci(answer, ['howlongtobeat', 'hltb'])
    mentions_main_story = has_any_ci(answer, ['main story', 'main'])

    game_count = count_valid_games(games_info)
    has_playtimes = False
    if games_info:
        playtime_checks = [
            looks_like_playtime(g.hltb_main_story) for g in [games_info.game1, games_info.game2, games_info.game3] if g and g.hltb_main_story
        ]
        has_playtimes = len(playtime_checks) >= 2

    hltb_perception_ok = mentions_hltb and has_playtimes
    evaluator.add_custom_node(
        result=bool(hltb_perception_ok),
        id="hltb_perception_playtime",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story playtime data for selected games",
        parent=hltb_node,
        critical=False
    )

    # Check if 20-hour criterion is addressed
    mentions_20_hours = has_any_ci(answer, ['20 hour', '20 hr', '20h', 'under 20', 'less than 20'])
    evaluator.add_custom_node(
        result=bool(mentions_20_hours),
        id="hltb_20hour_criterion",
        desc="Addresses the 20-hour or less Main Story playtime preference",
        parent=hltb_node,
        critical=False
    )

    # 3.3 Steam section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam Deck compatibility verification",
        parent=root,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Verify Steam Deck compatibility
    mentions_steam = has_any_ci(answer, ['steam'])
    mentions_deck_verified = has_any_ci(answer, ['deck verified', 'deck compatibility', 'verified', 'steam deck'])

    has_deck_statuses = False
    if games_info:
        status_checks = [
            looks_like_deck_status(g.steam_deck_status) for g in [games_info.game1, games_info.game2, games_info.game3] if g and g.steam_deck_status
        ]
        has_deck_statuses = len(status_checks) >= 2

    steam_perception_ok = mentions_steam and has_deck_statuses
    evaluator.add_custom_node(
        result=bool(steam_perception_ok),
        id="steam_perception_deck_status",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Verify Steam Deck compatibility status (Verified icon) for each game",
        parent=steam_node,
        critical=False
    )

    # 3.4 Output completeness check
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Output contains all required information for the 3 games",
        parent=root,
        critical=False
    )

    # Check that 3 games are reported
    has_three_games = game_count >= 3
    evaluator.add_custom_node(
        result=bool(has_three_games),
        id="output_three_games",
        desc="Output includes 3 game recommendations",
        parent=output_node,
        critical=False
    )

    # Check that each game has the required fields
    all_games = [games_info.game1, games_info.game2, games_info.game3] if games_info else []
    complete_games = 0
    for game in all_games:
        if game and game.title:
            has_title = bool(game.title.strip())
            has_metascore = looks_like_metascore(game.metascore)
            has_playtime = looks_like_playtime(game.hltb_main_story)
            has_deck = looks_like_deck_status(game.steam_deck_status)
            if has_title and has_metascore and has_playtime and has_deck:
                complete_games += 1

    output_fields_ok = complete_games >= 2
    evaluator.add_custom_node(
        result=bool(output_fields_ok),
        id="output_required_fields",
        desc="Each game includes title, Metascore, HLTB Main Story length, and Steam Deck status",
        parent=output_node,
        critical=False
    )

    # Check metascore threshold is met
    metascore_ok = False
    if games_info:
        metascore_checks = [
            check_metascore_threshold(g.metascore) for g in all_games if g and g.metascore
        ]
        metascore_ok = len(metascore_checks) >= 2 and all(metascore_checks[:2])

    evaluator.add_custom_node(
        result=bool(metascore_ok),
        id="output_metascore_threshold",
        desc="Games meet the Metascore ≥ 85 requirement",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
