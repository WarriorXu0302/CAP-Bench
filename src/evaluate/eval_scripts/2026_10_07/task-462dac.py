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
TASK_ID = "task-462dac"
TASK_DESCRIPTION = "I am preparing for a live-streamed game series titled '2025's Most Playable Action-Adventure Games'. I need to select 3 high-quality and trending games for in-depth review.\n\nFirst, go to Steam and search for 'Action-Adventure' games. Filter by the 'Action' and 'Adventure' tags, then sort by 'Sales' to identify the top 10 games on the current bestseller list. Record the game names, current prices, and discount information.\n\nNext, for each of these 10 games, visit Metacritic to find their Metascore (media rating) and User Score for the PC platform. Retain only games with a Metascore of ≥75 AND a User Score of ≥7.5.\n\nThen, go to HowLongToBeat to query the 'Main Story' completion time and 'Completionist' time for these filtered games. Further filter to include only games with a 'Main Story' completion time between 15-40 hours (games that are too short won't provide enough content for a series, and games that are too long might cause viewer drop-off).\n\nFinally, search IGN for these remaining games to confirm if a review article exists. Record the IGN score, and check the completeness of the Wiki guide page (specifically, look for detailed quest walkthroughs and collectible location guides).\n\nPlease output a comparison table for the finally filtered games, including:\n*   Game Name\n*   Steam Price & Discount\n*   Metacritic Media Score\n*   Metacritic User Score\n*   Main Story Completion Time\n*   Completionist Time\n*   IGN Score\n*   IGN Guide Completeness (Complete/Partial/None)\n*   Steam Detail Page Link\n*   Metacritic Detail Page Link\n*   HowLongToBeat Detail Page Link\n*   IGN Review Link\n\nFrom this list, recommend the 3 most suitable games for the live stream series, explaining the reasons based on dimensions such as scores, playtime, trending potential, and guide support."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SteamGamesInfo(BaseModel):
    """Steam games list extracted from the answer"""
    game_names: Optional[List[str]] = Field(default_factory=list)
    num_games: Optional[int] = None


class GameEntry(BaseModel):
    """A single game entry with all required information"""
    game_name: Optional[str] = None
    steam_price: Optional[str] = None
    steam_discount: Optional[str] = None
    metacritic_metascore: Optional[str] = None
    metacritic_userscore: Optional[str] = None
    main_story_hours: Optional[str] = None
    completionist_hours: Optional[str] = None
    ign_score: Optional[str] = None
    ign_guide_completeness: Optional[str] = None
    steam_link: Optional[str] = None
    metacritic_link: Optional[str] = None
    hltb_link: Optional[str] = None
    ign_review_link: Optional[str] = None


class GamesTable(BaseModel):
    """All games table extracted from the answer"""
    games: Optional[List[GameEntry]] = Field(default_factory=list)


class RecommendationsInfo(BaseModel):
    """Recommendations extracted from the answer"""
    recommended_games: Optional[List[str]] = Field(default_factory=list)
    has_reasoning: Optional[bool] = False


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_steam_games() -> str:
    return """
Extract the Steam games list from the answer. The user should have identified 10 games from Steam's Action-Adventure bestseller list.

Return:
- game_names: list of game names mentioned (up to 10)
- num_games: the count of games in the initial list

If not present, return empty list and null.
"""


def prompt_extract_games_table() -> str:
    return """
Extract the comparison table of filtered games from the answer. For each game in the final table, extract:

- game_name
- steam_price (price text with currency)
- steam_discount (discount percentage or 'no discount')
- metacritic_metascore (the Metascore value)
- metacritic_userscore (the User Score value)
- main_story_hours (main story completion time)
- completionist_hours (completionist time)
- ign_score (IGN review score)
- ign_guide_completeness (Complete/Partial/None)
- steam_link (Steam detail page URL)
- metacritic_link (Metacritic detail page URL)
- hltb_link (HowLongToBeat detail page URL)
- ign_review_link (IGN review URL)

Return a list of game entries. If any field is missing for a game, set it to null.
"""


def prompt_extract_recommendations() -> str:
    return """
Extract the final 3 game recommendations from the answer.

Return:
- recommended_games: list of 3 recommended game names
- has_reasoning: true if the answer provides reasoning based on scores, playtime, trending potential, and guide support; false otherwise

If not present, return empty list and false.
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
    return contains_digits(text) and has_any_ci(text, ['$', '€', '¥', 'usd', 'eur', 'free', 'price'])


def looks_like_discount(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['%', 'off', 'discount', 'sale', 'no discount', 'none'])


def looks_like_metascore(text: Optional[str]) -> bool:
    if not text:
        return False
    score = extract_float(text)
    return score is not None and 0 <= score <= 100


def looks_like_userscore(text: Optional[str]) -> bool:
    if not text:
        return False
    score = extract_float(text)
    return score is not None and 0 <= score <= 10


def looks_like_hours(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['hour', 'hr', 'h'])


def looks_like_ign_score(text: Optional[str]) -> bool:
    if not text:
        return False
    score = extract_float(text)
    return score is not None and 0 <= score <= 10


def looks_like_completeness(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['complete', 'partial', 'none'])


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http', 'www', '.com'])


def check_metascore_threshold(text: Optional[str], threshold: float = 75.0) -> bool:
    score = extract_float(text)
    return score is not None and score >= threshold


def check_userscore_threshold(text: Optional[str], threshold: float = 7.5) -> bool:
    score = extract_float(text)
    return score is not None and score >= threshold


def check_hours_in_range(text: Optional[str], min_hours: float = 15.0, max_hours: float = 40.0) -> bool:
    hours = extract_float(text)
    return hours is not None and min_hours <= hours <= max_hours


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
        strategy=AggregationStrategy.SEQUENTIAL,
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
        prompt=prompt_extract_steam_games(),
        template_class=SteamGamesInfo,
        extraction_name="steam_games_info"
    )

    games_table = await evaluator.extract(
        prompt=prompt_extract_games_table(),
        template_class=GamesTable,
        extraction_name="games_comparison_table"
    )

    recommendations = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=RecommendationsInfo,
        extraction_name="recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Steam section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam Action-Adventure games bestseller list acquisition",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Multi-tag filtering
    steam_tags_ok = has_any_ci(answer, ['action', 'adventure']) and has_any_ci(answer, ['tag', 'filter'])
    evaluator.add_custom_node(
        result=bool(steam_tags_ok),
        id="steam_action_multi_tag",
        desc="[Action Node] store.steampowered.com:F1:A1 - Filter by Action and Adventure tags on Steam",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A4 - Sort by sales
    steam_sort_ok = has_any_ci(answer, ['sales', 'bestseller', 'top selling', 'sort'])
    evaluator.add_custom_node(
        result=bool(steam_sort_ok),
        id="steam_action_sort_sales",
        desc="[Action Node] store.steampowered.com:F1:A4 - Sort by Sales to identify bestsellers",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Discount identification
    has_discount_info = any(
        game.steam_discount and looks_like_discount(game.steam_discount)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    evaluator.add_custom_node(
        result=bool(has_discount_info),
        id="steam_perception_discount",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Identify discount information for games",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P3 - Extract list information
    has_game_names = steam_info and steam_info.game_names and len(steam_info.game_names) > 0
    has_prices = any(
        game.steam_price and looks_like_price(game.steam_price)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    list_extraction_ok = has_game_names and has_prices
    evaluator.add_custom_node(
        result=bool(list_extraction_ok),
        id="steam_perception_list_extraction",
        desc="[Perception Node] store.steampowered.com:F1:P3 - Extract game names, prices, and discount information from search results",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A8 - Pagination
    got_10_games = steam_info and steam_info.num_games and steam_info.num_games >= 10
    evaluator.add_custom_node(
        result=bool(got_10_games),
        id="steam_action_pagination",
        desc="[Action Node] store.steampowered.com:F1:A8 - Navigate through pages to collect top 10 games",
        parent=steam_node,
        critical=False
    )

    # 3.2 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic score filtering for PC platform",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A1 - Platform filtering
    metacritic_pc_filter_ok = has_any_ci(answer, ['metacritic']) and has_any_ci(answer, ['pc', 'platform'])
    evaluator.add_custom_node(
        result=bool(metacritic_pc_filter_ok),
        id="metacritic_action_platform_filter",
        desc="[Action Node] metacritic.comgame:F1:A1 - Filter for PC platform on Metacritic",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.comgame:F2:A3 - Platform version switching
    metacritic_pc_switch_ok = has_any_ci(answer, ['pc version', 'pc platform', 'platform switch'])
    evaluator.add_custom_node(
        result=bool(metacritic_pc_switch_ok or metacritic_pc_filter_ok),
        id="metacritic_action_platform_switch",
        desc="[Action Node] metacritic.comgame:F2:A3 - Switch to PC platform version for each game",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F2:P1 - Metascore identification
    has_valid_metascores = any(
        game.metacritic_metascore and looks_like_metascore(game.metacritic_metascore) and
        check_metascore_threshold(game.metacritic_metascore, 75.0)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    evaluator.add_custom_node(
        result=bool(has_valid_metascores),
        id="metacritic_perception_metascore",
        desc="[Perception Node] metacritic.comgame:F2:P1 - Identify Metascore ≥75 for games",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F2:P2 - User Score identification
    has_valid_userscores = any(
        game.metacritic_userscore and looks_like_userscore(game.metacritic_userscore) and
        check_userscore_threshold(game.metacritic_userscore, 7.5)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    evaluator.add_custom_node(
        result=bool(has_valid_userscores),
        id="metacritic_perception_userscore",
        desc="[Perception Node] metacritic.comgame:F2:P2 - Identify User Score ≥7.5 for games",
        parent=metacritic_node,
        critical=False
    )

    # 3.3 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="hltb_section",
        desc="HowLongToBeat playtime filtering",
        parent=root,
        critical=False
    )

    # [Action Node] howlongtobeat.com:F1:A1 - Game search
    hltb_search_ok = has_any_ci(answer, ['howlongtobeat', 'hltb']) and has_any_ci(answer, ['search', 'query'])
    evaluator.add_custom_node(
        result=bool(hltb_search_ok),
        id="hltb_action_search",
        desc="[Action Node] howlongtobeat.com:F1:A1 - Search for games on HowLongToBeat",
        parent=hltb_node,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F2:P1 - Playtime data identification
    has_main_story_in_range = any(
        game.main_story_hours and looks_like_hours(game.main_story_hours) and
        check_hours_in_range(game.main_story_hours, 15.0, 40.0)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    has_completionist_time = any(
        game.completionist_hours and looks_like_hours(game.completionist_hours)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    playtime_extraction_ok = has_main_story_in_range and has_completionist_time
    evaluator.add_custom_node(
        result=bool(playtime_extraction_ok),
        id="hltb_perception_playtime",
        desc="[Perception Node] howlongtobeat.com:F2:P1 - Extract Main Story (15-40h) and Completionist times",
        parent=hltb_node,
        critical=False
    )

    # 3.4 IGN section
    ign_node = evaluator.add_sequential(
        id="ign_section",
        desc="IGN review and Wiki guide verification",
        parent=root,
        critical=False
    )

    # [Action Node] ign.com:F1:A1 - Navigation menu search
    ign_search_ok = has_any_ci(answer, ['ign']) and has_any_ci(answer, ['search', 'review', 'wiki'])
    evaluator.add_custom_node(
        result=bool(ign_search_ok),
        id="ign_action_search",
        desc="[Action Node] ign.com:F1:A1 - Navigate to IGN and search for games",
        parent=ign_node,
        critical=False
    )

    # [Action Node] ign.com:F2:A3 - Tab switching for Wiki
    ign_wiki_tab_ok = has_any_ci(answer, ['wiki', 'guide', 'walkthrough', 'tab'])
    evaluator.add_custom_node(
        result=bool(ign_wiki_tab_ok),
        id="ign_action_wiki_tab",
        desc="[Action Node] ign.com:F2:A3 - Switch to Wiki tab to check guide completeness",
        parent=ign_node,
        critical=False
    )

    # [Perception Node] ign.com:F2:P3 - Review score extraction
    has_ign_scores = any(
        game.ign_score and looks_like_ign_score(game.ign_score)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    evaluator.add_custom_node(
        result=bool(has_ign_scores),
        id="ign_perception_score",
        desc="[Perception Node] ign.com:F2:P3 - Extract IGN review scores",
        parent=ign_node,
        critical=False
    )

    # Check guide completeness (non-prefixed node)
    has_guide_completeness = any(
        game.ign_guide_completeness and looks_like_completeness(game.ign_guide_completeness)
        for game in (games_table.games if games_table and games_table.games else [])
    )
    evaluator.add_custom_node(
        result=bool(has_guide_completeness),
        id="ign_guide_completeness_check",
        desc="Check Wiki guide completeness (Complete/Partial/None) with quest walkthroughs and collectible guides",
        parent=ign_node,
        critical=False
    )

    # 3.5 Output format and recommendations
    output_node = evaluator.add_sequential(
        id="output_section",
        desc="Comparison table output and recommendations",
        parent=root,
        critical=False
    )

    # Check table completeness
    has_complete_table = (
        games_table and
        games_table.games and
        len(games_table.games) > 0 and
        all(
            game.game_name and
            game.steam_price and
            game.metacritic_metascore and
            game.metacritic_userscore and
            game.main_story_hours
            for game in games_table.games
        )
    )
    evaluator.add_custom_node(
        result=bool(has_complete_table),
        id="output_table_complete",
        desc="Comparison table includes all required fields for filtered games",
        parent=output_node,
        critical=False
    )

    # Check for links
    has_all_links = (
        games_table and
        games_table.games and
        any(
            game.steam_link and looks_like_url(game.steam_link) and
            game.metacritic_link and looks_like_url(game.metacritic_link) and
            game.hltb_link and looks_like_url(game.hltb_link) and
            game.ign_review_link and looks_like_url(game.ign_review_link)
            for game in games_table.games
        )
    )
    evaluator.add_custom_node(
        result=bool(has_all_links),
        id="output_links_included",
        desc="Table includes Steam, Metacritic, HowLongToBeat, and IGN links",
        parent=output_node,
        critical=False
    )

    # Check recommendations
    has_3_recommendations = (
        recommendations and
        recommendations.recommended_games and
        len(recommendations.recommended_games) == 3
    )
    evaluator.add_custom_node(
        result=bool(has_3_recommendations),
        id="output_recommendations_count",
        desc="Provides exactly 3 game recommendations",
        parent=output_node,
        critical=False
    )

    # Check reasoning
    has_reasoning = recommendations and recommendations.has_reasoning
    reasoning_keywords_ok = has_any_ci(answer, ['score', 'playtime', 'trending', 'guide', 'reason', 'because', 'suitable'])
    evaluator.add_custom_node(
        result=bool(has_reasoning and reasoning_keywords_ok),
        id="output_recommendations_reasoning",
        desc="Recommendations include reasoning based on scores, playtime, trending potential, and guide support",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
