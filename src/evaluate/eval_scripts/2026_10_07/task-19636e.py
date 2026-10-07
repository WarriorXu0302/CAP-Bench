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
TASK_ID = "task-19636e"
TASK_DESCRIPTION = "I'm looking to purchase a highly replayable JRPG on Steam during the ongoing sales, with a budget of under 100 RMB. Please help me filter games on Steam that are tagged 'JRPG', priced under 100 RMB, and have a review rating of 'Very Positive' or higher. To ensure the gameplay isn't too short, please browse through several pages of the search results, select 3 promising titles, and then check their 'Main Story' completion times on HowLongToBeat. Finally, output a table listing the names of these 3 games, their current Steam price, review rating status, and Main Story length. Also, tell me which game offers the best 'price-per-hour' value."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Information about a single game extracted from the answer"""
    game_name: Optional[str] = None
    price_text: Optional[str] = None
    review_rating: Optional[str] = None
    main_story_length: Optional[str] = None


class GamesTable(BaseModel):
    """Table of games extracted from the answer"""
    games: List[GameInfo] = Field(default_factory=list)
    best_value_game: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games_table() -> str:
    return """
Extract the table of 3 JRPG games from the answer. For each game, extract:
- game_name: the name of the game
- price_text: the current Steam price exactly as stated (include currency/units)
- review_rating: the review rating status (e.g., "Very Positive", "Overwhelmingly Positive")
- main_story_length: the Main Story completion time from HowLongToBeat exactly as stated (include units)

Also extract:
- best_value_game: which game is stated as having the best 'price-per-hour' value

If any field is missing, set it to null. If fewer than 3 games are present, return what is available.
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


def looks_like_price_rmb(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['rmb', '¥', 'yuan', 'cny']) or contains_digits(text)


def looks_like_positive_review(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['very positive', 'overwhelmingly positive', 'mostly positive', 'positive'])


def looks_like_time_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['hour', 'hours', 'hr', 'hrs', 'h', 'minute', 'minutes', 'min'])


def mentions_steam(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return ci_contains(answer_text, 'steam')


def mentions_jrpg_tag(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return ci_contains(answer_text, 'jrpg')


def mentions_price_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['100', 'price', 'under', 'below', 'budget'])


def mentions_review_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['very positive', 'review', 'rating'])


def mentions_pagination(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['page', 'pages', 'browse', 'next', 'several'])


def mentions_howlongtobeat(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['howlongtobeat', 'how long to beat', 'hltb'])


def mentions_main_story(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['main story', 'main game', 'story length'])


def has_table_structure(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    # Check for markdown table, HTML table, or structured list
    has_markdown_table = '|' in answer_text and ('---' in answer_text or '|-' in answer_text)
    has_html_table = has_any_ci(answer_text, ['<table>', '<tr>', '<td>'])
    has_structured_list = answer_text.count('\n') >= 5 and contains_digits(answer_text)
    return has_markdown_table or has_html_table or has_structured_list


def mentions_price_per_hour(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['price-per-hour', 'price per hour', 'best value', 'value', 'cost per hour'])


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
    games_table = await evaluator.extract(
        prompt=prompt_extract_games_table(),
        template_class=GamesTable,
        extraction_name="games_table"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Steam filtering section
    steam_node = evaluator.add_sequential(
        id="steam_filtering_section",
        desc="Steam search and filtering for JRPG games under 100 RMB with Very Positive or higher ratings",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Multi-select filtering (JRPG tag)
    jrpg_tag_action_ok = mentions_steam(answer) and mentions_jrpg_tag(answer)
    evaluator.add_custom_node(
        result=bool(jrpg_tag_action_ok),
        id="steam_action_jrpg_tag",
        desc="[Action Node] store.steampowered.com:F1:A1 - Apply JRPG tag filter in Steam search",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A5 - Range/price slider (under 100 RMB)
    price_filter_action_ok = mentions_steam(answer) and mentions_price_filter(answer)
    evaluator.add_custom_node(
        result=bool(price_filter_action_ok),
        id="steam_action_price_filter",
        desc="[Action Node] store.steampowered.com:F1:A5 - Set price filter to under 100 RMB using price slider or input",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P3 - List content understanding (review ratings)
    review_filter_perception_ok = mentions_review_filter(answer)
    evaluator.add_custom_node(
        result=bool(review_filter_perception_ok),
        id="steam_perception_review_ratings",
        desc="[Perception Node] store.steampowered.com:F1:P3 - Identify and filter games with 'Very Positive' or higher review ratings from the list",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A8 - Pagination
    pagination_action_ok = mentions_pagination(answer)
    evaluator.add_custom_node(
        result=bool(pagination_action_ok),
        id="steam_action_pagination",
        desc="[Action Node] store.steampowered.com:F1:A8 - Browse through multiple pages of search results",
        parent=steam_node,
        critical=False
    )

    # Additional check: Selected 3 games
    has_three_games = games_table and games_table.games and len(games_table.games) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_games),
        id="steam_selected_three_games",
        desc="Selected 3 promising game titles from the search results",
        parent=steam_node,
        critical=False
    )

    # 3.2 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="howlongtobeat_section",
        desc="HowLongToBeat queries for Main Story completion times",
        parent=root,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story duration
    hltb_mention_ok = mentions_howlongtobeat(answer)
    main_story_mention_ok = mentions_main_story(answer)
    evaluator.add_custom_node(
        result=bool(hltb_mention_ok and main_story_mention_ok),
        id="hltb_action_query",
        desc="Query HowLongToBeat for Main Story completion times for the selected games",
        parent=hltb_node,
        critical=False
    )

    # Check that Main Story durations are extracted for games
    if games_table and games_table.games:
        games_with_duration = [g for g in games_table.games if looks_like_time_duration(g.main_story_length)]
        has_duration_data = len(games_with_duration) >= 2
    else:
        has_duration_data = False

    evaluator.add_custom_node(
        result=bool(has_duration_data),
        id="hltb_perception_main_story_durations",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story completion time data for the games",
        parent=hltb_node,
        critical=False
    )

    # 3.3 Output and analysis section
    output_node = evaluator.add_parallel(
        id="output_section",
        desc="Final output table and price-per-hour analysis",
        parent=root,
        critical=False
    )

    # Check table structure
    table_structure_ok = has_table_structure(answer)
    evaluator.add_custom_node(
        result=bool(table_structure_ok),
        id="output_table_structure",
        desc="Output contains a structured table with game information",
        parent=output_node,
        critical=False
    )

    # Validate table content: game names, prices, ratings, durations
    if games_table and games_table.games and len(games_table.games) >= 3:
        valid_games = []
        for game in games_table.games[:3]:
            name_ok = bool(game.game_name and game.game_name.strip())
            price_ok = looks_like_price_rmb(game.price_text)
            rating_ok = looks_like_positive_review(game.review_rating)
            duration_ok = looks_like_time_duration(game.main_story_length)
            if name_ok and price_ok and rating_ok and duration_ok:
                valid_games.append(game)
        table_content_ok = len(valid_games) >= 2
    else:
        table_content_ok = False

    evaluator.add_custom_node(
        result=bool(table_content_ok),
        id="output_table_content_valid",
        desc="Table contains valid data: game names, Steam prices (RMB), review ratings, and Main Story lengths",
        parent=output_node,
        critical=False
    )

    # Check price-per-hour analysis
    price_per_hour_ok = mentions_price_per_hour(answer)
    best_value_identified = bool(games_table and games_table.best_value_game and games_table.best_value_game.strip())
    evaluator.add_custom_node(
        result=bool(price_per_hour_ok and best_value_identified),
        id="output_price_per_hour_analysis",
        desc="Identifies which game offers the best price-per-hour value",
        parent=output_node,
        critical=False
    )

    # Additional check: All prices are under 100 RMB
    if games_table and games_table.games:
        prices_under_100 = []
        for game in games_table.games:
            price_num = extract_float(game.price_text)
            if price_num is not None:
                prices_under_100.append(price_num < 100)
        all_under_budget = len(prices_under_100) >= 2 and all(prices_under_100)
    else:
        all_under_budget = False

    evaluator.add_custom_node(
        result=bool(all_under_budget),
        id="output_prices_within_budget",
        desc="All listed games have prices under 100 RMB as requested",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
