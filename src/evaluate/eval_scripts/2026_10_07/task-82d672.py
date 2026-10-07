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
TASK_ID = "task-82d672"
TASK_DESCRIPTION = "During a Steam sale event, I'd like to find some engaging RPGs to play during my leisure time.\n\nFirst, on Steam, filter for RPG games that are currently on sale (Special Offers), have a user review rating of 'Very Positive' or higher, and are priced between $10 and $30. Sort these results by user reviews and identify the top 5 games.\n\nNext, on HowLongToBeat, find the 'Main + Extra' average completion time for each of these 5 games.\n\nFinally, using the current discounted price from Steam, calculate the 'cost per hour of gameplay' (Price / Duration) for each game, and on Metacritic, confirm the Metascore rating for these 5 games.\n\nFor each of the 5 games, output the following: Game Title, Steam Current Price, Steam Review Status, HowLongToBeat Duration, Calculated Cost Per Hour, Metacritic Score, and links to their respective platform detail pages."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Information extracted for a single game from the answer"""
    game_title: Optional[str] = None
    steam_current_price: Optional[str] = None
    steam_review_status: Optional[str] = None
    hltb_duration: Optional[str] = None
    cost_per_hour: Optional[str] = None
    metacritic_score: Optional[str] = None
    steam_link: Optional[str] = None
    hltb_link: Optional[str] = None
    metacritic_link: Optional[str] = None


class GamesCollection(BaseModel):
    """Collection of games extracted from the answer"""
    games: List[GameInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games_from_answer() -> str:
    return """
Extract all games mentioned in the answer with their details. For each game, extract:

- game_title: the game's title exactly as written
- steam_current_price: the current discounted price from Steam (include currency symbol if present)
- steam_review_status: the user review status from Steam (e.g., 'Very Positive', 'Overwhelmingly Positive')
- hltb_duration: the 'Main + Extra' completion time from HowLongToBeat (include units if present)
- cost_per_hour: the calculated cost per hour value (include currency symbol if present)
- metacritic_score: the Metascore rating from Metacritic
- steam_link: URL to the Steam detail page
- hltb_link: URL to the HowLongToBeat detail page
- metacritic_link: URL to the Metacritic detail page

If any field is missing for a game, set it to null. Return all games found in the answer.
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
    num = extract_float(text)
    if num is None:
        return False
    has_currency = has_any_ci(text, ['$', 'usd', 'dollar'])
    return num >= 10 and num <= 30 and has_currency


def looks_like_review_status(text: Optional[str]) -> bool:
    if not text:
        return False
    valid_statuses = ['very positive', 'overwhelmingly positive', 'mostly positive', 'positive']
    return has_any_ci(text, valid_statuses)


def looks_like_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['hour', 'hr', 'h'])


def looks_like_cost_per_hour(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    has_currency = has_any_ci(text, ['$', 'usd', 'dollar'])
    return has_currency


def looks_like_metacritic_score(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 100


def is_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and ci_contains(text, 'http')


def mentions_rpg(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['rpg', 'role-playing', 'role playing'])


def mentions_sale_or_discount(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['sale', 'discount', 'special offer', 'on sale'])


def mentions_sorting_by_reviews(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['sort', 'sorted', 'user review'])


def count_games_in_list(games: List[GameInfo]) -> int:
    valid_games = [g for g in games if g.game_title and g.game_title.strip()]
    return len(valid_games)


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
    Restrict evaluator.verify to at most one usage (we'll not use it here).
    Favor lenient, fault-tolerant checks and allow partial credit.
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
        template_class=GamesCollection,
        extraction_name="games_collection"
    )

    games_list = games_info.games if games_info and games_info.games else []
    num_games = count_games_in_list(games_list)

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Steam filtering and sorting section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam filtering for RPG games on sale with specific criteria",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Filter for RPG games
    rpg_filter_ok = mentions_rpg(answer) and num_games > 0
    evaluator.add_custom_node(
        result=bool(rpg_filter_ok),
        id="steam_action_rpg_filter",
        desc="[Action Node] store.steampowered.com:F1:A1 - Filter for RPG game type",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A2 - Filter for Special Offers (on sale)
    sale_filter_ok = mentions_sale_or_discount(answer)
    evaluator.add_custom_node(
        result=bool(sale_filter_ok),
        id="steam_action_sale_filter",
        desc="[Action Node] store.steampowered.com:F1:A2 - Filter for games currently on sale (Special Offers)",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A5 - Price range filter ($10-$30)
    prices_in_range = 0
    for game in games_list:
        if looks_like_price(game.steam_current_price):
            prices_in_range += 1
    price_range_ok = prices_in_range >= 3
    evaluator.add_custom_node(
        result=bool(price_range_ok),
        id="steam_action_price_range",
        desc="[Action Node] store.steampowered.com:F1:A5 - Filter for games priced between $10 and $30",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A4 - Sort by user reviews
    sort_by_reviews_ok = mentions_sorting_by_reviews(answer)
    evaluator.add_custom_node(
        result=bool(sort_by_reviews_ok),
        id="steam_action_sort_reviews",
        desc="[Action Node] store.steampowered.com:F1:A4 - Sort results by user reviews",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Review status is 'Very Positive' or higher
    high_review_count = 0
    for game in games_list:
        if looks_like_review_status(game.steam_review_status):
            high_review_count += 1
    review_status_ok = high_review_count >= 3
    evaluator.add_custom_node(
        result=bool(review_status_ok),
        id="steam_perception_review_status",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Games have review rating of 'Very Positive' or higher",
        parent=steam_node,
        critical=False
    )

    # Check for top 5 games (non-prefixed)
    top_5_ok = num_games >= 5
    evaluator.add_custom_node(
        result=bool(top_5_ok),
        id="steam_top_5_games",
        desc="Identified top 5 games from filtered results",
        parent=steam_node,
        critical=False
    )

    # 3.2 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="hltb_section",
        desc="HowLongToBeat duration lookup for Main + Extra completion time",
        parent=root,
        critical=False
    )

    # [Action Node] howlongtobeat.com:F1:A1 - Search for each game
    hltb_search_ok = has_any_ci(answer, ['howlongtobeat', 'hltb']) and num_games > 0
    evaluator.add_custom_node(
        result=bool(hltb_search_ok),
        id="hltb_action_search",
        desc="[Action Node] howlongtobeat.com:F1:A1 - Search for and select each of the 5 games on HowLongToBeat",
        parent=hltb_node,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main + Extra duration
    duration_count = 0
    for game in games_list:
        if looks_like_duration(game.hltb_duration):
            duration_count += 1
    duration_ok = duration_count >= 3
    evaluator.add_custom_node(
        result=bool(duration_ok),
        id="hltb_perception_duration",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract 'Main + Extra' average completion time for games",
        parent=hltb_node,
        critical=False
    )

    # Check for Main + Extra mention (non-prefixed)
    main_extra_mention = has_any_ci(answer, ['main + extra', 'main+extra', 'main and extra'])
    evaluator.add_custom_node(
        result=bool(main_extra_mention),
        id="hltb_mentions_main_extra",
        desc="Mentions 'Main + Extra' completion time category",
        parent=hltb_node,
        critical=False
    )

    # 3.3 Cost per hour calculation section
    calculation_node = evaluator.add_sequential(
        id="calculation_section",
        desc="Calculate cost per hour of gameplay (Price / Duration)",
        parent=root,
        critical=False
    )

    cost_per_hour_count = 0
    for game in games_list:
        if looks_like_cost_per_hour(game.cost_per_hour):
            cost_per_hour_count += 1
    cost_calculation_ok = cost_per_hour_count >= 3
    evaluator.add_custom_node(
        result=bool(cost_calculation_ok),
        id="cost_per_hour_calculation",
        desc="Calculate cost per hour (Price / Duration) for each game",
        parent=calculation_node,
        critical=False
    )

    # 3.4 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic Metascore confirmation for games",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.comgame:F1:A1 - Search for each game
    metacritic_search_ok = has_any_ci(answer, ['metacritic']) and num_games > 0
    evaluator.add_custom_node(
        result=bool(metacritic_search_ok),
        id="metacritic_action_search",
        desc="[Action Node] metacritic.comgame:F1:A1 - Search for and locate each of the 5 games on Metacritic",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F1:P1 - Extract Metascore rating
    metascore_count = 0
    for game in games_list:
        if looks_like_metacritic_score(game.metacritic_score):
            metascore_count += 1
    metascore_ok = metascore_count >= 3
    evaluator.add_custom_node(
        result=bool(metascore_ok),
        id="metacritic_perception_score",
        desc="[Perception Node] metacritic.comgame:F1:P1 - Extract Metascore rating (0-100) for games",
        parent=metacritic_node,
        critical=False
    )

    # 3.5 Output completeness section
    output_node = evaluator.add_sequential(
        id="output_section",
        desc="Output completeness with all required fields and links",
        parent=root,
        critical=False
    )

    # Check for required output fields
    complete_games = 0
    for game in games_list:
        fields_present = [
            bool(game.game_title and game.game_title.strip()),
            looks_like_price(game.steam_current_price),
            looks_like_review_status(game.steam_review_status),
            looks_like_duration(game.hltb_duration),
            looks_like_cost_per_hour(game.cost_per_hour),
            looks_like_metacritic_score(game.metacritic_score)
        ]
        if sum(fields_present) >= 5:
            complete_games += 1

    output_fields_ok = complete_games >= 3
    evaluator.add_custom_node(
        result=bool(output_fields_ok),
        id="output_required_fields",
        desc="Output includes all required fields: Game Title, Steam Current Price, Steam Review Status, HowLongToBeat Duration, Cost Per Hour, Metacritic Score",
        parent=output_node,
        critical=False
    )

    # Check for platform links
    games_with_steam_links = sum(1 for g in games_list if is_valid_url(g.steam_link, 'steampowered.com'))
    games_with_hltb_links = sum(1 for g in games_list if is_valid_url(g.hltb_link, 'howlongtobeat.com'))
    games_with_metacritic_links = sum(1 for g in games_list if is_valid_url(g.metacritic_link, 'metacritic.com'))

    links_ok = (games_with_steam_links >= 2) and (games_with_hltb_links >= 2) and (games_with_metacritic_links >= 2)
    evaluator.add_custom_node(
        result=bool(links_ok),
        id="output_platform_links",
        desc="Output includes links to Steam, HowLongToBeat, and Metacritic detail pages for games",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
