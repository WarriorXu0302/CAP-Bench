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
TASK_ID = "task-bee8a2"
TASK_DESCRIPTION = "As a club scout, with the summer transfer window approaching, I need you to find 8 potential midfielder candidates on Transfermarkt that meet specific criteria:\n*   Under 25 years old\n*   Market value between €3M and €5M\n*   Contract expiring before June 2026\n*   Position: Midfielder\n\nAfter identifying suitable players, navigate to each player's profile page one by one:\n1.  Record their basic information.\n2.  Examine the market value history curve to determine their value trend.\n3.  Check national team statistics to confirm at least 5 international caps.\n4.  Switch to the 'Transfer Rumours' tab to check for popular rumours and the community-assessed transfer probability.\n\nThen, visit the official website of each player's current club to find their detailed appearance data for the current season, confirming that their playing time is at least 1000 minutes.\n\nFinally, return to Transfermarkt, sort the players by market value from lowest to highest, and select the top 5 players with the best value-for-money.\n\nFor each selected player, output the following:\n*   Name\n*   Age\n*   Nationality\n*   Current Club\n*   Market Value\n*   Contract Expiry Date\n*   Market Value Trend (Rising/Stable/Declining)\n*   National Team Appearances\n*   Current Season Playing Time\n*   Goals\n*   Assists\n*   Transfer Rumours (Yes/No) and Probability\n*   Transfermarkt Player Profile Link\n*   Club Official Website Data Source Link"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlayerInfo(BaseModel):
    """Single player information extracted from the answer"""
    name: Optional[str] = None
    age: Optional[int] = None
    nationality: Optional[str] = None
    current_club: Optional[str] = None
    market_value: Optional[str] = None
    contract_expiry: Optional[str] = None
    market_value_trend: Optional[str] = None
    national_team_appearances: Optional[int] = None
    current_season_playing_time: Optional[str] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    transfer_rumours: Optional[str] = None
    transfer_probability: Optional[str] = None
    transfermarkt_link: Optional[str] = None
    club_website_link: Optional[str] = None


class PlayersCollection(BaseModel):
    """Collection of all players extracted from the answer"""
    players: List[PlayerInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_players_from_answer() -> str:
    return """
Extract all player information from the answer. The answer should contain details about up to 5 selected midfielder candidates.

For each player, extract:
- name: player's full name
- age: player's age (integer)
- nationality: player's nationality
- current_club: name of the player's current club
- market_value: market value as stated (include currency if present)
- contract_expiry: contract expiry date as stated
- market_value_trend: the trend description (e.g., "Rising", "Stable", "Declining")
- national_team_appearances: number of national team caps (integer)
- current_season_playing_time: playing time in minutes as stated
- goals: number of goals (integer)
- assists: number of assists (integer)
- transfer_rumours: whether transfer rumours exist (e.g., "Yes", "No")
- transfer_probability: transfer probability if mentioned (e.g., percentage or "N/A")
- transfermarkt_link: URL to the player's Transfermarkt profile
- club_website_link: URL to the club's official website data source

If any field is missing for a player, set it to null. Return all players found in the answer.
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


def looks_like_euro_value(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['€', 'eur', 'million', 'm'])


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        r'(january|february|march|april|may|june|july|august|september|october|november|december)'
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def looks_like_transfermarkt_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'transfermarkt' in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or 'www.' in text.lower())


def is_valid_trend(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['rising', 'stable', 'declining', 'up', 'down', 'flat'])


def date_before_june_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    # Lenient check: look for year <= 2026 and if 2026, month should suggest before June
    year_match = re.search(r'20(\d{2})', text)
    if not year_match:
        return False
    year = int(year_match.group(1)) + 2000
    if year < 2026:
        return True
    if year == 2026:
        # Check if month is mentioned and is before June
        month_before_june = has_any_ci(text, ['jan', 'feb', 'mar', 'apr', 'may', 'january', 'february', 'march', 'april'])
        return month_before_june
    return False


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
    players_data = await evaluator.extract(
        prompt=prompt_extract_players_from_answer(),
        template_class=PlayersCollection,
        extraction_name="players_collection"
    )

    players = players_data.players if players_data and players_data.players else []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Initial search and filtering section
    search_filter_node = evaluator.add_sequential(
        id="search_filter_section",
        desc="Transfermarkt search and multi-criteria filtering to find 8 midfielder candidates",
        parent=root,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A1 - Position filter (Midfielder)
    position_mentioned = has_any_ci(answer, ['midfielder', 'midfield'])
    evaluator.add_custom_node(
        result=bool(position_mentioned and len(players) > 0),
        id="filter_position",
        desc="[Action Node] transfermarkt.com:F1:A1 - Filter by position: Midfielder",
        parent=search_filter_node,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A5 - Age range filter (under 25)
    ages_valid = all(p.age and p.age <= 25 for p in players if p.age is not None)
    age_mentioned = has_any_ci(answer, ['age', '25', 'under 25'])
    evaluator.add_custom_node(
        result=bool(ages_valid and len(players) > 0 and age_mentioned),
        id="filter_age",
        desc="[Action Node] transfermarkt.com:F1:A5 - Filter by age: under 25 years old",
        parent=search_filter_node,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A4 - Market value range filter (€3M-€5M)
    market_values_in_range = []
    for p in players:
        if p.market_value:
            val = extract_float(p.market_value)
            if val and 3 <= val <= 5:
                market_values_in_range.append(True)
            elif val:
                market_values_in_range.append(False)

    market_value_ok = len(market_values_in_range) > 0 and all(market_values_in_range)
    evaluator.add_custom_node(
        result=bool(market_value_ok and has_any_ci(answer, ['market value', '€', 'million'])),
        id="filter_market_value",
        desc="[Action Node] transfermarkt.com:F1:A4 - Filter by market value: €3M to €5M range",
        parent=search_filter_node,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A6 - Contract expiry date filter (before June 2026)
    contracts_valid = all(
        date_before_june_2026(p.contract_expiry)
        for p in players
        if p.contract_expiry is not None
    )
    contract_mentioned = has_any_ci(answer, ['contract', 'expir', '2026'])
    evaluator.add_custom_node(
        result=bool(contracts_valid and len(players) > 0 and contract_mentioned),
        id="filter_contract_expiry",
        desc="[Action Node] transfermarkt.com:F1:A6 - Filter by contract expiry: before June 2026",
        parent=search_filter_node,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A17 - Search result pagination (finding 8 candidates)
    found_eight_candidates = has_any_ci(answer, ['8', 'eight']) or len(players) >= 5
    evaluator.add_custom_node(
        result=bool(found_eight_candidates),
        id="search_pagination",
        desc="[Action Node] transfermarkt.com:F1:A17 - Navigate through search results to find 8 candidates",
        parent=search_filter_node,
        critical=False
    )

    # 3.2 Player detail exploration section
    detail_exploration_node = evaluator.add_sequential(
        id="detail_exploration_section",
        desc="Navigate to each player's profile page and extract detailed information",
        parent=root,
        critical=False
    )

    # [Action Node] transfermarkt.com:F1:A19 - Click into player detail pages
    has_transfermarkt_links = any(
        p.transfermarkt_link and looks_like_transfermarkt_url(p.transfermarkt_link)
        for p in players
    )
    evaluator.add_custom_node(
        result=bool(has_transfermarkt_links and len(players) > 0),
        id="navigate_to_detail_pages",
        desc="[Action Node] transfermarkt.com:F1:A19 - Click into each player's profile page from search results",
        parent=detail_exploration_node,
        critical=False
    )

    # [Perception Node] transfermarkt.com:F2:P1 - Market value trend determination
    trends_valid = all(
        is_valid_trend(p.market_value_trend)
        for p in players
        if p.market_value_trend is not None
    )
    trend_mentioned = has_any_ci(answer, ['trend', 'rising', 'stable', 'declining'])
    evaluator.add_custom_node(
        result=bool(trends_valid and len(players) > 0 and trend_mentioned),
        id="perceive_market_value_trend",
        desc="[Perception Node] transfermarkt.com:F2:P1 - Examine market value history curve to determine trend",
        parent=detail_exploration_node,
        critical=False
    )

    # [Perception Node] transfermarkt.com:F2:P3 - National team statistics (at least 5 caps)
    national_team_valid = all(
        p.national_team_appearances and p.national_team_appearances >= 5
        for p in players
        if p.national_team_appearances is not None
    )
    national_team_mentioned = has_any_ci(answer, ['national team', 'international', 'caps'])
    evaluator.add_custom_node(
        result=bool(national_team_valid and len(players) > 0 and national_team_mentioned),
        id="perceive_national_team_stats",
        desc="[Perception Node] transfermarkt.com:F2:P3 - Check national team statistics for at least 5 caps",
        parent=detail_exploration_node,
        critical=False
    )

    # [Action Node] transfermarkt.com:F2:A8 - Switch to Transfer Rumours tab
    rumours_mentioned = has_any_ci(answer, ['rumour', 'rumor', 'transfer'])
    has_rumour_data = any(p.transfer_rumours is not None for p in players)
    evaluator.add_custom_node(
        result=bool(rumours_mentioned and has_rumour_data),
        id="switch_to_rumours_tab",
        desc="[Action Node] transfermarkt.com:F2:A8 - Switch to 'Transfer Rumours' tab",
        parent=detail_exploration_node,
        critical=False
    )

    # [Perception Node] transfermarkt.com:F6:P13 - Transfer rumour probability perception
    has_probability_data = any(p.transfer_probability is not None for p in players)
    probability_mentioned = has_any_ci(answer, ['probability', '%', 'percent', 'chance'])
    evaluator.add_custom_node(
        result=bool(has_probability_data and (probability_mentioned or rumours_mentioned)),
        id="perceive_transfer_probability",
        desc="[Perception Node] transfermarkt.com:F6:P13 - Extract community-assessed transfer probability from rumours",
        parent=detail_exploration_node,
        critical=False
    )

    # 3.3 Club website verification section
    club_website_node = evaluator.add_sequential(
        id="club_website_section",
        desc="Visit each player's club official website to verify playing time",
        parent=root,
        critical=False
    )

    has_club_links = any(p.club_website_link is not None for p in players)
    playing_time_valid = all(
        p.current_season_playing_time and extract_float(p.current_season_playing_time) and extract_float(p.current_season_playing_time) >= 1000
        for p in players
        if p.current_season_playing_time is not None
    )
    evaluator.add_custom_node(
        result=bool(has_club_links and playing_time_valid and len(players) > 0),
        id="verify_playing_time_club_website",
        desc="Visit club official websites to verify current season playing time (at least 1000 minutes)",
        parent=club_website_node,
        critical=False
    )

    # 3.4 Final sorting and selection section
    final_selection_node = evaluator.add_sequential(
        id="final_selection_section",
        desc="Sort by market value and select top 5 value-for-money players",
        parent=root,
        critical=False
    )

    # [Action Node] transfermarkt.com:F7:A12 - Sort by market value (low to high)
    market_values_sorted = True
    player_values = []
    for p in players:
        if p.market_value:
            val = extract_float(p.market_value)
            if val:
                player_values.append(val)

    if len(player_values) >= 2:
        market_values_sorted = all(player_values[i] <= player_values[i+1] for i in range(len(player_values)-1))

    evaluator.add_custom_node(
        result=bool(market_values_sorted and len(players) > 0),
        id="sort_by_market_value",
        desc="[Action Node] transfermarkt.com:F7:A12 - Sort players by market value from lowest to highest",
        parent=final_selection_node,
        critical=False
    )

    selected_five = len(players) == 5 or has_any_ci(answer, ['top 5', 'best 5', 'five'])
    evaluator.add_custom_node(
        result=bool(selected_five and len(players) > 0),
        id="select_top_five",
        desc="Select top 5 players with best value-for-money",
        parent=final_selection_node,
        critical=False
    )

    # 3.5 Output completeness section
    output_node = evaluator.add_parallel(
        id="output_completeness_section",
        desc="Verify all required output fields are present for each selected player",
        parent=root,
        critical=False
    )

    required_fields_present = len(players) > 0 and all(
        p.name and p.age and p.nationality and p.current_club and
        p.market_value and p.contract_expiry and p.market_value_trend and
        p.national_team_appearances and p.current_season_playing_time and
        p.transfermarkt_link
        for p in players
    )

    evaluator.add_custom_node(
        result=bool(required_fields_present),
        id="all_required_fields",
        desc="All required fields present for each selected player",
        parent=output_node,
        critical=False
    )

    has_performance_stats = any(
        (p.goals is not None or p.assists is not None)
        for p in players
    )
    evaluator.add_custom_node(
        result=bool(has_performance_stats),
        id="performance_statistics",
        desc="Performance statistics (goals/assists) included",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
