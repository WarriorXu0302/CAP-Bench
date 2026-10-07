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
TASK_ID = "task-42c042"
TASK_DESCRIPTION = "I'd like to purchase three highly replayable and well-regarded games during the current Steam sale. My budget is limited, with each game not exceeding 50 USD.\n\nFirst, navigate to Steam's 'Specials' or 'Deals' section. Find several discounted games that are tagged with both 'RPG' and 'Indie'. Make sure to browse through multiple pages, not just the first one. Keep a record of these candidate games' names and their discounted prices.\n\nNext, visit HowLongToBeat.com to check the 'Main Story' duration for each candidate game. Eliminate any 'quick-play' games with a Main Story duration shorter than 15 hours. Subsequently, go to Metacritic.com for the remaining games. Check if their Metascore is 80 or above, or if they have received the 'Must-Play' golden badge.\n\nFinally, present the top three 'high-value' games that meet these criteria. For each selected game, provide its name, Steam discounted price, HLTB Main Story duration, Metacritic score, and your recommendation reason. If three perfect matches cannot be found, two suitable recommendations will suffice."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CandidateGames(BaseModel):
    """Candidate games extracted from Steam with RPG+Indie tags"""
    game_names: Optional[List[str]] = Field(default_factory=list)
    game_prices: Optional[List[str]] = Field(default_factory=list)
    num_games: Optional[int] = 0


class HLTBDurations(BaseModel):
    """Main Story durations from HowLongToBeat"""
    games_with_durations: Optional[List[Dict[str, str]]] = Field(default_factory=list)


class MetacriticScores(BaseModel):
    """Metacritic scores and Must-Play badges"""
    games_with_scores: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class FinalRecommendations(BaseModel):
    """Final game recommendations with all details"""
    recommendations: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    num_recommendations: Optional[int] = 0


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate_games() -> str:
    return """
Extract the candidate games the user found on Steam that are tagged with both 'RPG' and 'Indie' from the answer.

Return:
- game_names: list of game names mentioned as candidates
- game_prices: list of their discounted prices as written (include currency symbols if present)
- num_games: total count of candidate games mentioned

If no games are mentioned, return empty lists and 0 for num_games.
"""


def prompt_extract_hltb_durations() -> str:
    return """
Extract the HowLongToBeat Main Story durations mentioned in the answer for the candidate games.

Return:
- games_with_durations: list of dictionaries, each with 'game_name' and 'main_story_hours' keys

If no durations are mentioned, return an empty list.
"""


def prompt_extract_metacritic_scores() -> str:
    return """
Extract the Metacritic scores and Must-Play badge information mentioned in the answer.

Return:
- games_with_scores: list of dictionaries, each with 'game_name', 'metascore' (as string), and 'has_must_play_badge' (as boolean) keys

If no scores are mentioned, return an empty list.
"""


def prompt_extract_final_recommendations() -> str:
    return """
Extract the final game recommendations from the answer. Each recommendation should include the game name, Steam price, HLTB Main Story duration, Metacritic score, and recommendation reason.

Return:
- recommendations: list of dictionaries with keys 'game_name', 'steam_price', 'hltb_duration', 'metacritic_score', 'reason'
- num_recommendations: count of final recommendations (should be 2 or 3)

If no final recommendations are present, return empty list and 0.
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
    has_number = contains_digits(text)
    has_currency = has_any_ci(text, ['$', 'usd', 'dollar', '€', '£', 'eur', 'gbp'])
    return has_number and (has_currency or True)


def looks_like_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_time_unit = has_any_ci(text, ['hour', 'hours', 'hr', 'hrs', 'h'])
    return has_number and has_time_unit


def extract_hours(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h)', text.lower())
    if not m:
        return extract_float(text)
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_metascore(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 100


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
    candidates = await evaluator.extract(
        prompt=prompt_extract_candidate_games(),
        template_class=CandidateGames,
        extraction_name="candidate_games"
    )

    hltb_data = await evaluator.extract(
        prompt=prompt_extract_hltb_durations(),
        template_class=HLTBDurations,
        extraction_name="hltb_durations"
    )

    metacritic_data = await evaluator.extract(
        prompt=prompt_extract_metacritic_scores(),
        template_class=MetacriticScores,
        extraction_name="metacritic_scores"
    )

    final_recs = await evaluator.extract(
        prompt=prompt_extract_final_recommendations(),
        template_class=FinalRecommendations,
        extraction_name="final_recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Steam section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam Specials/Deals browsing with RPG+Indie tag filtering",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Multi-select RPG+Indie tags
    steam_mentions = has_any_ci(answer, ['steam'])
    specials_mentions = has_any_ci(answer, ['specials', 'deals', 'sale'])
    rpg_mentions = has_any_ci(answer, ['rpg'])
    indie_mentions = has_any_ci(answer, ['indie', 'independent'])
    tags_both = rpg_mentions and indie_mentions

    evaluator.add_custom_node(
        result=bool(steam_mentions and specials_mentions and tags_both),
        id="steam_action_tag_filtering",
        desc="[Action Node] store.steampowered.com:F1:A1 - Navigate to Steam Specials/Deals and filter by both RPG and Indie tags",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A8 - Browse multiple pages
    multiple_pages = has_any_ci(answer, ['multiple pages', 'several pages', 'page 2', 'page 3', 'next page', 'browsed through', 'not just the first'])

    evaluator.add_custom_node(
        result=bool(multiple_pages),
        id="steam_action_pagination",
        desc="[Action Node] store.steampowered.com:F1:A8 - Browse through multiple pages, not just the first one",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F1:P1 - Identify discounted prices
    has_candidates = candidates and candidates.num_games and candidates.num_games > 0
    prices_recorded = has_candidates and candidates.game_prices and len(candidates.game_prices) > 0
    prices_valid = False
    if prices_recorded:
        prices_valid = any(looks_like_price(p) for p in candidates.game_prices)

    evaluator.add_custom_node(
        result=bool(has_candidates and prices_recorded and prices_valid),
        id="steam_perception_discount_prices",
        desc="[Perception Node] store.steampowered.com:F1:P1 - Extract discounted prices for candidate games",
        parent=steam_node,
        critical=False
    )

    # Additional check: recorded game names
    names_recorded = has_candidates and candidates.game_names and len(candidates.game_names) > 0
    evaluator.add_custom_node(
        result=bool(names_recorded),
        id="steam_recorded_game_names",
        desc="Recorded candidate game names from Steam",
        parent=steam_node,
        critical=False
    )

    # 3.2 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="hltb_section",
        desc="HowLongToBeat Main Story duration checking",
        parent=root,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story durations
    hltb_mentions = has_any_ci(answer, ['howlongtobeat', 'hltb'])
    main_story_mentions = has_any_ci(answer, ['main story', 'main-story', 'main'])
    has_durations = hltb_data and hltb_data.games_with_durations and len(hltb_data.games_with_durations) > 0
    durations_valid = False
    if has_durations:
        durations_valid = any(
            looks_like_duration(d.get('main_story_hours', ''))
            for d in hltb_data.games_with_durations
        )

    evaluator.add_custom_node(
        result=bool(hltb_mentions and main_story_mentions and has_durations and durations_valid),
        id="hltb_perception_main_story",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract Main Story duration for each candidate game",
        parent=hltb_node,
        critical=False
    )

    # Check for 15-hour filtering
    fifteen_hour_filter = has_any_ci(answer, ['15 hour', '15-hour', 'shorter than 15', 'less than 15', 'at least 15'])
    evaluator.add_custom_node(
        result=bool(fifteen_hour_filter),
        id="hltb_filter_15hours",
        desc="Applied the 15-hour Main Story duration filter to eliminate quick-play games",
        parent=hltb_node,
        critical=False
    )

    # 3.3 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic score and Must-Play badge checking",
        parent=root,
        critical=False
    )

    # Check Metascore (80+)
    metacritic_mentions = has_any_ci(answer, ['metacritic'])
    metascore_mentions = has_any_ci(answer, ['metascore', 'score'])
    has_scores = metacritic_data and metacritic_data.games_with_scores and len(metacritic_data.games_with_scores) > 0
    scores_valid = False
    if has_scores:
        scores_valid = any(
            looks_like_metascore(d.get('metascore', ''))
            for d in metacritic_data.games_with_scores
        )

    evaluator.add_custom_node(
        result=bool(metacritic_mentions and metascore_mentions and has_scores and scores_valid),
        id="metacritic_perception_metascore",
        desc="Check Metacritic Metascore for remaining games (80+ criterion)",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.comgame:F1:P9 - Identify Must-Play badge
    must_play_mentions = has_any_ci(answer, ['must-play', 'must play', 'golden badge', 'gold badge'])
    badge_checked = False
    if has_scores:
        badge_checked = any(
            isinstance(d.get('has_must_play_badge'), bool)
            for d in metacritic_data.games_with_scores
        )

    evaluator.add_custom_node(
        result=bool(must_play_mentions or badge_checked),
        id="metacritic_perception_must_play_badge",
        desc="[Perception Node] metacritic.comgame:F1:P9 - Identify Must-Play golden badge for games",
        parent=metacritic_node,
        critical=False
    )

    # 3.4 Final recommendations section
    final_node = evaluator.add_sequential(
        id="final_recommendations_section",
        desc="Final top 3 (or 2) high-value game recommendations",
        parent=root,
        critical=False
    )

    # Check if final recommendations are present
    has_final = final_recs and final_recs.num_recommendations and final_recs.num_recommendations >= 2
    rec_count_ok = has_final and 2 <= final_recs.num_recommendations <= 3

    evaluator.add_custom_node(
        result=bool(rec_count_ok),
        id="final_recommendations_count",
        desc="Provided 2-3 final game recommendations",
        parent=final_node,
        critical=False
    )

    # Check completeness of each recommendation
    complete_recs = False
    if has_final and final_recs.recommendations:
        complete_recs = all(
            rec.get('game_name') and
            rec.get('steam_price') and
            rec.get('hltb_duration') and
            rec.get('metacritic_score') and
            rec.get('reason')
            for rec in final_recs.recommendations
        )

    evaluator.add_custom_node(
        result=bool(complete_recs),
        id="final_recommendations_completeness",
        desc="Each recommendation includes game name, Steam price, HLTB duration, Metacritic score, and reason",
        parent=final_node,
        critical=False
    )

    # Check price constraint (under 50 USD)
    price_constraint_ok = False
    if has_final and final_recs.recommendations:
        all_under_50 = True
        for rec in final_recs.recommendations:
            price_text = rec.get('steam_price', '')
            price_num = extract_float(price_text)
            if price_num is not None and price_num > 50:
                all_under_50 = False
                break
        price_constraint_ok = all_under_50

    evaluator.add_custom_node(
        result=bool(price_constraint_ok),
        id="final_price_constraint",
        desc="All recommended games are priced under 50 USD",
        parent=final_node,
        critical=False
    )

    # Check if recommendations meet criteria (15+ hours, 80+ score or Must-Play)
    criteria_met = False
    if has_final and final_recs.recommendations:
        all_meet_criteria = True
        for rec in final_recs.recommendations:
            duration_text = rec.get('hltb_duration', '')
            duration_hours = extract_hours(duration_text)
            score_text = rec.get('metacritic_score', '')
            score_num = extract_float(score_text)

            duration_ok = duration_hours is not None and duration_hours >= 15
            score_ok = score_num is not None and score_num >= 80
            must_play_ok = has_any_ci(rec.get('reason', ''), ['must-play', 'must play'])

            if not (duration_ok and (score_ok or must_play_ok)):
                all_meet_criteria = False
                break
        criteria_met = all_meet_criteria

    evaluator.add_custom_node(
        result=bool(criteria_met),
        id="final_criteria_validation",
        desc="Recommended games meet the criteria: 15+ hours main story, 80+ Metascore or Must-Play badge",
        parent=final_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
