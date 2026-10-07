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
TASK_ID = "task-11ffd9"
TASK_DESCRIPTION = "As an avid NBA 2K player, I feel Klay Thompson's recent three-point rating in the game is too low, and I intend to post on a forum to discuss this.\n\nFirst, please check IGN for recent news on NBA 2K25 player rating updates or find relevant top-player rating articles. Confirm Klay Thompson's current three-point rating and note if the articles mention any changes to his numerical value.\n\nNext, go to Basketball-Reference to retrieve his real-world statistics from his most recent 20 regular season games. I specifically need his '3-point field goal percentage (3P%)' and '3-point attempts per game (3PA)'. It would be even better if you could find shot charts or zone shooting percentages for these 20 games. This information will help me ascertain whether his performance has genuinely declined or if the game's rating is unwarranted.\n\nFinally, synthesize the data from both sources and inform me whether his in-game rating accurately reflects his recent real-world performance."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class KlayThompsonGameRating(BaseModel):
    """Klay Thompson's three-point rating extracted from IGN NBA 2K25 coverage"""
    three_point_rating: Optional[str] = None
    rating_change_mentioned: Optional[bool] = None


class KlayThompsonRealStats(BaseModel):
    """Klay Thompson's real-world stats from Basketball-Reference for recent 20 games"""
    three_point_percentage: Optional[str] = None
    three_point_attempts_per_game: Optional[str] = None
    shot_chart_or_zone_info_present: Optional[bool] = None


class RatingAnalysis(BaseModel):
    """Analysis comparing game rating to real performance"""
    comparison_provided: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_game_rating_from_answer() -> str:
    return """
Extract Klay Thompson's three-point rating in NBA 2K25 as reported in the answer from IGN coverage.

Return:
- three_point_rating: the numerical three-point rating value exactly as stated (e.g., "88", "87", etc.). If not present, set null.
- rating_change_mentioned: true if the answer mentions any changes or updates to his rating, false if no change mentioned, null if unclear.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_real_stats_from_answer() -> str:
    return """
From the answer, extract Klay Thompson's real-world statistics from Basketball-Reference for his most recent 20 regular season games:

- three_point_percentage: his 3P% exactly as stated (include percentage symbol if present).
- three_point_attempts_per_game: his 3PA exactly as stated (include any units if present).
- shot_chart_or_zone_info_present: true if the answer mentions shot charts, zone shooting percentages, or heat maps; false otherwise.

If any field is missing, set it to null.
"""


def prompt_extract_analysis_from_answer() -> str:
    return """
From the answer, determine if a comparison or analysis was provided between Klay Thompson's in-game NBA 2K rating and his recent real-world performance.

Return:
- comparison_provided: true if the answer synthesizes both sources and provides a conclusion about whether the rating reflects his performance; false otherwise.
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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_rating(text: Optional[str]) -> bool:
    """Check if text looks like a game rating (typically 0-99 range)"""
    if not text:
        return False
    num = extract_number(text)
    if num is None:
        return False
    return 0 <= num <= 99


def looks_like_percentage(text: Optional[str]) -> bool:
    """Check if text looks like a percentage (has number and % or mentions percentage)"""
    if not text:
        return False
    has_num = contains_digits(text)
    has_percent = '%' in text or ci_contains(text, 'percent')
    return has_num and has_percent


def looks_like_attempts_per_game(text: Optional[str]) -> bool:
    """Check if text looks like attempts per game (has a number)"""
    if not text:
        return False
    return contains_digits(text)


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
    game_rating_info = await evaluator.extract(
        prompt=prompt_extract_game_rating_from_answer(),
        template_class=KlayThompsonGameRating,
        extraction_name="klay_game_rating"
    )

    real_stats_info = await evaluator.extract(
        prompt=prompt_extract_real_stats_from_answer(),
        template_class=KlayThompsonRealStats,
        extraction_name="klay_real_stats"
    )

    analysis_info = await evaluator.extract(
        prompt=prompt_extract_analysis_from_answer(),
        template_class=RatingAnalysis,
        extraction_name="rating_analysis"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IGN section - NBA 2K25 rating lookup
    ign_node = evaluator.add_sequential(
        id="ign_section",
        desc="IGN: NBA 2K25 player rating lookup for Klay Thompson",
        parent=root,
        critical=False
    )

    # Check if IGN was visited and NBA 2K25 content accessed
    ign_visited = has_any_ci(answer, ['ign']) and has_any_ci(answer, ['nba 2k', '2k25', 'nba2k'])
    evaluator.add_custom_node(
        result=bool(ign_visited),
        id="ign_site_visited",
        desc="Visited IGN and accessed NBA 2K25 related content",
        parent=ign_node,
        critical=False
    )

    # [Action Node] ign.com:F1:A11 - Scroll through article to find specific rating
    # This ability is triggered implicitly by needing to find Klay's specific rating within articles
    scroll_implied = (has_any_ci(answer, ['klay thompson', 'thompson']) and
                      game_rating_info and game_rating_info.three_point_rating is not None)
    evaluator.add_custom_node(
        result=bool(scroll_implied),
        id="ign_scroll_for_rating",
        desc="[Action Node] ign.com:F1:A11 - Scroll through long article content to locate Klay Thompson's specific three-point rating value",
        parent=ign_node,
        critical=False
    )

    # Extract and validate the three-point rating
    rating_extracted = looks_like_rating(game_rating_info.three_point_rating) if game_rating_info else False
    evaluator.add_custom_node(
        result=bool(rating_extracted),
        id="ign_rating_extracted",
        desc="Extract Klay Thompson's three-point rating from IGN article",
        parent=ign_node,
        critical=False
    )

    # Check if rating changes were noted
    changes_noted = bool(game_rating_info and game_rating_info.rating_change_mentioned)
    evaluator.add_custom_node(
        result=bool(changes_noted),
        id="ign_rating_changes_noted",
        desc="Note if articles mention any changes to Klay Thompson's rating value",
        parent=ign_node,
        critical=False
    )

    # 3.2 Basketball-Reference section - Real-world stats
    bbref_node = evaluator.add_sequential(
        id="basketball_reference_section",
        desc="Basketball-Reference: Klay Thompson's recent 20-game statistics",
        parent=root,
        critical=False
    )

    # Check if Basketball-Reference was accessed
    bbref_visited = has_any_ci(answer, ['basketball-reference', 'basketball reference', 'bbref'])
    evaluator.add_custom_node(
        result=bool(bbref_visited),
        id="bbref_site_visited",
        desc="Visited Basketball-Reference for Klay Thompson's statistics",
        parent=bbref_node,
        critical=False
    )

    # [Action Node] basketball-reference.com:F1:A6 - Select specific rows (last 20 games)
    # This ability is triggered by the requirement to filter/select the most recent 20 games
    twenty_games_filtered = (has_any_ci(answer, ['20 games', 'recent 20', 'last 20', 'most recent']) and
                             real_stats_info and (real_stats_info.three_point_percentage or real_stats_info.three_point_attempts_per_game))
    evaluator.add_custom_node(
        result=bool(twenty_games_filtered),
        id="bbref_select_20_games",
        desc="[Action Node] basketball-reference.com:F1:A6 - Select or filter data for the most recent 20 regular season games in Game Log",
        parent=bbref_node,
        critical=False
    )

    # [Perception Node] basketball-reference.com:F1:P1 - Extract 3P% and 3PA from dense table
    # This tests the ability to locate correct columns in a data-dense statistics table
    three_p_pct_ok = looks_like_percentage(real_stats_info.three_point_percentage) if real_stats_info else False
    three_pa_ok = looks_like_attempts_per_game(real_stats_info.three_point_attempts_per_game) if real_stats_info else False

    evaluator.add_custom_node(
        result=bool(three_p_pct_ok and three_pa_ok),
        id="bbref_extract_3p_stats",
        desc="[Perception Node] basketball-reference.com:F1:P1 - Extract 3P% and 3PA from dense statistical table for the 20-game span",
        parent=bbref_node,
        critical=False
    )

    # [Perception Node] basketball-reference.com:F1:P5 - Find and interpret shot charts/zone percentages
    # This tests visual perception of shot chart graphics and understanding color/zone meanings
    shot_chart_found = bool(real_stats_info and real_stats_info.shot_chart_or_zone_info_present)
    evaluator.add_custom_node(
        result=bool(shot_chart_found),
        id="bbref_shot_chart_perception",
        desc="[Perception Node] basketball-reference.com:F1:P5 - Locate and interpret shot charts or zone shooting percentages with visual/color understanding",
        parent=bbref_node,
        critical=False
    )

    # 3.3 Synthesis and Analysis
    analysis_node = evaluator.add_sequential(
        id="analysis_section",
        desc="Synthesis: Compare in-game rating to real-world performance",
        parent=root,
        critical=False
    )

    # Check if both data sources are mentioned together
    both_sources = (has_any_ci(answer, ['ign', 'nba 2k']) and
                    has_any_ci(answer, ['basketball-reference', 'basketball reference']))
    evaluator.add_custom_node(
        result=bool(both_sources),
        id="both_sources_present",
        desc="Answer references both IGN game rating and Basketball-Reference real stats",
        parent=analysis_node,
        critical=False
    )

    # Check if comparison/conclusion was provided
    comparison_made = bool(analysis_info and analysis_info.comparison_provided)
    evaluator.add_custom_node(
        result=bool(comparison_made),
        id="rating_comparison_provided",
        desc="Synthesize data from both sources and provide conclusion about rating accuracy",
        parent=analysis_node,
        critical=False
    )

    # Optional: Check if the answer discusses whether rating reflects performance
    discusses_accuracy = has_any_ci(answer, ['accurately', 'reflect', 'warranted', 'justified', 'appropriate', 'too low', 'too high'])
    evaluator.add_custom_node(
        result=bool(discusses_accuracy),
        id="discusses_rating_accuracy",
        desc="Discusses whether the in-game rating accurately reflects real-world performance",
        parent=analysis_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
