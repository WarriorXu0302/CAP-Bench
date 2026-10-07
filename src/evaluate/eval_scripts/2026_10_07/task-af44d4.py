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
TASK_ID = "task-af44d4"
TASK_DESCRIPTION = "I'm a huge Lakers fan and plan to attend an away game in February 2026. Please check the Los Angeles Lakers' 2025-26 season schedule on Basketball-Reference.com to find their first away game in February. Once the opponent and arena are identified, locate the arena on Google Maps. Then, find a hotel within a 15-minute walking distance (approximately 0.8 miles) of the arena with a rating of 4.0 or higher. After confirming the hotel's location, use maps or search engines to find the arena's official website. Review their audience entry and security regulations, specifically focusing on backpack size restrictions. Finally, provide me with the game date, opponent, arena name, the selected hotel's name and rating, the precise walking time from the hotel to the arena, and the detailed backpack security policy of the arena."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Game details extracted from the answer"""
    game_date: Optional[str] = None
    opponent: Optional[str] = None
    arena_name: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    hotel_rating: Optional[str] = None


class WalkingInfo(BaseModel):
    """Walking distance/time information extracted from the answer"""
    walking_time: Optional[str] = None
    walking_distance: Optional[str] = None


class SecurityPolicy(BaseModel):
    """Arena security policy details extracted from the answer"""
    backpack_policy: Optional[str] = None
    arena_website: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_game_info() -> str:
    return """
Extract the Lakers' first February 2026 away game information from the answer.

Return:
- game_date: the date of the game exactly as stated (e.g., "February 3, 2026", "Feb 3", "2/3/2026"). If not present, set null.
- opponent: the opposing team name exactly as written. If not present, set null.
- arena_name: the arena/venue name exactly as written. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_hotel_info() -> str:
    return """
Extract the hotel information from the answer.

Return:
- hotel_name: the name of the selected hotel exactly as stated. If not present, set null.
- hotel_rating: the hotel's rating exactly as stated (e.g., "4.5", "4.3 stars"). If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_walking_info() -> str:
    return """
Extract the walking distance and time information between the hotel and arena from the answer.

Return:
- walking_time: the walking time exactly as stated (e.g., "12 minutes", "10 min"). If not present, set null.
- walking_distance: the walking distance exactly as stated (e.g., "0.6 miles", "0.7 mi"). If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_security_policy() -> str:
    return """
Extract the arena's backpack security policy from the answer.

Return:
- backpack_policy: the backpack size restrictions or security policy text exactly as stated. If not present, set null.
- arena_website: the arena's official website URL if mentioned. If not present, set null.

If any field is missing, set it to null.
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


def looks_like_february_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    has_feb = any(x in t for x in ['february', 'feb'])
    has_2026 = '2026' in t
    return has_feb or has_2026


def looks_like_rating_above_4(text: Optional[str]) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    if rating is None:
        return False
    return rating >= 4.0


def looks_like_walking_time_minutes(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_time_unit = has_any_ci(text, ['min', 'minute', 'minutes'])
    return has_number and has_time_unit


def looks_like_distance_miles(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_dist_unit = has_any_ci(text, ['mile', 'miles', 'mi'])
    return has_number and has_dist_unit


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://|www\.', text, re.IGNORECASE))


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
    Restrict evaluator.verify to at most one usage.
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
    game_info = await evaluator.extract(
        prompt=prompt_extract_game_info(),
        template_class=GameInfo,
        extraction_name="game_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    walking_info = await evaluator.extract(
        prompt=prompt_extract_walking_info(),
        template_class=WalkingInfo,
        extraction_name="walking_info"
    )

    security_policy = await evaluator.extract(
        prompt=prompt_extract_security_policy(),
        template_class=SecurityPolicy,
        extraction_name="security_policy"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Basketball-Reference section
    basketball_ref_node = evaluator.add_sequential(
        id="basketball_reference_section",
        desc="Basketball-Reference.com - Lakers 2025-26 schedule navigation and game identification",
        parent=root,
        critical=False
    )

    # [Action Node] basketball-reference.com:F2:A3 - Tab navigation to Schedule
    schedule_tab_ok = has_any_ci(answer, ['basketball-reference', 'basketball reference']) and has_any_ci(answer, ['schedule'])
    evaluator.add_custom_node(
        result=bool(schedule_tab_ok),
        id="basketball_ref_schedule_tab",
        desc="[Action Node] basketball-reference.com:F2:A3 - Navigate to the Schedule tab for Lakers 2025-26 season",
        parent=basketball_ref_node,
        critical=False
    )

    # [Perception Node] basketball-reference.com:F2:P1 - Table data understanding
    game_date_ok = bool(game_info.game_date and looks_like_february_2026(game_info.game_date))
    opponent_ok = bool(game_info.opponent and game_info.opponent.strip())
    away_game_ok = has_any_ci(answer, ['away', '@', 'road'])

    evaluator.add_custom_node(
        result=bool(game_date_ok and opponent_ok and away_game_ok),
        id="basketball_ref_first_feb_away",
        desc="[Perception Node] basketball-reference.com:F2:P1 - Identify first February 2026 away game from schedule table",
        parent=basketball_ref_node,
        critical=False
    )

    # 3.2 Google Maps - Arena location section
    maps_arena_node = evaluator.add_sequential(
        id="maps_arena_section",
        desc="Google Maps - Arena location and hotel search",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search for arena
    arena_name_ok = bool(game_info.arena_name and game_info.arena_name.strip())
    maps_search_ok = has_any_ci(answer, ['google maps', 'maps']) and arena_name_ok

    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="maps_arena_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for and locate the arena on Google Maps",
        parent=maps_arena_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P3 - Hotel rating extraction
    hotel_name_ok = bool(hotel_info.hotel_name and hotel_info.hotel_name.strip())
    hotel_rating_ok = looks_like_rating_above_4(hotel_info.hotel_rating)

    evaluator.add_custom_node(
        result=bool(hotel_name_ok and hotel_rating_ok),
        id="maps_hotel_rating",
        desc="[Perception Node] maps.google.com:F5:P3 - Extract hotel rating (4.0 or higher)",
        parent=maps_arena_node,
        critical=False
    )

    # 3.3 Google Maps - Route planning section
    maps_route_node = evaluator.add_sequential(
        id="maps_route_section",
        desc="Google Maps - Walking route planning and distance verification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection (walking)
    walking_mode_ok = has_any_ci(answer, ['walk', 'walking', 'on foot'])

    evaluator.add_custom_node(
        result=bool(walking_mode_ok),
        id="maps_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking mode for route planning",
        parent=maps_route_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    walking_time_ok = looks_like_walking_time_minutes(walking_info.walking_time)
    walking_dist_ok = looks_like_distance_miles(walking_info.walking_distance)
    route_planned_ok = walking_time_ok or walking_dist_ok

    evaluator.add_custom_node(
        result=bool(route_planned_ok),
        id="maps_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Plan walking route from hotel to arena",
        parent=maps_route_node,
        critical=False
    )

    # Extra check: 15-minute constraint verification (non-prefixed)
    time_value = extract_float(walking_info.walking_time) if walking_info.walking_time else None
    within_15min = time_value is not None and time_value <= 15

    evaluator.add_custom_node(
        result=bool(within_15min),
        id="maps_15min_constraint",
        desc="Verify walking time is within 15 minutes as required",
        parent=maps_route_node,
        critical=False
    )

    # 3.4 Arena official website section
    arena_website_node = evaluator.add_sequential(
        id="arena_website_section",
        desc="Arena official website - Security policy retrieval",
        parent=root,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P3 - Extract official website from Maps
    arena_website_ok = looks_like_url(security_policy.arena_website)
    website_mention_ok = has_any_ci(answer, ['official', 'website', 'site'])

    evaluator.add_custom_node(
        result=bool(arena_website_ok and website_mention_ok),
        id="maps_arena_website",
        desc="[Perception Node] maps.google.com:F5:P3 - Extract arena official website from Maps details",
        parent=arena_website_node,
        critical=False
    )

    # Security policy perception (non-prefixed)
    backpack_policy_ok = bool(security_policy.backpack_policy and security_policy.backpack_policy.strip())
    security_keywords_ok = has_any_ci(answer, ['backpack', 'bag', 'security', 'policy', 'restriction'])

    evaluator.add_custom_node(
        result=bool(backpack_policy_ok and security_keywords_ok),
        id="arena_backpack_policy",
        desc="Extract detailed backpack size restrictions from arena security policy",
        parent=arena_website_node,
        critical=False
    )

    # 3.5 Completeness check (non-prefixed)
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall completeness of required information",
        parent=root,
        critical=False
    )

    all_game_info_present = game_date_ok and opponent_ok and arena_name_ok
    evaluator.add_custom_node(
        result=bool(all_game_info_present),
        id="complete_game_info",
        desc="All game information provided (date, opponent, arena)",
        parent=completeness_node,
        critical=False
    )

    all_hotel_info_present = hotel_name_ok and hotel_rating_ok and route_planned_ok
    evaluator.add_custom_node(
        result=bool(all_hotel_info_present),
        id="complete_hotel_info",
        desc="All hotel information provided (name, rating, walking time/distance)",
        parent=completeness_node,
        critical=False
    )

    security_info_present = backpack_policy_ok
    evaluator.add_custom_node(
        result=bool(security_info_present),
        id="complete_security_info",
        desc="Arena security policy information provided",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
