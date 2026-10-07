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
TASK_ID = "task-883008"
TASK_DESCRIPTION = 'I’m a die-hard San Francisco Giants fan and I’m planning a pilgrimage to Oracle Park during the next ticket-available part of the season.  \nPlease first check a **rolling 15-day window** in that period and identify which Giants home games are available. I want to attend a game against the **Los Angeles Dodgers** or **New York Mets** if possible (prefer a weekend game). If there is no game against those two teams, pick any **weekend home game**. If no suitable home game exists in the current 15-day window, move forward to the **next 15-day window within the same season** and continue searching.\n\nAfter confirming the game date, use **Oracle Park as the center point** and find a **3-star or higher hotel within a 15-minute walk** on Google Maps. Then check that hotel on TripAdvisor, filter reviews mentioning **“safety/safe”** or **“noise/quiet,”** and determine whether the sentiment is positive. Finally, find a **well-rated (4.0+ stars) seafood or American restaurant within a 10-minute walk** of the ballpark for a post-game celebration.\n\nOutput required:\n- Game date and opponent  \n- Ballpark location  \n- Hotel name and star rating  \n- Walking time from hotel to ballpark  \n- Summary of TripAdvisor safety/noise review findings  \n- Recommended restaurant name and rating  \n- Verification links for each step'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Game details extracted from the answer"""
    game_date: Optional[str] = None
    opponent: Optional[str] = None
    is_home_game: Optional[bool] = None
    is_weekend: Optional[bool] = None


class BallparkInfo(BaseModel):
    """Ballpark location extracted from the answer"""
    ballpark_name: Optional[str] = None
    ballpark_location: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    star_rating: Optional[str] = None
    walking_time_to_ballpark: Optional[str] = None


class TripAdvisorReviewInfo(BaseModel):
    """TripAdvisor review findings extracted from the answer"""
    safety_review_summary: Optional[str] = None
    noise_review_summary: Optional[str] = None
    overall_sentiment: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Restaurant details extracted from the answer"""
    restaurant_name: Optional[str] = None
    rating: Optional[str] = None
    cuisine_type: Optional[str] = None
    walking_time_from_ballpark: Optional[str] = None


class VerificationLinks(BaseModel):
    """Verification links extracted from the answer"""
    espn_link: Optional[str] = None
    google_maps_hotel_link: Optional[str] = None
    tripadvisor_hotel_link: Optional[str] = None
    restaurant_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_game_info() -> str:
    return """
Extract the Giants home game information from the answer:

- game_date: the exact date of the selected game
- opponent: the opposing team name
- is_home_game: whether it's explicitly mentioned or confirmed as a home game (true/false)
- is_weekend: whether the game is on a weekend (Saturday or Sunday) (true/false)

If any field is missing, set it to null.
"""


def prompt_extract_ballpark_info() -> str:
    return """
Extract the ballpark location information from the answer:

- ballpark_name: the name of the ballpark (should be Oracle Park)
- ballpark_location: the address or location description

If any field is missing, set it to null.
"""


def prompt_extract_hotel_info() -> str:
    return """
Extract the hotel information from the answer:

- hotel_name: the name of the recommended hotel
- star_rating: the star rating (e.g., "3-star", "4-star", or numeric like "3", "4")
- walking_time_to_ballpark: the walking time from hotel to Oracle Park (include units if present)

If any field is missing, set it to null.
"""


def prompt_extract_tripadvisor_review_info() -> str:
    return """
Extract the TripAdvisor review analysis from the answer:

- safety_review_summary: summary of reviews mentioning safety or safe
- noise_review_summary: summary of reviews mentioning noise or quiet
- overall_sentiment: whether the sentiment is positive, negative, or mixed

If any field is missing, set it to null.
"""


def prompt_extract_restaurant_info() -> str:
    return """
Extract the restaurant information from the answer:

- restaurant_name: the name of the recommended restaurant
- rating: the rating (should be 4.0 or higher)
- cuisine_type: the type of cuisine (should mention seafood or American)
- walking_time_from_ballpark: the walking time from Oracle Park to the restaurant

If any field is missing, set it to null.
"""


def prompt_extract_verification_links() -> str:
    return """
Extract all verification links mentioned in the answer:

- espn_link: link to ESPN schedule or game information
- google_maps_hotel_link: link to Google Maps for the hotel
- tripadvisor_hotel_link: link to TripAdvisor for the hotel
- restaurant_link: link to the restaurant (Google Maps or TripAdvisor)

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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def looks_like_time_minutes(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute'])


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.strip().startswith('http://') or text.strip().startswith('https://')


def mentions_dodgers_or_mets(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['dodgers', 'mets'])


def mentions_weekend(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['weekend', 'saturday', 'sunday'])


def mentions_oracle_park(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['oracle park'])


def mentions_giants(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['giants', 'sf giants', 'san francisco giants'])


def mentions_home_game(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['home game', 'home', 'vs', 'vs.'])


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
    game_info = await evaluator.extract(
        prompt=prompt_extract_game_info(),
        template_class=GameInfo,
        extraction_name="game_info"
    )

    ballpark_info = await evaluator.extract(
        prompt=prompt_extract_ballpark_info(),
        template_class=BallparkInfo,
        extraction_name="ballpark_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    tripadvisor_review_info = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_review_info(),
        template_class=TripAdvisorReviewInfo,
        extraction_name="tripadvisor_review_info"
    )

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant_info(),
        template_class=RestaurantInfo,
        extraction_name="restaurant_info"
    )

    verification_links = await evaluator.extract(
        prompt=prompt_extract_verification_links(),
        template_class=VerificationLinks,
        extraction_name="verification_links"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 ESPN schedule section
    espn_node = evaluator.add_sequential(
        id="espn_schedule_section",
        desc="ESPN - Find Giants home games in rolling 15-day windows",
        parent=root,
        critical=False
    )

    # [Action Node] espn.com:F5:A2 - Multi-condition filtering (date range and home team)
    espn_mentions = has_any_ci(answer, ['espn'])
    date_range_mentioned = has_any_ci(answer, ['15-day', '15 day', 'rolling window']) or contains_digits(answer)
    home_game_ok = mentions_home_game(answer) or (game_info and game_info.is_home_game)

    evaluator.add_custom_node(
        result=bool(espn_mentions and date_range_mentioned and home_game_ok),
        id="espn_action_multi_filter",
        desc="[Action Node] espn.com:F5:A2 - Filter games by date range and home team (Giants)",
        parent=espn_node,
        critical=False
    )

    # [Action Node] espn.com:F5:A6 - Tab switching to view different dates
    schedule_navigation_ok = has_any_ci(answer, ['schedule', 'calendar', 'date']) or (game_info and game_info.game_date)

    evaluator.add_custom_node(
        result=bool(schedule_navigation_ok),
        id="espn_action_tab_switch",
        desc="[Action Node] espn.com:F5:A6 - Navigate through date tabs to find games in the specified window",
        parent=espn_node,
        critical=False
    )

    # Check if game meets preference criteria (Dodgers/Mets or weekend)
    opponent_preference_ok = False
    if game_info and game_info.opponent:
        opponent_preference_ok = has_any_ci(game_info.opponent, ['dodgers', 'mets'])

    weekend_preference_ok = False
    if game_info and game_info.is_weekend:
        weekend_preference_ok = True

    evaluator.add_custom_node(
        result=bool(opponent_preference_ok or weekend_preference_ok),
        id="espn_game_preference_match",
        desc="Selected game matches preference (Dodgers/Mets or weekend home game)",
        parent=espn_node,
        critical=False
    )

    # 3.2 Google Maps hotel section
    maps_hotel_node = evaluator.add_sequential(
        id="google_maps_hotel_section",
        desc="Google Maps - Find 3+ star hotel within 15-minute walk of Oracle Park",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    maps_mentioned = has_any_ci(answer, ['google maps', 'maps'])
    oracle_park_center = mentions_oracle_park(answer)
    hotel_search_ok = hotel_info and hotel_info.hotel_name

    evaluator.add_custom_node(
        result=bool(maps_mentioned and oracle_park_center and hotel_search_ok),
        id="maps_action_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Set Oracle Park as starting point and search for hotels",
        parent=maps_hotel_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Walking mode selection
    walking_time = extract_int(hotel_info.walking_time_to_ballpark if hotel_info else None)
    walking_ok = walking_time is not None and walking_time <= 15
    walking_mentioned = has_any_ci(answer, ['walk', 'walking'])

    evaluator.add_custom_node(
        result=bool(walking_ok and walking_mentioned),
        id="maps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Calculate walking time (should be ≤15 minutes)",
        parent=maps_hotel_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Star rating perception
    star_rating_num = extract_float(hotel_info.star_rating if hotel_info else None)
    star_rating_ok = star_rating_num is not None and star_rating_num >= 3
    star_mentioned = has_any_ci(answer, ['star', 'rating'])

    evaluator.add_custom_node(
        result=bool(star_rating_ok and star_mentioned),
        id="maps_perception_star_rating",
        desc="[Perception Node] maps.google.com:F1:P1 - Identify hotel with 3+ star rating",
        parent=maps_hotel_node,
        critical=False
    )

    # 3.3 TripAdvisor hotel review section
    tripadvisor_hotel_node = evaluator.add_sequential(
        id="tripadvisor_hotel_section",
        desc="TripAdvisor - Verify hotel safety and noise reviews",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A1 - Multi-field search
    tripadvisor_mentioned = has_any_ci(answer, ['tripadvisor'])
    hotel_search_tripadvisor = tripadvisor_mentioned and hotel_search_ok

    evaluator.add_custom_node(
        result=bool(hotel_search_tripadvisor),
        id="tripadvisor_action_hotel_search",
        desc="[Action Node] tripadvisor.com:F1:A1 - Search for the selected hotel on TripAdvisor",
        parent=tripadvisor_hotel_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F2:A10 - Expand/filter reviews
    safety_review_ok = tripadvisor_review_info and tripadvisor_review_info.safety_review_summary
    noise_review_ok = tripadvisor_review_info and tripadvisor_review_info.noise_review_summary
    review_keywords_ok = has_any_ci(answer, ['safety', 'safe', 'noise', 'quiet'])

    evaluator.add_custom_node(
        result=bool((safety_review_ok or noise_review_ok) and review_keywords_ok),
        id="tripadvisor_action_review_filter",
        desc="[Action Node] tripadvisor.com:F2:A10 - Filter or expand reviews mentioning safety/noise keywords",
        parent=tripadvisor_hotel_node,
        critical=False
    )

    # Check sentiment analysis
    sentiment_ok = tripadvisor_review_info and has_any_ci(tripadvisor_review_info.overall_sentiment, ['positive'])

    evaluator.add_custom_node(
        result=bool(sentiment_ok),
        id="tripadvisor_sentiment_positive",
        desc="TripAdvisor reviews show positive sentiment regarding safety/noise",
        parent=tripadvisor_hotel_node,
        critical=False
    )

    # 3.4 Restaurant search section
    restaurant_node = evaluator.add_sequential(
        id="restaurant_section",
        desc="Find 4.0+ rated seafood or American restaurant within 10-minute walk",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-condition filtering (rating and cuisine)
    restaurant_found = restaurant_info and restaurant_info.restaurant_name
    rating_num = extract_float(restaurant_info.rating if restaurant_info else None)
    rating_ok = rating_num is not None and rating_num >= 4.0
    cuisine_ok = restaurant_info and has_any_ci(restaurant_info.cuisine_type, ['seafood', 'american'])

    evaluator.add_custom_node(
        result=bool(restaurant_found and rating_ok and cuisine_ok),
        id="restaurant_action_multi_filter",
        desc="[Action Node] tripadvisor.com:F1:A4 - Filter restaurants by rating (4.0+) and cuisine type (seafood/American)",
        parent=restaurant_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P8 - Map location awareness
    restaurant_walking_time = extract_int(restaurant_info.walking_time_from_ballpark if restaurant_info else None)
    restaurant_location_ok = restaurant_walking_time is not None and restaurant_walking_time <= 10
    ballpark_proximity_mentioned = has_any_ci(answer, ['ballpark', 'oracle park', 'near', 'walk'])

    evaluator.add_custom_node(
        result=bool(restaurant_location_ok and ballpark_proximity_mentioned),
        id="restaurant_perception_location",
        desc="[Perception Node] tripadvisor.com:F2:P8 - Verify restaurant is within 10-minute walk of Oracle Park",
        parent=restaurant_node,
        critical=False
    )

    # 3.5 Verification links section
    verification_node = evaluator.add_parallel(
        id="verification_links_section",
        desc="Verification links for each step",
        parent=root,
        critical=False
    )

    espn_link_ok = is_valid_url(verification_links.espn_link if verification_links else None)
    evaluator.add_custom_node(
        result=bool(espn_link_ok),
        id="verification_espn_link",
        desc="ESPN schedule verification link provided",
        parent=verification_node,
        critical=False
    )

    maps_link_ok = is_valid_url(verification_links.google_maps_hotel_link if verification_links else None)
    evaluator.add_custom_node(
        result=bool(maps_link_ok),
        id="verification_maps_link",
        desc="Google Maps hotel verification link provided",
        parent=verification_node,
        critical=False
    )

    tripadvisor_link_ok = is_valid_url(verification_links.tripadvisor_hotel_link if verification_links else None)
    evaluator.add_custom_node(
        result=bool(tripadvisor_link_ok),
        id="verification_tripadvisor_link",
        desc="TripAdvisor hotel verification link provided",
        parent=verification_node,
        critical=False
    )

    restaurant_link_ok = is_valid_url(verification_links.restaurant_link if verification_links else None)
    evaluator.add_custom_node(
        result=bool(restaurant_link_ok),
        id="verification_restaurant_link",
        desc="Restaurant verification link provided",
        parent=verification_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
