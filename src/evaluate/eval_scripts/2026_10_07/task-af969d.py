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
TASK_ID = "task-af969d"
TASK_DESCRIPTION = 'I’m a San Francisco 49ers fan and plan to attend two home games in person this coming NFL regular season in December. First, check ESPN for the 49ers’ schedule for that season, identify two December home games at Levi’s Stadium, and record each game’s date and opponent. For the first game, use Google Maps to plan a driving route from San Francisco International Airport (SFO) to Levi’s Stadium and confirm the estimated driving time under smooth traffic conditions. Finally, search on Airbnb for accommodations near the stadium (Santa Clara area): set check-in to the day before the first game and check-out to the day after the game (2 nights total), apply a price filter of USD 150–400 per night, and select the “Wi-Fi” amenity. Find 3 listings tagged as “Guest Favorite.” If there are fewer than two home games in December for that season, supplement with the nearest home game(s) in November or January, while keeping all subsequent steps unchanged. Output: the dates and opponents of the two games, driving time from SFO to the stadium, and for 3 recommended listings, the name, nightly price, whether it is a Guest Favorite, and the Airbnb detail page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Information about a 49ers home game"""
    game_date: Optional[str] = None
    opponent: Optional[str] = None


class GamesSchedule(BaseModel):
    """Two home games extracted from the answer"""
    game1: Optional[GameInfo] = None
    game2: Optional[GameInfo] = None


class DrivingInfo(BaseModel):
    """Driving time from SFO to Levi's Stadium"""
    driving_time: Optional[str] = None


class AirbnbListing(BaseModel):
    """Single Airbnb listing details"""
    name: Optional[str] = None
    nightly_price: Optional[str] = None
    is_guest_favorite: Optional[bool] = None
    detail_link: Optional[str] = None


class AirbnbListings(BaseModel):
    """Three Airbnb listings extracted from the answer"""
    listing1: Optional[AirbnbListing] = None
    listing2: Optional[AirbnbListing] = None
    listing3: Optional[AirbnbListing] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games_from_answer() -> str:
    return """
Extract information about two San Francisco 49ers home games at Levi's Stadium from the answer.

Return:
- game1: the first game with its date and opponent
- game2: the second game with its date and opponent

For each game:
- game_date: the date exactly as written (include month and day/year if present)
- opponent: the opposing team name

If any field is missing in the answer, set it to null.
"""


def prompt_extract_driving_from_answer() -> str:
    return """
From the answer, extract the driving time from San Francisco International Airport (SFO) to Levi's Stadium.

Return:
- driving_time: the estimated driving time exactly as stated (include units like "min" or "minutes" if present)

If not present, set it to null.
"""


def prompt_extract_airbnb_from_answer() -> str:
    return """
From the answer, extract details about three Airbnb listings near Levi's Stadium / Santa Clara.

Return:
- listing1, listing2, listing3: each containing:
  - name: the listing name/title
  - nightly_price: the price per night (include currency if present)
  - is_guest_favorite: true if marked as "Guest Favorite", false otherwise
  - detail_link: the Airbnb URL for the listing

If any field is missing, set it to null (for is_guest_favorite, set to false if unclear).
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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_december_date(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['dec', 'december', '12/']) and contains_digits(text)


def looks_like_home_game(answer: str, game_date: Optional[str]) -> bool:
    if not game_date:
        return False
    context_window = answer.lower()
    return has_any_ci(context_window, ['home', 'levi', 'levis', 'stadium'])


def looks_like_driving_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour', 'hr'])


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar'])


def price_in_range(text: Optional[str], min_val: float = 150, max_val: float = 400) -> bool:
    if not text:
        return False
    price = extract_float(text)
    if price is None:
        return False
    return min_val <= price <= max_val


def looks_like_airbnb_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'airbnb' in text.lower() and ('http' in text.lower() or 'www' in text.lower())


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
    games_info = await evaluator.extract(
        prompt=prompt_extract_games_from_answer(),
        template_class=GamesSchedule,
        extraction_name="games_schedule"
    )

    driving_info = await evaluator.extract(
        prompt=prompt_extract_driving_from_answer(),
        template_class=DrivingInfo,
        extraction_name="driving_info"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_from_answer(),
        template_class=AirbnbListings,
        extraction_name="airbnb_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 ESPN section
    espn_node = evaluator.add_sequential(
        id="espn_section",
        desc="ESPN 49ers schedule - identify two December home games",
        parent=root,
        critical=False
    )

    # [Action Node] espn.com:F5:A2 - Multi-condition filtering (select 49ers schedule)
    espn_action_search_ok = (has_any_ci(answer, ['espn']) and
                             has_any_ci(answer, ['49ers', 'forty-niners', 'san francisco']))
    evaluator.add_custom_node(
        result=bool(espn_action_search_ok),
        id="espn_action_search",
        desc="[Action Node] espn.com:F5:A2 - Navigate to ESPN and locate the 49ers schedule",
        parent=espn_node,
        critical=False
    )

    # [Perception Node] espn.com:F5:P12 - Schedule information recognition (identify dates and home/away)
    game1 = games_info.game1 if games_info else None
    game2 = games_info.game2 if games_info else None

    game1_date_ok = game1 and game1.game_date and contains_digits(game1.game_date)
    game1_opponent_ok = game1 and game1.opponent and len(game1.opponent.strip()) > 0
    game2_date_ok = game2 and game2.game_date and contains_digits(game2.game_date)
    game2_opponent_ok = game2 and game2.opponent and len(game2.opponent.strip()) > 0

    games_identified_ok = game1_date_ok and game1_opponent_ok and game2_date_ok and game2_opponent_ok

    evaluator.add_custom_node(
        result=bool(games_identified_ok),
        id="espn_perception_schedule",
        desc="[Perception Node] espn.com:F5:P12 - Identify two home games with dates and opponents (December preferred, November/January if needed)",
        parent=espn_node,
        critical=False
    )

    # Additional check: mentions December or home context
    december_context = has_any_ci(answer, ['december', 'dec'])
    home_context = has_any_ci(answer, ['home', 'levi', 'levis stadium'])
    evaluator.add_custom_node(
        result=bool(december_context or home_context),
        id="espn_mentions_december_home",
        desc="Mentions December and/or home games at Levi's Stadium",
        parent=espn_node,
        critical=False
    )

    # 3.2 Google Maps section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps driving route from SFO to Levi's Stadium",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    maps_action_route_ok = (has_any_ci(answer, ['google maps', 'maps.google']) and
                            has_any_ci(answer, ['sfo', 'san francisco international', 'airport']) and
                            has_any_ci(answer, ['levi', 'stadium']))
    evaluator.add_custom_node(
        result=bool(maps_action_route_ok),
        id="maps_action_route",
        desc="[Action Node] maps.google.com:F2:A5 - Plan a route from SFO to Levi's Stadium on Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection (driving)
    driving_time = driving_info.driving_time if driving_info else None
    driving_time_ok = looks_like_driving_time(driving_time)
    driving_mode_ok = has_any_ci(answer, ['driv', 'car'])

    evaluator.add_custom_node(
        result=bool(driving_time_ok and driving_mode_ok),
        id="maps_action_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode and extract estimated driving time",
        parent=maps_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb search for accommodations near Levi's Stadium with filters",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Location search filter
    airbnb_location_ok = (has_any_ci(answer, ['airbnb']) and
                          has_any_ci(answer, ['santa clara', 'levi', 'stadium']))
    evaluator.add_custom_node(
        result=bool(airbnb_location_ok),
        id="airbnb_action_location",
        desc="[Action Node] airbnb.com:F1:A10 - Search Airbnb for accommodations near Levi's Stadium (Santa Clara area)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date range selection
    checkin_context = has_any_ci(answer, ['check-in', 'checkin', 'check in', 'day before'])
    nights_context = has_any_ci(answer, ['2 night', 'two night'])
    dates_ok = checkin_context or nights_context

    evaluator.add_custom_node(
        result=bool(dates_ok),
        id="airbnb_action_dates",
        desc="[Action Node] airbnb.com:F1:A6 - Set check-in to day before first game and check-out day after (2 nights)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price range slider
    listing1 = airbnb_info.listing1 if airbnb_info else None
    listing2 = airbnb_info.listing2 if airbnb_info else None
    listing3 = airbnb_info.listing3 if airbnb_info else None

    price1_ok = listing1 and listing1.nightly_price and price_in_range(listing1.nightly_price)
    price2_ok = listing2 and listing2.nightly_price and price_in_range(listing2.nightly_price)
    price3_ok = listing3 and listing3.nightly_price and price_in_range(listing3.nightly_price)

    price_filter_ok = (price1_ok or price2_ok or price3_ok) or has_any_ci(answer, ['150', '400', 'price filter'])

    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="airbnb_action_price",
        desc="[Action Node] airbnb.com:F2:A13 - Apply price filter of USD 150-400 per night",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Amenities multi-select
    wifi_filter_ok = has_any_ci(answer, ['wi-fi', 'wifi', 'wireless'])
    evaluator.add_custom_node(
        result=bool(wifi_filter_ok),
        id="airbnb_action_amenities",
        desc="[Action Node] airbnb.com:F2:A11 - Select Wi-Fi amenity filter",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P3 - Status awareness (Guest Favorite)
    gf1 = listing1 and listing1.is_guest_favorite
    gf2 = listing2 and listing2.is_guest_favorite
    gf3 = listing3 and listing3.is_guest_favorite

    guest_favorite_ok = (gf1 or gf2 or gf3) or has_any_ci(answer, ['guest favorite'])

    evaluator.add_custom_node(
        result=bool(guest_favorite_ok),
        id="airbnb_perception_guest_favorite",
        desc="[Perception Node] airbnb.com:F1:P3 - Identify listings tagged as Guest Favorite",
        parent=airbnb_node,
        critical=False
    )

    # Check completeness of listing information
    listing1_complete = (listing1 and listing1.name and listing1.nightly_price and
                        listing1.detail_link and looks_like_airbnb_url(listing1.detail_link))
    listing2_complete = (listing2 and listing2.name and listing2.nightly_price and
                        listing2.detail_link and looks_like_airbnb_url(listing2.detail_link))
    listing3_complete = (listing3 and listing3.name and listing3.nightly_price and
                        listing3.detail_link and looks_like_airbnb_url(listing3.detail_link))

    three_listings_ok = listing1_complete and listing2_complete and listing3_complete

    evaluator.add_custom_node(
        result=bool(three_listings_ok),
        id="airbnb_three_listings",
        desc="Provide 3 complete listings with name, price, Guest Favorite status, and Airbnb detail link",
        parent=airbnb_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
