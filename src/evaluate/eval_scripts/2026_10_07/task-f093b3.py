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
TASK_ID = "task-f093b3"
TASK_DESCRIPTION = 'I am planning an off-road photography trip to Moab, UT in my Jeep next weekend.\n\nFirst, use AllTrails to find an "Off-road driving" trail near Moab, UT. The trail must meet the following criteria: "Hard" difficulty, a rating above 4.5, and a length exceeding 10 miles. Select the trail with the highest number of reviews.\n\nAfter identifying the trail, go to Weather.com to check the probability of rain in Moab for the upcoming Saturday or Sunday. Confirm that the rain probability is below 20% for safety.\n\nFinally, on Google Maps, plan a driving route from the "Moab Information Center" to the trailhead of the selected off-road route. Utilize the "Search along the route" feature to locate a gas station with a rating of 4.0 or higher along this planned route.\n\nOutput: Trail Name, AllTrails Rating, Trail Length, Weather.com Rain Probability, Selected Gas Station Name and Rating, Google Maps Route Link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TrailInfo(BaseModel):
    """Trail information extracted from AllTrails"""
    trail_name: Optional[str] = None
    rating: Optional[str] = None
    length: Optional[str] = None
    difficulty: Optional[str] = None
    review_count: Optional[str] = None


class WeatherInfo(BaseModel):
    """Weather information extracted from Weather.com"""
    rain_probability: Optional[str] = None
    day_of_week: Optional[str] = None


class GasStationInfo(BaseModel):
    """Gas station information from Google Maps"""
    gas_station_name: Optional[str] = None
    gas_station_rating: Optional[str] = None


class RouteInfo(BaseModel):
    """Route information from Google Maps"""
    route_link: Optional[str] = None
    start_location: Optional[str] = None
    end_location: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trail_info() -> str:
    return """
Extract the AllTrails trail information from the answer for the off-road driving trail near Moab, UT.

Return:
- trail_name: the name of the trail exactly as written
- rating: the AllTrails rating exactly as written (include numbers and units if present)
- length: the trail length exactly as written (include units if present)
- difficulty: the difficulty level exactly as written
- review_count: the number of reviews exactly as written if mentioned

If any field is missing, set it to null.
"""


def prompt_extract_weather_info() -> str:
    return """
Extract the Weather.com rain probability information for Moab from the answer.

Return:
- rain_probability: the rain probability exactly as written (include percentage if present)
- day_of_week: which day (Saturday or Sunday) the rain probability refers to if mentioned

If any field is missing, set it to null.
"""


def prompt_extract_gas_station_info() -> str:
    return """
Extract the gas station information from Google Maps from the answer.

Return:
- gas_station_name: the name of the gas station exactly as written
- gas_station_rating: the rating of the gas station exactly as written (include numbers if present)

If any field is missing, set it to null.
"""


def prompt_extract_route_info() -> str:
    return """
Extract the Google Maps route information from the answer.

Return:
- route_link: the Google Maps route URL/link exactly as written
- start_location: the starting point of the route if mentioned
- end_location: the ending point of the route if mentioned

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


def looks_like_trail_name(text: Optional[str]) -> bool:
    if not text:
        return False
    # Known Moab trails or reasonable trail name format
    known_trails = ["fins and things", "hell's revenge", "hells revenge", "poison spider", "steel bender"]
    if has_any_ci(text, known_trails):
        return True
    # Generic trail name check: contains letters and reasonable length
    return len(text.strip()) > 3 and bool(re.search(r'[a-zA-Z]', text))


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_length_miles(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    has_mile_unit = has_any_ci(text, ['mile', 'mi', 'miles'])
    return has_mile_unit or num > 5


def looks_like_rain_probability(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    has_percent = '%' in text or ci_contains(text, 'percent')
    return (0 <= num <= 100) and (has_percent or num <= 100)


def looks_like_gas_station_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'http' in text.lower() or 'maps.google' in text.lower() or 'goo.gl' in text.lower()


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
    trail_info = await evaluator.extract(
        prompt=prompt_extract_trail_info(),
        template_class=TrailInfo,
        extraction_name="trail_info"
    )

    weather_info = await evaluator.extract(
        prompt=prompt_extract_weather_info(),
        template_class=WeatherInfo,
        extraction_name="weather_info"
    )

    gas_station_info = await evaluator.extract(
        prompt=prompt_extract_gas_station_info(),
        template_class=GasStationInfo,
        extraction_name="gas_station_info"
    )

    route_info = await evaluator.extract(
        prompt=prompt_extract_route_info(),
        template_class=RouteInfo,
        extraction_name="route_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 AllTrails section
    alltrails_node = evaluator.add_sequential(
        id="alltrails_section",
        desc="AllTrails off-road trail search and selection near Moab, UT",
        parent=root,
        critical=False
    )

    # [Action Node] alltrails.com:F1:A1 - Filter and sort trails
    trail_name_ok = looks_like_trail_name(trail_info.trail_name)
    rating_num = extract_float(trail_info.rating)
    rating_ok = rating_num is not None and rating_num > 4.5
    difficulty_ok = ci_contains(trail_info.difficulty, 'hard')
    moab_mentioned = has_any_ci(answer, ['moab'])

    filter_sort_ok = trail_name_ok and rating_ok and difficulty_ok and moab_mentioned

    evaluator.add_custom_node(
        result=bool(filter_sort_ok),
        id="alltrails_filter_sort",
        desc="[Action Node] alltrails.com:F1:A1 - Filter by Hard difficulty, rating above 4.5, and select trail with highest reviews",
        parent=alltrails_node,
        critical=False
    )

    # [Perception Node] alltrails.com:F1:P1 - Extract trail length
    length_num = extract_float(trail_info.length)
    length_ok = length_num is not None and length_num > 10
    length_format_ok = looks_like_length_miles(trail_info.length)

    evaluator.add_custom_node(
        result=bool(length_ok and length_format_ok),
        id="alltrails_length_perception",
        desc="[Perception Node] alltrails.com:F1:P1 - Extract trail length exceeding 10 miles",
        parent=alltrails_node,
        critical=False
    )

    # Additional check: mentions off-road driving
    offroad_mentioned = has_any_ci(answer, ['off-road', 'offroad', 'off road'])
    evaluator.add_custom_node(
        result=bool(offroad_mentioned),
        id="alltrails_offroad_context",
        desc="Mentions off-road driving context in trail search",
        parent=alltrails_node,
        critical=False
    )

    # 3.2 Weather.com section
    weather_node = evaluator.add_sequential(
        id="weather_section",
        desc="Weather.com rain probability check for Moab weekend",
        parent=root,
        critical=False
    )

    # [Perception Node] weather.com:F1:P1 - Extract rain probability for specific day
    rain_prob_num = extract_float(weather_info.rain_probability)
    rain_prob_format_ok = looks_like_rain_probability(weather_info.rain_probability)
    rain_prob_value_ok = rain_prob_num is not None and rain_prob_num < 20
    weekend_mentioned = has_any_ci(answer, ['saturday', 'sunday', 'weekend'])

    evaluator.add_custom_node(
        result=bool(rain_prob_format_ok and rain_prob_value_ok and weekend_mentioned),
        id="weather_rain_probability",
        desc="[Perception Node] weather.com:F1:P1 - Extract rain probability below 20% for upcoming Saturday or Sunday",
        parent=weather_node,
        critical=False
    )

    # Additional check: mentions Weather.com specifically
    weather_com_mentioned = has_any_ci(answer, ['weather.com', 'weather com'])
    evaluator.add_custom_node(
        result=bool(weather_com_mentioned),
        id="weather_source_mentioned",
        desc="Mentions Weather.com as the weather source",
        parent=weather_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps route planning and gas station search",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Route planning
    route_link_ok = looks_like_url(route_info.route_link)
    start_ok = ci_contains(route_info.start_location, 'moab information center') or ci_contains(answer, 'moab information center')
    end_mentions_trail = ci_contains(route_info.end_location, 'trailhead') or ci_contains(answer, 'trailhead')

    route_planning_ok = route_link_ok and start_ok

    evaluator.add_custom_node(
        result=bool(route_planning_ok),
        id="googlemaps_route_planning",
        desc="[Action Node] maps.google.com:F1:A2 - Plan route from Moab Information Center to trail trailhead",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F11:A20 - Search along route
    gas_name_ok = gas_station_info.gas_station_name is not None and len(str(gas_station_info.gas_station_name).strip()) > 0
    along_route_mentioned = has_any_ci(answer, ['along', 'route', 'search along'])

    evaluator.add_custom_node(
        result=bool(gas_name_ok and along_route_mentioned),
        id="googlemaps_search_along_route",
        desc="[Action Node] maps.google.com:F11:A20 - Use search along route feature to find gas station",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Extract gas station rating
    gas_rating_num = extract_float(gas_station_info.gas_station_rating)
    gas_rating_format_ok = looks_like_gas_station_rating(gas_station_info.gas_station_rating)
    gas_rating_value_ok = gas_rating_num is not None and gas_rating_num >= 4.0

    evaluator.add_custom_node(
        result=bool(gas_rating_format_ok and gas_rating_value_ok),
        id="googlemaps_gas_rating_perception",
        desc="[Perception Node] maps.google.com:F1:P1 - Extract gas station rating of 4.0 or higher",
        parent=maps_node,
        critical=False
    )

    # Additional check: mentions Google Maps
    google_maps_mentioned = has_any_ci(answer, ['google maps', 'google map'])
    evaluator.add_custom_node(
        result=bool(google_maps_mentioned),
        id="googlemaps_source_mentioned",
        desc="Mentions Google Maps as the navigation source",
        parent=maps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
