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
TASK_ID = "task-522faf"
TASK_DESCRIPTION = 'I’m planning to drive an EV from Redding, California to Eugene, Oregon over a weekend next month. Since winter low temperatures can significantly reduce range, I need a dynamic charging plan.\n\nFirst, use Google Maps to map this route, note the total mileage, and select 3 major intermediate cities along I-5 as potential stops (e.g., Mt. Shasta, Ashland, Roseburg).\n\nNext, check the **Daily** weather forecast on AccuWeather for those 3 cities for the **departure day**. Focus on the **RealFeel® high temperature** and whether any precipitation/snow is forecast. Rule: if a city’s RealFeel® high is below **45°F**, or if the forecast includes any form of precipitation (**Rain/Snow/Ice**), mark that city as **“High Energy Consumption Risk.”**\n\nFor cities marked as **“High Energy Consumption Risk,”** return to Google Maps and find a nearby Tesla Supercharger with a rating above **4.5**. If no Tesla Supercharger above 4.5 exists in that city, choose the highest-rated Tesla Supercharger within the city and note **“Below 4.5”** in the output.\n\nOutput required:\n- Total trip distance  \n- For each of the 3 cities: city name, AccuWeather RealFeel® high on departure day, precipitation forecast details, risk status (Yes/No)  \n- For each risk city: selected Supercharger name, rating, and Google Maps link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TripDistance(BaseModel):
    """Total trip distance extracted from the answer"""
    distance_text: Optional[str] = None


class CityWeatherInfo(BaseModel):
    """Weather and risk information for intermediate cities"""
    city1_name: Optional[str] = None
    city1_realfeel_high: Optional[str] = None
    city1_precipitation: Optional[str] = None
    city1_risk_status: Optional[str] = None

    city2_name: Optional[str] = None
    city2_realfeel_high: Optional[str] = None
    city2_precipitation: Optional[str] = None
    city2_risk_status: Optional[str] = None

    city3_name: Optional[str] = None
    city3_realfeel_high: Optional[str] = None
    city3_precipitation: Optional[str] = None
    city3_risk_status: Optional[str] = None


class SuperchargerInfo(BaseModel):
    """Supercharger information for high-risk cities"""
    superchargers_found: Optional[List[Dict[str, str]]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_distance_from_answer() -> str:
    return """
Extract the total trip distance from Redding, California to Eugene, Oregon as reported in the answer.

Return:
- distance_text: the distance exactly as stated (include units like miles, mi, km if present).

If not present, set it to null.
"""


def prompt_extract_city_weather_from_answer() -> str:
    return """
Extract the weather information for the 3 intermediate cities from the answer.

For each city, extract:
- city name
- RealFeel® high temperature on departure day
- precipitation forecast details (Rain/Snow/Ice or None)
- risk status (Yes/No or High Energy Consumption Risk/No Risk)

Return the information for city1, city2, and city3. If any field is missing, set it to null.
"""


def prompt_extract_supercharger_from_answer() -> str:
    return """
Extract information about Tesla Superchargers found for high-risk cities from the answer.

For each supercharger mentioned, extract:
- city_name: the city where the supercharger is located
- supercharger_name: the name of the Tesla Supercharger
- rating: the rating (e.g., "4.6", "4.5", "Below 4.5")
- maps_link: the Google Maps link if provided

Return a list of superchargers_found. If none are mentioned, return an empty list.
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


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['mile', 'mi', 'km', 'kilometer'])


def is_valid_distance_range(text: Optional[str]) -> bool:
    """Check if distance is in reasonable range for Redding to Eugene (220-240 miles)"""
    num = extract_float(text)
    if num is None:
        return False
    # Accept 220-240 miles or 350-390 km
    if 220 <= num <= 240:
        return True
    if 350 <= num <= 390:
        return True
    return False


def looks_like_temperature(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return True


def is_below_45f(temp_text: Optional[str]) -> Optional[bool]:
    """Check if temperature is below 45°F"""
    num = extract_float(temp_text)
    if num is None:
        return None
    return num < 45


def mentions_precipitation(precip_text: Optional[str]) -> bool:
    if not precip_text:
        return False
    return has_any_ci(precip_text, ['rain', 'snow', 'ice', 'precipitation', 'sleet', 'freezing'])


def looks_like_risk_status(status_text: Optional[str]) -> bool:
    if not status_text:
        return False
    return has_any_ci(status_text, ['yes', 'no', 'risk', 'high energy consumption'])


def extract_rating_value(rating_text: Optional[str]) -> Optional[float]:
    if not rating_text:
        return None
    # Handle "Below 4.5" case
    if ci_contains(rating_text, 'below'):
        return 0.0  # Indicator that it's below threshold
    num = extract_float(rating_text)
    return num


def is_rating_above_45(rating_text: Optional[str]) -> Optional[bool]:
    """Check if rating is above 4.5 or noted as below"""
    if ci_contains(rating_text, 'below 4.5'):
        return False
    num = extract_rating_value(rating_text)
    if num is None:
        return None
    if num == 0.0:  # "Below" indicator
        return False
    return num > 4.5


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
    distance_info = await evaluator.extract(
        prompt=prompt_extract_distance_from_answer(),
        template_class=TripDistance,
        extraction_name="trip_distance"
    )

    city_weather_info = await evaluator.extract(
        prompt=prompt_extract_city_weather_from_answer(),
        template_class=CityWeatherInfo,
        extraction_name="city_weather_info"
    )

    supercharger_info = await evaluator.extract(
        prompt=prompt_extract_supercharger_from_answer(),
        template_class=SuperchargerInfo,
        extraction_name="supercharger_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Google Maps - Route Planning Section
    gmaps_route_node = evaluator.add_sequential(
        id="gmaps_route_planning",
        desc="Google Maps route planning from Redding to Eugene with intermediate cities",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A5 - Route planning
    route_planning_ok = (has_any_ci(answer, ['google maps', 'google map']) and
                        has_any_ci(answer, ['redding', 'eugene']) and
                        has_any_ci(answer, ['route', 'distance', 'mileage']))
    evaluator.add_custom_node(
        result=bool(route_planning_ok),
        id="gmaps_action_route_planning",
        desc="[Action Node] google.com/maps:F2:A5 - Plan route from Redding, CA to Eugene, OR",
        parent=gmaps_route_node,
        critical=False
    )

    # Check distance extraction and validity
    distance_extracted = looks_like_distance(distance_info.distance_text)
    distance_valid_range = is_valid_distance_range(distance_info.distance_text)

    evaluator.add_custom_node(
        result=bool(distance_extracted and distance_valid_range),
        id="gmaps_distance_valid",
        desc="Extract total trip distance in reasonable range (220-240 miles or 350-390 km)",
        parent=gmaps_route_node,
        critical=False
    )

    # Check intermediate cities selection
    cities_mentioned = 0
    city_names = [city_weather_info.city1_name, city_weather_info.city2_name, city_weather_info.city3_name]
    for city_name in city_names:
        if city_name and len(city_name.strip()) > 0:
            cities_mentioned += 1

    three_cities_ok = cities_mentioned >= 3
    i5_cities_ok = has_any_ci(answer, ['i-5', 'i5', 'interstate 5'])

    evaluator.add_custom_node(
        result=bool(three_cities_ok and i5_cities_ok),
        id="gmaps_intermediate_cities",
        desc="Select 3 major intermediate cities along I-5",
        parent=gmaps_route_node,
        critical=False
    )

    # 3.2 AccuWeather - Weather Forecast Section
    accuweather_node = evaluator.add_sequential(
        id="accuweather_weather_check",
        desc="AccuWeather daily forecast for 3 intermediate cities",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F1:A1 - Location search for cities
    accuweather_search_ok = (has_any_ci(answer, ['accuweather']) and
                            cities_mentioned >= 1)
    evaluator.add_custom_node(
        result=bool(accuweather_search_ok),
        id="accuweather_action_search",
        desc="[Action Node] accuweather.com:F1:A1 - Search for the 3 intermediate cities on AccuWeather",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F2:A2 - Tab switching to Daily
    daily_tab_ok = has_any_ci(answer, ['daily', 'daily forecast', 'daily weather'])
    evaluator.add_custom_node(
        result=bool(daily_tab_ok),
        id="accuweather_action_daily_tab",
        desc="[Action Node] accuweather.com:F2:A2 - Navigate to Daily weather forecast tab",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F2:P3 - Extract RealFeel® and precipitation
    realfeel_mentioned = has_any_ci(answer, ['realfeel', 'real feel'])

    city_data_complete = 0
    for i, (name, temp, precip) in enumerate([
        (city_weather_info.city1_name, city_weather_info.city1_realfeel_high, city_weather_info.city1_precipitation),
        (city_weather_info.city2_name, city_weather_info.city2_realfeel_high, city_weather_info.city2_precipitation),
        (city_weather_info.city3_name, city_weather_info.city3_realfeel_high, city_weather_info.city3_precipitation)
    ]):
        if name and looks_like_temperature(temp) and precip is not None:
            city_data_complete += 1

    weather_data_ok = realfeel_mentioned and city_data_complete >= 2

    evaluator.add_custom_node(
        result=bool(weather_data_ok),
        id="accuweather_perception_weather_data",
        desc="[Perception Node] accuweather.com:F2:P3 - Extract RealFeel® high temperature and precipitation for the 3 cities",
        parent=accuweather_node,
        critical=False
    )

    # Check risk assessment logic
    risk_logic_applied = False
    risk_statuses = [city_weather_info.city1_risk_status,
                    city_weather_info.city2_risk_status,
                    city_weather_info.city3_risk_status]

    for status in risk_statuses:
        if looks_like_risk_status(status):
            risk_logic_applied = True
            break

    evaluator.add_custom_node(
        result=bool(risk_logic_applied),
        id="risk_assessment_logic",
        desc="Apply risk assessment logic based on temperature (<45°F) and precipitation",
        parent=accuweather_node,
        critical=False
    )

    # 3.3 Google Maps - Supercharger Search Section
    gmaps_supercharger_node = evaluator.add_sequential(
        id="gmaps_supercharger_search",
        desc="Google Maps search for Tesla Superchargers in high-risk cities",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F1:A2 - Location search for Superchargers
    supercharger_search_ok = (has_any_ci(answer, ['tesla supercharger', 'supercharger']) and
                             len(supercharger_info.superchargers_found) > 0)
    evaluator.add_custom_node(
        result=bool(supercharger_search_ok),
        id="gmaps_action_supercharger_search",
        desc="[Action Node] google.com/maps:F1:A2 - Search for Tesla Superchargers near high-risk cities",
        parent=gmaps_supercharger_node,
        critical=False
    )

    # [Perception Node] google.com/maps:F1:P3 - Rating extraction and filtering
    rating_check_ok = False
    has_maps_link = False

    for sc in supercharger_info.superchargers_found:
        rating_text = sc.get('rating', '')
        if rating_text:
            rating_check_ok = True
        maps_link = sc.get('maps_link', '')
        if maps_link and ('google.com/maps' in maps_link.lower() or 'maps.google.com' in maps_link.lower()):
            has_maps_link = True

    evaluator.add_custom_node(
        result=bool(rating_check_ok),
        id="gmaps_perception_rating_filter",
        desc="[Perception Node] google.com/maps:F1:P3 - Extract and verify Supercharger ratings (above 4.5 or highest available)",
        parent=gmaps_supercharger_node,
        critical=False
    )

    # Check for Google Maps links
    evaluator.add_custom_node(
        result=bool(has_maps_link),
        id="gmaps_links_provided",
        desc="Provide Google Maps links for selected Superchargers",
        parent=gmaps_supercharger_node,
        critical=False
    )

    # 3.4 Output Completeness Check
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required output elements are present",
        parent=root,
        critical=False
    )

    # Total distance provided
    evaluator.add_custom_node(
        result=bool(distance_info.distance_text is not None),
        id="output_total_distance",
        desc="Output includes total trip distance",
        parent=output_node,
        critical=False
    )

    # City information completeness
    evaluator.add_custom_node(
        result=bool(city_data_complete >= 3),
        id="output_city_info_complete",
        desc="Output includes name, RealFeel® high, precipitation, and risk status for all 3 cities",
        parent=output_node,
        critical=False
    )

    # Supercharger information for risk cities
    supercharger_info_complete = len(supercharger_info.superchargers_found) > 0
    evaluator.add_custom_node(
        result=bool(supercharger_info_complete),
        id="output_supercharger_info",
        desc="Output includes Supercharger name, rating, and link for high-risk cities",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
