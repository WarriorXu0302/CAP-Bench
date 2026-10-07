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
TASK_ID = "task-e14616"
TASK_DESCRIPTION = "I am planning to move to the Travis Heights neighborhood in Austin, Texas (ZIP code 78704) and am concerned about environmental health issues in the vicinity.\n\nFirst, navigate to the **EPA ECHO** website. Use the facility search function to find all facilities in the 'Austin, TX' area. From the results, filter for facilities that are part of the 'Air' regulatory program and are currently in 'Significant Violation' status. Identify the facility whose 'Last Inspection Date' is closest to the current date. Record its name and detailed address.\n\nNext, go to **Google Maps** to calculate the driving distance from this violating facility to **Big Stacy Pool** (located within the Travis Heights neighborhood).\n\nFinally, visit **AccuWeather** to view the current month's weather forecast for Austin, TX, and count the number of days with 'Rain/Showers' predicted for the month.\n\nOutput: EPA Facility Name, EPA Facility Address, Last Inspection Date, Violation Status (confirm if 'Significant Violation'), Driving Distance to Big Stacy Pool, Number of Predicted Rainy Days this Month, EPA Facility Details Page Link, AccuWeather Monthly Forecast Page Link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EPAFacilityInfo(BaseModel):
    """EPA facility information extracted from the answer"""
    facility_name: Optional[str] = None
    facility_address: Optional[str] = None
    last_inspection_date: Optional[str] = None
    violation_status: Optional[str] = None
    epa_details_link: Optional[str] = None


class MapsDistanceInfo(BaseModel):
    """Google Maps distance information extracted from the answer"""
    driving_distance: Optional[str] = None


class WeatherForecastInfo(BaseModel):
    """AccuWeather forecast information extracted from the answer"""
    rainy_days_count: Optional[str] = None
    accuweather_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_epa_facility() -> str:
    return """
Extract the EPA ECHO facility details from the answer for the facility in Austin, TX with Air violations:

Return:
- facility_name: the name of the EPA facility exactly as stated
- facility_address: the detailed address of the facility
- last_inspection_date: the last inspection date exactly as stated
- violation_status: the violation status (should mention 'Significant Violation' or similar)
- epa_details_link: the URL to the EPA facility details page

If any field is missing, set it to null.
"""


def prompt_extract_maps_distance() -> str:
    return """
Extract the Google Maps driving distance from the answer:

Return:
- driving_distance: the driving distance from the EPA facility to Big Stacy Pool exactly as stated (include units)

If missing, set it to null.
"""


def prompt_extract_weather_forecast() -> str:
    return """
Extract the AccuWeather forecast information from the answer:

Return:
- rainy_days_count: the number of days with rain/showers predicted for the month
- accuweather_link: the URL to the AccuWeather monthly forecast page

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


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for typical address components
    has_street = has_any_ci(text, ['st', 'street', 'ave', 'avenue', 'rd', 'road', 'dr', 'drive', 'blvd', 'lane', 'ln'])
    has_city = ci_contains(text, 'austin')
    has_state = has_any_ci(text, ['tx', 'texas'])
    has_zip = bool(re.search(r'\b\d{5}\b', text))
    return (has_street or has_city or has_state or has_zip) and len(text) > 10


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        r'\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
    ]
    return any(re.search(p, text.lower()) for p in date_patterns)


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    # Check for distance units
    return has_any_ci(text, ['mi', 'mile', 'miles', 'km', 'kilometer', 'kilometers'])


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text.lower()))


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
    epa_info = await evaluator.extract(
        prompt=prompt_extract_epa_facility(),
        template_class=EPAFacilityInfo,
        extraction_name="epa_facility_info"
    )

    maps_info = await evaluator.extract(
        prompt=prompt_extract_maps_distance(),
        template_class=MapsDistanceInfo,
        extraction_name="maps_distance_info"
    )

    weather_info = await evaluator.extract(
        prompt=prompt_extract_weather_forecast(),
        template_class=WeatherForecastInfo,
        extraction_name="weather_forecast_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 EPA ECHO section
    epa_node = evaluator.add_sequential(
        id="epa_echo_section",
        desc="EPA ECHO facility search and violation identification",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F1:A2 - Geographic location search for Austin, TX
    epa_location_search_ok = (
        has_any_ci(answer, ['epa', 'echo']) and
        has_any_ci(answer, ['austin']) and
        has_any_ci(answer, ['tx', 'texas'])
    )
    evaluator.add_custom_node(
        result=bool(epa_location_search_ok),
        id="epa_location_search",
        desc="[Action Node] epa.gov:F1:A2 - Search for facilities in Austin, TX area",
        parent=epa_node,
        critical=False
    )

    # Check if address is in Austin area (o2 verification)
    address_in_austin = (
        epa_info.facility_address and
        looks_like_address(epa_info.facility_address) and
        has_any_ci(epa_info.facility_address, ['austin']) and
        has_any_ci(epa_info.facility_address, ['tx', 'texas'])
    )
    evaluator.add_custom_node(
        result=bool(address_in_austin),
        id="epa_address_verification",
        desc="Facility address is located in Austin, TX area",
        parent=epa_node,
        critical=False
    )

    # [Action Node] epa.gov:F1:A3 - Filter by Air regulatory program
    epa_air_filter_ok = has_any_ci(answer, ['air'])
    evaluator.add_custom_node(
        result=bool(epa_air_filter_ok),
        id="epa_air_filter",
        desc="[Action Node] epa.gov:F1:A3 - Filter for facilities in Air regulatory program",
        parent=epa_node,
        critical=False
    )

    # [Action Node] epa.gov:F1:A5 - Filter by Significant Violation status
    violation_is_significant = (
        epa_info.violation_status and
        has_any_ci(epa_info.violation_status, ['significant', 'violation'])
    )
    evaluator.add_custom_node(
        result=bool(violation_is_significant),
        id="epa_violation_filter",
        desc="[Action Node] epa.gov:F1:A5 - Filter for Significant Violation status",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F1:P1 - Identify facility with most recent inspection date
    has_recent_inspection = (
        epa_info.last_inspection_date and
        looks_like_date(epa_info.last_inspection_date)
    )
    evaluator.add_custom_node(
        result=bool(has_recent_inspection),
        id="epa_recent_inspection",
        desc="[Perception Node] epa.gov:F1:P1 - Identify facility with most recent Last Inspection Date",
        parent=epa_node,
        critical=False
    )

    # Check facility name is provided
    has_facility_name = bool(epa_info.facility_name and len(epa_info.facility_name.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(has_facility_name),
        id="epa_facility_name",
        desc="EPA facility name is provided",
        parent=epa_node,
        critical=False
    )

    # Check EPA details link is provided (o7 verification)
    has_epa_link = (
        epa_info.epa_details_link and
        looks_like_url(epa_info.epa_details_link) and
        has_any_ci(epa_info.epa_details_link, ['epa', 'echo'])
    )
    evaluator.add_custom_node(
        result=bool(has_epa_link),
        id="epa_details_link",
        desc="EPA facility details page link is provided",
        parent=epa_node,
        critical=False
    )

    # 3.2 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps distance calculation",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Search for locations (facility and Big Stacy Pool)
    maps_search_ok = (
        has_any_ci(answer, ['google', 'maps']) and
        has_any_ci(answer, ['big stacy pool', 'big stacy', 'stacy pool'])
    )
    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="maps_location_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for EPA facility and Big Stacy Pool locations",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Extract driving distance
    distance_extracted = (
        maps_info.driving_distance and
        looks_like_distance(maps_info.driving_distance)
    )
    evaluator.add_custom_node(
        result=bool(distance_extracted),
        id="maps_distance_extraction",
        desc="[Perception Node] maps.google.com:F1:P1 - Extract driving distance between locations",
        parent=maps_node,
        critical=False
    )

    # Check distance is reasonable for same-city travel
    distance_num = extract_number(maps_info.driving_distance)
    distance_reasonable = False
    if distance_num is not None:
        # Assume miles; reasonable same-city distance is 0.1 to 50 miles
        if 0.1 <= distance_num <= 50:
            distance_reasonable = True
        # Could be kilometers; check if 0.2 to 80 km
        elif 0.2 <= distance_num <= 80:
            distance_reasonable = True
    evaluator.add_custom_node(
        result=bool(distance_reasonable),
        id="maps_distance_reasonable",
        desc="Distance value is reasonable for same-city travel",
        parent=maps_node,
        critical=False
    )

    # 3.3 AccuWeather section
    weather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather monthly forecast for Austin, TX",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F6:A10 - Navigate to monthly forecast view
    accuweather_monthly_ok = (
        has_any_ci(answer, ['accuweather']) and
        has_any_ci(answer, ['monthly', 'month']) and
        has_any_ci(answer, ['austin'])
    )
    evaluator.add_custom_node(
        result=bool(accuweather_monthly_ok),
        id="accuweather_monthly_view",
        desc="[Action Node] accuweather.com:F6:A10 - Navigate to monthly forecast view for Austin, TX",
        parent=weather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F6:P11 - Count rainy days from weather icons
    rainy_count_extracted = (
        weather_info.rainy_days_count and
        contains_digits(weather_info.rainy_days_count)
    )
    evaluator.add_custom_node(
        result=bool(rainy_count_extracted),
        id="accuweather_rainy_days_count",
        desc="[Perception Node] accuweather.com:F6:P11 - Count days with rain/showers predicted",
        parent=weather_node,
        critical=False
    )

    # Verify count is a non-negative integer
    rainy_count_num = extract_number(weather_info.rainy_days_count)
    count_valid = rainy_count_num is not None and rainy_count_num >= 0 and rainy_count_num == int(rainy_count_num)
    evaluator.add_custom_node(
        result=bool(count_valid),
        id="accuweather_count_valid",
        desc="Rainy days count is a valid non-negative integer",
        parent=weather_node,
        critical=False
    )

    # Check AccuWeather link is provided
    has_weather_link = (
        weather_info.accuweather_link and
        looks_like_url(weather_info.accuweather_link) and
        has_any_ci(weather_info.accuweather_link, ['accuweather'])
    )
    evaluator.add_custom_node(
        result=bool(has_weather_link),
        id="accuweather_link",
        desc="AccuWeather monthly forecast page link is provided",
        parent=weather_node,
        critical=False
    )

    # Check mentions of rain/showers context
    mentions_rain = has_any_ci(answer, ['rain', 'shower', 'precipitation'])
    evaluator.add_custom_node(
        result=bool(mentions_rain),
        id="accuweather_rain_context",
        desc="Mentions rain/showers in the forecast context",
        parent=weather_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
