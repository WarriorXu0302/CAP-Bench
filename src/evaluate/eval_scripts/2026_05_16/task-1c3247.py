import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-1c3247"
TASK_DESCRIPTION = """
I'm planning a short getaway for the upcoming weekend. However, I suffer from severe allergies and am currently deciding between Austin, TX, and Seattle, WA.

First, go to AccuWeather to check the "Health & Activities" forecast for those days in both cities. Compare the risk levels for "Mold" or "Pollen". For added precaution, also check the current Air Quality Index (AQI) values for both cities on EPA AirNow. Based on this data, select the city with air quality more suitable for my condition.

Once the city is determined, search on TripAdvisor for three hotels in that city. Requirements: Check-in dates should correspond to my travel dates, the price must be between $200 and $450 per night, hotels must be 4-star rated or higher, and must include "Air Conditioning" facilities to prevent allergy issues from open windows. Finally, sort by "Traveler Ranked" and list the top 3 hotels.

Output: Mold/Pollen levels for both locations, AQI values for both locations, the final chosen city, names of the recommended hotels, their prices, star ratings, whether they include air conditioning, and their detail page links.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AccuWeatherCityData(BaseModel):
    city: Optional[str] = None
    allergen: Optional[str] = None  # 'Mold' or 'Pollen' (or both)
    risk_level: Optional[str] = None  # e.g., 'Low', 'Moderate', 'High', etc.
    has_health_activities: Optional[bool] = False


class AccuWeatherData(BaseModel):
    cities: List[AccuWeatherCityData] = Field(default_factory=list)


class CityAQI(BaseModel):
    city: Optional[str] = None
    aqi_value: Optional[float] = None
    aqi_category: Optional[str] = None  # e.g., 'Good', 'Moderate'


class AQIData(BaseModel):
    cities: List[CityAQI] = Field(default_factory=list)


class Decision(BaseModel):
    chosen_city: Optional[str] = None
    rationale: Optional[str] = None


class Hotel(BaseModel):
    name: Optional[str] = None
    price_per_night: Optional[float] = None
    star_rating: Optional[float] = None
    includes_air_conditioning: Optional[bool] = None
    url: Optional[str] = None
    city: Optional[str] = None
    checkin_date: Optional[str] = None
    checkout_date: Optional[str] = None


class HotelsList(BaseModel):
    city: Optional[str] = None
    hotels: List[Hotel] = Field(default_factory=list)
    sorting: Optional[str] = None  # e.g., 'Traveler Ranked' if mentioned


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_accuweather_data() -> str:
    return """
    From the provided answer, extract the allergy-related data gathered from AccuWeather for both Austin, TX and Seattle, WA.

    For each of the two cities (Austin and Seattle), extract:
    - city: the city name as referenced in the answer
    - allergen: which allergen was used for comparison (e.g., 'Mold' or 'Pollen'); if both are mentioned, pick the one actually compared
    - risk_level: the risk level for that allergen across the travel dates (choose the worst level mentioned if multiple; use plain text like 'Low', 'Moderate', 'High', 'Very High', 'Extreme', etc.)
    - has_health_activities: true if the answer explicitly mentions the 'Health & Activities' section/tab of AccuWeather for this city, otherwise false

    Return exactly two entries, one for Austin and one for Seattle, if possible. If any field is missing, set it to null or false.
    """


def prompt_extract_airnow_aqi() -> str:
    return """
    From the provided answer, extract the current Air Quality Index (AQI) information the user obtained from EPA AirNow for both Austin, TX and Seattle, WA.

    For each city, extract:
    - city: the city name
    - aqi_value: the numeric AQI value (convert to a number if possible; otherwise set null)
    - aqi_category: the AQI category text if mentioned (e.g., 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', etc.)

    Return a list with entries for Austin and Seattle. If any field is missing, set it to null.
    """


def prompt_extract_decision() -> str:
    return """
    Extract the final chosen city for the trip and any rationale text given in the answer.

    Fields:
    - chosen_city: the city name selected (e.g., 'Austin' or 'Seattle')
    - rationale: the textual explanation for why this city was chosen, if any
    """


def prompt_extract_hotels(chosen_city: str) -> str:
    return f"""
    From the provided answer, extract the top three recommended hotels in the chosen city "{chosen_city}" as listed by the assistant.

    Extract:
    - city: the city these hotels are in (if given once in the answer, include it here)
    - sorting: any sorting label mentioned for the list (e.g., 'Traveler Ranked'), if any
    - hotels: a list of up to three hotels with
        - name: hotel name
        - price_per_night: numeric price per night (convert '$250' to 250.0 if possible)
        - star_rating: numeric star rating (e.g., 4.0, 4.5, 5.0)
        - includes_air_conditioning: true if the answer states 'Air Conditioning' is available for this hotel, else false
        - url: the direct URL to the hotel's detail page on TripAdvisor
        - city: the city for the specific hotel if provided
        - checkin_date: the check-in date (or date range start) if provided in the answer
        - checkout_date: the check-out date (or date range end) if provided

    If any field is missing for a hotel, set it to null. Return exactly what is present in the answer; do not fabricate.
    """


# --------------------------------------------------------------------------- #
# Helper functions for lenient evaluation                                     #
# --------------------------------------------------------------------------- #
def safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def contains_any(text: Optional[str], keywords: List[str]) -> bool:
    t = safe_lower(text)
    return any(k.lower() in t for k in keywords)


def risk_to_score(risk_level: Optional[str]) -> int:
    # Map common risk descriptors to a numeric scale (lower is better)
    rl = safe_lower(risk_level)
    if "very low" in rl:
        return 1
    if "low" in rl:
        return 2
    if "moderate" in rl:
        return 3
    if "high" in rl and "very" not in rl:
        return 4
    if "very high" in rl or "extreme" in rl:
        return 5
    # Unknown risk level: neutral
    return 3


def aqi_to_score(aqi_value: Optional[float]) -> int:
    # Convert numeric AQI to rough category score (lower is better)
    try:
        if aqi_value is None:
            return 3
        v = float(aqi_value)
        if v <= 50:
            return 1  # Good
        if v <= 100:
            return 2  # Moderate
        if v <= 150:
            return 3  # USG
        if v <= 200:
            return 4  # Unhealthy
        if v <= 300:
            return 5  # Very Unhealthy
        return 6  # Hazardous
    except Exception:
        return 3


def get_city_entry_by_name(entries: List[AccuWeatherCityData], target: str) -> Optional[AccuWeatherCityData]:
    tgt = safe_lower(target)
    for e in entries:
        if tgt in safe_lower(e.city):
            return e
    return None


def get_aqi_entry_by_name(entries: List[CityAQI], target: str) -> Optional[CityAQI]:
    tgt = safe_lower(target)
    for e in entries:
        if tgt in safe_lower(e.city):
            return e
    return None


def price_in_range(price: Optional[float], low: float = 200.0, high: float = 450.0) -> bool:
    try:
        if price is None:
            return False
        v = float(price)
        return low - 1e-6 <= v <= high + 1e-6
    except Exception:
        return False


def star_ok(star: Optional[float]) -> bool:
    try:
        if star is None:
            return False
        return float(star) >= 4.0 - 1e-6
    except Exception:
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

    # -------- 2. Extract structured info from the answer ----------------- #
    # Decision (chosen city)
    decision: Decision = await evaluator.extract(
        prompt=prompt_extract_decision(),
        template_class=Decision,
        extraction_name="decision"
    )

    chosen_city = decision.chosen_city or ""

    # AccuWeather allergy info
    accuweather_data: AccuWeatherData = await evaluator.extract(
        prompt=prompt_extract_accuweather_data(),
        template_class=AccuWeatherData,
        extraction_name="accuweather_data"
    )

    # AQI data from AirNow
    aqi_data: AQIData = await evaluator.extract(
        prompt=prompt_extract_airnow_aqi(),
        template_class=AQIData,
        extraction_name="aqi_data"
    )

    # Hotels (top 3) in chosen city
    hotels_list: HotelsList = await evaluator.extract(
        prompt=prompt_extract_hotels(chosen_city if chosen_city else "the chosen city"),
        template_class=HotelsList,
        extraction_name="hotels_list"
    )

    logger.info(f"Chosen city extracted: {chosen_city}")
    logger.info(f"AccuWeather entries: {len(accuweather_data.cities)}; AQI entries: {len(aqi_data.cities)}; Hotels: {len(hotels_list.hotels)}")

    # -------- 3. Build verification tree --------------------------------- #
    # 3.1 AccuWeather verification
    accu_node = evaluator.add_sequential(
        id="accuweather_verification",
        desc="Verify AccuWeather allergy data usage and extraction",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F1:A1 - Location search for both cities
    # Condition: both Austin and Seattle present AND answer references AccuWeather or Health & Activities
    austin_entry = get_city_entry_by_name(accuweather_data.cities, "Austin")
    seattle_entry = get_city_entry_by_name(accuweather_data.cities, "Seattle")
    accu_location_ok = bool(austin_entry and seattle_entry) and contains_any(answer, ["accuweather", "health & activities"])

    evaluator.add_custom_node(
        result=accu_location_ok,
        id="accu_loc_search",
        desc="[Action Node] accuweather.com:F1:A1 - Confirm location search on AccuWeather for Austin and Seattle",
        parent=accu_node,
        critical=False
    )

    # [Action Node] accuweather.com:F7:A2 - Tab switch to Health & Activities
    # Condition: 'Health & Activities' mentioned in answer or has_health_activities True in any entry
    has_health_tab = contains_any(answer, ["health & activities", "health and activities"]) or any(
        bool(c.has_health_activities) for c in accuweather_data.cities
    )
    evaluator.add_custom_node(
        result=has_health_tab,
        id="accu_tab_switch",
        desc="[Action Node] accuweather.com:F7:A2 - Use the 'Health & Activities' tab",
        parent=accu_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F7:P10 - List content understanding (risk levels)
    # Condition: risk level includes Low/Moderate/High etc. and allergen mentions Mold/Pollen
    def risk_valid(entry: Optional[AccuWeatherCityData]) -> bool:
        if not entry:
            return False
        risk = safe_lower(entry.risk_level)
        allergen = safe_lower(entry.allergen)
        has_risk_word = any(w in risk for w in ["low", "moderate", "high", "very high", "extreme"])
        has_allergen = any(a in allergen for a in ["mold", "pollen"])
        return has_risk_word and has_allergen

    accu_list_understanding_ok = risk_valid(austin_entry) and risk_valid(seattle_entry)
    evaluator.add_custom_node(
        result=accu_list_understanding_ok,
        id="accu_list_content",
        desc="[Perception Node] accuweather.com:F7:P10 - Understand and extract Mold/Pollen risk levels",
        parent=accu_node,
        critical=False
    )

    # 3.2 AirNow AQI verification (general)
    aqi_node = evaluator.add_sequential(
        id="airnow_verification",
        desc="Verify AirNow AQI values extracted for both cities",
        parent=root,
        critical=False
    )

    aqi_austin = get_aqi_entry_by_name(aqi_data.cities, "Austin")
    aqi_seattle = get_aqi_entry_by_name(aqi_data.cities, "Seattle")

    def aqi_numeric_ok(entry: Optional[CityAQI]) -> bool:
        try:
            if entry is None or entry.aqi_value is None:
                return False
            v = float(entry.aqi_value)
            # leniently accept values within a reasonable AQI range
            return 0 <= v <= 500
        except Exception:
            return False

    airnow_ok = aqi_numeric_ok(aqi_austin) and aqi_numeric_ok(aqi_seattle)
    evaluator.add_custom_node(
        result=airnow_ok,
        id="airnow_aqi_values",
        desc="Confirm numeric AQI values (0-500) present for both Austin and Seattle",
        parent=aqi_node,
        critical=False
    )

    # 3.3 Decision verification (general)
    decision_node = evaluator.add_sequential(
        id="decision_verification",
        desc="Verify chosen city aligns with allergy and AQI data (lenient)",
        parent=root,
        critical=False
    )

    chosen_city_lc = safe_lower(chosen_city)
    chosen_city_in_set = chosen_city_lc in ["austin", "seattle"]

    # Compute lenient alignment of chosen city with risk and AQI (if available)
    austin_risk_score = risk_to_score(austin_entry.risk_level if austin_entry else None)
    seattle_risk_score = risk_to_score(seattle_entry.risk_level if seattle_entry else None)
    austin_aqi_score = aqi_to_score(aqi_austin.aqi_value if aqi_austin else None)
    seattle_aqi_score = aqi_to_score(aqi_seattle.aqi_value if aqi_seattle else None)

    austin_total = austin_risk_score + austin_aqi_score
    seattle_total = seattle_risk_score + seattle_aqi_score

    aligns = False
    if chosen_city_in_set:
        if chosen_city_lc == "austin":
            aligns = austin_total <= seattle_total or austin_risk_score <= seattle_risk_score or austin_aqi_score <= seattle_aqi_score
        elif chosen_city_lc == "seattle":
            aligns = seattle_total <= austin_total or seattle_risk_score <= austin_risk_score or seattle_aqi_score <= austin_aqi_score

    # Lenient fallback: if rationale mentions allergies or better air quality, accept presence
    rationale_mentions = contains_any(decision.rationale, ["allerg", "air quality", "aqi", "mold", "pollen"])

    evaluator.add_custom_node(
        result=bool(chosen_city_in_set),
        id="decision_city_present",
        desc="Chosen city is either Austin or Seattle",
        parent=decision_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(aligns or rationale_mentions),
        id="decision_alignment",
        desc="Chosen city reasonably aligns with lower allergy risk and/or AQI (lenient or rationale-based)",
        parent=decision_node,
        critical=False
    )

    # 3.4 TripAdvisor verification
    trip_node = evaluator.add_sequential(
        id="tripadvisor_verification",
        desc="Verify TripAdvisor hotel selection meets constraints",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A1 - Multi-field search in chosen city
    def hotel_city_match(h: Hotel, tgt_city: str) -> bool:
        if contains_any(h.city, [tgt_city]):
            return True
        if contains_any(h.name, [tgt_city]):
            return True
        if contains_any(h.url, [tgt_city.replace(" ", "-")]) or contains_any(h.url, [tgt_city]):
            return True
        return False

    hotels = hotels_list.hotels or []
    hotels_count_ok = len(hotels) >= 3
    city_match_count = sum(1 for h in hotels if hotel_city_match(h, chosen_city)) if chosen_city else 0
    multi_field_search_ok = hotels_count_ok and (city_match_count >= 2 or contains_any(hotels_list.city, [chosen_city]))

    evaluator.add_custom_node(
        result=multi_field_search_ok,
        id="trip_multi_field_search",
        desc="[Action Node] tripadvisor.com:F1:A1 - Multi-field search results show hotels in the chosen city",
        parent=trip_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A2 - Date range selection
    # Accept if any check-in/out dates appear either at list level or per-hotel, or answer mentions 'check-in'/'upcoming weekend'
    has_dates = bool(hotels_list and (contains_any(hotels_list.city, ["weekend"]) or contains_any(answer, ["check-in", "check in", "upcoming weekend"])))
    if not has_dates:
        for h in hotels:
            if h.checkin_date or h.checkout_date:
                has_dates = True
                break

    evaluator.add_custom_node(
        result=has_dates,
        id="trip_dates_selected",
        desc="[Action Node] tripadvisor.com:F1:A2 - Check-in/out dates selected for the search (lenient)",
        parent=trip_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-condition filters (price 200-450, star >=4, AC)
    def hotel_meets_filters(h: Hotel) -> bool:
        return price_in_range(h.price_per_night) and star_ok(h.star_rating) and bool(h.includes_air_conditioning)

    hotels_meeting_filters = sum(1 for h in hotels if hotel_meets_filters(h))
    # Lenient: pass if at least 2 out of 3 hotels meet all filters
    filters_ok = hotels_meeting_filters >= 2

    evaluator.add_custom_node(
        result=filters_ok,
        id="trip_multi_filters",
        desc="[Action Node] tripadvisor.com:F1:A4 - Apply combined filters: $200-450, 4+ stars, AC (lenient)",
        parent=trip_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A5 - Option dropdown 'Traveler Ranked' sorting
    traveler_ranked_mentioned = contains_any(hotels_list.sorting, ["traveler ranked"]) or contains_any(answer, ["traveler ranked"])
    evaluator.add_custom_node(
        result=traveler_ranked_mentioned,
        id="trip_sort_traveler_ranked",
        desc="[Action Node] tripadvisor.com:F1:A5 - Sort by 'Traveler Ranked'",
        parent=trip_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F1:P2 - Star rating recognition (>= 4)
    stars_ok_count = sum(1 for h in hotels if star_ok(h.star_rating))
    stars_ok = stars_ok_count >= 2  # lenient
    evaluator.add_custom_node(
        result=stars_ok,
        id="trip_star_badge",
        desc="[Perception Node] tripadvisor.com:F1:P2 - Recognize and report 4-star or higher ratings",
        parent=trip_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P10 - Facility info 'Air Conditioning'
    ac_ok_count = sum(1 for h in hotels if bool(h.includes_air_conditioning))
    ac_ok = ac_ok_count >= 2  # lenient
    evaluator.add_custom_node(
        result=ac_ok,
        id="trip_facility_ac",
        desc="[Perception Node] tripadvisor.com:F2:P10 - Identify 'Air Conditioning' facility for hotels",
        parent=trip_node,
        critical=False
    )

    # General checks: top 3 hotels have names, prices, star ratings, and detail page links
    def hotel_has_basic_fields(h: Hotel) -> bool:
        return bool(h.name) and bool(h.url) and (h.price_per_night is not None) and (h.star_rating is not None)

    basic_ok_count = sum(1 for h in hotels if hotel_has_basic_fields(h))
    evaluator.add_custom_node(
        result=basic_ok_count >= 2,
        id="trip_hotels_basic_fields",
        desc="At least two of the top three hotels include name, price, star rating, and detail URL",
        parent=trip_node,
        critical=False
    )

    # General check: all hotel prices are within 200-450 (lenient: at least 2/3)
    prices_ok_count = sum(1 for h in hotels if price_in_range(h.price_per_night))
    evaluator.add_custom_node(
        result=prices_ok_count >= 2,
        id="trip_prices_in_range",
        desc="At least two hotels have price per night within $200-$450",
        parent=trip_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()