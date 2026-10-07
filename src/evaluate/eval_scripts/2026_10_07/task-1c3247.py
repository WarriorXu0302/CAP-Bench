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
TASK_ID = "task-1c3247"
TASK_DESCRIPTION = 'I’m planning a short getaway on a weekend next month (check in on Friday, check out on Sunday), but I have fairly severe allergies and am deciding between Austin, TX and Seattle, WA. First, check AccuWeather’s **Health & Activities** forecast for both cities for that travel weekend and compare the **Mold** or **Pollen** risk levels. If the forecast for that weekend is not yet available, use the nearest available date within the next two weeks and clearly note the date used. For extra assurance, then check EPA AirNow for the current **AQI** values in both cities. Based on this data, choose the city with more allergy-friendly air conditions.\n\nAfter selecting the city, go to TripAdvisor and find 3 hotels in that city. Requirements: stay dates must match my trip window (Friday to Sunday of next month), nightly price must be between **$200 and $450**, hotels must be **4-star or above**, and they must include **Air Conditioning** to reduce allergy risk from open windows. Finally, sort by **Traveler Ranked** and list the top 3 hotels.\n\nOutput required: mold/pollen levels for both cities, AQI values for both cities, the final selected city, and for each recommended hotel: name, price, star rating, whether air conditioning is included, and the detail page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CityAllergyData(BaseModel):
    """Allergy data extracted for a city"""
    city_name: Optional[str] = None
    mold_level: Optional[str] = None
    pollen_level: Optional[str] = None
    aqi_value: Optional[str] = None


class SelectedCity(BaseModel):
    """The city selected based on allergy conditions"""
    selected_city: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel information extracted"""
    name: Optional[str] = None
    price: Optional[str] = None
    star_rating: Optional[str] = None
    has_air_conditioning: Optional[bool] = None
    detail_link: Optional[str] = None


class HotelsOutput(BaseModel):
    """All hotels extracted from answer"""
    hotels: Optional[List[HotelInfo]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_austin_data() -> str:
    return """
Extract the allergy-related data for Austin, TX from the answer:

- city_name: should be "Austin" or "Austin, TX"
- mold_level: the mold risk level (e.g., Low, Moderate, High) from AccuWeather Health & Activities forecast
- pollen_level: the pollen risk level (e.g., Low, Moderate, High) from AccuWeather Health & Activities forecast
- aqi_value: the AQI value from EPA AirNow for Austin

If any field is missing, set it to null.
"""


def prompt_extract_seattle_data() -> str:
    return """
Extract the allergy-related data for Seattle, WA from the answer:

- city_name: should be "Seattle" or "Seattle, WA"
- mold_level: the mold risk level (e.g., Low, Moderate, High) from AccuWeather Health & Activities forecast
- pollen_level: the pollen risk level (e.g., Low, Moderate, High) from AccuWeather Health & Activities forecast
- aqi_value: the AQI value from EPA AirNow for Seattle

If any field is missing, set it to null.
"""


def prompt_extract_selected_city() -> str:
    return """
Extract which city was selected as more allergy-friendly from the answer:

- selected_city: the name of the chosen city (Austin or Seattle)

If not clearly stated, set it to null.
"""


def prompt_extract_hotels() -> str:
    return """
Extract the list of recommended hotels from the answer. For each hotel, extract:

- name: hotel name
- price: nightly price as stated (include currency/units if present)
- star_rating: star rating (e.g., "4", "4.5", "5")
- has_air_conditioning: true if air conditioning is mentioned as included, false otherwise
- detail_link: the TripAdvisor detail page link/URL

Return a list of hotels. If none are present, return an empty list.
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


def has_risk_level(text: Optional[str]) -> bool:
    """Check if text contains a valid risk level indicator"""
    if not text:
        return False
    return has_any_ci(text, ['low', 'moderate', 'high', 'very high', 'extreme'])


def in_price_range(price_text: Optional[str], min_price: float = 200.0, max_price: float = 450.0) -> bool:
    """Check if price is within specified range"""
    if not price_text:
        return False
    price_val = extract_float(price_text)
    if price_val is None:
        return False
    return min_price <= price_val <= max_price


def is_four_star_or_above(rating_text: Optional[str]) -> bool:
    """Check if star rating is 4 or above"""
    if not rating_text:
        return False
    rating_val = extract_float(rating_text)
    if rating_val is None:
        return False
    return rating_val >= 4.0


def mentions_both_cities(answer_text: str) -> bool:
    """Check if answer mentions both Austin and Seattle"""
    has_austin = has_any_ci(answer_text, ['austin'])
    has_seattle = has_any_ci(answer_text, ['seattle'])
    return has_austin and has_seattle


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
    austin_data = await evaluator.extract(
        prompt=prompt_extract_austin_data(),
        template_class=CityAllergyData,
        extraction_name="austin_allergy_data"
    )

    seattle_data = await evaluator.extract(
        prompt=prompt_extract_seattle_data(),
        template_class=CityAllergyData,
        extraction_name="seattle_allergy_data"
    )

    selected_city_data = await evaluator.extract(
        prompt=prompt_extract_selected_city(),
        template_class=SelectedCity,
        extraction_name="selected_city"
    )

    hotels_data = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsOutput,
        extraction_name="hotels_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 AccuWeather section
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather Health & Activities forecast for Austin and Seattle",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F1:A1 - Search for cities
    cities_searched = mentions_both_cities(answer)
    evaluator.add_custom_node(
        result=bool(cities_searched),
        id="accuweather_search_cities",
        desc="[Action Node] accuweather.com:F1:A1 - Search for both Austin, TX and Seattle, WA on AccuWeather",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F7:A2 - Navigate to Health & Activities tab
    health_activities_mentioned = has_any_ci(answer, ['health', 'activities', 'health & activities', 'health and activities'])
    evaluator.add_custom_node(
        result=bool(health_activities_mentioned),
        id="accuweather_health_activities_tab",
        desc="[Action Node] accuweather.com:F7:A2 - Navigate to the Health & Activities tab for forecast data",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F7:P10 - Extract mold/pollen levels for both cities
    austin_has_risk = has_risk_level(austin_data.mold_level) or has_risk_level(austin_data.pollen_level)
    seattle_has_risk = has_risk_level(seattle_data.mold_level) or has_risk_level(seattle_data.pollen_level)
    both_cities_have_data = austin_has_risk and seattle_has_risk

    evaluator.add_custom_node(
        result=bool(both_cities_have_data),
        id="accuweather_extract_risk_levels",
        desc="[Perception Node] accuweather.com:F7:P10 - Extract mold or pollen risk levels (Low/Moderate/High) for both cities",
        parent=accuweather_node,
        critical=False
    )

    # 3.2 EPA AirNow section
    airnow_node = evaluator.add_sequential(
        id="airnow_section",
        desc="EPA AirNow AQI values for Austin and Seattle",
        parent=root,
        critical=False
    )

    # Check if AQI values are present for both cities
    austin_has_aqi = austin_data.aqi_value and contains_digits(austin_data.aqi_value)
    seattle_has_aqi = seattle_data.aqi_value and contains_digits(seattle_data.aqi_value)
    both_aqi_present = austin_has_aqi and seattle_has_aqi

    airnow_mentioned = has_any_ci(answer, ['airnow', 'air now', 'epa'])
    evaluator.add_custom_node(
        result=bool(airnow_mentioned and both_aqi_present),
        id="airnow_extract_aqi",
        desc="Extract current AQI values from EPA AirNow for both cities",
        parent=airnow_node,
        critical=False
    )

    # 3.3 City selection
    city_selection_node = evaluator.add_sequential(
        id="city_selection",
        desc="Select the more allergy-friendly city based on data",
        parent=root,
        critical=False
    )

    city_selected = selected_city_data and selected_city_data.selected_city and selected_city_data.selected_city.strip()
    city_is_austin_or_seattle = False
    if city_selected:
        city_is_austin_or_seattle = has_any_ci(selected_city_data.selected_city, ['austin', 'seattle'])

    evaluator.add_custom_node(
        result=bool(city_selected and city_is_austin_or_seattle),
        id="city_decision_made",
        desc="A clear city choice (Austin or Seattle) is made based on allergy data",
        parent=city_selection_node,
        critical=False
    )

    # 3.4 TripAdvisor hotel search
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor hotel search and filtering in the selected city",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A1 - Multi-field search with city
    tripadvisor_mentioned = has_any_ci(answer, ['tripadvisor'])
    evaluator.add_custom_node(
        result=bool(tripadvisor_mentioned and city_selected),
        id="tripadvisor_search_city",
        desc="[Action Node] tripadvisor.com:F1:A1 - Search for hotels in the selected city on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A2 - Date range selection (Friday to Sunday)
    dates_mentioned = has_any_ci(answer, ['friday', 'sunday', 'check in', 'check out', 'weekend'])
    evaluator.add_custom_node(
        result=bool(dates_mentioned),
        id="tripadvisor_date_selection",
        desc="[Action Node] tripadvisor.com:F1:A2 - Select check-in (Friday) and check-out (Sunday) dates",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-condition filtering (price, star rating, amenities)
    price_filter_mentioned = has_any_ci(answer, ['200', '450', 'price'])
    star_filter_mentioned = has_any_ci(answer, ['4-star', '4 star', 'four star', 'star rating'])
    ac_filter_mentioned = has_any_ci(answer, ['air conditioning', 'ac', 'air conditioned'])

    filters_applied = price_filter_mentioned and star_filter_mentioned and ac_filter_mentioned
    evaluator.add_custom_node(
        result=bool(filters_applied),
        id="tripadvisor_apply_filters",
        desc="[Action Node] tripadvisor.com:F1:A4 - Apply filters: price ($200-$450), 4+ stars, air conditioning",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A5 - Sort by Traveler Ranked
    sort_mentioned = has_any_ci(answer, ['traveler ranked', 'ranked', 'sorted', 'sort'])
    evaluator.add_custom_node(
        result=bool(sort_mentioned),
        id="tripadvisor_sort_traveler_ranked",
        desc="[Action Node] tripadvisor.com:F1:A5 - Sort hotels by Traveler Ranked",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.5 Hotel results validation
    hotel_results_node = evaluator.add_sequential(
        id="hotel_results",
        desc="Validate extracted hotel information meets requirements",
        parent=root,
        critical=False
    )

    # Check that at least 3 hotels are provided
    hotels_list = hotels_data.hotels if hotels_data and hotels_data.hotels else []
    has_three_hotels = len(hotels_list) >= 3

    evaluator.add_custom_node(
        result=bool(has_three_hotels),
        id="hotel_count_check",
        desc="At least 3 hotels are listed in the output",
        parent=hotel_results_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F1:P2 - Star rating badge recognition
    hotels_with_valid_stars = 0
    for hotel in hotels_list[:3]:
        if is_four_star_or_above(hotel.star_rating):
            hotels_with_valid_stars += 1

    star_rating_ok = hotels_with_valid_stars >= 3 if has_three_hotels else hotels_with_valid_stars == len(hotels_list) and len(hotels_list) > 0

    evaluator.add_custom_node(
        result=bool(star_rating_ok),
        id="tripadvisor_star_rating_perception",
        desc="[Perception Node] tripadvisor.com:F1:P2 - All listed hotels have 4-star or above rating",
        parent=hotel_results_node,
        critical=False
    )

    # Check price range compliance
    hotels_with_valid_price = 0
    for hotel in hotels_list[:3]:
        if in_price_range(hotel.price):
            hotels_with_valid_price += 1

    price_range_ok = hotels_with_valid_price >= 3 if has_three_hotels else hotels_with_valid_price == len(hotels_list) and len(hotels_list) > 0

    evaluator.add_custom_node(
        result=bool(price_range_ok),
        id="hotel_price_range_check",
        desc="All listed hotels have prices between $200 and $450",
        parent=hotel_results_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P10 - Air conditioning amenity identification
    hotels_with_ac = 0
    for hotel in hotels_list[:3]:
        if hotel.has_air_conditioning:
            hotels_with_ac += 1

    ac_ok = hotels_with_ac >= 3 if has_three_hotels else hotels_with_ac == len(hotels_list) and len(hotels_list) > 0

    evaluator.add_custom_node(
        result=bool(ac_ok),
        id="tripadvisor_ac_amenity_perception",
        desc="[Perception Node] tripadvisor.com:F2:P10 - All listed hotels include air conditioning amenity",
        parent=hotel_results_node,
        critical=False
    )

    # Check that detail links are provided
    hotels_with_links = 0
    for hotel in hotels_list[:3]:
        if hotel.detail_link and hotel.detail_link.strip():
            hotels_with_links += 1

    links_ok = hotels_with_links >= 3 if has_three_hotels else hotels_with_links == len(hotels_list) and len(hotels_list) > 0

    evaluator.add_custom_node(
        result=bool(links_ok),
        id="hotel_detail_links_check",
        desc="All listed hotels include detail page links",
        parent=hotel_results_node,
        critical=False
    )

    # 3.6 Output completeness check
    output_node = evaluator.add_sequential(
        id="output_completeness",
        desc="Check that all required output fields are present",
        parent=root,
        critical=False
    )

    has_comparison_data = both_cities_have_data and both_aqi_present
    has_selected_city_output = city_selected and city_is_austin_or_seattle
    has_hotel_output = has_three_hotels

    evaluator.add_custom_node(
        result=bool(has_comparison_data),
        id="output_comparison_data",
        desc="Output includes mold/pollen levels and AQI for both cities",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_selected_city_output),
        id="output_selected_city",
        desc="Output clearly states which city was selected",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_hotel_output),
        id="output_hotel_details",
        desc="Output includes name, price, star rating, AC status, and link for each hotel",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
