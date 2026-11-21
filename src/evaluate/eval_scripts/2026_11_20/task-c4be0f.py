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
TASK_ID = "task-c4be0f"
TASK_DESCRIPTION = "I am planning a trip from New York to Paris for an upcoming month and I am looking for an economical flight option with a comfortable layover experience.\n\nFirst, use Google Flights to search for connecting flights from New York to Paris (allowing one layover). Set the layover duration between 3 and 8 hours. Identify the 3 cheapest options, noting the flight numbers, layover airport, layover duration, and price.\n\nThen, go to Skyscanner and input the same flight route and dates to verify the prices of these 3 options. If Skyscanner offers a cheaper equivalent option, also record it.\n\nNext, for each layover airport involved in these options (e.g., Frankfurt, Amsterdam, London Heathrow), search Google Maps to view the terminal layout. Determine if a terminal change is required for the transfer and estimate the walking distance.\n\nThen, use Yelp to search for restaurants within each airport (using the search query 'airport name + restaurant'). Filter for restaurants with a rating of 4 stars or higher and located in the post-security area. Recommend 2 restaurants for each airport.\n\nFinally, use TripAdvisor to search for lounges in each airport (using the search query 'airport name + lounge'). Check ratings, user reviews, and access policies (Priority Pass, credit card benefits, or paid access). Recommend 1-2 of the best-reviewed lounges for each airport.\n\nOutput for each flight option: flight number, layover airport name, layover duration, Google Flights price, Skyscanner price comparison result, whether a terminal change is required and estimated walking distance, recommended restaurant names and Yelp ratings, recommended lounge names and TripAdvisor ratings and access methods, and links to detail pages on each platform."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightOption(BaseModel):
    """A single flight option extracted from the answer"""
    flight_numbers: Optional[str] = None
    layover_airport: Optional[str] = None
    layover_duration: Optional[str] = None
    google_flights_price: Optional[str] = None


class PriceComparison(BaseModel):
    """Price comparison results from Skyscanner"""
    skyscanner_prices_mentioned: Optional[bool] = None
    has_price_comparison: Optional[bool] = None


class AirportInfo(BaseModel):
    """Airport terminal and facilities information"""
    airport_name: Optional[str] = None
    terminal_change_info: Optional[str] = None
    walking_distance: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Restaurant recommendations"""
    restaurant_names: Optional[List[str]] = Field(default_factory=list)
    ratings_mentioned: Optional[bool] = None


class LoungeInfo(BaseModel):
    """Lounge recommendations"""
    lounge_names: Optional[List[str]] = Field(default_factory=list)
    access_methods_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight_options() -> str:
    return """
Extract information about the 3 cheapest flight options from New York to Paris mentioned in the answer.

For each option, identify:
- flight_numbers: the flight numbers mentioned (e.g., "AA123, BA456")
- layover_airport: the airport where the layover occurs (e.g., "Frankfurt", "Amsterdam")
- layover_duration: the duration of the layover (e.g., "5 hours", "3h 30m")
- google_flights_price: the price from Google Flights (e.g., "$650", "€580")

Return information for at least one option. Set fields to null if not found.
"""


def prompt_extract_price_comparison() -> str:
    return """
From the answer, determine if Skyscanner price verification was performed.

Return:
- skyscanner_prices_mentioned: true if Skyscanner prices are mentioned, false otherwise
- has_price_comparison: true if there is any comparison between Google Flights and Skyscanner prices, false otherwise

Set to null if unclear.
"""


def prompt_extract_airport_info() -> str:
    return """
Extract information about airport terminal layouts and transfers from the answer.

Look for:
- airport_name: name of any layover airport mentioned (e.g., "Frankfurt Airport")
- terminal_change_info: whether terminal changes are required (e.g., "terminal change required", "same terminal")
- walking_distance: estimated walking distance or time (e.g., "10 minute walk", "500 meters")

Set fields to null if not found.
"""


def prompt_extract_restaurant_info() -> str:
    return """
Extract restaurant recommendations from the answer.

Return:
- restaurant_names: list of restaurant names mentioned (at least one if any are present)
- ratings_mentioned: true if any Yelp ratings (numeric or star ratings) are mentioned, false otherwise

Set to null or empty list if not found.
"""


def prompt_extract_lounge_info() -> str:
    return """
Extract lounge recommendations from the answer.

Return:
- lounge_names: list of lounge names mentioned (at least one if any are present)
- access_methods_mentioned: true if access methods (Priority Pass, credit card, paid access) are mentioned, false otherwise

Set to null or empty list if not found.
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


def looks_like_flight_number(text: Optional[str]) -> bool:
    if not text:
        return False
    # Flight numbers typically have letters and numbers
    has_letter = bool(re.search(r'[A-Za-z]', text))
    has_digit = bool(re.search(r'\d', text))
    return has_letter and has_digit


def looks_like_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    # Duration patterns: "3 hours", "5h", "3-8 hours", etc.
    return has_any_ci(text, ['hour', 'hr', 'h']) or (contains_digits(text) and re.search(r'\d+[-–]\d+', text))


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Price indicators: currency symbols or digits with currency codes
    return contains_digits(text) and has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp'])


def extract_numeric_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Look for patterns like "4.5", "4", "4.5 stars"
    match = re.search(r'(\d+(\.\d+)?)\s*(star|rating)?', text.lower())
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return None


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
    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_options(),
        template_class=FlightOption,
        extraction_name="flight_options"
    )

    price_comparison = await evaluator.extract(
        prompt=prompt_extract_price_comparison(),
        template_class=PriceComparison,
        extraction_name="price_comparison"
    )

    airport_info = await evaluator.extract(
        prompt=prompt_extract_airport_info(),
        template_class=AirportInfo,
        extraction_name="airport_info"
    )

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant_info(),
        template_class=RestaurantInfo,
        extraction_name="restaurant_info"
    )

    lounge_info = await evaluator.extract(
        prompt=prompt_extract_lounge_info(),
        template_class=LoungeInfo,
        extraction_name="lounge_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Google Flights section
    google_flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights search for New York to Paris connecting flights",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F1:A1 - Form input with autocomplete
    flights_input_ok = (has_any_ci(answer, ['google flights', 'google.com/flights']) and
                        has_any_ci(answer, ['new york']) and
                        has_any_ci(answer, ['paris']))
    evaluator.add_custom_node(
        result=bool(flights_input_ok),
        id="google_flights_input_route",
        desc="[Action Node] google.comflights:F1:A1 - Enter departure (New York) and destination (Paris) in search form",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - Date selection
    date_selection_ok = has_any_ci(answer, ['march', '2026', 'upcoming month']) or contains_digits(answer)
    evaluator.add_custom_node(
        result=bool(date_selection_ok),
        id="google_flights_date_selection",
        desc="[Action Node] google.comflights:F1:A2 - Select travel dates for March 2026",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - Multi-condition filter (layover duration 3-8 hours)
    layover_filter_ok = (flight_info and flight_info.layover_duration and
                         looks_like_duration(flight_info.layover_duration))
    evaluator.add_custom_node(
        result=bool(layover_filter_ok),
        id="google_flights_layover_filter",
        desc="[Action Node] google.comflights:F3:A5 - Set layover duration filter between 3 and 8 hours",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A15 - Sort by price
    price_sorting_ok = (flight_info and flight_info.google_flights_price and
                        looks_like_price(flight_info.google_flights_price))
    evaluator.add_custom_node(
        result=bool(price_sorting_ok),
        id="google_flights_price_sort",
        desc="[Action Node] google.comflights:F1:A15 - Sort results by price to find 3 cheapest options",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F6:A14 - Click to expand flight details
    flight_details_ok = (flight_info and flight_info.flight_numbers and
                         looks_like_flight_number(flight_info.flight_numbers))
    evaluator.add_custom_node(
        result=bool(flight_details_ok),
        id="google_flights_expand_details",
        desc="[Action Node] google.comflights:F6:A14 - Click flight card to view detailed information",
        parent=google_flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P11 - Extract layover duration
    layover_duration_ok = (flight_info and flight_info.layover_duration and
                           looks_like_duration(flight_info.layover_duration))
    evaluator.add_custom_node(
        result=bool(layover_duration_ok),
        id="google_flights_layover_duration",
        desc="[Perception Node] google.comflights:F6:P11 - Extract layover duration from flight details",
        parent=google_flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F1:P1 - Extract price data
    price_extraction_ok = (flight_info and flight_info.google_flights_price and
                           looks_like_price(flight_info.google_flights_price))
    evaluator.add_custom_node(
        result=bool(price_extraction_ok),
        id="google_flights_price_extraction",
        desc="[Perception Node] google.comflights:F1:P1 - Extract price values from flight listings",
        parent=google_flights_node,
        critical=False
    )

    # 3.2 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner price verification for the same route",
        parent=root,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A1 - Form input and search
    skyscanner_input_ok = (price_comparison and price_comparison.skyscanner_prices_mentioned and
                           has_any_ci(answer, ['skyscanner']))
    evaluator.add_custom_node(
        result=bool(skyscanner_input_ok),
        id="skyscanner_search_input",
        desc="[Action Node] skyscanner.com:F1:A1 - Enter same route (New York to Paris) in Skyscanner search",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A7 - Sort by price
    skyscanner_sort_ok = (price_comparison and price_comparison.has_price_comparison)
    evaluator.add_custom_node(
        result=bool(skyscanner_sort_ok),
        id="skyscanner_price_sort",
        desc="[Action Node] skyscanner.com:F1:A7 - Sort by price to find cheapest options for verification",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F1:P1 - Extract and compare prices
    skyscanner_price_perception_ok = (price_comparison and
                                      price_comparison.skyscanner_prices_mentioned and
                                      price_comparison.has_price_comparison)
    evaluator.add_custom_node(
        result=bool(skyscanner_price_perception_ok),
        id="skyscanner_price_perception",
        desc="[Perception Node] skyscanner.com:F1:P1 - Extract Skyscanner prices and compare with Google Flights",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F1:P2 - Match flight times and options
    skyscanner_time_match_ok = (price_comparison and price_comparison.skyscanner_prices_mentioned and
                                flight_info and flight_info.flight_numbers)
    evaluator.add_custom_node(
        result=bool(skyscanner_time_match_ok),
        id="skyscanner_time_matching",
        desc="[Perception Node] skyscanner.com:F1:P2 - Match flight times to verify same options as Google Flights",
        parent=skyscanner_node,
        critical=False
    )

    # 3.3 Google Maps section
    google_maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps airport terminal layout investigation",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Search for airport location
    maps_search_ok = (airport_info and airport_info.airport_name and
                      has_any_ci(answer, ['google maps', 'maps.google']))
    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="google_maps_airport_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for layover airport on Google Maps",
        parent=google_maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F3:A9 - Zoom and pan to view terminals
    maps_zoom_ok = (airport_info and airport_info.terminal_change_info)
    evaluator.add_custom_node(
        result=bool(maps_zoom_ok),
        id="google_maps_zoom_terminals",
        desc="[Action Node] maps.google.com:F3:A9 - Zoom and pan map to view terminal layout details",
        parent=google_maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P12 - Identify terminal markers
    terminal_markers_ok = (airport_info and airport_info.terminal_change_info and
                           has_any_ci(airport_info.terminal_change_info, ['terminal', 'change', 'same']))
    evaluator.add_custom_node(
        result=bool(terminal_markers_ok),
        id="google_maps_terminal_markers",
        desc="[Perception Node] maps.google.com:F3:P12 - Identify terminal buildings and connections on map",
        parent=google_maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P13 - Estimate walking distance
    distance_perception_ok = (airport_info and airport_info.walking_distance and
                              (contains_digits(airport_info.walking_distance) or
                               has_any_ci(airport_info.walking_distance, ['walk', 'distance', 'meter', 'minute'])))
    evaluator.add_custom_node(
        result=bool(distance_perception_ok),
        id="google_maps_distance_estimate",
        desc="[Perception Node] maps.google.com:F3:P13 - Estimate walking distance from map scale",
        parent=google_maps_node,
        critical=False
    )

    # 3.4 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp restaurant search at layover airports",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Keyword search
    yelp_search_ok = (restaurant_info and restaurant_info.restaurant_names and
                      len(restaurant_info.restaurant_names) > 0 and
                      has_any_ci(answer, ['yelp']))
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_restaurant_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for restaurants using 'airport name + restaurant' query",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-condition filter (4+ stars, post-security)
    yelp_filter_ok = (restaurant_info and restaurant_info.ratings_mentioned and
                      has_any_ci(answer, ['4 star', '4-star', '4+', 'post-security', 'after security']))
    evaluator.add_custom_node(
        result=bool(yelp_filter_ok),
        id="yelp_rating_filter",
        desc="[Action Node] yelp.com:F1:A2 - Filter for 4+ star ratings and post-security location",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A3 - Sort by rating
    yelp_sort_ok = (restaurant_info and restaurant_info.ratings_mentioned)
    evaluator.add_custom_node(
        result=bool(yelp_sort_ok),
        id="yelp_rating_sort",
        desc="[Action Node] yelp.com:F1:A3 - Sort by rating to find best restaurants",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Extract restaurant names and basic info
    yelp_name_perception_ok = (restaurant_info and restaurant_info.restaurant_names and
                               len(restaurant_info.restaurant_names) >= 1)
    evaluator.add_custom_node(
        result=bool(yelp_name_perception_ok),
        id="yelp_restaurant_names",
        desc="[Perception Node] yelp.com:F1:P1 - Extract restaurant names from search results",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P7 - Verify post-security location
    yelp_location_perception_ok = (restaurant_info and restaurant_info.restaurant_names and
                                   has_any_ci(answer, ['post-security', 'after security', 'airside']))
    evaluator.add_custom_node(
        result=bool(yelp_location_perception_ok),
        id="yelp_location_verification",
        desc="[Perception Node] yelp.com:F2:P7 - Verify restaurants are in post-security area",
        parent=yelp_node,
        critical=False
    )

    # 3.5 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor lounge search at layover airports",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A1 - Search for lounges
    tripadvisor_search_ok = (lounge_info and lounge_info.lounge_names and
                             len(lounge_info.lounge_names) > 0 and
                             has_any_ci(answer, ['tripadvisor']))
    evaluator.add_custom_node(
        result=bool(tripadvisor_search_ok),
        id="tripadvisor_lounge_search",
        desc="[Action Node] tripadvisor.com:F4:A1 - Search for lounges using 'airport name + lounge' query",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A5 - Sort by rating
    tripadvisor_sort_ok = (lounge_info and lounge_info.lounge_names and
                           has_any_ci(answer, ['rating', 'review', 'best']))
    evaluator.add_custom_node(
        result=bool(tripadvisor_sort_ok),
        id="tripadvisor_rating_sort",
        desc="[Action Node] tripadvisor.com:F4:A5 - Sort by rating to find best-reviewed lounges",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F4:P14 - Extract lounge list information
    tripadvisor_list_perception_ok = (lounge_info and lounge_info.lounge_names and
                                      len(lounge_info.lounge_names) >= 1)
    evaluator.add_custom_node(
        result=bool(tripadvisor_list_perception_ok),
        id="tripadvisor_lounge_names",
        desc="[Perception Node] tripadvisor.com:F4:P14 - Extract lounge names and ratings from search results",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F3:A10 - Expand details for access policy
    tripadvisor_expand_ok = (lounge_info and lounge_info.access_methods_mentioned)
    evaluator.add_custom_node(
        result=bool(tripadvisor_expand_ok),
        id="tripadvisor_expand_details",
        desc="[Action Node] tripadvisor.com:F3:A10 - Click to view detailed access policy information",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F3:P6 - Extract access policy from reviews
    tripadvisor_access_perception_ok = (lounge_info and lounge_info.access_methods_mentioned and
                                        has_any_ci(answer, ['priority pass', 'credit card', 'paid', 'access']))
    evaluator.add_custom_node(
        result=bool(tripadvisor_access_perception_ok),
        id="tripadvisor_access_policy",
        desc="[Perception Node] tripadvisor.com:F3:P6 - Extract access methods (Priority Pass, credit card, paid) from reviews",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.6 Overall completeness checks (non-prefixed nodes)
    completeness_node = evaluator.add_parallel(
        id="overall_completeness",
        desc="Overall task completeness checks",
        parent=root,
        critical=False
    )

    # Check if 3 flight options are provided
    three_options_ok = has_any_ci(answer, ['option 1', 'option 2', 'option 3']) or \
                       has_any_ci(answer, ['first option', 'second option', 'third option']) or \
                       (answer.count('flight') >= 3 and answer.count('price') >= 3)
    evaluator.add_custom_node(
        result=bool(three_options_ok),
        id="three_options_provided",
        desc="Provides information for 3 different flight options",
        parent=completeness_node,
        critical=False
    )

    # Check if recommendations are provided (2 restaurants per airport, 1-2 lounges)
    recommendations_ok = (restaurant_info and len(restaurant_info.restaurant_names) >= 2 and
                          lounge_info and len(lounge_info.lounge_names) >= 1)
    evaluator.add_custom_node(
        result=bool(recommendations_ok),
        id="recommendations_provided",
        desc="Provides restaurant and lounge recommendations for layover airports",
        parent=completeness_node,
        critical=False
    )

    # Check if links are provided
    links_ok = has_any_ci(answer, ['http', 'www', '.com', 'link'])
    evaluator.add_custom_node(
        result=bool(links_ok),
        id="platform_links_provided",
        desc="Includes links to detail pages on various platforms",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
