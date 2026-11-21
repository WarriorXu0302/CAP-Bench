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
TASK_ID = "task-198aa0"
TASK_DESCRIPTION = 'I\'d like to get away for a short break, departing from New York (JFK) next Friday and returning on Sunday.\n\nFirst, please use the "Explore everywhere" feature on Skyscanner, only looking for direct flights, to find the 3 cheapest specific destination cities in different countries (choosing the cheapest city from each country).\n\nOnce these 3 cities are decided, go to TripAdvisor to find a restaurant for each city. The requirements are: it must specialize in local cuisine (e.g., if going to Mexico, look for tacos), and have a rating of 4.5 stars or higher. Most importantly, please open and check the most recent reviews to ensure no one is complaining that it\'s a "tourist rip-off" or "tourist trap".\n\nFinally, send me these 3 options, including the city, round-trip flight price, and the recommended restaurant.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DestinationCity(BaseModel):
    """A single destination city with its details"""
    city_name: Optional[str] = None
    country: Optional[str] = None
    flight_price: Optional[str] = None
    restaurant_name: Optional[str] = None
    restaurant_rating: Optional[str] = None
    cuisine_type: Optional[str] = None


class ExtractedDestinations(BaseModel):
    """All three destination cities extracted from the answer"""
    destinations: List[DestinationCity] = Field(default_factory=list)
    mentions_direct_flights: Optional[bool] = None
    mentions_skyscanner: Optional[bool] = None
    mentions_tripadvisor: Optional[bool] = None
    mentions_tourist_trap_check: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_destinations_from_answer() -> str:
    return """
Extract the 3 destination city options provided in the answer.

For each destination, extract:
- city_name: the specific city name
- country: the country where the city is located
- flight_price: the round-trip flight price exactly as stated (include currency if present)
- restaurant_name: the recommended restaurant name
- restaurant_rating: the restaurant's rating exactly as stated
- cuisine_type: the type of cuisine the restaurant specializes in

Also extract these flags:
- mentions_direct_flights: true if the answer mentions direct flights or non-stop flights
- mentions_skyscanner: true if the answer mentions Skyscanner
- mentions_tripadvisor: true if the answer mentions TripAdvisor
- mentions_tourist_trap_check: true if the answer mentions checking reviews for tourist traps or tourist rip-offs

If any field is missing, set it to null or false for boolean fields.
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
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_currency = has_any_ci(text, ['$', 'usd', 'dollar', '€', 'eur', '£', 'gbp'])
    return has_number and has_currency


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def has_valid_cities(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    valid_count = sum(1 for d in destinations if d.city_name and d.city_name.strip())
    return valid_count >= 3


def has_different_countries(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    countries = set()
    for d in destinations:
        if d.country and d.country.strip():
            countries.add(d.country.lower().strip())
    return len(countries) >= 3


def has_valid_prices(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    valid_count = sum(1 for d in destinations if looks_like_price(d.flight_price))
    return valid_count >= 3


def has_valid_restaurants(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    valid_count = sum(1 for d in destinations if d.restaurant_name and d.restaurant_name.strip())
    return valid_count >= 3


def has_valid_ratings(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    valid_count = sum(1 for d in destinations if looks_like_rating(d.restaurant_rating))
    return valid_count >= 3


def has_high_ratings(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    high_rating_count = 0
    for d in destinations:
        if looks_like_rating(d.restaurant_rating):
            num = extract_float(d.restaurant_rating)
            if num is not None and num >= 4.5:
                high_rating_count += 1
    return high_rating_count >= 3


def has_cuisine_types(destinations: List[DestinationCity]) -> bool:
    if not destinations or len(destinations) < 3:
        return False
    valid_count = sum(1 for d in destinations if d.cuisine_type and d.cuisine_type.strip())
    return valid_count >= 3


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_destinations_from_answer(),
        template_class=ExtractedDestinations,
        extraction_name="destinations_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner flight search with Explore Everywhere and direct flights filter",
        parent=root,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A5 - Direct flights filter
    direct_flights_ok = (
        extracted.mentions_direct_flights or
        has_any_ci(answer, ['direct flight', 'non-stop', 'nonstop', 'direct only'])
    )
    evaluator.add_custom_node(
        result=bool(direct_flights_ok),
        id="skyscanner_action_direct_flights",
        desc="[Action Node] skyscanner.com:F1:A5 - Filter to show only direct flights using checkbox or filter",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F1:P1 - Identify 3 cheapest cities from different countries
    cities_ok = has_valid_cities(extracted.destinations)
    countries_ok = has_different_countries(extracted.destinations)
    prices_ok = has_valid_prices(extracted.destinations)

    evaluator.add_custom_node(
        result=bool(cities_ok and countries_ok and prices_ok),
        id="skyscanner_perception_cheapest_cities",
        desc="[Perception Node] skyscanner.com:F1:P1 - Identify 3 cheapest destination cities from different countries with their prices",
        parent=skyscanner_node,
        critical=False
    )

    # Additional context checks
    skyscanner_mentioned = extracted.mentions_skyscanner or has_any_ci(answer, ['skyscanner'])
    evaluator.add_custom_node(
        result=bool(skyscanner_mentioned),
        id="skyscanner_mentioned",
        desc="Mentions using Skyscanner for flight search",
        parent=skyscanner_node,
        critical=False
    )

    explore_everywhere_ok = has_any_ci(answer, ['explore everywhere', 'everywhere'])
    evaluator.add_custom_node(
        result=bool(explore_everywhere_ok),
        id="skyscanner_explore_everywhere",
        desc="Mentions using the 'Explore everywhere' feature",
        parent=skyscanner_node,
        critical=False
    )

    jfk_mentioned = has_any_ci(answer, ['jfk', 'new york'])
    evaluator.add_custom_node(
        result=bool(jfk_mentioned),
        id="skyscanner_departure_jfk",
        desc="Mentions departure from JFK or New York",
        parent=skyscanner_node,
        critical=False
    )

    # 3.2 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor restaurant search with local cuisine, high ratings, and review verification",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F6:A6 - Filter by local cuisine type
    has_restaurants = has_valid_restaurants(extracted.destinations)
    has_cuisines = has_cuisine_types(extracted.destinations)

    evaluator.add_custom_node(
        result=bool(has_restaurants and has_cuisines),
        id="tripadvisor_action_cuisine_filter",
        desc="[Action Node] tripadvisor.com:F6:A6 - Filter restaurants by local/regional cuisine type for each destination",
        parent=tripadvisor_node,
        critical=False
    )

    # Rating filter check
    has_ratings = has_valid_ratings(extracted.destinations)
    high_ratings = has_high_ratings(extracted.destinations)

    evaluator.add_custom_node(
        result=bool(has_ratings and high_ratings),
        id="tripadvisor_rating_filter",
        desc="Filter restaurants with rating of 4.5 stars or higher",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F3:P6 - Semantic understanding of reviews
    tourist_trap_check_ok = (
        extracted.mentions_tourist_trap_check or
        has_any_ci(answer, ['tourist trap', 'tourist rip', 'rip-off', 'ripoff', 'review', 'recent review'])
    )

    evaluator.add_custom_node(
        result=bool(tourist_trap_check_ok),
        id="tripadvisor_perception_tourist_trap",
        desc="[Perception Node] tripadvisor.com:F3:P6 - Verify reviews do not mention tourist traps or rip-offs through semantic understanding",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F3:A10 - Expand review text if needed
    review_interaction_ok = has_any_ci(answer, [
        'review', 'checked', 'checked review', 'read review',
        'most recent', 'latest review', 'opened', 'expanded'
    ])

    evaluator.add_custom_node(
        result=bool(review_interaction_ok),
        id="tripadvisor_action_expand_reviews",
        desc="[Action Node] tripadvisor.com:F3:A10 - Open and read full review text (expand if truncated)",
        parent=tripadvisor_node,
        critical=False
    )

    # Additional context checks
    tripadvisor_mentioned = extracted.mentions_tripadvisor or has_any_ci(answer, ['tripadvisor', 'trip advisor'])
    evaluator.add_custom_node(
        result=bool(tripadvisor_mentioned),
        id="tripadvisor_mentioned",
        desc="Mentions using TripAdvisor for restaurant search",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.3 Final deliverable check
    deliverable_node = evaluator.add_parallel(
        id="deliverable_section",
        desc="Final deliverable with all 3 destination options complete",
        parent=root,
        critical=False
    )

    all_three_complete = (
        has_valid_cities(extracted.destinations) and
        has_different_countries(extracted.destinations) and
        has_valid_prices(extracted.destinations) and
        has_valid_restaurants(extracted.destinations)
    )

    evaluator.add_custom_node(
        result=bool(all_three_complete),
        id="deliverable_three_options",
        desc="Provides all 3 complete destination options with city, price, and restaurant",
        parent=deliverable_node,
        critical=False
    )

    local_cuisine_check = has_cuisines and has_any_ci(answer, ['local', 'traditional', 'authentic'])
    evaluator.add_custom_node(
        result=bool(local_cuisine_check),
        id="deliverable_local_cuisine",
        desc="Restaurants specialize in local or traditional cuisine",
        parent=deliverable_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
