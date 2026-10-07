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
TASK_ID = "task-bd7f2b"
TASK_DESCRIPTION = "As a huge fan of 'The White Lotus', I'm planning a spring trip to Sicily. First, please go to IMDb to find information on the filming locations for Season 2 and identify the real name and exact town of the iconic seaside hotel featured in the show. Next, locate this hotel on Google Maps. To ensure proximity to a key attraction, calculate the walking time from this hotel to the famous local landmark, 'Teatro Antico di Taormina' (Ancient Greek Theater). Finally, since the original hotel is too expensive, I'd like to find more affordable alternatives on Airbnb. Search for accommodations in the same town, with the following filters: price per night between $200 and $400, a rating of 4.8 or higher, and 'Kitchen' facilities are a must. Please find 3 listings that are within a 15-minute walk of the original hotel (you'll need to verify the distance using Airbnb maps or Google Maps).\n\n---\n\n**Output Requirements:**\n\n1.  Real name of the hotel from the show and its town.\n2.  Google Maps walking time from the hotel to the Ancient Greek Theater.\n3.  For each of the 3 selected listings: Name, Price per night, Rating, whether it includes a kitchen, and the Airbnb listing URL.\n4.  For each listing, state its walking time to the original hotel (verify it's within 15 minutes)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    """Hotel information extracted from the answer"""
    hotel_name: Optional[str] = None
    town_name: Optional[str] = None


class WalkingTime(BaseModel):
    """Walking time from hotel to theater"""
    walking_time_text: Optional[str] = None


class AirbnbListing(BaseModel):
    """Single Airbnb listing details"""
    name: Optional[str] = None
    price_per_night_text: Optional[str] = None
    rating_text: Optional[str] = None
    has_kitchen: Optional[bool] = None
    listing_url: Optional[str] = None
    walking_time_to_hotel_text: Optional[str] = None


class AirbnbListings(BaseModel):
    """All Airbnb listings extracted from the answer"""
    listings: List[AirbnbListing] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
Extract the hotel name and town name from the answer that the user found from IMDb for 'The White Lotus' Season 2 filming location.

Return:
- hotel_name: the real name of the hotel (e.g., "San Domenico Palace")
- town_name: the exact town where the hotel is located (e.g., "Taormina")

If any field is missing, set it to null.
"""


def prompt_extract_walking_time() -> str:
    return """
Extract the walking time from the hotel to 'Teatro Antico di Taormina' (Ancient Greek Theater) that the user calculated using Google Maps.

Return:
- walking_time_text: the walking time exactly as stated (e.g., "10 minutes", "8 min", "5-7 minutes")

If not present, set it to null.
"""


def prompt_extract_airbnb_listings() -> str:
    return """
Extract all Airbnb listings from the answer. The user should have provided 3 listings with details.

For each listing, extract:
- name: the listing name/title
- price_per_night_text: the price per night exactly as stated (include currency if present)
- rating_text: the rating exactly as stated
- has_kitchen: true if the listing mentions having a kitchen, false otherwise
- listing_url: the Airbnb listing URL
- walking_time_to_hotel_text: the walking time from this listing to the original hotel

Return a list of up to 3 listings. If fewer listings are found, return what's available.
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


def extract_minutes(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    num = extract_float(text)
    if num is None:
        return None
    return num


def looks_like_hotel_name(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should have some reasonable length and not be just numbers
    text_clean = text.strip()
    return len(text_clean) > 3 and not text_clean.isdigit()


def looks_like_town_name(text: Optional[str]) -> bool:
    if not text:
        return False
    text_clean = text.strip()
    return len(text_clean) > 2 and not text_clean.isdigit()


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute'])


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'dollar', 'usd', '€', 'euro'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'airbnb.com', 'www.'])


def price_in_range(price_text: Optional[str], min_price: float, max_price: float) -> bool:
    if not price_text:
        return False
    price = extract_float(price_text)
    if price is None:
        return False
    return min_price <= price <= max_price


def rating_above_threshold(rating_text: Optional[str], threshold: float) -> bool:
    if not rating_text:
        return False
    rating = extract_float(rating_text)
    if rating is None:
        return False
    return rating >= threshold


def walking_time_within_limit(time_text: Optional[str], limit_minutes: float) -> bool:
    if not time_text:
        return False
    minutes = extract_minutes(time_text)
    if minutes is None:
        return False
    return minutes <= limit_minutes


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
    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    walking_time_info = await evaluator.extract(
        prompt=prompt_extract_walking_time(),
        template_class=WalkingTime,
        extraction_name="walking_time_to_theater"
    )

    airbnb_listings_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_listings(),
        template_class=AirbnbListings,
        extraction_name="airbnb_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb - Find filming location for The White Lotus Season 2",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Navigate to Season 2 filming locations
    imdb_action_ok = (has_any_ci(answer, ['imdb']) and
                      has_any_ci(answer, ['white lotus']) and
                      has_any_ci(answer, ['season 2', 'season two']))
    evaluator.add_custom_node(
        result=bool(imdb_action_ok),
        id="imdb_action_season2",
        desc="[Action Node] imdb.com:F1:A1 - Navigate to IMDb and find Season 2 filming location information",
        parent=imdb_node,
        critical=False
    )

    # Check hotel name extraction - output o1
    hotel_name_ok = looks_like_hotel_name(hotel_info.hotel_name)
    evaluator.add_custom_node(
        result=bool(hotel_name_ok),
        id="imdb_hotel_name_extracted",
        desc="Hotel name extracted from IMDb (output o1: should be 'San Domenico Palace')",
        parent=imdb_node,
        critical=False
    )

    # Check town name extraction
    town_name_ok = looks_like_town_name(hotel_info.town_name)
    evaluator.add_custom_node(
        result=bool(town_name_ok),
        id="imdb_town_name_extracted",
        desc="Town name extracted from IMDb (should be 'Taormina')",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Google Maps section
    gmaps_node = evaluator.add_sequential(
        id="gmaps_section",
        desc="Google Maps - Calculate walking time from hotel to Ancient Greek Theater",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input route planning
    gmaps_action_route_ok = (has_any_ci(answer, ['google maps']) and
                             has_any_ci(answer, ['teatro antico', 'ancient greek theater', 'greek theater']))
    evaluator.add_custom_node(
        result=bool(gmaps_action_route_ok),
        id="gmaps_action_route_input",
        desc="[Action Node] maps.google.com:F2:A5 - Input route planning from hotel to Teatro Antico di Taormina",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select walking mode
    walking_mode_ok = (has_any_ci(answer, ['walk', 'walking']) and
                       looks_like_walking_time(walking_time_info.walking_time_text))
    evaluator.add_custom_node(
        result=bool(walking_mode_ok),
        id="gmaps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking mode for route calculation",
        parent=gmaps_node,
        critical=False
    )

    # Check walking time extraction - output o3
    walking_time_extracted = looks_like_walking_time(walking_time_info.walking_time_text)
    evaluator.add_custom_node(
        result=bool(walking_time_extracted),
        id="gmaps_walking_time_extracted",
        desc="Walking time to theater extracted (output o3: should be reasonable time like 5-15 min)",
        parent=gmaps_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb - Find 3 listings with filters and distance verification",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price filter
    airbnb_price_filter_ok = has_any_ci(answer, ['200', '400']) and has_any_ci(answer, ['price', '$'])
    evaluator.add_custom_node(
        result=bool(airbnb_price_filter_ok),
        id="airbnb_action_price_filter",
        desc="[Action Node] airbnb.com:F2:A13 - Apply price filter ($200-$400 per night)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Kitchen facility filter
    airbnb_kitchen_filter_ok = has_any_ci(answer, ['kitchen'])
    evaluator.add_custom_node(
        result=bool(airbnb_kitchen_filter_ok),
        id="airbnb_action_kitchen_filter",
        desc="[Action Node] airbnb.com:F2:A11 - Apply Kitchen facility filter",
        parent=airbnb_node,
        critical=False
    )

    # Check if we have 3 listings
    num_listings = len(airbnb_listings_info.listings) if airbnb_listings_info.listings else 0
    has_three_listings = num_listings >= 3
    evaluator.add_custom_node(
        result=bool(has_three_listings),
        id="airbnb_three_listings_found",
        desc="Found 3 Airbnb listings as required",
        parent=airbnb_node,
        critical=False
    )

    # For each listing, check individual requirements
    for idx, listing in enumerate(airbnb_listings_info.listings[:3], 1):
        listing_node = evaluator.add_parallel(
            id=f"airbnb_listing_{idx}",
            desc=f"Airbnb Listing {idx} - Verification",
            parent=airbnb_node,
            critical=False
        )

        # Check name
        name_ok = bool(listing.name and listing.name.strip())
        evaluator.add_custom_node(
            result=bool(name_ok),
            id=f"listing_{idx}_name",
            desc=f"Listing {idx}: Name provided",
            parent=listing_node,
            critical=False
        )

        # Check price - output o5
        price_ok = looks_like_price(listing.price_per_night_text)
        price_in_range_ok = price_in_range(listing.price_per_night_text, 200, 400)
        evaluator.add_custom_node(
            result=bool(price_ok and price_in_range_ok),
            id=f"listing_{idx}_price",
            desc=f"Listing {idx}: Price per night in $200-$400 range (output o5)",
            parent=listing_node,
            critical=False
        )

        # [Perception Node] airbnb.com:F1:P3 - Rating check - output o6
        rating_ok = looks_like_rating(listing.rating_text)
        rating_threshold_ok = rating_above_threshold(listing.rating_text, 4.8)
        evaluator.add_custom_node(
            result=bool(rating_ok and rating_threshold_ok),
            id=f"listing_{idx}_rating",
            desc=f"[Perception Node] airbnb.com:F1:P3 - Listing {idx}: Rating >= 4.8 (output o6)",
            parent=listing_node,
            critical=False
        )

        # Check kitchen - output o7
        kitchen_ok = listing.has_kitchen is True
        evaluator.add_custom_node(
            result=bool(kitchen_ok),
            id=f"listing_{idx}_kitchen",
            desc=f"Listing {idx}: Kitchen facility confirmed (output o7)",
            parent=listing_node,
            critical=False
        )

        # Check URL
        url_ok = looks_like_url(listing.listing_url)
        evaluator.add_custom_node(
            result=bool(url_ok),
            id=f"listing_{idx}_url",
            desc=f"Listing {idx}: Airbnb listing URL provided",
            parent=listing_node,
            critical=False
        )

        # [Perception Node] airbnb.com:F4:P16 - Distance verification - output o8
        distance_ok = looks_like_walking_time(listing.walking_time_to_hotel_text)
        within_15min = walking_time_within_limit(listing.walking_time_to_hotel_text, 15)
        evaluator.add_custom_node(
            result=bool(distance_ok and within_15min),
            id=f"listing_{idx}_distance",
            desc=f"[Perception Node] airbnb.com:F4:P16 - Listing {idx}: Walking time to hotel <= 15 minutes (output o8)",
            parent=listing_node,
            critical=False
        )

    # Overall check: mentions using map for distance verification
    map_verification_ok = has_any_ci(answer, ['map', 'distance', 'walk'])
    evaluator.add_custom_node(
        result=bool(map_verification_ok),
        id="airbnb_map_verification_mentioned",
        desc="Distance verification using Airbnb maps or Google Maps mentioned",
        parent=airbnb_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
