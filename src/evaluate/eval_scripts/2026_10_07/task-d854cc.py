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
TASK_ID = "task-d854cc"
TASK_DESCRIPTION = "My partner and I are planning a road trip to Big Sur next weekend, and we'd like to rent a convertible from San Francisco International Airport (SFO). First, search Expedia for car rental information for next weekend. Filter the results by the 'Convertible' category. Tell me the specific car model that is primarily displayed (e.g., Mustang or Camaro).\n\nOnce you have the car model, go to CarAndDriver and search for its latest review. Find the specific trunk volume data (in cubic feet) and check if the review mentions anything about its ability to accommodate 'carry-on' luggage.\n\nFinally, go to Yelp and find a restaurant in the Big Sur area with a rating of 4 stars or higher. It must have an 'Ocean View'. Additionally, please thoroughly browse the review section to find and quote a user review that explicitly praises 'convenient parking' or mentions 'dedicated parking lot.' I don't want to worry about parking a convertible on the street."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ExpediaConvertible(BaseModel):
    """Convertible car model extracted from Expedia search"""
    car_model: Optional[str] = None
    pickup_location: Optional[str] = None
    rental_dates: Optional[str] = None


class CarAndDriverReview(BaseModel):
    """Car review details from CarAndDriver"""
    car_model_searched: Optional[str] = None
    trunk_volume_text: Optional[str] = None
    carry_on_mention: Optional[str] = None


class YelpRestaurant(BaseModel):
    """Restaurant information from Yelp"""
    restaurant_name: Optional[str] = None
    location: Optional[str] = None
    rating_text: Optional[str] = None
    ocean_view_mention: Optional[str] = None
    parking_review_quote: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_expedia_convertible() -> str:
    return """
Extract the user's reported Expedia car rental search results for SFO for next weekend.

Return:
- car_model: the specific convertible car model name mentioned (e.g., "Mustang", "Camaro", "Ford Mustang"). If not present, set null.
- pickup_location: any mention of SFO or San Francisco International Airport. If not present, set null.
- rental_dates: any mention of "next weekend" or specific dates. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_caranddriver_review() -> str:
    return """
From the answer, extract the CarAndDriver review details:

- car_model_searched: the car model that was searched on CarAndDriver. If not present, set null.
- trunk_volume_text: the trunk volume in cubic feet exactly as stated (include units if present). If not present, set null.
- carry_on_mention: any text mentioning "carry-on" or "carry on" luggage in relation to trunk capacity. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_yelp_restaurant() -> str:
    return """
From the answer, extract the Yelp restaurant information for Big Sur:

- restaurant_name: the name of the restaurant found. If not present, set null.
- location: any mention of "Big Sur" location. If not present, set null.
- rating_text: the rating information (should be 4 stars or higher). If not present, set null.
- ocean_view_mention: any mention of "ocean view" or similar. If not present, set null.
- parking_review_quote: the exact quoted user review text that mentions "convenient parking" or "dedicated parking lot" or "parking lot". If not present, set null.

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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_cubic_feet(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['cu ft', 'cubic ft', 'cubic feet', 'ft³', 'ft^3', 'cu. ft.'])


def looks_like_rating_4_plus(text: Optional[str]) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    if rating is None:
        return False
    return rating >= 4.0


def has_parking_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['convenient parking', 'dedicated parking', 'parking lot', 'easy parking', 'ample parking'])


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
    Restrict evaluator.verify to at most one usage (we'll not use it here).
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
    expedia_info = await evaluator.extract(
        prompt=prompt_extract_expedia_convertible(),
        template_class=ExpediaConvertible,
        extraction_name="expedia_convertible"
    )

    caranddriver_info = await evaluator.extract(
        prompt=prompt_extract_caranddriver_review(),
        template_class=CarAndDriverReview,
        extraction_name="caranddriver_review"
    )

    yelp_info = await evaluator.extract(
        prompt=prompt_extract_yelp_restaurant(),
        template_class=YelpRestaurant,
        extraction_name="yelp_restaurant"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Expedia section
    expedia_node = evaluator.add_sequential(
        id="expedia_section",
        desc="Expedia car rental search for convertible at SFO for next weekend",
        parent=root,
        critical=False
    )

    # [Action Node] expedia.com:F3:A9 - Date selection / view switching
    expedia_date_action_ok = (has_any_ci(answer, ['expedia']) and
                              has_any_ci(answer, ['next weekend', 'weekend']))
    evaluator.add_custom_node(
        result=bool(expedia_date_action_ok),
        id="expedia_action_date_selection",
        desc="[Action Node] expedia.com:F3:A9 - Navigate to Expedia and select dates for next weekend rental",
        parent=expedia_node,
        critical=False
    )

    # [Action Node] expedia.com:F3:A22 - Scroll to load / filter by Convertible
    expedia_filter_action_ok = (has_any_ci(answer, ['convertible']) and
                                has_any_ci(answer, ['filter', 'category', 'type']))
    evaluator.add_custom_node(
        result=bool(expedia_filter_action_ok),
        id="expedia_action_filter_convertible",
        desc="[Action Node] expedia.com:F3:A22 - Filter car rental results by 'Convertible' category (may require scrolling)",
        parent=expedia_node,
        critical=False
    )

    # Check for specific car model extraction
    car_model_ok = bool(expedia_info and expedia_info.car_model and expedia_info.car_model.strip())
    sfo_mention_ok = has_any_ci(answer, ['sfo', 'san francisco international', 'san francisco airport'])

    evaluator.add_custom_node(
        result=bool(car_model_ok),
        id="expedia_perception_car_model",
        desc="Extract the specific convertible car model name from Expedia results",
        parent=expedia_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(sfo_mention_ok),
        id="expedia_mentions_sfo",
        desc="Mentions SFO or San Francisco International Airport as pickup location",
        parent=expedia_node,
        critical=False
    )

    # 3.2 CarAndDriver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="CarAndDriver review for the convertible model found on Expedia",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search for the car model
    caranddriver_search_action_ok = (has_any_ci(answer, ['caranddriver', 'car and driver']) and
                                     (car_model_ok or has_any_ci(answer, ['search', 'review'])))
    evaluator.add_custom_node(
        result=bool(caranddriver_search_action_ok),
        id="caranddriver_action_search",
        desc="[Action Node] caranddriver.com:F2:A1 - Search for the convertible model on CarAndDriver",
        parent=caranddriver_node,
        critical=False
    )

    # Extract trunk volume
    trunk_num = extract_float(caranddriver_info.trunk_volume_text)
    trunk_units_ok = looks_like_cubic_feet(caranddriver_info.trunk_volume_text)
    trunk_ok = (trunk_num is not None) and trunk_units_ok

    evaluator.add_custom_node(
        result=bool(trunk_ok),
        id="caranddriver_perception_trunk_volume",
        desc="Extract trunk volume in cubic feet from CarAndDriver review",
        parent=caranddriver_node,
        critical=False
    )

    # Check for carry-on luggage mention
    carry_on_ok = bool(caranddriver_info and caranddriver_info.carry_on_mention and
                       has_any_ci(caranddriver_info.carry_on_mention, ['carry-on', 'carry on']))

    evaluator.add_custom_node(
        result=bool(carry_on_ok),
        id="caranddriver_perception_carry_on",
        desc="Check if review mentions ability to accommodate 'carry-on' luggage",
        parent=caranddriver_node,
        critical=False
    )

    # 3.3 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp restaurant search in Big Sur with ocean view and parking verification",
        parent=root,
        critical=False
    )

    # Basic search and filtering
    yelp_search_ok = (has_any_ci(answer, ['yelp']) and
                      has_any_ci(answer, ['big sur']))
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_action_search_bigsur",
        desc="Search for restaurants in Big Sur area on Yelp",
        parent=yelp_node,
        critical=False
    )

    # Check rating 4+ and ocean view
    restaurant_name_ok = bool(yelp_info and yelp_info.restaurant_name and yelp_info.restaurant_name.strip())
    rating_ok = looks_like_rating_4_plus(yelp_info.rating_text)
    ocean_view_ok = bool(yelp_info and yelp_info.ocean_view_mention and
                         has_any_ci(yelp_info.ocean_view_mention, ['ocean view', 'ocean-view', 'oceanview']))

    evaluator.add_custom_node(
        result=bool(restaurant_name_ok and rating_ok),
        id="yelp_perception_restaurant_rating",
        desc="Find restaurant with 4 stars or higher rating",
        parent=yelp_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(ocean_view_ok),
        id="yelp_perception_ocean_view",
        desc="Verify restaurant has 'Ocean View' feature",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F2:A10 - Browse review section / tab switching
    yelp_review_browse_action_ok = (has_any_ci(answer, ['review', 'reviews']) and
                                    has_any_ci(answer, ['browse', 'check', 'look', 'search', 'find']))
    evaluator.add_custom_node(
        result=bool(yelp_review_browse_action_ok),
        id="yelp_action_browse_reviews",
        desc="[Action Node] yelp.com:F2:A10 - Browse the review section to find parking-related reviews (may require tab switching or scrolling)",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] - Extract parking review quote
    parking_quote_ok = bool(yelp_info and yelp_info.parking_review_quote and
                           yelp_info.parking_review_quote.strip())
    parking_keywords_ok = has_parking_keywords(yelp_info.parking_review_quote)

    evaluator.add_custom_node(
        result=bool(parking_quote_ok and parking_keywords_ok),
        id="yelp_perception_parking_review",
        desc="[Perception Node] Extract and quote user review mentioning 'convenient parking' or 'dedicated parking lot'",
        parent=yelp_node,
        critical=False
    )

    # Additional lenient check: mentions quote or specific review
    quote_mention_ok = has_any_ci(answer, ['quote', 'review says', 'user said', 'reviewer', 'customer'])
    evaluator.add_custom_node(
        result=bool(quote_mention_ok),
        id="yelp_mentions_quote",
        desc="Answer includes quoted or referenced user review text",
        parent=yelp_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
