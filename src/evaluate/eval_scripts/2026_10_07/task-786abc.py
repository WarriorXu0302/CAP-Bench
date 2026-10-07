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
TASK_ID = "task-786abc"
TASK_DESCRIPTION = "I have an onsite interview next week at a tech company in Seattle and need to book a 3-night stay.\nThe company is Amazon Web Services. First, please confirm their specific headquarters address in Seattle on LinkedIn.\nThen, search for Airbnb listings near that address. The check-in dates should be next Monday to Thursday. Find 3 options that have high ratings and a convenient commute to the company.\nI'd also like to understand the amenities around these listings, as I might move there if I get an offer. Therefore, I want to know if the area is suitable for long-term living.\nFinally, please provide a comparison of these 3 listings, including: distance from the company, commute time, nightly price, rating, and nearby amenities like restaurants and supermarkets."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AWSAddressInfo(BaseModel):
    """AWS Seattle headquarters address extracted from the answer"""
    address_text: Optional[str] = None
    linkedin_mentioned: Optional[bool] = None


class AirbnbSearchInfo(BaseModel):
    """Airbnb search parameters and results extracted from the answer"""
    location_searched: Optional[str] = None
    checkin_date: Optional[str] = None
    checkout_date: Optional[str] = None
    number_of_listings_found: Optional[int] = None


class ListingComparison(BaseModel):
    """Comparison data for Airbnb listings extracted from the answer"""
    listing_1_name: Optional[str] = None
    listing_1_distance: Optional[str] = None
    listing_1_commute_time: Optional[str] = None
    listing_1_price: Optional[str] = None
    listing_1_rating: Optional[str] = None
    listing_1_amenities: Optional[str] = None

    listing_2_name: Optional[str] = None
    listing_2_distance: Optional[str] = None
    listing_2_commute_time: Optional[str] = None
    listing_2_price: Optional[str] = None
    listing_2_rating: Optional[str] = None
    listing_2_amenities: Optional[str] = None

    listing_3_name: Optional[str] = None
    listing_3_distance: Optional[str] = None
    listing_3_commute_time: Optional[str] = None
    listing_3_price: Optional[str] = None
    listing_3_rating: Optional[str] = None
    listing_3_amenities: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_aws_address() -> str:
    return """
Extract the AWS Seattle headquarters address information from the answer.

Return:
- address_text: the specific address mentioned for AWS Seattle headquarters (street address, city, etc.)
- linkedin_mentioned: true if the answer mentions using LinkedIn to find this information, false otherwise

If any field is missing in the answer, set it to null.
"""


def prompt_extract_airbnb_search() -> str:
    return """
Extract the Airbnb search parameters and results from the answer.

Return:
- location_searched: the location or address used to search for Airbnb listings
- checkin_date: the check-in date mentioned (e.g., "next Monday", specific date)
- checkout_date: the check-out date mentioned (e.g., "Thursday", specific date)
- number_of_listings_found: the number of listings found or presented (usually 3 as requested)

If any field is missing, set it to null.
"""


def prompt_extract_listing_comparison() -> str:
    return """
Extract the comparison data for the 3 Airbnb listings from the answer.

For each listing (1, 2, and 3), extract:
- name: the listing name or title
- distance: distance from AWS headquarters
- commute_time: estimated commute time to AWS
- price: nightly price
- rating: the rating score
- amenities: nearby amenities like restaurants, supermarkets, etc.

If any field is missing for any listing, set it to null.
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


def extract_numbers(text: Optional[str]) -> List[float]:
    if not text:
        return []
    matches = re.findall(r'\d+(?:\.\d+)?', text)
    try:
        return [float(m) for m in matches]
    except Exception:
        return []


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for common address components
    has_number = contains_digits(text)
    has_location_words = has_any_ci(text, ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'way', 'seattle', 'wa'])
    return has_number and has_location_words


def looks_like_date_reference(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'next week', '/', '-'])


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['$', 'usd', 'dollar', 'price', 'night']) and contains_digits(text)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    numbers = extract_numbers(text)
    if not numbers:
        return False
    # Ratings typically range from 1-5 or are presented as percentages
    return any(0 <= n <= 5 for n in numbers) or any(n > 5 for n in numbers)


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['mile', 'mi', 'km', 'kilometer', 'meter', 'away', 'from']) and contains_digits(text)


def looks_like_commute_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['minute', 'min', 'hour', 'hr', 'walk', 'drive', 'commute']) and contains_digits(text)


def mentions_amenities(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['restaurant', 'supermarket', 'grocery', 'store', 'cafe', 'shop', 'amenity', 'amenities'])


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
    aws_info = await evaluator.extract(
        prompt=prompt_extract_aws_address(),
        template_class=AWSAddressInfo,
        extraction_name="aws_address_info"
    )

    search_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_search(),
        template_class=AirbnbSearchInfo,
        extraction_name="airbnb_search_info"
    )

    comparison_info = await evaluator.extract(
        prompt=prompt_extract_listing_comparison(),
        template_class=ListingComparison,
        extraction_name="listing_comparison"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 LinkedIn section - confirm AWS address
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="LinkedIn company page - AWS Seattle headquarters address",
        parent=root,
        critical=False
    )

    # [Perception Node] linkedin.com:F9:P13 - Extract company location information
    linkedin_mentioned = bool(aws_info and aws_info.linkedin_mentioned)
    address_valid = looks_like_address(aws_info.address_text) if aws_info else False
    aws_mentioned = has_any_ci(answer, ['aws', 'amazon web services'])

    evaluator.add_custom_node(
        result=bool(linkedin_mentioned and address_valid and aws_mentioned),
        id="linkedin_perception_aws_address",
        desc="[Perception Node] linkedin.com:F9:P13 - Extract AWS Seattle headquarters specific address from company page",
        parent=linkedin_node,
        critical=False
    )

    # 3.2 Airbnb search section
    airbnb_search_node = evaluator.add_sequential(
        id="airbnb_search_section",
        desc="Airbnb search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A38 - Geographic location search
    location_used = bool(search_info and search_info.location_searched)
    address_in_search = has_any_ci(answer, ['search', 'near', 'location', 'address']) if aws_info and aws_info.address_text else False

    evaluator.add_custom_node(
        result=bool(location_used and address_in_search),
        id="airbnb_action_location_search",
        desc="[Action Node] airbnb.com:F1:A38 - Search Airbnb using the AWS address location",
        parent=airbnb_search_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date range selection
    checkin_valid = looks_like_date_reference(search_info.checkin_date) if search_info else False
    checkout_valid = looks_like_date_reference(search_info.checkout_date) if search_info else False
    monday_to_thursday = has_any_ci(answer, ['monday', 'thursday']) or has_any_ci(answer, ['3 night', 'three night'])

    evaluator.add_custom_node(
        result=bool(checkin_valid and checkout_valid and monday_to_thursday),
        id="airbnb_action_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Select check-in (Monday) and check-out (Thursday) dates",
        parent=airbnb_search_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A26 - Scroll to browse multiple listings
    multiple_listings = (search_info and search_info.number_of_listings_found and search_info.number_of_listings_found >= 3) if search_info else False
    scroll_implied = has_any_ci(answer, ['listings', 'options', 'found', 'browse', 'compare'])

    evaluator.add_custom_node(
        result=bool(multiple_listings or scroll_implied),
        id="airbnb_action_scroll_browse",
        desc="[Action Node] airbnb.com:F1:A26 - Scroll through listings to find high-rated options",
        parent=airbnb_search_node,
        critical=False
    )

    # 3.3 Listing details section
    listing_details_node = evaluator.add_sequential(
        id="listing_details_section",
        desc="Airbnb listing details and amenities",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F3:A19 - Click into listing details
    has_detailed_info = bool(comparison_info and (comparison_info.listing_1_name or comparison_info.listing_1_distance))
    click_implied = has_any_ci(answer, ['details', 'detail page', 'clicked', 'view', 'check'])

    evaluator.add_custom_node(
        result=bool(has_detailed_info or click_implied),
        id="airbnb_action_click_details",
        desc="[Action Node] airbnb.com:F3:A19 - Click into listing detail pages to view location and amenities",
        parent=listing_details_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F4:P16 - Understand location and distance
    has_distance_info = False
    has_commute_info = False
    if comparison_info:
        distances = [comparison_info.listing_1_distance, comparison_info.listing_2_distance, comparison_info.listing_3_distance]
        commutes = [comparison_info.listing_1_commute_time, comparison_info.listing_2_commute_time, comparison_info.listing_3_commute_time]
        has_distance_info = any(looks_like_distance(d) for d in distances if d)
        has_commute_info = any(looks_like_commute_time(c) for c in commutes if c)

    evaluator.add_custom_node(
        result=bool(has_distance_info and has_commute_info),
        id="airbnb_perception_location",
        desc="[Perception Node] airbnb.com:F4:P16 - Extract distance and commute time relative to AWS headquarters",
        parent=listing_details_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F3:P23 - Understand reviews and amenities
    has_rating_info = False
    has_amenity_info = False
    if comparison_info:
        ratings = [comparison_info.listing_1_rating, comparison_info.listing_2_rating, comparison_info.listing_3_rating]
        amenities = [comparison_info.listing_1_amenities, comparison_info.listing_2_amenities, comparison_info.listing_3_amenities]
        has_rating_info = any(looks_like_rating(r) for r in ratings if r)
        has_amenity_info = any(mentions_amenities(a) for a in amenities if a)

    high_rating_mentioned = has_any_ci(answer, ['high rating', 'highly rated', 'top rated', 'good rating'])

    evaluator.add_custom_node(
        result=bool(has_rating_info and has_amenity_info and high_rating_mentioned),
        id="airbnb_perception_reviews_amenities",
        desc="[Perception Node] airbnb.com:F3:P23 - Extract ratings and understand nearby amenities from reviews/descriptions",
        parent=listing_details_node,
        critical=False
    )

    # 3.4 Comparison section
    comparison_node = evaluator.add_sequential(
        id="comparison_section",
        desc="Compare 3 listings with key metrics",
        parent=root,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P21 - Cross-listing comparison
    three_listings_presented = False
    if comparison_info:
        listing_count = sum([
            bool(comparison_info.listing_1_name),
            bool(comparison_info.listing_2_name),
            bool(comparison_info.listing_3_name)
        ])
        three_listings_presented = listing_count >= 3

    has_price_comparison = False
    if comparison_info:
        prices = [comparison_info.listing_1_price, comparison_info.listing_2_price, comparison_info.listing_3_price]
        has_price_comparison = sum(looks_like_price(p) for p in prices if p) >= 2

    comparison_mentioned = has_any_ci(answer, ['comparison', 'compare', 'versus', 'vs'])

    evaluator.add_custom_node(
        result=bool(three_listings_presented and has_price_comparison and comparison_mentioned),
        id="airbnb_perception_cross_comparison",
        desc="[Perception Node] airbnb.com:F1:P21 - Compare multiple listings across distance, commute, price, rating, and amenities",
        parent=comparison_node,
        critical=False
    )

    # Extra check: all required comparison fields present
    all_fields_present = False
    if comparison_info and three_listings_presented:
        listing_1_complete = all([
            comparison_info.listing_1_distance,
            comparison_info.listing_1_commute_time,
            comparison_info.listing_1_price,
            comparison_info.listing_1_rating,
            comparison_info.listing_1_amenities
        ])
        listing_2_complete = all([
            comparison_info.listing_2_distance,
            comparison_info.listing_2_commute_time,
            comparison_info.listing_2_price,
            comparison_info.listing_2_rating,
            comparison_info.listing_2_amenities
        ])
        listing_3_complete = all([
            comparison_info.listing_3_distance,
            comparison_info.listing_3_commute_time,
            comparison_info.listing_3_price,
            comparison_info.listing_3_rating,
            comparison_info.listing_3_amenities
        ])
        all_fields_present = listing_1_complete and listing_2_complete and listing_3_complete

    evaluator.add_custom_node(
        result=bool(all_fields_present),
        id="comparison_completeness",
        desc="All 5 requested comparison fields (distance, commute time, price, rating, amenities) provided for all 3 listings",
        parent=comparison_node,
        critical=False
    )

    # Extra check: mentions long-term living suitability
    long_term_mentioned = has_any_ci(answer, ['long-term', 'long term', 'move there', 'suitable for living', 'area is suitable'])
    evaluator.add_custom_node(
        result=bool(long_term_mentioned),
        id="long_term_living_context",
        desc="Addresses the long-term living suitability of the areas",
        parent=comparison_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
