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
TASK_ID = "task-2a9f91"
TASK_DESCRIPTION = 'I just received a software engineer offer from a tech company in the San Francisco Bay Area, with an annual salary of $120,000, and now I need to find housing. The company is Salesforce. First, I want to confirm the exact address of their San Francisco office (preferably from the company’s LinkedIn page; if LinkedIn does not show the full address, use Salesforce’s official office locations page or the official Google Maps location page to verify it, and cite the source).\n\nThen help me find apartments near that address with a budget of no more than $2,500/month, ideally with a driving commute of no more than 30 minutes, and it must be a 1-bedroom unit. My top priorities are an in-unit washer and dryer and a parking space, since I plan to drive to work.\n\nPlease try to find 3–5 listings that meet these criteria. If there are fewer than 3 under the strict requirements, keep the existing results and state the count—do not relax any of my filters. For each listing, calculate the total move-in cost, including deposit and parking fees, and estimate the monthly commute time.\n\nFinally, summarize which options offer the best value for money so I can plan in-person tours.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SalesforceAddress(BaseModel):
    """Salesforce SF office address extracted from the answer"""
    address_text: Optional[str] = None
    source_cited: Optional[str] = None


class ApartmentListing(BaseModel):
    """Single apartment listing details"""
    listing_name: Optional[str] = None
    monthly_rent: Optional[str] = None
    bedrooms: Optional[str] = None
    has_washer_dryer: Optional[bool] = None
    has_parking: Optional[bool] = None
    commute_time: Optional[str] = None
    total_move_in_cost: Optional[str] = None


class ApartmentListings(BaseModel):
    """All apartment listings extracted from the answer"""
    listings: List[ApartmentListing] = Field(default_factory=list)
    total_count: Optional[int] = None


class ValueSummary(BaseModel):
    """Value for money summary extracted from the answer"""
    summary_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_salesforce_address() -> str:
    return """
Extract the Salesforce San Francisco office address from the answer.

Return:
- address_text: the exact address text as stated in the answer (street, city, etc.). If not present, set null.
- source_cited: the source where the address was verified (e.g., "LinkedIn", "Salesforce official page", "Google Maps"). If not mentioned, set null.
"""


def prompt_extract_apartment_listings() -> str:
    return """
Extract all apartment listings mentioned in the answer that meet the user's criteria.

For each listing, extract:
- listing_name: the name or address of the apartment complex
- monthly_rent: the monthly rent amount as stated
- bedrooms: number of bedrooms (should be 1)
- has_washer_dryer: true if the listing mentions in-unit washer/dryer, false otherwise
- has_parking: true if the listing mentions parking availability, false otherwise
- commute_time: the estimated commute time to Salesforce office
- total_move_in_cost: the total move-in cost including deposit, parking fees, etc.

Also extract:
- total_count: the total number of listings found

If any field is missing, set it to null or appropriate default.
"""


def prompt_extract_value_summary() -> str:
    return """
Extract the summary or recommendation about which apartment options offer the best value for money from the answer.

Return:
- summary_text: the text summarizing the value comparison and recommendations. If not present, set null.
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


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain street indicators and city or state
    street_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 'drive', 'dr', 'tower', 'building']
    location_indicators = ['san francisco', 'sf', 'california', 'ca']
    has_street = has_any_ci(text, street_indicators)
    has_location = has_any_ci(text, location_indicators)
    has_number = contains_digits(text)
    return has_street and has_location and has_number


def looks_like_rent(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Should be in reasonable range (e.g., 1000-3000 for this task)
    return 1000 <= num <= 3000 and has_any_ci(text, ['$', 'dollar'])


def looks_like_commute_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour'])


def looks_like_move_in_cost(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num > 0 and has_any_ci(text, ['$', 'dollar'])


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
    salesforce_info = await evaluator.extract(
        prompt=prompt_extract_salesforce_address(),
        template_class=SalesforceAddress,
        extraction_name="salesforce_address"
    )

    listings_info = await evaluator.extract(
        prompt=prompt_extract_apartment_listings(),
        template_class=ApartmentListings,
        extraction_name="apartment_listings"
    )

    value_info = await evaluator.extract(
        prompt=prompt_extract_value_summary(),
        template_class=ValueSummary,
        extraction_name="value_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 LinkedIn section - Get Salesforce office address
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="Retrieve Salesforce San Francisco office address from LinkedIn or alternative sources",
        parent=root,
        critical=False
    )

    # [Perception Node] linkedin.com:F9:P13 - Extract company office location from LinkedIn
    address_ok = looks_like_address(salesforce_info.address_text)
    source_ok = salesforce_info.source_cited is not None and len(salesforce_info.source_cited.strip()) > 0
    salesforce_mentioned = has_any_ci(answer, ['salesforce'])

    evaluator.add_custom_node(
        result=bool(address_ok and source_ok and salesforce_mentioned),
        id="linkedin_perception_office_address",
        desc="[Perception Node] linkedin.com:F9:P13 - Extract and verify Salesforce SF office address with cited source",
        parent=linkedin_node,
        critical=False
    )

    # 3.2 Apartments.com section - Search and filter
    apartments_node = evaluator.add_sequential(
        id="apartments_section",
        desc="Search and filter apartments on Apartments.com based on criteria",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A9 - Use address as search center
    apartments_mention = has_any_ci(answer, ['apartments', 'apartment.com', 'apartments.com'])
    address_used_for_search = address_ok and apartments_mention

    evaluator.add_custom_node(
        result=bool(address_used_for_search),
        id="apartments_action_location_search",
        desc="[Action Node] apartments.com:F1:A9 - Use Salesforce office address as search center point",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A1 - Price range filter
    price_filter_mentioned = has_any_ci(answer, ['2500', '2,500', 'budget', 'price', 'rent'])
    max_price_respected = True
    if listings_info.listings:
        for listing in listings_info.listings:
            if listing.monthly_rent:
                rent_val = extract_float(listing.monthly_rent)
                if rent_val and rent_val > 2500:
                    max_price_respected = False
                    break

    evaluator.add_custom_node(
        result=bool(price_filter_mentioned and max_price_respected),
        id="apartments_action_price_filter",
        desc="[Action Node] apartments.com:F1:A1 - Apply price range filter (max $2500/month)",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Bedroom count selection
    bedroom_filter_mentioned = has_any_ci(answer, ['1 bedroom', '1-bedroom', '1bedroom', '1 bed'])
    all_one_bedroom = True
    if listings_info.listings:
        for listing in listings_info.listings:
            if listing.bedrooms and not has_any_ci(listing.bedrooms, ['1']):
                all_one_bedroom = False
                break

    evaluator.add_custom_node(
        result=bool(bedroom_filter_mentioned and all_one_bedroom),
        id="apartments_action_bedroom_filter",
        desc="[Action Node] apartments.com:F1:A2 - Select 1-bedroom filter",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F8:A32 - Commute time configuration
    commute_filter_mentioned = has_any_ci(answer, ['commute', '30 min', '30 minute', 'drive', 'driving'])
    all_within_30min = True
    if listings_info.listings:
        for listing in listings_info.listings:
            if listing.commute_time:
                time_val = extract_float(listing.commute_time)
                if time_val and time_val > 30:
                    all_within_30min = False
                    break

    evaluator.add_custom_node(
        result=bool(commute_filter_mentioned and all_within_30min),
        id="apartments_action_commute_filter",
        desc="[Action Node] apartments.com:F8:A32 - Configure commute time filter (30 min driving)",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F6:A4 - Amenities multi-select filter
    washer_dryer_mentioned = has_any_ci(answer, ['washer', 'dryer', 'laundry', 'in-unit'])
    parking_mentioned = has_any_ci(answer, ['parking', 'garage'])
    all_have_amenities = True
    if listings_info.listings:
        for listing in listings_info.listings:
            if not (listing.has_washer_dryer and listing.has_parking):
                all_have_amenities = False
                break

    evaluator.add_custom_node(
        result=bool(washer_dryer_mentioned and parking_mentioned and all_have_amenities),
        id="apartments_action_amenities_filter",
        desc="[Action Node] apartments.com:F6:A4 - Select in-unit washer/dryer and parking amenities",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A27 - Browse multiple pages for listings
    multiple_listings = listings_info.total_count is not None and listings_info.total_count >= 3
    pagination_implied = len(listings_info.listings) >= 3

    evaluator.add_custom_node(
        result=bool(multiple_listings or pagination_implied),
        id="apartments_action_pagination",
        desc="[Action Node] apartments.com:F1:A27 - Browse multiple pages to find 3-5 listings",
        parent=apartments_node,
        critical=False
    )

    # 3.3 Cost calculation section
    cost_node = evaluator.add_sequential(
        id="cost_calculation_section",
        desc="Calculate move-in costs for each listing",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F10:A23 - Open rent calculator
    calculator_usage = has_any_ci(answer, ['calculator', 'estimate', 'total cost', 'move-in cost', 'deposit'])

    evaluator.add_custom_node(
        result=bool(calculator_usage),
        id="apartments_action_open_calculator",
        desc="[Action Node] apartments.com:F10:A23 - Open rent/cost calculator for listings",
        parent=cost_node,
        critical=False
    )

    # [Action Node] apartments.com:F10:A24 - Configure lease options (parking, etc.)
    parking_fee_mentioned = has_any_ci(answer, ['parking fee', 'parking cost', 'parking charge'])

    evaluator.add_custom_node(
        result=bool(parking_fee_mentioned),
        id="apartments_action_configure_options",
        desc="[Action Node] apartments.com:F10:A24 - Configure parking and other lease options in calculator",
        parent=cost_node,
        critical=False
    )

    # [Perception Node] apartments.com:F10:P13 - Extract and calculate total costs
    all_have_move_in_cost = True
    deposit_mentioned = has_any_ci(answer, ['deposit', 'security'])
    if listings_info.listings:
        for listing in listings_info.listings:
            if not listing.total_move_in_cost or not looks_like_move_in_cost(listing.total_move_in_cost):
                all_have_move_in_cost = False
                break
    else:
        all_have_move_in_cost = False

    evaluator.add_custom_node(
        result=bool(all_have_move_in_cost and deposit_mentioned),
        id="apartments_perception_cost_breakdown",
        desc="[Perception Node] apartments.com:F10:P13 - Extract cost breakdown and calculate total move-in cost",
        parent=cost_node,
        critical=False
    )

    # 3.4 Location and commute analysis
    location_node = evaluator.add_sequential(
        id="location_analysis_section",
        desc="Analyze location and commute time for each listing",
        parent=root,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P4 - Understand map location relative to office
    all_have_commute_time = True
    if listings_info.listings:
        for listing in listings_info.listings:
            if not listing.commute_time or not looks_like_commute_time(listing.commute_time):
                all_have_commute_time = False
                break
    else:
        all_have_commute_time = False

    evaluator.add_custom_node(
        result=bool(all_have_commute_time),
        id="apartments_perception_map_location",
        desc="[Perception Node] apartments.com:F1:P4 - Understand map positions and estimate commute times",
        parent=location_node,
        critical=False
    )

    # 3.5 Value comparison section
    comparison_node = evaluator.add_sequential(
        id="value_comparison_section",
        desc="Compare listings and identify best value options",
        parent=root,
        critical=False
    )

    # [Perception Node] apartments.com:F2:P13 - Compare price, area, and features
    has_value_summary = value_info.summary_text is not None and len(value_info.summary_text.strip()) > 0
    comparison_mentioned = has_any_ci(answer, ['value', 'best', 'recommend', 'comparison', 'compare'])
    sufficient_listings = listings_info.total_count is not None and listings_info.total_count >= 3

    evaluator.add_custom_node(
        result=bool(has_value_summary and comparison_mentioned and sufficient_listings),
        id="apartments_perception_value_comparison",
        desc="[Perception Node] apartments.com:F2:P13 - Compare rent, features, and costs to identify best value",
        parent=comparison_node,
        critical=False
    )

    # Additional quality checks
    evaluator.add_custom_node(
        result=bool(listings_info.total_count is not None and 3 <= listings_info.total_count <= 5),
        id="listing_count_check",
        desc="Found 3-5 listings as requested (or stated actual count if fewer)",
        parent=comparison_node,
        critical=False
    )

    strict_criteria_maintained = has_any_ci(answer, ['strict', 'criteria', 'filter', 'requirement'])
    evaluator.add_custom_node(
        result=bool(strict_criteria_maintained),
        id="strict_criteria_maintained",
        desc="Maintained strict criteria without relaxing filters",
        parent=comparison_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
