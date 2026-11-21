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
TASK_ID = "task-19fcf3"
TASK_DESCRIPTION = "I have a $100,000 down payment and want to buy a condo in Nashville's The Gulch neighborhood for short-term rental investment.\n\nFirst, go to Redfin and search for 'Condo' in The Gulch area, with a price under $600,000. To ensure cash flow, please use the filter to find a property with relatively low HOA fees (preferably under $500/month). Record its total price and estimated monthly payment.\n\nNext, go to Airbnb, locate the same The Gulch area on the map, and search for accommodation prices for a weekend next month (e.g., Friday to Sunday). Set the filters to 'Entire place' and a rating of 4.8 or higher (or 'Guest Favorite' tag). Find 3 comparable listings with the same number of bedrooms as the Redfin property, and calculate their average nightly price.\n\nFinally, help me calculate: If the property is rented out for 20 days per month at this average nightly rate, can the income cover the Redfin property's monthly payment?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RedfinProperty(BaseModel):
    """Property details extracted from Redfin"""
    total_price: Optional[str] = None
    monthly_payment: Optional[str] = None
    hoa_fee: Optional[str] = None
    bedrooms: Optional[str] = None


class AirbnbComparables(BaseModel):
    """Airbnb comparable listings information"""
    listing_1_price: Optional[str] = None
    listing_2_price: Optional[str] = None
    listing_3_price: Optional[str] = None
    average_nightly_price: Optional[str] = None


class CashFlowCalculation(BaseModel):
    """Cash flow calculation result"""
    monthly_rental_income: Optional[str] = None
    monthly_payment: Optional[str] = None
    can_cover: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_redfin_property() -> str:
    return """
Extract the Redfin property details from the answer for a condo in Nashville's The Gulch neighborhood:

Return:
- total_price: the total price of the property exactly as stated (include currency symbols if present).
- monthly_payment: the estimated monthly payment exactly as stated (include currency symbols if present).
- hoa_fee: the HOA fee exactly as stated (include currency symbols and period if present, e.g., "$450/month").
- bedrooms: the number of bedrooms (e.g., "1", "2", "1 bedroom", "2 bedrooms").

If any field is missing in the answer, set it to null.
"""


def prompt_extract_airbnb_comparables() -> str:
    return """
Extract the Airbnb comparable listings information from the answer:

Return:
- listing_1_price: the nightly price of the first comparable listing exactly as stated.
- listing_2_price: the nightly price of the second comparable listing exactly as stated.
- listing_3_price: the nightly price of the third comparable listing exactly as stated.
- average_nightly_price: the calculated average nightly price exactly as stated.

If any field is missing, set it to null.
"""


def prompt_extract_cashflow_calculation() -> str:
    return """
Extract the cash flow calculation from the answer:

Return:
- monthly_rental_income: the calculated monthly rental income (20 days * average nightly rate) exactly as stated.
- monthly_payment: the Redfin property's monthly payment used in the calculation exactly as stated.
- can_cover: whether the rental income can cover the monthly payment (e.g., "yes", "no", "Yes, it can cover", etc.).

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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove common currency symbols and commas
    cleaned = re.sub(r'[\$,]', '', text)
    m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '$') or contains_digits(text))


def looks_like_hoa_fee(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_period_indicator = has_any_ci(text, ['/month', 'per month', 'month', 'monthly'])
    return has_number and (has_period_indicator or ci_contains(text, '$'))


def looks_like_bedroom_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) or has_any_ci(text, ['studio', 'bedroom', 'bed'])


def mentions_the_gulch(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['the gulch', 'gulch'])


def mentions_nashville(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['nashville'])


def mentions_airbnb_filters(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    entire_place = has_any_ci(answer_text, ['entire place', 'entire home'])
    rating = has_any_ci(answer_text, ['4.8', 'rating', 'guest favorite'])
    return entire_place and rating


def mentions_guest_favorite(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['guest favorite'])


def has_three_comparables(listing_1: Optional[str], listing_2: Optional[str], listing_3: Optional[str]) -> bool:
    return bool(listing_1 and listing_2 and listing_3)


def mentions_calculation(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['calculate', 'calculation', '20 days', 'rental income', 'cover'])


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
    redfin_info = await evaluator.extract(
        prompt=prompt_extract_redfin_property(),
        template_class=RedfinProperty,
        extraction_name="redfin_property"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_comparables(),
        template_class=AirbnbComparables,
        extraction_name="airbnb_comparables"
    )

    cashflow_info = await evaluator.extract(
        prompt=prompt_extract_cashflow_calculation(),
        template_class=CashFlowCalculation,
        extraction_name="cashflow_calculation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Redfin section
    redfin_node = evaluator.add_sequential(
        id="redfin_section",
        desc="Redfin property search in Nashville's The Gulch neighborhood",
        parent=root,
        critical=False
    )

    # [Action Node] redfin.com:F1:A18 - Search and filter for condo in The Gulch
    redfin_search_ok = (has_any_ci(answer, ['redfin']) and
                        mentions_the_gulch(answer) and
                        has_any_ci(answer, ['condo', 'condominium']))
    evaluator.add_custom_node(
        result=bool(redfin_search_ok),
        id="redfin_action_search_gulch",
        desc="[Action Node] redfin.com:F1:A18 - Search for 'Condo' in The Gulch area on Redfin",
        parent=redfin_node,
        critical=False
    )

    # Check price filter under $600,000
    price_filter_ok = has_any_ci(answer, ['600,000', '600000', 'under 600', 'price'])
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="redfin_price_filter",
        desc="Applies price filter under $600,000",
        parent=redfin_node,
        critical=False
    )

    # Check HOA fee filter
    hoa_filter_ok = (looks_like_hoa_fee(redfin_info.hoa_fee) and
                     has_any_ci(answer, ['hoa', 'hoa fee']))
    evaluator.add_custom_node(
        result=bool(hoa_filter_ok),
        id="redfin_hoa_filter",
        desc="Filters for properties with low HOA fees (preferably under $500/month)",
        parent=redfin_node,
        critical=False
    )

    # Extract property details
    total_price_ok = looks_like_price(redfin_info.total_price)
    monthly_payment_ok = looks_like_price(redfin_info.monthly_payment)
    property_details_ok = total_price_ok and monthly_payment_ok

    evaluator.add_custom_node(
        result=bool(property_details_ok),
        id="redfin_property_details",
        desc="Records total price and estimated monthly payment",
        parent=redfin_node,
        critical=False
    )

    # Extract bedroom count
    bedrooms_ok = looks_like_bedroom_count(redfin_info.bedrooms)
    evaluator.add_custom_node(
        result=bool(bedrooms_ok),
        id="redfin_bedroom_count",
        desc="Records the number of bedrooms for later matching with Airbnb",
        parent=redfin_node,
        critical=False
    )

    # 3.2 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb comparable listings search in The Gulch area",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A36 - Navigate to The Gulch on Airbnb map
    airbnb_map_ok = (has_any_ci(answer, ['airbnb']) and
                     mentions_the_gulch(answer) and
                     has_any_ci(answer, ['map', 'locate']))
    evaluator.add_custom_node(
        result=bool(airbnb_map_ok),
        id="airbnb_action_map_location",
        desc="[Action Node] airbnb.com:F4:A36 - Locate The Gulch area on Airbnb map",
        parent=airbnb_node,
        critical=False
    )

    # Check date filter (weekend next month)
    date_filter_ok = has_any_ci(answer, ['weekend', 'next month', 'friday', 'sunday'])
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="airbnb_date_filter",
        desc="Sets date filter for a weekend next month (Friday to Sunday)",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Apply multiple filters (Entire place + Rating)
    filters_ok = mentions_airbnb_filters(answer)
    evaluator.add_custom_node(
        result=bool(filters_ok),
        id="airbnb_action_filters",
        desc="[Action Node] airbnb.com:F2:A11 - Apply filters for 'Entire place' and rating 4.8 or higher",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F2:P3 - Identify Guest Favorite listings
    guest_favorite_ok = mentions_guest_favorite(answer)
    evaluator.add_custom_node(
        result=bool(guest_favorite_ok),
        id="airbnb_perception_guest_favorite",
        desc="[Perception Node] airbnb.com:F2:P3 - Identifies listings with 'Guest Favorite' tag",
        parent=airbnb_node,
        critical=False
    )

    # Check bedroom matching
    bedroom_match_ok = (bedrooms_ok and
                        has_any_ci(answer, ['same', 'comparable', 'matching', 'bedroom']))
    evaluator.add_custom_node(
        result=bool(bedroom_match_ok),
        id="airbnb_bedroom_matching",
        desc="Finds listings with the same number of bedrooms as the Redfin property",
        parent=airbnb_node,
        critical=False
    )

    # Check three comparable listings
    three_comparables_ok = has_three_comparables(
        airbnb_info.listing_1_price,
        airbnb_info.listing_2_price,
        airbnb_info.listing_3_price
    )
    evaluator.add_custom_node(
        result=bool(three_comparables_ok),
        id="airbnb_three_comparables",
        desc="Identifies 3 comparable listings",
        parent=airbnb_node,
        critical=False
    )

    # Check average calculation
    average_ok = looks_like_price(airbnb_info.average_nightly_price)
    evaluator.add_custom_node(
        result=bool(average_ok),
        id="airbnb_average_calculation",
        desc="Calculates the average nightly price from the 3 listings",
        parent=airbnb_node,
        critical=False
    )

    # 3.3 Cash flow calculation section
    calculation_node = evaluator.add_sequential(
        id="calculation_section",
        desc="Cash flow calculation for investment analysis",
        parent=root,
        critical=False
    )

    # Check calculation mention
    calc_mention_ok = mentions_calculation(answer)
    evaluator.add_custom_node(
        result=bool(calc_mention_ok),
        id="calculation_mention",
        desc="Performs cash flow calculation",
        parent=calculation_node,
        critical=False
    )

    # Check monthly rental income calculation (20 days)
    rental_income_ok = (looks_like_price(cashflow_info.monthly_rental_income) and
                        has_any_ci(answer, ['20 days', '20']))
    evaluator.add_custom_node(
        result=bool(rental_income_ok),
        id="calculation_rental_income",
        desc="Calculates monthly rental income based on 20 days at average nightly rate",
        parent=calculation_node,
        critical=False
    )

    # Check coverage determination
    can_cover_ok = (cashflow_info.can_cover is not None and
                    bool(cashflow_info.can_cover.strip()))
    evaluator.add_custom_node(
        result=bool(can_cover_ok),
        id="calculation_coverage_determination",
        desc="Determines whether rental income can cover the monthly payment",
        parent=calculation_node,
        critical=False
    )

    # Comprehensive calculation check
    comprehensive_calc_ok = (rental_income_ok and
                             looks_like_price(cashflow_info.monthly_payment) and
                             can_cover_ok)
    evaluator.add_custom_node(
        result=bool(comprehensive_calc_ok),
        id="calculation_comprehensive",
        desc="Provides complete analysis comparing rental income vs monthly payment",
        parent=calculation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
