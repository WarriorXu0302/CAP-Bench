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
TASK_ID = "task-7ae3f0"
TASK_DESCRIPTION = 'I\'ve recently received an offer from Tableau in Seattle, with the office situated on North 34th Street. I plan to start in early 2026 and need to find accommodation.\n\nFirst, please confirm the exact address of Tableau Software\'s headquarters in Fremont, Seattle.\n\nUsing this address as the destination, please plan a driving route on Google Maps for an 8:30 AM arrival on a Monday. Identify three specific neighborhoods where the peak-hour driving commute time falls between 25 and 35 minutes (inclusive) to be considered as prospective residential areas.\n\nThen, on Apartments.com, for each of these three identified neighborhoods, locate one 1-bedroom apartment that has a rating of 4.0 or higher, a rent between $2500 and $3200, and explicitly features "In Unit Washer & Dryer" amenities.\n\nOutput should include: Tableau\'s exact address; the names of the three prospective neighborhoods along with their corresponding estimated Google Maps commute times; and for each recommended apartment: its name, address, rent, rating, and the Apartments.com detail page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TableauAddress(BaseModel):
    """Tableau headquarters address extracted from the answer"""
    address: Optional[str] = None


class Neighborhood(BaseModel):
    """A single neighborhood with commute time"""
    name: Optional[str] = None
    commute_time_text: Optional[str] = None


class NeighborhoodList(BaseModel):
    """List of neighborhoods extracted from the answer"""
    neighborhoods: List[Neighborhood] = Field(default_factory=list)


class Apartment(BaseModel):
    """A single apartment recommendation"""
    name: Optional[str] = None
    address: Optional[str] = None
    rent_text: Optional[str] = None
    rating_text: Optional[str] = None
    detail_link: Optional[str] = None


class ApartmentList(BaseModel):
    """List of apartments extracted from the answer"""
    apartments: List[Apartment] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tableau_address() -> str:
    return """
Extract the exact address of Tableau Software's headquarters in Fremont, Seattle from the answer.

Return:
- address: the complete street address as stated (e.g., "1621 N 34th St, Seattle, WA 98103"). If not present, set null.
"""


def prompt_extract_neighborhoods() -> str:
    return """
From the answer, extract the three prospective neighborhoods identified along with their Google Maps commute times to Tableau's office.

For each neighborhood, return:
- name: the neighborhood name exactly as stated
- commute_time_text: the commute time for that neighborhood exactly as stated (include units like "minutes" or "min")

Return a list of neighborhoods. If fewer than three are mentioned, return what is available.
"""


def prompt_extract_apartments() -> str:
    return """
From the answer, extract the apartment recommendations provided for each neighborhood.

For each apartment, return:
- name: the apartment complex name
- address: the apartment address
- rent_text: the monthly rent exactly as stated (include $ if present)
- rating_text: the rating exactly as stated
- detail_link: the Apartments.com detail page URL

Return a list of apartments. If fewer than three are mentioned, return what is available.
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


def looks_like_seattle_address(text: Optional[str]) -> bool:
    if not text:
        return False
    has_street_number = contains_digits(text)
    has_street_indicator = has_any_ci(text, ['street', 'st', 'ave', 'avenue', 'blvd', 'boulevard', 'way', 'road', 'rd'])
    has_seattle = has_any_ci(text, ['seattle', 'wa'])
    return has_street_number and has_street_indicator and has_seattle


def looks_like_north_34th(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['34th', '34']) and has_any_ci(text, ['north', 'n '])


def extract_commute_minutes(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Try to extract a number followed by "min" or "minute"
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:min|minute)',
        r'(\d+(?:\.\d+)?)\s*(?:to|-)?\s*\d+\s*(?:min|minute)',  # range like "25-30 min"
    ]
    for pattern in patterns:
        m = re.search(pattern, text.lower())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # Fallback: just extract first number if "minute" or "min" is mentioned
    if has_any_ci(text, ['min', 'minute']):
        return extract_float(text)
    return None


def commute_in_range(commute_text: Optional[str], min_val: float = 25.0, max_val: float = 35.0) -> bool:
    minutes = extract_commute_minutes(commute_text)
    if minutes is None:
        return False
    return min_val <= minutes <= max_val


def extract_rent_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Look for dollar amount pattern like $2,500 or $2500 or 2500
    m = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
    if m:
        try:
            cleaned = m.group(1).replace(',', '')
            return float(cleaned)
        except Exception:
            pass
    return None


def rent_in_range(rent_text: Optional[str], min_val: float = 2500.0, max_val: float = 3200.0) -> bool:
    amount = extract_rent_amount(rent_text)
    if amount is None:
        return False
    return min_val <= amount <= max_val


def extract_rating_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Look for rating like "4.5" or "4.0" or "4"
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def rating_at_least(rating_text: Optional[str], min_rating: float = 4.0) -> bool:
    rating = extract_rating_value(rating_text)
    if rating is None:
        return False
    return rating >= min_rating


def looks_like_apartments_com_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['apartments.com'])


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
    tableau_info = await evaluator.extract(
        prompt=prompt_extract_tableau_address(),
        template_class=TableauAddress,
        extraction_name="tableau_address"
    )

    neighborhoods_info = await evaluator.extract(
        prompt=prompt_extract_neighborhoods(),
        template_class=NeighborhoodList,
        extraction_name="neighborhoods_list"
    )

    apartments_info = await evaluator.extract(
        prompt=prompt_extract_apartments(),
        template_class=ApartmentList,
        extraction_name="apartments_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Tableau Address Confirmation
    address_node = evaluator.add_sequential(
        id="tableau_address_section",
        desc="Confirm Tableau Software headquarters address in Fremont, Seattle",
        parent=root,
        critical=False
    )

    address_ok = looks_like_seattle_address(tableau_info.address)
    north_34th_ok = looks_like_north_34th(tableau_info.address) if tableau_info.address else False

    evaluator.add_custom_node(
        result=bool(address_ok and north_34th_ok),
        id="tableau_address_confirmed",
        desc="Tableau headquarters address on North 34th Street in Seattle is provided",
        parent=address_node,
        critical=False
    )

    # 3.2 Google Maps Route Planning
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning to identify neighborhoods with 25-35 minute commute",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input destination (Tableau address)
    maps_destination_ok = has_any_ci(answer, ['google maps']) and (
        has_any_ci(answer, ['destination', 'arrival', 'ending point']) or
        (tableau_info.address and has_any_ci(answer, ['route', 'commute']))
    )
    evaluator.add_custom_node(
        result=bool(maps_destination_ok),
        id="maps_action_destination",
        desc="[Action Node] maps.google.com:F2:A5 - Input Tableau address as destination for route planning",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode and arrival time (Monday 8:30 AM)
    driving_ok = has_any_ci(answer, ['driv'])
    arrival_time_ok = has_any_ci(answer, ['8:30', '8:30 am', 'monday'])
    evaluator.add_custom_node(
        result=bool(driving_ok and arrival_time_ok),
        id="maps_action_driving_time",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode with Monday 8:30 AM arrival time for peak-hour calculation",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Identify neighborhoods with 25-35 min commute
    neighborhoods = neighborhoods_info.neighborhoods if neighborhoods_info else []
    neighborhoods_count = len(neighborhoods)
    neighborhoods_in_range = [n for n in neighborhoods if commute_in_range(n.commute_time_text)]
    all_three_in_range = len(neighborhoods_in_range) >= 3

    evaluator.add_custom_node(
        result=bool(all_three_in_range),
        id="maps_perception_neighborhood_time_range",
        desc="[Perception Node] maps.google.com:F1:P1 - Identify three neighborhoods with commute times between 25-35 minutes (inclusive)",
        parent=maps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(neighborhoods_count >= 3),
        id="maps_three_neighborhoods_identified",
        desc="Three specific neighborhoods are identified and named",
        parent=maps_node,
        critical=False
    )

    # 3.3 Apartments.com Search
    apartments_node = evaluator.add_sequential(
        id="apartments_com_section",
        desc="Apartments.com search for 1-bedroom apartments with specific criteria in identified neighborhoods",
        parent=root,
        critical=False
    )

    apartments = apartments_info.apartments if apartments_info else []
    apartments_count = len(apartments)

    # [Action Node] apartments.com:F1:A30 - Location targeting (neighborhoods)
    location_targeting_ok = has_any_ci(answer, ['apartments.com']) and (
        any(has_any_ci(answer, [n.name]) for n in neighborhoods if n.name) if neighborhoods else False
    )
    evaluator.add_custom_node(
        result=bool(location_targeting_ok),
        id="apartments_action_location",
        desc="[Action Node] apartments.com:F1:A30 - Target searches to the three identified neighborhoods",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Bedroom filter (1 bedroom)
    bedroom_filter_ok = has_any_ci(answer, ['1 bedroom', '1-bedroom', '1 bed'])
    evaluator.add_custom_node(
        result=bool(bedroom_filter_ok),
        id="apartments_action_bedroom_filter",
        desc="[Action Node] apartments.com:F1:A2 - Filter for 1-bedroom apartments",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A1 - Price range filter ($2500-$3200)
    price_filter_ok = has_any_ci(answer, ['2500', '3200', 'rent'])
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="apartments_action_price_filter",
        desc="[Action Node] apartments.com:F1:A1 - Filter by rent range $2500-$3200",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A11 - Amenities filter (In Unit Washer & Dryer)
    amenities_filter_ok = has_any_ci(answer, ['washer', 'dryer', 'w/d', 'in unit'])
    evaluator.add_custom_node(
        result=bool(amenities_filter_ok),
        id="apartments_action_amenities_filter",
        desc="[Action Node] apartments.com:F1:A11 - Filter for 'In Unit Washer & Dryer' amenities",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A3 - Sort/filter by rating (4.0+)
    rating_filter_ok = has_any_ci(answer, ['rating', '4.0', 'review'])
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),
        id="apartments_action_rating_sort",
        desc="[Action Node] apartments.com:F1:A3 - Sort or filter by rating 4.0 or higher",
        parent=apartments_node,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P1 - Verify In Unit Washer & Dryer amenity
    washer_dryer_ok = all(
        has_any_ci(apt.detail_link, ['apartments.com']) and
        (has_any_ci(answer, ['washer', 'dryer']) or has_any_ci(str(apt.name), ['washer', 'dryer']))
        for apt in apartments
    ) if apartments else False

    evaluator.add_custom_node(
        result=bool(washer_dryer_ok and apartments_count >= 3),
        id="apartments_perception_washer_dryer",
        desc="[Perception Node] apartments.com:F1:P1 - Verify all recommended apartments explicitly feature In Unit Washer & Dryer",
        parent=apartments_node,
        critical=False
    )

    # 3.4 Apartment Output Validation
    output_node = evaluator.add_parallel(
        id="apartment_output_section",
        desc="Validate apartment output completeness and correctness",
        parent=apartments_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(apartments_count >= 3),
        id="three_apartments_provided",
        desc="Three apartments are provided (one per neighborhood)",
        parent=output_node,
        critical=False
    )

    # Check rent range compliance
    apartments_rent_ok = [apt for apt in apartments if rent_in_range(apt.rent_text)]
    evaluator.add_custom_node(
        result=bool(len(apartments_rent_ok) >= 3),
        id="apartments_rent_in_range",
        desc="All apartments have rent between $2500 and $3200",
        parent=output_node,
        critical=False
    )

    # Check rating compliance
    apartments_rating_ok = [apt for apt in apartments if rating_at_least(apt.rating_text)]
    evaluator.add_custom_node(
        result=bool(len(apartments_rating_ok) >= 3),
        id="apartments_rating_4_plus",
        desc="All apartments have rating of 4.0 or higher",
        parent=output_node,
        critical=False
    )

    # Check Apartments.com links provided
    apartments_links_ok = [apt for apt in apartments if looks_like_apartments_com_url(apt.detail_link)]
    evaluator.add_custom_node(
        result=bool(len(apartments_links_ok) >= 3),
        id="apartments_links_provided",
        desc="Apartments.com detail page links are provided for all apartments",
        parent=output_node,
        critical=False
    )

    # Check all required fields present
    apartments_complete = [
        apt for apt in apartments
        if apt.name and apt.address and apt.rent_text and apt.rating_text and apt.detail_link
    ]
    evaluator.add_custom_node(
        result=bool(len(apartments_complete) >= 3),
        id="apartments_all_fields",
        desc="All apartments have complete information (name, address, rent, rating, link)",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
