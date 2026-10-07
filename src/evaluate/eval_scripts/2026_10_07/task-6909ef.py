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
TASK_ID = "task-6909ef"
TASK_DESCRIPTION = 'I plan to relocate with my family to North Carolina within the next year, and I’m currently deciding between downtown Raleigh (ZIP 27601) and downtown Charlotte (ZIP 28202). Environmental health is my top priority. First, go to epa.gov and use the ECHO tool to find, for each of these ZIP code areas, the number of facilities that have had a **Significant Violation** in the past three years. Then use the **How’s My Waterway** tool to check the overall **Condition** of the major water bodies in each area.\n\nCompare these two sets of results and select the ZIP code area with fewer significant-violation facilities and better water quality conditions. Then go to Zillow and find 3 for-sale **Single Family Homes** in the selected ZIP code area, with at least 3 bedrooms, priced at no more than **$650,000**, sorted by **Newest**.\n\nIf fewer than 3 listings are available with the above filters, keep property type = Single Family, bedrooms ≥ 3, and sorting = Newest unchanged, and first relax the price cap (to **$700,000**). If there are still fewer than 3, return all available listings in that ZIP code under the current criteria and clearly state the final number found.\n\n**Output requirements:** Provide a comparison of significant-violation facility counts for the two ZIP codes, plus the names and conditions of the major water bodies. For listings in the selected area, include address, price, bedroom count, days on Zillow (or listing/publish date), and Zillow detail-page link, and indicate whether each listing meets the “≤ $650,000” requirement.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ZipCodeViolationData(BaseModel):
    """Violation data for a single ZIP code from EPA ECHO"""
    zip_code: Optional[str] = None
    significant_violation_count: Optional[int] = None


class WaterBodyData(BaseModel):
    """Water body condition data for a single ZIP code from How's My Waterway"""
    zip_code: Optional[str] = None
    water_body_names: Optional[List[str]] = Field(default_factory=list)
    conditions: Optional[List[str]] = Field(default_factory=list)


class SelectedZipCode(BaseModel):
    """The selected ZIP code after comparison"""
    selected_zip: Optional[str] = None
    reason: Optional[str] = None


class PropertyListing(BaseModel):
    """A single property listing from Zillow"""
    address: Optional[str] = None
    price: Optional[int] = None
    bedrooms: Optional[int] = None
    days_on_zillow: Optional[str] = None
    zillow_url: Optional[str] = None
    meets_650k_requirement: Optional[bool] = None


class AllListings(BaseModel):
    """All property listings extracted from the answer"""
    listings: Optional[List[PropertyListing]] = Field(default_factory=list)
    total_count: Optional[int] = None
    price_cap_relaxed: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_raleigh_violations() -> str:
    return """
Extract the EPA ECHO significant violation data for Raleigh ZIP code 27601 from the answer.

Return:
- zip_code: should be "27601"
- significant_violation_count: the number of facilities with significant violations in the past three years

If the information is missing, set fields to null.
"""


def prompt_extract_charlotte_violations() -> str:
    return """
Extract the EPA ECHO significant violation data for Charlotte ZIP code 28202 from the answer.

Return:
- zip_code: should be "28202"
- significant_violation_count: the number of facilities with significant violations in the past three years

If the information is missing, set fields to null.
"""


def prompt_extract_raleigh_waterway() -> str:
    return """
Extract the How's My Waterway data for Raleigh ZIP code 27601 from the answer.

Return:
- zip_code: should be "27601"
- water_body_names: list of major water body names mentioned
- conditions: list of condition statuses (e.g., "Good", "Impaired") for those water bodies

If the information is missing, set fields to empty lists or null.
"""


def prompt_extract_charlotte_waterway() -> str:
    return """
Extract the How's My Waterway data for Charlotte ZIP code 28202 from the answer.

Return:
- zip_code: should be "28202"
- water_body_names: list of major water body names mentioned
- conditions: list of condition statuses (e.g., "Good", "Impaired") for those water bodies

If the information is missing, set fields to empty lists or null.
"""


def prompt_extract_selected_zip() -> str:
    return """
Extract which ZIP code was selected after comparing the environmental data.

Return:
- selected_zip: either "27601" or "28202"
- reason: brief explanation of why this ZIP was chosen

If not explicitly stated, set to null.
"""


def prompt_extract_listings() -> str:
    return """
Extract all property listings from the answer.

For each listing, extract:
- address: full property address
- price: numeric price value
- bedrooms: number of bedrooms
- days_on_zillow: days on Zillow or listing date
- zillow_url: the Zillow detail page link
- meets_650k_requirement: whether the price is ≤ $650,000

Also extract:
- total_count: total number of listings found
- price_cap_relaxed: whether the price cap was relaxed to $700,000

If information is missing, use null or empty lists.
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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\d+', str(text))
    if not m:
        return None
    try:
        return int(m.group())
    except Exception:
        return None


def mentions_zip_code(answer: str, zip_code: str) -> bool:
    return zip_code in answer


def mentions_significant_violation(answer: str) -> bool:
    return has_any_ci(answer, ['significant violation', 'significant-violation'])


def mentions_water_condition(answer: str) -> bool:
    return has_any_ci(answer, ['condition', 'good', 'impaired', 'poor', 'excellent'])


def mentions_zillow(answer: str) -> bool:
    return has_any_ci(answer, ['zillow'])


def mentions_single_family(answer: str) -> bool:
    return has_any_ci(answer, ['single family', 'single-family'])


def mentions_bedrooms(answer: str) -> bool:
    return has_any_ci(answer, ['bedroom', 'bed', 'br'])


def mentions_price_cap(answer: str) -> bool:
    return has_any_ci(answer, ['650', '650000', '650,000', '$650'])


def mentions_newest_sort(answer: str) -> bool:
    return has_any_ci(answer, ['newest', 'most recent', 'latest', 'new'])


def is_valid_condition_status(status: Optional[str]) -> bool:
    if not status:
        return False
    valid_statuses = ['good', 'impaired', 'poor', 'excellent', 'fair', 'not assessed']
    return any(ci_contains(status, s) for s in valid_statuses)


def is_zillow_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'zillow.com' in url.lower()


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
    raleigh_violations = await evaluator.extract(
        prompt=prompt_extract_raleigh_violations(),
        template_class=ZipCodeViolationData,
        extraction_name="raleigh_violations"
    )

    charlotte_violations = await evaluator.extract(
        prompt=prompt_extract_charlotte_violations(),
        template_class=ZipCodeViolationData,
        extraction_name="charlotte_violations"
    )

    raleigh_waterway = await evaluator.extract(
        prompt=prompt_extract_raleigh_waterway(),
        template_class=WaterBodyData,
        extraction_name="raleigh_waterway"
    )

    charlotte_waterway = await evaluator.extract(
        prompt=prompt_extract_charlotte_waterway(),
        template_class=WaterBodyData,
        extraction_name="charlotte_waterway"
    )

    selected_zip = await evaluator.extract(
        prompt=prompt_extract_selected_zip(),
        template_class=SelectedZipCode,
        extraction_name="selected_zip"
    )

    listings = await evaluator.extract(
        prompt=prompt_extract_listings(),
        template_class=AllListings,
        extraction_name="listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 EPA ECHO Section
    echo_node = evaluator.add_sequential(
        id="epa_echo_section",
        desc="EPA ECHO significant violation queries for both ZIP codes",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F1:A2 - Geographic location input for ZIP codes
    zip_input_ok = mentions_zip_code(answer, "27601") and mentions_zip_code(answer, "28202")
    evaluator.add_custom_node(
        result=bool(zip_input_ok),
        id="echo_action_zip_input",
        desc="[Action Node] epa.gov:F1:A2 - Input geographic location (ZIP codes 27601 and 28202) in ECHO tool",
        parent=echo_node,
        critical=False
    )

    # [Action Node] epa.gov:F1:A5 - Filter by significant violations
    sig_violation_filter_ok = mentions_significant_violation(answer)
    evaluator.add_custom_node(
        result=bool(sig_violation_filter_ok),
        id="echo_action_sig_violation_filter",
        desc="[Action Node] epa.gov:F1:A5 - Apply 'Significant Violation' filter for past three years",
        parent=echo_node,
        critical=False
    )

    # [Perception Node] epa.gov:F1:P1 - Extract facility counts
    raleigh_count_ok = raleigh_violations and raleigh_violations.significant_violation_count is not None
    charlotte_count_ok = charlotte_violations and charlotte_violations.significant_violation_count is not None
    evaluator.add_custom_node(
        result=bool(raleigh_count_ok and charlotte_count_ok),
        id="echo_perception_facility_counts",
        desc="[Perception Node] epa.gov:F1:P1 - Extract number of facilities with significant violations for both ZIP codes",
        parent=echo_node,
        critical=False
    )

    # 3.2 How's My Waterway Section
    waterway_node = evaluator.add_sequential(
        id="waterway_section",
        desc="How's My Waterway queries for both ZIP codes",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F4:A13 - Input location in How's My Waterway
    waterway_input_ok = mentions_zip_code(answer, "27601") and mentions_zip_code(answer, "28202") and has_any_ci(answer, ['waterway', 'water'])
    evaluator.add_custom_node(
        result=bool(waterway_input_ok),
        id="waterway_action_location_input",
        desc="[Action Node] epa.gov:F4:A13 - Input location (ZIP codes) in How's My Waterway tool",
        parent=waterway_node,
        critical=False
    )

    # [Perception Node] epa.gov:F4:P7 - Extract water body condition status
    raleigh_water_ok = (raleigh_waterway and
                        raleigh_waterway.water_body_names and
                        len(raleigh_waterway.water_body_names) > 0 and
                        raleigh_waterway.conditions and
                        any(is_valid_condition_status(c) for c in raleigh_waterway.conditions))

    charlotte_water_ok = (charlotte_waterway and
                          charlotte_waterway.water_body_names and
                          len(charlotte_waterway.water_body_names) > 0 and
                          charlotte_waterway.conditions and
                          any(is_valid_condition_status(c) for c in charlotte_waterway.conditions))

    evaluator.add_custom_node(
        result=bool(raleigh_water_ok and charlotte_water_ok),
        id="waterway_perception_conditions",
        desc="[Perception Node] epa.gov:F4:P7 - Extract water body names and condition statuses (Good/Impaired/etc.) for both ZIP codes",
        parent=waterway_node,
        critical=False
    )

    # 3.3 Comparison and Selection
    comparison_node = evaluator.add_sequential(
        id="comparison_section",
        desc="Compare environmental data and select ZIP code",
        parent=root,
        critical=False
    )

    # Check if a valid ZIP was selected
    selection_ok = selected_zip and selected_zip.selected_zip in ["27601", "28202"]
    evaluator.add_custom_node(
        result=bool(selection_ok),
        id="comparison_selection_made",
        desc="Select one ZIP code based on fewer violations and better water quality",
        parent=comparison_node,
        critical=False
    )

    # Check if comparison was performed
    comparison_mentioned = has_any_ci(answer, ['compare', 'comparison', 'fewer', 'better', 'selected', 'chose'])
    evaluator.add_custom_node(
        result=bool(comparison_mentioned),
        id="comparison_rationale_provided",
        desc="Provide comparison rationale for ZIP code selection",
        parent=comparison_node,
        critical=False
    )

    # 3.4 Zillow Section
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow property search in selected ZIP code",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Search location (selected ZIP)
    zillow_location_ok = mentions_zillow(answer) and (mentions_zip_code(answer, "27601") or mentions_zip_code(answer, "28202"))
    evaluator.add_custom_node(
        result=bool(zillow_location_ok),
        id="zillow_action_location_search",
        desc="[Action Node] zillow.com:F1:A35 - Search for properties in the selected ZIP code area",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A8 - Filter by property type (Single Family)
    single_family_filter_ok = mentions_single_family(answer)
    evaluator.add_custom_node(
        result=bool(single_family_filter_ok),
        id="zillow_action_property_type_filter",
        desc="[Action Node] zillow.com:F2:A8 - Filter by property type = Single Family Homes",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A10 - Filter by bedrooms (≥3)
    bedrooms_filter_ok = mentions_bedrooms(answer) and has_any_ci(answer, ['3', 'three'])
    evaluator.add_custom_node(
        result=bool(bedrooms_filter_ok),
        id="zillow_action_bedrooms_filter",
        desc="[Action Node] zillow.com:F2:A10 - Filter by bedrooms ≥ 3",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A12 - Filter by price range (≤$650,000)
    price_filter_ok = mentions_price_cap(answer)
    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="zillow_action_price_filter",
        desc="[Action Node] zillow.com:F2:A12 - Filter by price ≤ $650,000 (or relaxed to $700,000 if needed)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A2 - Sort by Newest
    newest_sort_ok = mentions_newest_sort(answer)
    evaluator.add_custom_node(
        result=bool(newest_sort_ok),
        id="zillow_action_sort_newest",
        desc="[Action Node] zillow.com:F2:A2 - Sort results by Newest",
        parent=zillow_node,
        critical=False
    )

    # 3.5 Listing Results
    results_node = evaluator.add_sequential(
        id="listing_results_section",
        desc="Extract and verify property listing results",
        parent=root,
        critical=False
    )

    # Check if listings were extracted
    has_listings = listings and listings.listings and len(listings.listings) > 0
    evaluator.add_custom_node(
        result=bool(has_listings),
        id="listings_found",
        desc="At least one property listing was found and extracted",
        parent=results_node,
        critical=False
    )

    # Verify listing details are complete
    if has_listings:
        complete_listings = []
        for listing in listings.listings:
            if (listing.address and
                listing.price is not None and
                listing.bedrooms is not None and
                listing.zillow_url and
                is_zillow_url(listing.zillow_url)):
                complete_listings.append(listing)

        completeness_ok = len(complete_listings) > 0
        evaluator.add_custom_node(
            result=bool(completeness_ok),
            id="listings_complete_details",
            desc="Listings include address, price, bedrooms, and Zillow URL",
            parent=results_node,
            critical=False
        )

        # Check bedroom requirement (≥3)
        bedroom_requirement_met = all(
            listing.bedrooms >= 3
            for listing in complete_listings
            if listing.bedrooms is not None
        )
        evaluator.add_custom_node(
            result=bool(bedroom_requirement_met),
            id="listings_bedroom_requirement",
            desc="All listings have at least 3 bedrooms",
            parent=results_node,
            critical=False
        )

        # Check if price compliance is indicated
        price_compliance_indicated = any(
            listing.meets_650k_requirement is not None
            for listing in listings.listings
        )
        evaluator.add_custom_node(
            result=bool(price_compliance_indicated),
            id="listings_price_compliance_indicated",
            desc="Listings indicate whether they meet the ≤$650,000 requirement",
            parent=results_node,
            critical=False
        )

        # Check for days on Zillow or listing date
        has_timing_info = any(
            listing.days_on_zillow is not None
            for listing in listings.listings
        )
        evaluator.add_custom_node(
            result=bool(has_timing_info),
            id="listings_timing_info",
            desc="Listings include days on Zillow or listing date",
            parent=results_node,
            critical=False
        )

    # Check if answer mentions the final count
    mentions_final_count = has_any_ci(answer, ['found', 'listing', 'properties', 'homes', 'results'])
    evaluator.add_custom_node(
        result=bool(mentions_final_count),
        id="mentions_listing_count",
        desc="Answer states how many listings were found",
        parent=results_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
