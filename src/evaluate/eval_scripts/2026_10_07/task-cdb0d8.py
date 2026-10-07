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
TASK_ID = "task-cdb0d8"
TASK_DESCRIPTION = 'I’m moving to Austin, Texas for work in January next year, and my company is located on Congress Avenue downtown. My monthly rent budget is $1,200–$1,800, and I’m looking for a one-bedroom apartment. First, please find up to 5 apartments on Apartments.com that match the budget and unit type, and record each property’s community address and ZIP code. If fewer than 5 listings are available after filtering, record all available listings and note the actual count.\n\nThen, use the EPA’s **How’s My Waterway** tool to check water quality conditions for those communities, determine whether nearby water bodies have pollution issues, and record each community’s water quality status (**Good / Impaired / Insufficient Data**) and primary pollution sources (if any).\n\nNext, go to Data.gov and search for Texas crime statistics datasets, prioritizing datasets that include crime rates by ZIP code for the Austin area. Download the dataset in CSV format and identify the crime rate (crimes per 1,000 people) for the ZIP codes corresponding to these communities. If you cannot find data covering all ZIP codes, first record the ZIP codes and crime rates that can be matched, and clearly mark unmatched ZIP codes in the results.\n\nFinally, output a comparison table including: **Apartment Name, Community Address, ZIP Code, Monthly Rent, Apartments.com Link, Water Quality Status, Primary Pollution Source (if any), EPA Query Link, Crime Rate (per 1,000 people), Data.gov Dataset Link**.\n\nBased on comparable samples, recommend the 2 communities with the **lowest crime rates and good water quality**, and explain the reasons. If fewer than 2 communities meet both criteria, recommend according to the rule **“good water quality first, lower crime rate second”** and explain why there is a shortfall.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApartmentListing(BaseModel):
    """Single apartment listing details"""
    apartment_name: Optional[str] = None
    community_address: Optional[str] = None
    zip_code: Optional[str] = None
    monthly_rent: Optional[str] = None
    apartments_com_link: Optional[str] = None


class ApartmentListings(BaseModel):
    """Collection of apartment listings from Apartments.com"""
    listings: List[ApartmentListing] = Field(default_factory=list)
    total_count: Optional[int] = None


class WaterQualityInfo(BaseModel):
    """Water quality information for a single community"""
    apartment_name: Optional[str] = None
    water_quality_status: Optional[str] = None
    pollution_source: Optional[str] = None
    epa_query_link: Optional[str] = None


class WaterQualityData(BaseModel):
    """Collection of water quality data"""
    water_quality_records: List[WaterQualityInfo] = Field(default_factory=list)


class CrimeRateInfo(BaseModel):
    """Crime rate information for a single ZIP code"""
    zip_code: Optional[str] = None
    crime_rate: Optional[str] = None


class CrimeData(BaseModel):
    """Collection of crime rate data"""
    crime_records: List[CrimeRateInfo] = Field(default_factory=list)
    data_gov_link: Optional[str] = None


class RecommendationInfo(BaseModel):
    """Final recommendations"""
    recommended_communities: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartment_listings() -> str:
    return """
Extract all apartment listings from the answer that were found on Apartments.com.

For each listing, extract:
- apartment_name: the name of the apartment complex
- community_address: the full community address (street, city, state)
- zip_code: the ZIP code
- monthly_rent: the monthly rent amount as stated
- apartments_com_link: the Apartments.com link/URL for that listing

Also extract:
- total_count: the total number of listings found

Return all listings in the 'listings' array. If any field is missing, set it to null.
"""


def prompt_extract_water_quality() -> str:
    return """
Extract water quality information from the answer for each community checked via EPA's How's My Waterway tool.

For each community, extract:
- apartment_name: the apartment name (to match with listings)
- water_quality_status: the status (Good, Impaired, Insufficient Data, or similar)
- pollution_source: primary pollution source if mentioned, otherwise null or "None"
- epa_query_link: the EPA How's My Waterway link/URL used for that query

Return all records in the 'water_quality_records' array. If any field is missing, set it to null.
"""


def prompt_extract_crime_data() -> str:
    return """
Extract crime rate information from the answer for each ZIP code.

For each ZIP code, extract:
- zip_code: the ZIP code
- crime_rate: the crime rate (crimes per 1,000 people) as stated

Also extract:
- data_gov_link: the Data.gov dataset link/URL used

Return all records in the 'crime_records' array. If any field is missing, set it to null.
"""


def prompt_extract_recommendations() -> str:
    return """
Extract the final recommendations from the answer.

Extract:
- recommended_communities: list of recommended apartment/community names (up to 2)
- explanation: the explanation for the recommendations

If any field is missing, set it to null or empty list.
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


def looks_like_rent_in_range(rent_text: Optional[str], min_rent: int = 1200, max_rent: int = 1800) -> bool:
    if not rent_text:
        return False
    val = extract_float(rent_text)
    if val is None:
        return False
    return min_rent <= val <= max_rent


def looks_like_zip_code(text: Optional[str]) -> bool:
    if not text:
        return False
    # Match 5-digit ZIP codes
    return bool(re.search(r'\b\d{5}\b', text))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text))


def looks_like_water_quality_status(text: Optional[str]) -> bool:
    if not text:
        return False
    valid_statuses = ['good', 'impaired', 'insufficient data', '良好', '受损', '数据不足']
    return any(ci_contains(text, status) for status in valid_statuses)


def looks_like_crime_rate(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


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
    apartment_listings = await evaluator.extract(
        prompt=prompt_extract_apartment_listings(),
        template_class=ApartmentListings,
        extraction_name="apartment_listings"
    )

    water_quality_data = await evaluator.extract(
        prompt=prompt_extract_water_quality(),
        template_class=WaterQualityData,
        extraction_name="water_quality_data"
    )

    crime_data = await evaluator.extract(
        prompt=prompt_extract_crime_data(),
        template_class=CrimeData,
        extraction_name="crime_data"
    )

    recommendations = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=RecommendationInfo,
        extraction_name="recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Apartments.com section
    apartments_node = evaluator.add_sequential(
        id="apartments_com_section",
        desc="Apartments.com search and listing collection",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A1 - Price filter
    has_listings = apartment_listings and apartment_listings.listings and len(apartment_listings.listings) > 0
    rent_in_range_count = sum(1 for listing in (apartment_listings.listings if has_listings else [])
                               if looks_like_rent_in_range(listing.monthly_rent))
    price_filter_ok = has_listings and rent_in_range_count > 0

    evaluator.add_custom_node(
        result=bool(price_filter_ok),
        id="apartments_price_filter",
        desc="[Action Node] apartments.com:F1:A1 - Apply price filter to show apartments within $1,200-$1,800 range",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Bedrooms filter
    one_bedroom_ok = has_any_ci(answer, ['one-bedroom', '1 bedroom', '1bd', '1-bedroom'])

    evaluator.add_custom_node(
        result=bool(one_bedroom_ok),
        id="apartments_bedroom_filter",
        desc="[Action Node] apartments.com:F1:A2 - Apply bedroom filter to show one-bedroom apartments",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A28 - Scroll to load more listings
    found_multiple = has_listings and len(apartment_listings.listings) >= 2

    evaluator.add_custom_node(
        result=bool(found_multiple),
        id="apartments_scroll_listings",
        desc="[Action Node] apartments.com:F1:A28 - Browse through multiple listings (scroll/pagination)",
        parent=apartments_node,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P1 - Status markers recognition
    has_valid_links = has_listings and sum(1 for listing in apartment_listings.listings
                                           if looks_like_url(listing.apartments_com_link)) > 0

    evaluator.add_custom_node(
        result=bool(has_valid_links),
        id="apartments_status_markers",
        desc="[Perception Node] apartments.com:F1:P1 - Recognize status markers and identify quality listings with valid links",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F3:A12 - Click card to enter details
    has_addresses = has_listings and sum(1 for listing in apartment_listings.listings
                                         if listing.community_address and len(listing.community_address.strip()) > 10) > 0

    evaluator.add_custom_node(
        result=bool(has_addresses),
        id="apartments_click_details",
        desc="[Action Node] apartments.com:F3:A12 - Click listing cards to access detail pages for complete address information",
        parent=apartments_node,
        critical=False
    )

    # Check ZIP codes extracted
    has_zip_codes = has_listings and sum(1 for listing in apartment_listings.listings
                                         if looks_like_zip_code(listing.zip_code)) > 0

    evaluator.add_custom_node(
        result=bool(has_zip_codes),
        id="apartments_zip_codes",
        desc="Extract ZIP codes for all found apartments",
        parent=apartments_node,
        critical=False
    )

    # 3.2 EPA How's My Waterway section
    epa_node = evaluator.add_sequential(
        id="epa_section",
        desc="EPA How's My Waterway water quality checks",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F4:A13 - Address input query
    has_water_data = water_quality_data and water_quality_data.water_quality_records and len(water_quality_data.water_quality_records) > 0
    epa_address_input_ok = has_water_data and has_any_ci(answer, ["how's my waterway", "epa"])

    evaluator.add_custom_node(
        result=bool(epa_address_input_ok),
        id="epa_address_input",
        desc="[Action Node] epa.gov:F4:A13 - Input community addresses into EPA How's My Waterway search",
        parent=epa_node,
        critical=False
    )

    # [Action Node] epa.gov:F4:A14 - Tab switching for different water quality indicators
    mentions_tabs = has_any_ci(answer, ['overview', 'swimming', 'eating fish', 'aquatic life', 'drinking water'])

    evaluator.add_custom_node(
        result=bool(mentions_tabs),
        id="epa_tab_switching",
        desc="[Action Node] epa.gov:F4:A14 - Switch between tabs (Overview/Swimming/Eating Fish) to view water quality details",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F4:P7 - Water quality status label recognition
    water_status_count = sum(1 for record in (water_quality_data.water_quality_records if has_water_data else [])
                             if looks_like_water_quality_status(record.water_quality_status))
    water_status_ok = water_status_count > 0

    evaluator.add_custom_node(
        result=bool(water_status_ok),
        id="epa_status_recognition",
        desc="[Perception Node] epa.gov:F4:P7 - Recognize water quality status labels (Good/Impaired/Insufficient Data)",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F4:P8 - Pollution issues structured data recognition
    has_pollution_info = has_water_data and sum(1 for record in water_quality_data.water_quality_records
                                                 if record.pollution_source and len(record.pollution_source.strip()) > 0) > 0

    evaluator.add_custom_node(
        result=bool(has_pollution_info),
        id="epa_pollution_recognition",
        desc="[Perception Node] epa.gov:F4:P8 - Identify pollution sources from Identified Issues area",
        parent=epa_node,
        critical=False
    )

    # Check EPA query links
    has_epa_links = has_water_data and sum(1 for record in water_quality_data.water_quality_records
                                           if looks_like_url(record.epa_query_link)) > 0

    evaluator.add_custom_node(
        result=bool(has_epa_links),
        id="epa_query_links",
        desc="Record EPA How's My Waterway query links for each community",
        parent=epa_node,
        critical=False
    )

    # 3.3 Data.gov section
    datagov_node = evaluator.add_sequential(
        id="datagov_section",
        desc="Data.gov crime statistics dataset search and download",
        parent=root,
        critical=False
    )

    # [Action Node] data.gov:F1:A4 - Search result pagination
    datagov_search_ok = has_any_ci(answer, ['data.gov', 'data gov']) and has_any_ci(answer, ['crime', 'texas', 'austin'])

    evaluator.add_custom_node(
        result=bool(datagov_search_ok),
        id="datagov_pagination",
        desc="[Action Node] data.gov:F1:A4 - Browse search results pages to find appropriate crime dataset",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F2:A1 - Organization filter
    texas_agency_ok = has_any_ci(answer, ['texas', 'tx', 'state'])

    evaluator.add_custom_node(
        result=bool(texas_agency_ok),
        id="datagov_org_filter",
        desc="[Action Node] data.gov:F2:A1 - Apply organization filter to narrow to Texas-related agencies",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F2:A2 - Format filter
    csv_format_ok = has_any_ci(answer, ['csv'])

    evaluator.add_custom_node(
        result=bool(csv_format_ok),
        id="datagov_format_filter",
        desc="[Action Node] data.gov:F2:A2 - Apply format filter to show CSV-downloadable datasets",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F2:A3 - Tag filter
    crime_tag_ok = has_any_ci(answer, ['crime'])

    evaluator.add_custom_node(
        result=bool(crime_tag_ok),
        id="datagov_tag_filter",
        desc="[Action Node] data.gov:F2:A3 - Apply tag filter for crime-related datasets",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F4:A8 - Click to enter dataset details
    has_datagov_link = crime_data and looks_like_url(crime_data.data_gov_link)

    evaluator.add_custom_node(
        result=bool(has_datagov_link),
        id="datagov_click_details",
        desc="[Action Node] data.gov:F4:A8 - Click search result to view dataset detail page",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F4:P4 - Metadata table understanding
    has_crime_records = crime_data and crime_data.crime_records and len(crime_data.crime_records) > 0
    austin_zip_ok = has_any_ci(answer, ['austin', 'zip code', 'zipcode'])

    evaluator.add_custom_node(
        result=bool(has_crime_records and austin_zip_ok),
        id="datagov_metadata_understanding",
        desc="[Perception Node] data.gov:F4:P4 - Understand dataset metadata to verify geographic coverage and ZIP code availability",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F5:A8 - Download resource
    download_ok = has_crime_records and has_any_ci(answer, ['download'])

    evaluator.add_custom_node(
        result=bool(download_ok),
        id="datagov_download",
        desc="[Action Node] data.gov:F5:A8 - Download CSV dataset file",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F5:P3 - Resource format selection understanding
    crime_rate_count = sum(1 for record in (crime_data.crime_records if has_crime_records else [])
                           if looks_like_crime_rate(record.crime_rate))
    crime_extraction_ok = crime_rate_count > 0

    evaluator.add_custom_node(
        result=bool(crime_extraction_ok),
        id="datagov_format_selection",
        desc="[Perception Node] data.gov:F5:P3 - Identify correct CSV format in Downloads & Resources table and extract crime rate data",
        parent=datagov_node,
        critical=False
    )

    # 3.4 Final output and recommendation section
    output_node = evaluator.add_sequential(
        id="output_section",
        desc="Final comparison table and recommendations",
        parent=root,
        critical=False
    )

    # Check comparison table completeness
    table_ok = (has_listings and has_water_data and has_crime_records and
                has_any_ci(answer, ['table', 'comparison', '对比']))

    evaluator.add_custom_node(
        result=bool(table_ok),
        id="comparison_table",
        desc="Output comprehensive comparison table with all required fields",
        parent=output_node,
        critical=False
    )

    # Check recommendations
    has_recommendations = recommendations and recommendations.recommended_communities and len(recommendations.recommended_communities) > 0
    has_explanation = recommendations and recommendations.explanation and len(recommendations.explanation.strip()) > 0
    recommendation_ok = has_recommendations and has_explanation

    evaluator.add_custom_node(
        result=bool(recommendation_ok),
        id="final_recommendations",
        desc="Provide recommendations for top 2 communities with lowest crime rates and good water quality, with explanations",
        parent=output_node,
        critical=False
    )

    # Check recommendation criteria mentioned
    criteria_ok = has_any_ci(answer, ['crime rate', 'water quality', '犯罪率', '水质'])

    evaluator.add_custom_node(
        result=bool(criteria_ok),
        id="recommendation_criteria",
        desc="Explain recommendation logic based on crime rate and water quality criteria",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
