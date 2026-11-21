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
TASK_ID = "task-acba34"
TASK_DESCRIPTION = 'I am considering purchasing a property in the **77012 ZIP code area of Houston, TX**, and would like to conduct an in-depth environmental and community background check.\n\nFirst, use **Google Maps** to confirm the general location of this ZIP code area. Identify and list the names of three prominent industrial or chemical-related facilities (e.g., oil refineries, chemical plants, or large warehouses) located within or immediately adjacent to this area.\n\nNext, visit the **EPA ECHO** website. For each of these three facilities, research their compliance records. Specifically look for "Significant Non-Compliance" incidents or formal enforcement actions within the last three years. Record their latest compliance status.\n\nFinally, go to **Census QuickFacts** to retrieve the latest census data for Houston City, Texas, and Harris County, Texas (where the 77012 ZIP code is located). Compare the "Persons in poverty, percent" and "Percentage of persons with a bachelor\'s degree or higher" for both.\n\n**Output:** The names of the three facilities, their Google Maps location links, their EPA compliance status (indicating "Significant Non-Compliance" if applicable), and the date of their latest compliance assessment. Additionally, provide comparative data for Harris County and Houston City regarding the poverty rate and the percentage of residents with a bachelor\'s degree or higher. If a specific facility\'s records cannot be found on EPA ECHO, please state the reason.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    """Information about a single facility"""
    name: Optional[str] = None
    location_link: Optional[str] = None
    compliance_status: Optional[str] = None
    compliance_date: Optional[str] = None
    not_found_reason: Optional[str] = None


class FacilitiesExtracted(BaseModel):
    """Three facilities extracted from the answer"""
    facility_1: Optional[FacilityInfo] = None
    facility_2: Optional[FacilityInfo] = None
    facility_3: Optional[FacilityInfo] = None


class CensusData(BaseModel):
    """Census demographic data extracted from the answer"""
    harris_county_poverty_percent: Optional[str] = None
    harris_county_bachelors_percent: Optional[str] = None
    houston_city_poverty_percent: Optional[str] = None
    houston_city_bachelors_percent: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities_from_answer() -> str:
    return """
Extract information about the three industrial/chemical facilities from the 77012 ZIP code area mentioned in the answer.

For each facility, extract:
- name: the facility name exactly as stated
- location_link: the Google Maps link if provided
- compliance_status: the EPA compliance status (e.g., "In Compliance", "Significant Non-Compliance", "SNC")
- compliance_date: the date of the latest compliance assessment
- not_found_reason: if EPA records were not found, the reason stated

Return facility_1, facility_2, and facility_3. If any facility or field is missing, set it to null.
"""


def prompt_extract_census_from_answer() -> str:
    return """
Extract the Census QuickFacts data for Harris County and Houston City from the answer.

Return:
- harris_county_poverty_percent: poverty rate for Harris County
- harris_county_bachelors_percent: percentage with bachelor's degree or higher for Harris County
- houston_city_poverty_percent: poverty rate for Houston City
- houston_city_bachelors_percent: percentage with bachelor's degree or higher for Houston City

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


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text, re.IGNORECASE))


def looks_like_maps_link(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['maps.google', 'google.com/maps', 'goo.gl/maps'])


def looks_like_compliance_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['compliance', 'snc', 'significant non-compliance', 'violation', 'enforcement'])


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats: "2023-01-15", "01/15/2023", "January 15, 2023", etc.
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b'
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in date_patterns)


def looks_like_percentage(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '%') or ci_contains(text, 'percent'))


def count_facilities_in_answer(answer: str, facilities: FacilitiesExtracted) -> int:
    count = 0
    if facilities.facility_1 and facilities.facility_1.name:
        count += 1
    if facilities.facility_2 and facilities.facility_2.name:
        count += 1
    if facilities.facility_3 and facilities.facility_3.name:
        count += 1
    return count


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
    Restrict evaluator.verify to at most one usage (not used here).
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
    facilities = await evaluator.extract(
        prompt=prompt_extract_facilities_from_answer(),
        template_class=FacilitiesExtracted,
        extraction_name="facilities_extracted"
    )

    census_data = await evaluator.extract(
        prompt=prompt_extract_census_from_answer(),
        template_class=CensusData,
        extraction_name="census_data_extracted"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps - Locate 77012 ZIP code and identify industrial facilities",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search and positioning
    maps_search_ok = (has_any_ci(answer, ['google maps']) and
                      has_any_ci(answer, ['77012']))
    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="maps_action_location_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search and locate the 77012 ZIP code area on Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A17 - Browse and page through to find facilities
    facility_count = count_facilities_in_answer(answer, facilities)
    found_three_facilities = (facility_count == 3)
    evaluator.add_custom_node(
        result=bool(found_three_facilities),
        id="maps_action_browse_facilities",
        desc="[Action Node] maps.google.com:F1:A17 - Browse/navigate the map area to identify exactly 3 industrial or chemical facilities",
        parent=maps_node,
        critical=False
    )

    # Check if facilities have names and location links
    facilities_have_names = (facilities.facility_1 and facilities.facility_1.name and
                            facilities.facility_2 and facilities.facility_2.name and
                            facilities.facility_3 and facilities.facility_3.name)

    facilities_have_links = False
    if facilities.facility_1 and facilities.facility_1.location_link:
        if looks_like_maps_link(facilities.facility_1.location_link):
            facilities_have_links = True
    if facilities.facility_2 and facilities.facility_2.location_link:
        if looks_like_maps_link(facilities.facility_2.location_link):
            facilities_have_links = True
    if facilities.facility_3 and facilities.facility_3.location_link:
        if looks_like_maps_link(facilities.facility_3.location_link):
            facilities_have_links = True

    evaluator.add_custom_node(
        result=bool(facilities_have_names and facilities_have_links),
        id="maps_facilities_with_links",
        desc="Facilities have names and at least one has a Google Maps location link",
        parent=maps_node,
        critical=False
    )

    # 3.2 EPA ECHO section
    epa_node = evaluator.add_sequential(
        id="epa_echo_section",
        desc="EPA ECHO - Research compliance records for the three facilities",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F1:A1 - Multi-dimensional search switching
    epa_search_ok = has_any_ci(answer, ['epa echo', 'epa', 'echo'])
    evaluator.add_custom_node(
        result=bool(epa_search_ok),
        id="epa_action_search_switch",
        desc="[Action Node] epa.gov:F1:A1 - Navigate to EPA ECHO and switch search mode to search by facility name/ID",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F1:P2 - Status label recognition
    status_count = 0
    if facilities.facility_1 and looks_like_compliance_status(facilities.facility_1.compliance_status):
        status_count += 1
    if facilities.facility_2 and looks_like_compliance_status(facilities.facility_2.compliance_status):
        status_count += 1
    if facilities.facility_3 and looks_like_compliance_status(facilities.facility_3.compliance_status):
        status_count += 1

    found_statuses = status_count >= 2  # At least 2 out of 3
    evaluator.add_custom_node(
        result=bool(found_statuses),
        id="epa_perception_status_labels",
        desc="[Perception Node] epa.gov:F1:P2 - Identify and extract compliance status labels (e.g., In Compliance, SNC) for facilities",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F1:P3 - Temporal data understanding (last 3 years, dates)
    date_count = 0
    if facilities.facility_1 and looks_like_date(facilities.facility_1.compliance_date):
        date_count += 1
    if facilities.facility_2 and looks_like_date(facilities.facility_2.compliance_date):
        date_count += 1
    if facilities.facility_3 and looks_like_date(facilities.facility_3.compliance_date):
        date_count += 1

    has_temporal_data = date_count >= 2
    mentions_three_years = has_any_ci(answer, ['three years', '3 years', 'last three', 'past three'])
    evaluator.add_custom_node(
        result=bool(has_temporal_data and mentions_three_years),
        id="epa_perception_temporal_data",
        desc="[Perception Node] epa.gov:F1:P3 - Extract compliance assessment dates and identify records within the last 3 years",
        parent=epa_node,
        critical=False
    )

    # Check for "not found" handling
    handles_not_found = False
    if facilities.facility_1 and facilities.facility_1.not_found_reason:
        handles_not_found = True
    if facilities.facility_2 and facilities.facility_2.not_found_reason:
        handles_not_found = True
    if facilities.facility_3 and facilities.facility_3.not_found_reason:
        handles_not_found = True

    evaluator.add_custom_node(
        result=bool(handles_not_found or found_statuses),
        id="epa_handles_missing_records",
        desc="Appropriately handles cases where EPA records cannot be found by stating the reason",
        parent=epa_node,
        critical=False
    )

    # 3.3 Census QuickFacts section
    census_node = evaluator.add_sequential(
        id="census_quickfacts_section",
        desc="Census QuickFacts - Retrieve and compare demographic data for Harris County and Houston City",
        parent=root,
        critical=False
    )

    # [Action Node] census.gov:F3:A1 - Multi-region comparison
    census_search_ok = has_any_ci(answer, ['census', 'quickfacts'])
    mentions_both_regions = (has_any_ci(answer, ['harris county']) and
                            has_any_ci(answer, ['houston city', 'houston,']))
    evaluator.add_custom_node(
        result=bool(census_search_ok and mentions_both_regions),
        id="census_action_multi_region",
        desc="[Action Node] census.gov:F3:A1 - Access Census QuickFacts and add both Harris County and Houston City for comparison",
        parent=census_node,
        critical=False
    )

    # [Perception Node] census.gov:F3:P1 - Table data extraction
    harris_poverty_ok = looks_like_percentage(census_data.harris_county_poverty_percent)
    harris_bachelors_ok = looks_like_percentage(census_data.harris_county_bachelors_percent)
    houston_poverty_ok = looks_like_percentage(census_data.houston_city_poverty_percent)
    houston_bachelors_ok = looks_like_percentage(census_data.houston_city_bachelors_percent)

    extracted_all_census = (harris_poverty_ok and harris_bachelors_ok and
                           houston_poverty_ok and houston_bachelors_ok)
    evaluator.add_custom_node(
        result=bool(extracted_all_census),
        id="census_perception_table_extraction",
        desc="[Perception Node] census.gov:F3:P1 - Extract poverty rate and bachelor's degree percentage from Census tables for both regions",
        parent=census_node,
        critical=False
    )

    # Check for comparison/contrast language
    mentions_comparison = has_any_ci(answer, ['compare', 'comparison', 'versus', 'vs', 'compared to', 'higher', 'lower'])
    evaluator.add_custom_node(
        result=bool(mentions_comparison and extracted_all_census),
        id="census_provides_comparison",
        desc="Provides comparative analysis of the demographic data between Harris County and Houston City",
        parent=census_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
