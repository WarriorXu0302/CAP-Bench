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
TASK_ID = "task-315905"
TASK_DESCRIPTION = 'Our team is evaluating environmental compliance risk in Harris County, Texas. Please use the EPA ECHO system and prioritize list-based results (do not rely on the map view) to identify one chemical or manufacturing facility in the county that is currently in **Significant Non-Compliance (SNC)** status. First, open the facility ranked at the top of the list, review its detailed report, and determine the specific chemical pollutant for which it is in violation (e.g., benzene, lead, particulate matter).\n\nIf the map page appears blank, flickers, or fails to load reliably, switch to an accessible list/table results page and continue the screening there. If the first facility’s detail page does not clearly name a pollutant, move on to the next facility, and continue until you find one with an explicit pollutant record.\n\nAfter obtaining the pollutant name, conduct a technical search in the USPTO patent database. Find **two U.S. patents granted after 2021** that are specifically aimed at treating or removing that pollutant. Finally, report the facility name, the violating pollutant, and the titles of the two patents you found.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    """Facility information extracted from the answer"""
    facility_name: Optional[str] = None
    county_mention: Optional[str] = None
    snc_status_mention: Optional[str] = None


class PollutantInfo(BaseModel):
    """Pollutant information extracted from the answer"""
    pollutant_name: Optional[str] = None


class PatentInfo(BaseModel):
    """Patent information extracted from the answer"""
    patent_1_title: Optional[str] = None
    patent_1_year: Optional[str] = None
    patent_2_title: Optional[str] = None
    patent_2_year: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_from_answer() -> str:
    return """
Extract the EPA ECHO facility information reported in the answer for Harris County, Texas.

Return:
- facility_name: the name of the chemical or manufacturing facility identified as being in Significant Non-Compliance (SNC) status. If not present, set null.
- county_mention: any mention of Harris County or the county context. If not present, set null.
- snc_status_mention: any mention of SNC, Significant Non-Compliance, or compliance status. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_pollutant_from_answer() -> str:
    return """
From the answer, extract the specific chemical pollutant for which the facility is in violation.

Return:
- pollutant_name: the name of the chemical pollutant (e.g., benzene, lead, particulate matter, PM2.5, etc.) exactly as stated. If not present, set null.

If the field is missing, set it to null.
"""


def prompt_extract_patents_from_answer() -> str:
    return """
From the answer, extract information about the two U.S. patents found in the USPTO database.

Return:
- patent_1_title: the title of the first patent exactly as stated. If not present, set null.
- patent_1_year: the grant year of the first patent. If not present, set null.
- patent_2_title: the title of the second patent exactly as stated. If not present, set null.
- patent_2_year: the grant year of the second patent. If not present, set null.

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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(20\d{2})', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_year_after_2021(text: Optional[str]) -> bool:
    year = extract_year(text)
    if year is None:
        return False
    return year > 2021


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
    facility_info = await evaluator.extract(
        prompt=prompt_extract_facility_from_answer(),
        template_class=FacilityInfo,
        extraction_name="facility_info"
    )

    pollutant_info = await evaluator.extract(
        prompt=prompt_extract_pollutant_from_answer(),
        template_class=PollutantInfo,
        extraction_name="pollutant_info"
    )

    patent_info = await evaluator.extract(
        prompt=prompt_extract_patents_from_answer(),
        template_class=PatentInfo,
        extraction_name="patent_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 EPA ECHO section
    epa_echo_node = evaluator.add_sequential(
        id="epa_echo_section",
        desc="EPA ECHO facility search and compliance review for Harris County, Texas",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F1:A2 - Geographic location input (Harris County, Texas)
    location_input_ok = (
        has_any_ci(answer, ['harris county', 'harris', 'texas']) and
        has_any_ci(answer, ['epa echo', 'echo', 'epa'])
    )
    evaluator.add_custom_node(
        result=bool(location_input_ok),
        id="epa_location_input",
        desc="[Action Node] epa.gov:F1:A2 - Input geographic location (Harris County, Texas) in EPA ECHO search",
        parent=epa_echo_node,
        critical=False
    )

    # [Action Node] epa.gov:F1:A5 - Compliance status filter (SNC)
    snc_filter_ok = has_any_ci(answer, ['snc', 'significant non-compliance', 'significant noncompliance', 'non-compliance'])
    evaluator.add_custom_node(
        result=bool(snc_filter_ok),
        id="epa_snc_filter",
        desc="[Action Node] epa.gov:F1:A5 - Apply Significant Non-Compliance (SNC) status filter",
        parent=epa_echo_node,
        critical=False
    )

    # Check facility name is present
    facility_name_ok = bool(facility_info and facility_info.facility_name and facility_info.facility_name.strip())
    evaluator.add_custom_node(
        result=bool(facility_name_ok),
        id="epa_facility_identified",
        desc="Facility name identified from the list",
        parent=epa_echo_node,
        critical=False
    )

    # [Perception Node] epa.gov:F1:P16 - Extract pollutant from detailed report
    pollutant_name_ok = bool(pollutant_info and pollutant_info.pollutant_name and pollutant_info.pollutant_name.strip())
    detail_page_ok = has_any_ci(answer, ['detail', 'report', 'violation', 'facility'])

    evaluator.add_custom_node(
        result=bool(pollutant_name_ok and detail_page_ok),
        id="epa_pollutant_extraction",
        desc="[Perception Node] epa.gov:F1:P16 - Extract specific chemical pollutant name from facility detailed report",
        parent=epa_echo_node,
        critical=False
    )

    # Optional: mentions list-based results (non-prefixed)
    list_based_ok = has_any_ci(answer, ['list', 'table', 'results', 'ranked', 'top of the list'])
    evaluator.add_custom_node(
        result=bool(list_based_ok),
        id="epa_list_based_mention",
        desc="Mentions using list-based or table results (not relying solely on map view)",
        parent=epa_echo_node,
        critical=False
    )

    # 3.2 USPTO patent search section
    uspto_node = evaluator.add_sequential(
        id="uspto_section",
        desc="USPTO patent database search for pollutant treatment technologies",
        parent=root,
        critical=False
    )

    # [Action Node] uspto.gov:F1:A16 - Database selection (USPAT for granted patents)
    uspto_access_ok = has_any_ci(answer, ['uspto', 'patent', 'u.s. patent', 'us patent'])
    granted_ok = has_any_ci(answer, ['grant', 'issued', 'uspat']) or not has_any_ci(answer, ['application', 'pending'])

    evaluator.add_custom_node(
        result=bool(uspto_access_ok and granted_ok),
        id="uspto_database_selection",
        desc="[Action Node] uspto.gov:F1:A16 - Access USPTO and select granted U.S. patents database (USPAT)",
        parent=uspto_node,
        critical=False
    )

    # [Action Node] uspto.gov:F1:A2 - Date range filter (after 2021)
    date_filter_ok = has_any_ci(answer, ['2021', 'after 2021', '2022', '2023', '2024', '2025'])
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="uspto_date_filter",
        desc="[Action Node] uspto.gov:F1:A2 - Apply date range filter for patents granted after 2021",
        parent=uspto_node,
        critical=False
    )

    # Check pollutant name used in search
    if pollutant_info and pollutant_info.pollutant_name:
        pollutant_in_search = ci_contains(answer, pollutant_info.pollutant_name.strip())
    else:
        pollutant_in_search = False

    evaluator.add_custom_node(
        result=bool(pollutant_in_search),
        id="uspto_pollutant_keyword",
        desc="Pollutant name from EPA ECHO used as search keyword in USPTO",
        parent=uspto_node,
        critical=False
    )

    # Check two patents found
    patent_1_ok = bool(patent_info and patent_info.patent_1_title and patent_info.patent_1_title.strip())
    patent_2_ok = bool(patent_info and patent_info.patent_2_title and patent_info.patent_2_title.strip())

    evaluator.add_custom_node(
        result=bool(patent_1_ok),
        id="uspto_patent_1_found",
        desc="First patent title identified",
        parent=uspto_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(patent_2_ok),
        id="uspto_patent_2_found",
        desc="Second patent title identified",
        parent=uspto_node,
        critical=False
    )

    # Check patent years are after 2021
    patent_1_year_ok = is_year_after_2021(patent_info.patent_1_year) if patent_info else False
    patent_2_year_ok = is_year_after_2021(patent_info.patent_2_year) if patent_info else False

    evaluator.add_custom_node(
        result=bool(patent_1_year_ok),
        id="uspto_patent_1_year_valid",
        desc="First patent granted after 2021",
        parent=uspto_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(patent_2_year_ok),
        id="uspto_patent_2_year_valid",
        desc="Second patent granted after 2021",
        parent=uspto_node,
        critical=False
    )

    # Check treatment/removal focus
    treatment_keywords = ['treat', 'treating', 'treatment', 'remov', 'removing', 'removal', 'abat', 'control', 'reduc', 'filter', 'captur', 'scrub']
    treatment_focus_ok = has_any_ci(answer, treatment_keywords)

    evaluator.add_custom_node(
        result=bool(treatment_focus_ok),
        id="uspto_treatment_focus",
        desc="Patents focus on treating or removing the pollutant",
        parent=uspto_node,
        critical=False
    )

    # 3.3 Information flow validation
    flow_node = evaluator.add_parallel(
        id="information_flow_section",
        desc="Validate information flow from EPA ECHO to USPTO",
        parent=root,
        critical=False
    )

    # Check complete workflow
    complete_workflow = (
        facility_name_ok and
        pollutant_name_ok and
        patent_1_ok and
        patent_2_ok and
        pollutant_in_search
    )

    evaluator.add_custom_node(
        result=bool(complete_workflow),
        id="complete_workflow",
        desc="Complete workflow: facility → pollutant → patents with proper information flow",
        parent=flow_node,
        critical=False
    )

    # Check final report contains all required elements
    final_report_ok = (
        facility_name_ok and
        pollutant_name_ok and
        patent_1_ok and
        patent_2_ok
    )

    evaluator.add_custom_node(
        result=bool(final_report_ok),
        id="final_report_complete",
        desc="Final report includes facility name, pollutant, and two patent titles",
        parent=flow_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
