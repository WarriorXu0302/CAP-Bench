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
TASK_ID = "task-4aed73"
TASK_DESCRIPTION = 'I’m planning a backpacking trip to Thailand, but I take Sertraline and Omeprazole daily and I’m worried about drug interactions.  \nPlease first check the CDC website to identify which vaccines are listed for travel to Thailand as **“Routine”** and which are **“Recommended.”**  \n\nPlease pay special attention to the **malaria prevention** guidance and note down the names of all antimalarial medications mentioned on the page.  \n\nThen, go to the **Drugs.com Interaction Checker** tool and enter my two daily medications together with all the CDC-mentioned antimalarial drugs to evaluate whether there are any serious interactions.  \n\nFinally, tell me which antimalarial option has the **lowest side-effect risk** for me (i.e., the fewest interactions or the mildest interaction severity). If there is a tie, list all tied options and explain the interaction severity basis for each.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CDCVaccineInfo(BaseModel):
    """Vaccine information extracted from the answer for Thailand travel"""
    routine_vaccines: Optional[List[str]] = Field(default_factory=list)
    recommended_vaccines: Optional[List[str]] = Field(default_factory=list)


class CDCMalariaInfo(BaseModel):
    """Malaria prevention medication information extracted from the answer"""
    antimalarial_medications: Optional[List[str]] = Field(default_factory=list)


class DrugInteractionInfo(BaseModel):
    """Drug interaction check results extracted from the answer"""
    sertraline_mentioned: Optional[bool] = None
    omeprazole_mentioned: Optional[bool] = None
    antimalarials_checked: Optional[List[str]] = Field(default_factory=list)
    interaction_results: Optional[str] = None


class RecommendationInfo(BaseModel):
    """Antimalarial recommendation extracted from the answer"""
    recommended_antimalarial: Optional[str] = None
    rationale: Optional[str] = None
    severity_explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cdc_vaccines() -> str:
    return """
Extract the CDC vaccine information for travel to Thailand from the answer.

Return:
- routine_vaccines: a list of vaccine names classified as "Routine" for Thailand travel. If none mentioned, return empty list.
- recommended_vaccines: a list of vaccine names classified as "Recommended" for Thailand travel. If none mentioned, return empty list.

If the answer does not contain this information, return empty lists.
"""


def prompt_extract_cdc_malaria() -> str:
    return """
Extract the antimalarial medications mentioned on the CDC page for Thailand from the answer.

Return:
- antimalarial_medications: a list of all antimalarial drug names mentioned. If none found, return empty list.

If the answer does not contain this information, return an empty list.
"""


def prompt_extract_drug_interactions() -> str:
    return """
Extract the drug interaction checking information from the answer.

Return:
- sertraline_mentioned: true if Sertraline was mentioned as being checked, false otherwise
- omeprazole_mentioned: true if Omeprazole was mentioned as being checked, false otherwise
- antimalarials_checked: list of antimalarial drug names that were entered into the interaction checker
- interaction_results: a summary of the interaction results found (any text describing the severity levels or findings)

If any field is missing, set appropriate defaults (false for booleans, empty list for lists, null for text).
"""


def prompt_extract_recommendation() -> str:
    return """
Extract the antimalarial recommendation from the answer.

Return:
- recommended_antimalarial: the name(s) of the antimalarial medication(s) recommended as having the lowest side-effect risk
- rationale: the explanation for why this option was chosen
- severity_explanation: details about interaction severity levels for the recommended option(s)

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


def mentions_thailand(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'thailand')


def mentions_cdc(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['cdc', 'centers for disease control'])


def mentions_drugs_com(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['drugs.com', 'drugs com', 'drug interaction checker'])


def mentions_vaccine_categories(text: Optional[str]) -> bool:
    if not text:
        return False
    routine = has_any_ci(text, ['routine'])
    recommended = has_any_ci(text, ['recommended'])
    return routine and recommended


def mentions_malaria(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['malaria', 'antimalarial'])


def mentions_severity_levels(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['major', 'moderate', 'minor', 'severity', 'serious', 'interaction'])


def has_antimalarial_drugs(drugs: Optional[List[str]]) -> bool:
    if not drugs:
        return False
    # Common antimalarials that might appear
    common_antimalarials = ['mefloquine', 'doxycycline', 'atovaquone', 'proguanil', 'chloroquine', 'primaquine', 'malarone']
    drugs_lower = [d.lower() for d in drugs]
    return any(any(am in drug for am in common_antimalarials) for drug in drugs_lower)


def count_non_empty_items(items: Optional[List[str]]) -> int:
    if not items:
        return 0
    return len([item for item in items if item and item.strip()])


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
    vaccine_info = await evaluator.extract(
        prompt=prompt_extract_cdc_vaccines(),
        template_class=CDCVaccineInfo,
        extraction_name="cdc_vaccine_info"
    )

    malaria_info = await evaluator.extract(
        prompt=prompt_extract_cdc_malaria(),
        template_class=CDCMalariaInfo,
        extraction_name="cdc_malaria_info"
    )

    interaction_info = await evaluator.extract(
        prompt=prompt_extract_drug_interactions(),
        template_class=DrugInteractionInfo,
        extraction_name="drug_interaction_info"
    )

    recommendation_info = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=RecommendationInfo,
        extraction_name="recommendation_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 CDC vaccine and malaria information section
    cdc_section = evaluator.add_sequential(
        id="cdc_section",
        desc="CDC website information for Thailand travel - vaccines and malaria prevention",
        parent=root,
        critical=False
    )

    # [Action Node] cdc.gov:F3:A2 - Navigate to CDC and select Thailand as destination
    cdc_navigation_ok = mentions_cdc(answer) and mentions_thailand(answer)
    evaluator.add_custom_node(
        result=bool(cdc_navigation_ok),
        id="cdc_action_select_thailand",
        desc="[Action Node] cdc.gov:F3:A2 - Navigate to CDC website and select Thailand as travel destination",
        parent=cdc_section,
        critical=False
    )

    # [Perception Node] cdc.gov:F3:P1 - Identify and distinguish Routine vs Recommended vaccines
    has_routine = count_non_empty_items(vaccine_info.routine_vaccines) > 0
    has_recommended = count_non_empty_items(vaccine_info.recommended_vaccines) > 0
    mentions_categories = mentions_vaccine_categories(answer)
    vaccine_classification_ok = has_routine and has_recommended and mentions_categories

    evaluator.add_custom_node(
        result=bool(vaccine_classification_ok),
        id="cdc_perception_vaccine_categories",
        desc="[Perception Node] cdc.gov:F3:P1 - Identify and distinguish between 'Routine' and 'Recommended' vaccine categories",
        parent=cdc_section,
        critical=False
    )

    # Additional check: extracted malaria medications
    has_antimalarials = (malaria_info and
                        malaria_info.antimalarial_medications and
                        len(malaria_info.antimalarial_medications) > 0)
    mentions_malaria_context = mentions_malaria(answer)

    evaluator.add_custom_node(
        result=bool(has_antimalarials and mentions_malaria_context),
        id="cdc_malaria_medications_extracted",
        desc="Extracted antimalarial medication names from CDC malaria prevention guidance",
        parent=cdc_section,
        critical=False
    )

    # 3.2 Drugs.com interaction checker section
    drugscom_section = evaluator.add_sequential(
        id="drugscom_section",
        desc="Drugs.com interaction checker for daily medications and antimalarials",
        parent=root,
        critical=False
    )

    # [Action Node] drugs.com:F3:A5 - Enter multiple medications into interaction checker
    mentions_drugscom = mentions_drugs_com(answer)
    sertraline_entered = interaction_info and interaction_info.sertraline_mentioned
    omeprazole_entered = interaction_info and interaction_info.omeprazole_mentioned
    antimalarials_entered = (interaction_info and
                            interaction_info.antimalarials_checked and
                            len(interaction_info.antimalarials_checked) > 0)

    multiple_drug_entry_ok = (mentions_drugscom and
                             sertraline_entered and
                             omeprazole_entered and
                             antimalarials_entered)

    evaluator.add_custom_node(
        result=bool(multiple_drug_entry_ok),
        id="drugscom_action_multiple_inputs",
        desc="[Action Node] drugs.com:F3:A5 - Enter multiple medications (Sertraline, Omeprazole, and antimalarials) into interaction checker",
        parent=drugscom_section,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P10 - Identify and compare interaction severity levels
    has_interaction_results = interaction_info and interaction_info.interaction_results
    mentions_severity = mentions_severity_levels(answer)
    severity_recognition_ok = has_interaction_results and mentions_severity

    evaluator.add_custom_node(
        result=bool(severity_recognition_ok),
        id="drugscom_perception_severity_levels",
        desc="[Perception Node] drugs.com:F3:P10 - Identify and compare interaction severity levels (Major/Moderate/Minor)",
        parent=drugscom_section,
        critical=False
    )

    # 3.3 Final recommendation section
    recommendation_section = evaluator.add_sequential(
        id="recommendation_section",
        desc="Final antimalarial recommendation based on interaction analysis",
        parent=root,
        critical=False
    )

    # Check if a clear recommendation was made
    has_recommendation = (recommendation_info and
                         recommendation_info.recommended_antimalarial and
                         recommendation_info.recommended_antimalarial.strip())
    has_rationale = (recommendation_info and
                    recommendation_info.rationale and
                    recommendation_info.rationale.strip())

    evaluator.add_custom_node(
        result=bool(has_recommendation),
        id="recommendation_provided",
        desc="Provided a specific antimalarial recommendation with lowest side-effect risk",
        parent=recommendation_section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_rationale),
        id="recommendation_rationale",
        desc="Explained the rationale for the recommendation based on interaction severity",
        parent=recommendation_section,
        critical=False
    )

    # Check if severity explanation was provided
    has_severity_explanation = (recommendation_info and
                               recommendation_info.severity_explanation and
                               recommendation_info.severity_explanation.strip())

    evaluator.add_custom_node(
        result=bool(has_severity_explanation),
        id="severity_explanation_provided",
        desc="Explained interaction severity levels for the recommended option(s)",
        parent=recommendation_section,
        critical=False
    )

    # Overall completeness check
    addresses_tie_scenario = has_any_ci(answer, ['tie', 'tied', 'equal', 'same'])
    evaluator.add_custom_node(
        result=bool(has_recommendation and has_rationale),
        id="complete_recommendation",
        desc="Provided complete recommendation with clear lowest-risk option and justification",
        parent=recommendation_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
