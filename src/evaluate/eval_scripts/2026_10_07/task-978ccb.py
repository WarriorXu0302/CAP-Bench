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
TASK_ID = "task-978ccb"
TASK_DESCRIPTION = 'I have a 65-year-old patient who is currently taking 7 prescription medications and 3 supplements simultaneously. The specific list includes: Prescription medications: Aspirin, Warfarin, Simvastatin, Amlodipine, Metformin, Lisinopril, Omeprazole; Supplements: St. John\'s Wort, Fish Oil, CoQ10. I need to conduct a comprehensive safety assessment of this medication and supplement regimen.\n\nFirst, go to Drugs.com to batch-check the interactions of these 10 medications/supplements. Record all Major and Moderate level interaction combinations along with their specific risk descriptions.\n\nNext, go to WebMD to individually verify the top 5 high-risk combinations (prioritizing by severity). Confirm if the risk level and clinical significance are consistent.\n\nThen, search PubMed for clinical evidence regarding these high-risk combinations. Use generic drug names in combination for the search (e.g., "Warfarin AND St. John\'s Wort AND interaction"). Filter for clinical trials or case reports published within the last 5 years, finding at least 3 most relevant articles.\n\nFinally, consult Mayo Clinic\'s medication guidelines for elderly patients, specifically focusing on special considerations and dosage adjustment recommendations for Warfarin and Simvastatin in the geriatric population.\n\n**Output:**\n(1) A list of all Major and Moderate level interactions detected by Drugs.com, including the drug combination, severity label, risk description, and a link to the Drugs.com interaction report page.\n(2) The top 5 high-risk combinations verified by WebMD, including the drug combination, the risk level displayed by WebMD, whether it is consistent with Drugs.com, and a link to the WebMD interaction checker result page.\n(3) Three clinical evidence articles found on PubMed, including title, author(s), publication year, journal name, PMID, link to the article details page, and key findings from the abstract.\n(4) Mayo Clinic\'s medication recommendations for elderly patients, including specific precautions and dosage adjustment suggestions for Warfarin and Simvastatin respectively in the geriatric population, and a link to the corresponding medication page on Mayo Clinic.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugsComInteractions(BaseModel):
    """Interaction information extracted from Drugs.com batch check"""
    major_interactions: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    moderate_interactions: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    drugscom_link: Optional[str] = None


class WebMDVerification(BaseModel):
    """WebMD verification results for top 5 high-risk combinations"""
    verified_combinations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class PubMedEvidence(BaseModel):
    """PubMed clinical evidence articles"""
    articles: Optional[List[Dict[str, str]]] = Field(default_factory=list)


class MayoClinicGuidelines(BaseModel):
    """Mayo Clinic guidelines for elderly patients"""
    warfarin_precautions: Optional[str] = None
    warfarin_dosage: Optional[str] = None
    simvastatin_precautions: Optional[str] = None
    simvastatin_dosage: Optional[str] = None
    warfarin_link: Optional[str] = None
    simvastatin_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_drugscom_interactions() -> str:
    return """
Extract the Drugs.com interaction check results from the answer.

Return:
- major_interactions: list of dicts with keys 'combination', 'severity', 'description' for each Major interaction
- moderate_interactions: list of dicts with keys 'combination', 'severity', 'description' for each Moderate interaction
- drugscom_link: the link to the Drugs.com interaction report page

If any field is missing, set it to null or empty list.
"""


def prompt_extract_webmd_verification() -> str:
    return """
Extract the WebMD verification results for the top 5 high-risk drug combinations from the answer.

Return:
- verified_combinations: list of dicts with keys 'combination', 'webmd_risk_level', 'consistency_with_drugscom', 'webmd_link' for each of the 5 combinations

If any field is missing, set it to null or empty list.
"""


def prompt_extract_pubmed_evidence() -> str:
    return """
Extract the PubMed clinical evidence articles from the answer.

Return:
- articles: list of dicts with keys 'title', 'authors', 'year', 'journal', 'pmid', 'link', 'key_findings' for each article (should be at least 3)

If any field is missing, set it to null or empty list.
"""


def prompt_extract_mayo_guidelines() -> str:
    return """
Extract the Mayo Clinic medication guidelines for elderly patients from the answer.

Return:
- warfarin_precautions: special precautions for Warfarin in elderly patients
- warfarin_dosage: dosage adjustment recommendations for Warfarin in elderly patients
- simvastatin_precautions: special precautions for Simvastatin in elderly patients
- simvastatin_dosage: dosage adjustment recommendations for Simvastatin in elderly patients
- warfarin_link: link to the Warfarin page on Mayo Clinic
- simvastatin_link: link to the Simvastatin page on Mayo Clinic

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


def count_medications_mentioned(text: Optional[str]) -> int:
    if not text:
        return 0
    medications = ['aspirin', 'warfarin', 'simvastatin', 'amlodipine', 'metformin',
                   'lisinopril', 'omeprazole', 'st. john\'s wort', 'st john\'s wort',
                   'fish oil', 'coq10']
    count = 0
    for med in medications:
        if ci_contains(text, med):
            count += 1
    return count


def has_severity_labels(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['major', 'moderate'])


def looks_like_year_range_2020_2025(text: Optional[str]) -> bool:
    if not text:
        return False
    years = re.findall(r'\b(202[0-5])\b', text)
    return len(years) > 0


def has_pmid(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'PMID|pmid|\bPM\d+', text, re.IGNORECASE))


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
    drugscom_info = await evaluator.extract(
        prompt=prompt_extract_drugscom_interactions(),
        template_class=DrugsComInteractions,
        extraction_name="drugscom_interactions"
    )

    webmd_info = await evaluator.extract(
        prompt=prompt_extract_webmd_verification(),
        template_class=WebMDVerification,
        extraction_name="webmd_verification"
    )

    pubmed_info = await evaluator.extract(
        prompt=prompt_extract_pubmed_evidence(),
        template_class=PubMedEvidence,
        extraction_name="pubmed_evidence"
    )

    mayo_info = await evaluator.extract(
        prompt=prompt_extract_mayo_guidelines(),
        template_class=MayoClinicGuidelines,
        extraction_name="mayo_guidelines"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Drugs.com section
    drugscom_node = evaluator.add_sequential(
        id="drugscom_section",
        desc="Drugs.com batch interaction check for 10 medications/supplements",
        parent=root,
        critical=False
    )

    # [Action Node] drugs.com:F3:A5 - Batch input drug names
    med_count = count_medications_mentioned(answer)
    batch_input_ok = med_count >= 8 and ci_contains(answer, 'drugs.com')
    evaluator.add_custom_node(
        result=bool(batch_input_ok),
        id="drugscom_action_batch_input",
        desc="[Action Node] drugs.com:F3:A5 - Batch input the 10 medication/supplement names into Drugs.com",
        parent=drugscom_node,
        critical=False
    )

    # [Action Node] drugs.com:F3:A6 - Add drugs to check list
    add_button_ok = med_count >= 8 and (has_any_ci(answer, ['add', 'added']) or ci_contains(answer, 'check'))
    evaluator.add_custom_node(
        result=bool(add_button_ok),
        id="drugscom_action_add_drugs",
        desc="[Action Node] drugs.com:F3:A6 - Add each drug to the interaction check list",
        parent=drugscom_node,
        critical=False
    )

    # [Action Node] drugs.com:F3:A13 - Execute batch interaction check
    check_executed = ci_contains(answer, 'drugs.com') and has_severity_labels(answer)
    evaluator.add_custom_node(
        result=bool(check_executed),
        id="drugscom_action_check_interactions",
        desc="[Action Node] drugs.com:F3:A13 - Execute the batch interaction check (click Check Interactions)",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P10 - Identify severity labels
    major_found = len(drugscom_info.major_interactions) > 0 if drugscom_info.major_interactions else False
    moderate_found = len(drugscom_info.moderate_interactions) > 0 if drugscom_info.moderate_interactions else False
    severity_labels_ok = major_found or moderate_found
    evaluator.add_custom_node(
        result=bool(severity_labels_ok),
        id="drugscom_perception_severity_labels",
        desc="[Perception Node] drugs.com:F3:P10 - Identify and distinguish Major and Moderate severity labels",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P4 - Understand detailed risk descriptions
    has_descriptions = False
    if drugscom_info.major_interactions:
        has_descriptions = any(d.get('description') for d in drugscom_info.major_interactions)
    if not has_descriptions and drugscom_info.moderate_interactions:
        has_descriptions = any(d.get('description') for d in drugscom_info.moderate_interactions)
    evaluator.add_custom_node(
        result=bool(has_descriptions),
        id="drugscom_perception_risk_descriptions",
        desc="[Perception Node] drugs.com:F3:P4 - Understand and extract detailed risk descriptions for interactions",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P20 - Identify interaction count statistics
    total_interactions = (len(drugscom_info.major_interactions) if drugscom_info.major_interactions else 0) + \
                        (len(drugscom_info.moderate_interactions) if drugscom_info.moderate_interactions else 0)
    count_identified = total_interactions > 0
    evaluator.add_custom_node(
        result=bool(count_identified),
        id="drugscom_perception_interaction_count",
        desc="[Perception Node] drugs.com:F3:P20 - Identify the count of Major and Moderate interactions",
        parent=drugscom_node,
        critical=False
    )

    # Drugscom link provided
    drugscom_link_ok = bool(drugscom_info.drugscom_link and drugscom_info.drugscom_link.strip())
    evaluator.add_custom_node(
        result=bool(drugscom_link_ok),
        id="drugscom_link_provided",
        desc="Provides link to Drugs.com interaction report page",
        parent=drugscom_node,
        critical=False
    )

    # 3.2 WebMD section
    webmd_node = evaluator.add_sequential(
        id="webmd_section",
        desc="WebMD verification of top 5 high-risk drug combinations",
        parent=root,
        critical=False
    )

    # [Action Node] webmd.com:F4:A4 - Input drug combinations for checking
    webmd_mentioned = ci_contains(answer, 'webmd')
    top_5_ok = webmd_info.verified_combinations and len(webmd_info.verified_combinations) >= 5
    evaluator.add_custom_node(
        result=bool(webmd_mentioned and top_5_ok),
        id="webmd_action_input_combinations",
        desc="[Action Node] webmd.com:F4:A4 - Input the top 5 high-risk drug combinations into WebMD for verification",
        parent=webmd_node,
        critical=False
    )

    # [Perception Node] webmd.com:F4:P4 - Understand risk levels and descriptions
    has_risk_levels = False
    if webmd_info.verified_combinations:
        has_risk_levels = any(c.get('webmd_risk_level') for c in webmd_info.verified_combinations)
    evaluator.add_custom_node(
        result=bool(has_risk_levels),
        id="webmd_perception_risk_levels",
        desc="[Perception Node] webmd.com:F4:P4 - Understand WebMD risk levels and clinical descriptions for each combination",
        parent=webmd_node,
        critical=False
    )

    # [Perception Node] webmd.com:F4:P5 - Identify severity labels and compare with Drugs.com
    has_consistency_check = False
    if webmd_info.verified_combinations:
        has_consistency_check = any(c.get('consistency_with_drugscom') is not None for c in webmd_info.verified_combinations)
    evaluator.add_custom_node(
        result=bool(has_consistency_check),
        id="webmd_perception_consistency",
        desc="[Perception Node] webmd.com:F4:P5 - Compare WebMD severity labels with Drugs.com results for consistency",
        parent=webmd_node,
        critical=False
    )

    # WebMD links provided
    webmd_links_ok = False
    if webmd_info.verified_combinations:
        webmd_links_ok = any(c.get('webmd_link') for c in webmd_info.verified_combinations)
    evaluator.add_custom_node(
        result=bool(webmd_links_ok),
        id="webmd_links_provided",
        desc="Provides links to WebMD interaction checker result pages",
        parent=webmd_node,
        critical=False
    )

    # 3.3 PubMed section
    pubmed_node = evaluator.add_sequential(
        id="pubmed_section",
        desc="PubMed clinical evidence search for high-risk drug combinations",
        parent=root,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Search with drug combination queries
    pubmed_mentioned = ci_contains(answer, 'pubmed')
    has_search_terms = has_any_ci(answer, ['and', 'interaction', 'warfarin', 'st. john'])
    evaluator.add_custom_node(
        result=bool(pubmed_mentioned and has_search_terms),
        id="pubmed_action_search",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Search PubMed using drug combination queries with AND operators",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Set date range filter (last 5 years)
    date_filter_ok = looks_like_year_range_2020_2025(answer) or has_any_ci(answer, ['last 5 years', 'past 5 years', '5 years'])
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="pubmed_action_date_filter",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Apply date range filter for publications within the last 5 years",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Filter by article type
    article_type_filter = has_any_ci(answer, ['clinical trial', 'case report'])
    evaluator.add_custom_node(
        result=bool(article_type_filter),
        id="pubmed_action_article_type_filter",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Filter by article type (Clinical Trial or Case Report)",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Understand literature list information
    articles_found = pubmed_info.articles and len(pubmed_info.articles) >= 3
    has_basic_info = False
    if pubmed_info.articles:
        has_basic_info = any(a.get('title') and a.get('authors') for a in pubmed_info.articles)
    evaluator.add_custom_node(
        result=bool(articles_found and has_basic_info),
        id="pubmed_perception_article_info",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Extract and understand basic information from at least 3 articles",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F3:A9 - Click to article detail pages
    has_links = False
    if pubmed_info.articles:
        has_links = any(a.get('link') for a in pubmed_info.articles)
    evaluator.add_custom_node(
        result=bool(has_links),
        id="pubmed_action_detail_pages",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F3:A9 - Access article detail pages to retrieve full information",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F3:P4 - Extract key findings from abstracts
    has_key_findings = False
    if pubmed_info.articles:
        has_key_findings = any(a.get('key_findings') for a in pubmed_info.articles)
    evaluator.add_custom_node(
        result=bool(has_key_findings),
        id="pubmed_perception_key_findings",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F3:P4 - Extract key findings about drug interactions from article abstracts",
        parent=pubmed_node,
        critical=False
    )

    # PMID provided
    pmid_ok = has_pmid(answer)
    evaluator.add_custom_node(
        result=bool(pmid_ok),
        id="pubmed_pmid_provided",
        desc="Provides PMID for the articles",
        parent=pubmed_node,
        critical=False
    )

    # 3.4 Mayo Clinic section
    mayo_node = evaluator.add_sequential(
        id="mayo_section",
        desc="Mayo Clinic medication guidelines for elderly patients (Warfarin and Simvastatin)",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F3:A2 - Navigate to drug information page
    mayo_mentioned = ci_contains(answer, 'mayo clinic')
    drugs_section = has_any_ci(answer, ['drug', 'medication', 'supplement'])
    evaluator.add_custom_node(
        result=bool(mayo_mentioned and drugs_section),
        id="mayo_action_navigate_drugs",
        desc="[Action Node] mayoclinic.org:F3:A2 - Navigate to Mayo Clinic Drugs & Supplements section",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F3:A5 - Search for specific drugs
    warfarin_mentioned = ci_contains(answer, 'warfarin')
    simvastatin_mentioned = ci_contains(answer, 'simvastatin')
    evaluator.add_custom_node(
        result=bool(mayo_mentioned and warfarin_mentioned and simvastatin_mentioned),
        id="mayo_action_find_drugs",
        desc="[Action Node] mayoclinic.org:F3:A5 - Locate Warfarin and Simvastatin using alphabetical index or search",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F3:A6 - Scroll through long drug pages
    elderly_mentioned = has_any_ci(answer, ['elderly', 'geriatric', 'older', '65'])
    precautions_mentioned = has_any_ci(answer, ['precaution', 'dosage', 'dosing', 'adjustment'])
    evaluator.add_custom_node(
        result=bool(elderly_mentioned and precautions_mentioned),
        id="mayo_action_scroll_sections",
        desc="[Action Node] mayoclinic.org:F3:A6 - Scroll through drug pages to find Precautions and Dosing sections",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F3:P5 - Understand dosage information
    warfarin_dosage_ok = bool(mayo_info.warfarin_dosage and mayo_info.warfarin_dosage.strip())
    simvastatin_dosage_ok = bool(mayo_info.simvastatin_dosage and mayo_info.simvastatin_dosage.strip())
    evaluator.add_custom_node(
        result=bool(warfarin_dosage_ok and simvastatin_dosage_ok),
        id="mayo_perception_dosage",
        desc="[Perception Node] mayoclinic.org:F3:P5 - Extract dosage adjustment recommendations for elderly patients",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F3:P6 - Understand precautions and side effects
    warfarin_precautions_ok = bool(mayo_info.warfarin_precautions and mayo_info.warfarin_precautions.strip())
    simvastatin_precautions_ok = bool(mayo_info.simvastatin_precautions and mayo_info.simvastatin_precautions.strip())
    evaluator.add_custom_node(
        result=bool(warfarin_precautions_ok and simvastatin_precautions_ok),
        id="mayo_perception_precautions",
        desc="[Perception Node] mayoclinic.org:F3:P6 - Extract special precautions for elderly patients",
        parent=mayo_node,
        critical=False
    )

    # Mayo Clinic links provided
    mayo_links_ok = bool(mayo_info.warfarin_link and mayo_info.simvastatin_link)
    evaluator.add_custom_node(
        result=bool(mayo_links_ok),
        id="mayo_links_provided",
        desc="Provides links to Mayo Clinic medication pages for Warfarin and Simvastatin",
        parent=mayo_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
