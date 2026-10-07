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
TASK_ID = "task-933495"
TASK_DESCRIPTION = 'I’m researching the medicinal value of **Ginkgo** for improving cognitive function and want to conduct a complete literature review from botany to clinical application.  \n\nFirst, go to the **Kew Plants database** to confirm Ginkgo’s accepted scientific name, taxonomic status, and synonyms (to avoid confusion with related species), and record the scientific name, family/genus, and native distribution.  \n\nThen, use this scientific name to search **PubMed** for pharmacological studies on cognitive function or neuroprotection. Filter for **systematic reviews or meta-analyses** published since **2020** with **free full text**, and try to find **3 highly cited papers**. Record each paper’s title, authors, PMID, and main conclusions. If fewer than 3 are available, supplement with review articles on the same topic and indicate whether each is a systematic review/meta-analysis.  \n\nNext, go to **Mayo Clinic** to check clinical application information for Ginkgo as an herbal supplement, including the level of evidence for cognitive improvement, as well as side effects and medication warnings.  \n\nFinally, use **Drugs.com** to check interactions between Ginkgo and common prescription drugs (such as warfarin and aspirin), and record interactions rated **Major** or **Moderate**, along with their explanations.  \n\n**Output required:**  \n- Ginkgo scientific name, family/genus, Kew page link, and native distribution.  \n- For 3 PubMed papers (or the actual number available): title, authors, PMID, publication year, main conclusions, PubMed link, and whether each is free full text and whether it is a systematic review/meta-analysis.  \n- Mayo Clinic’s evidence-level assessment for Ginkgo in cognitive improvement, side-effect list, and Mayo page link.  \n- From Drugs.com interaction results: drug names with Major/Moderate interactions, severity level, interaction description, and interaction page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class KewInformation(BaseModel):
    """Botanical information from Kew Plants database"""
    scientific_name: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    native_distribution: Optional[str] = None
    kew_page_link: Optional[str] = None


class PubMedPaper(BaseModel):
    """Single PubMed paper details"""
    title: Optional[str] = None
    authors: Optional[str] = None
    pmid: Optional[str] = None
    publication_year: Optional[str] = None
    main_conclusions: Optional[str] = None
    pubmed_link: Optional[str] = None
    is_free_full_text: Optional[bool] = None
    is_systematic_review_or_meta_analysis: Optional[bool] = None


class PubMedPapers(BaseModel):
    """Collection of PubMed papers"""
    papers: List[PubMedPaper] = Field(default_factory=list)


class MayoClinicInfo(BaseModel):
    """Mayo Clinic clinical information"""
    evidence_level_for_cognitive_improvement: Optional[str] = None
    side_effects_list: Optional[List[str]] = Field(default_factory=list)
    mayo_page_link: Optional[str] = None


class DrugInteraction(BaseModel):
    """Single drug interaction detail"""
    drug_name: Optional[str] = None
    severity_level: Optional[str] = None
    interaction_description: Optional[str] = None


class DrugsComInfo(BaseModel):
    """Drugs.com interaction information"""
    interactions: List[DrugInteraction] = Field(default_factory=list)
    interaction_page_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_kew_info() -> str:
    return """
Extract the Kew Plants database information about Ginkgo from the answer.

Return:
- scientific_name: the accepted scientific name (e.g., "Ginkgo biloba")
- family: the taxonomic family
- genus: the taxonomic genus
- native_distribution: the native geographic distribution
- kew_page_link: the URL to the Kew species page

If any field is missing, set it to null.
"""


def prompt_extract_pubmed_papers() -> str:
    return """
Extract all PubMed papers mentioned in the answer about Ginkgo and cognitive function.

For each paper, extract:
- title: the paper title
- authors: the author names or list
- pmid: the PubMed ID
- publication_year: the year of publication
- main_conclusions: the main findings or conclusions
- pubmed_link: the URL to the PubMed entry
- is_free_full_text: whether free full text is available (true/false)
- is_systematic_review_or_meta_analysis: whether it's a systematic review or meta-analysis (true/false)

Return a list of papers. If no papers are mentioned, return an empty list.
"""


def prompt_extract_mayo_clinic_info() -> str:
    return """
Extract Mayo Clinic information about Ginkgo from the answer.

Return:
- evidence_level_for_cognitive_improvement: the evidence level or assessment for cognitive improvement
- side_effects_list: a list of side effects mentioned
- mayo_page_link: the URL to the Mayo Clinic page

If any field is missing, set it to null or empty list.
"""


def prompt_extract_drugs_com_info() -> str:
    return """
Extract Drugs.com drug interaction information from the answer.

For each interaction with Major or Moderate severity, extract:
- drug_name: the name of the interacting drug
- severity_level: "Major" or "Moderate"
- interaction_description: the explanation of the interaction

Also extract:
- interaction_page_link: the URL to the Drugs.com interaction results page

Return the interactions as a list. If none found, return an empty list.
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


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http' in text.lower() or 'www' in text.lower())


def looks_like_pmid(text: Optional[str]) -> bool:
    if not text:
        return False
    # PMID should be numeric, often 7-8 digits
    return bool(re.search(r'\d{6,9}', str(text)))


def looks_like_year_since_2020(text: Optional[str]) -> bool:
    if not text:
        return False
    match = re.search(r'\b(202[0-9])\b', str(text))
    if match:
        year = int(match.group(1))
        return year >= 2020
    return False


def is_severity_major_or_moderate(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return 'major' in t or 'moderate' in t


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
    kew_info = await evaluator.extract(
        prompt=prompt_extract_kew_info(),
        template_class=KewInformation,
        extraction_name="kew_information"
    )

    pubmed_papers = await evaluator.extract(
        prompt=prompt_extract_pubmed_papers(),
        template_class=PubMedPapers,
        extraction_name="pubmed_papers"
    )

    mayo_info = await evaluator.extract(
        prompt=prompt_extract_mayo_clinic_info(),
        template_class=MayoClinicInfo,
        extraction_name="mayo_clinic_info"
    )

    drugs_info = await evaluator.extract(
        prompt=prompt_extract_drugs_com_info(),
        template_class=DrugsComInfo,
        extraction_name="drugs_com_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Kew Plants database section
    kew_node = evaluator.add_sequential(
        id="kew_section",
        desc="Kew Plants database - Ginkgo botanical information",
        parent=root,
        critical=False
    )

    # [Action Node] powo.science.kew.org:F2:A1 - Search for Ginkgo
    kew_search_ok = has_any_ci(answer, ['kew']) and has_any_ci(answer, ['ginkgo'])
    evaluator.add_custom_node(
        result=bool(kew_search_ok),
        id="kew_action_search",
        desc="[Action Node] powo.science.kew.org:F2:A1 - Search for Ginkgo in Kew Plants database",
        parent=kew_node,
        critical=False
    )

    # [Action Node] powo.science.kew.org:F3:A10 - Click species card
    kew_species_click_ok = (kew_info.kew_page_link is not None and
                            looks_like_url(kew_info.kew_page_link, 'powo.science.kew.org'))
    evaluator.add_custom_node(
        result=bool(kew_species_click_ok),
        id="kew_action_species_click",
        desc="[Action Node] powo.science.kew.org:F3:A10 - Click on the Ginkgo species card to enter details page",
        parent=kew_node,
        critical=False
    )

    # [Perception Node] powo.science.kew.org:F3:P12 - Extract taxonomic information
    taxonomic_info_ok = (kew_info.scientific_name is not None and
                        kew_info.family is not None and
                        kew_info.genus is not None and
                        has_any_ci(kew_info.scientific_name, ['ginkgo']))
    evaluator.add_custom_node(
        result=bool(taxonomic_info_ok),
        id="kew_perception_taxonomic",
        desc="[Perception Node] powo.science.kew.org:F3:P12 - Extract scientific name, family, and genus information",
        parent=kew_node,
        critical=False
    )

    # [Perception Node] powo.science.kew.org:F4:P12 - Extract native distribution
    distribution_ok = kew_info.native_distribution is not None and len(str(kew_info.native_distribution).strip()) > 0
    evaluator.add_custom_node(
        result=bool(distribution_ok),
        id="kew_perception_distribution",
        desc="[Perception Node] powo.science.kew.org:F4:P12 - Extract native geographic distribution information",
        parent=kew_node,
        critical=False
    )

    # 3.2 PubMed section
    pubmed_node = evaluator.add_sequential(
        id="pubmed_section",
        desc="PubMed pharmacological studies on Ginkgo and cognitive function",
        parent=root,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F2:A1 - Advanced search with scientific name
    pubmed_search_ok = (has_any_ci(answer, ['pubmed']) and
                       has_any_ci(answer, ['ginkgo']) and
                       has_any_ci(answer, ['cognitive', 'neuroprotect']))
    evaluator.add_custom_node(
        result=bool(pubmed_search_ok),
        id="pubmed_action_search",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F2:A1 - Search PubMed with Ginkgo scientific name and cognitive function keywords",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Filter by date range (since 2020)
    has_papers_since_2020 = any(looks_like_year_since_2020(p.publication_year) for p in pubmed_papers.papers if p.publication_year)
    evaluator.add_custom_node(
        result=bool(has_papers_since_2020),
        id="pubmed_action_date_filter",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Filter for publications since 2020",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Filter by article type (systematic review/meta-analysis)
    has_systematic_reviews = any(p.is_systematic_review_or_meta_analysis for p in pubmed_papers.papers if p.is_systematic_review_or_meta_analysis is not None)
    evaluator.add_custom_node(
        result=bool(has_systematic_reviews),
        id="pubmed_action_article_type",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Filter for systematic reviews or meta-analyses",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F6:A5 - Filter by free full text availability
    has_free_full_text = any(p.is_free_full_text for p in pubmed_papers.papers if p.is_free_full_text is not None)
    evaluator.add_custom_node(
        result=bool(has_free_full_text),
        id="pubmed_action_free_text",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F6:A5 - Filter for free full text availability",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A6 - Sort by relevance/citation
    papers_sorted_ok = len(pubmed_papers.papers) > 0
    evaluator.add_custom_node(
        result=bool(papers_sorted_ok),
        id="pubmed_action_sort",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A6 - Sort results by relevance or citation count to find highly cited papers",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A9 - Click paper titles to view details
    has_paper_links = any(p.pubmed_link is not None for p in pubmed_papers.papers)
    evaluator.add_custom_node(
        result=bool(has_paper_links),
        id="pubmed_action_paper_click",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A9 - Click on paper titles to access detailed information",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Extract basic paper information
    basic_info_ok = all(
        p.title is not None and p.authors is not None and looks_like_pmid(p.pmid)
        for p in pubmed_papers.papers
    ) if len(pubmed_papers.papers) > 0 else False
    evaluator.add_custom_node(
        result=bool(basic_info_ok),
        id="pubmed_perception_basic_info",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Extract title, authors, and PMID for each paper",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P3 - Extract main conclusions from abstracts
    conclusions_ok = all(
        p.main_conclusions is not None and len(str(p.main_conclusions).strip()) > 10
        for p in pubmed_papers.papers
    ) if len(pubmed_papers.papers) > 0 else False
    evaluator.add_custom_node(
        result=bool(conclusions_ok),
        id="pubmed_perception_conclusions",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P3 - Extract main conclusions from paper abstracts",
        parent=pubmed_node,
        critical=False
    )

    # Additional check: Found approximately 3 papers (lenient: 2-4 papers acceptable)
    paper_count_ok = 2 <= len(pubmed_papers.papers) <= 4
    evaluator.add_custom_node(
        result=bool(paper_count_ok),
        id="pubmed_paper_count",
        desc="Found approximately 3 highly cited papers (2-4 acceptable)",
        parent=pubmed_node,
        critical=False
    )

    # 3.3 Mayo Clinic section
    mayo_node = evaluator.add_sequential(
        id="mayo_section",
        desc="Mayo Clinic clinical application information for Ginkgo",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A2 - Navigate to health information section
    mayo_nav_ok = has_any_ci(answer, ['mayo clinic', 'mayo']) and has_any_ci(answer, ['ginkgo'])
    evaluator.add_custom_node(
        result=bool(mayo_nav_ok),
        id="mayo_action_navigation",
        desc="[Action Node] mayoclinic.org:F2:A2 - Navigate to Mayo Clinic health information section",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A3 - Browse herb/supplement list and find Ginkgo
    mayo_ginkgo_found = (mayo_info.mayo_page_link is not None and
                        looks_like_url(mayo_info.mayo_page_link, 'mayoclinic.org'))
    evaluator.add_custom_node(
        result=bool(mayo_ginkgo_found),
        id="mayo_action_find_ginkgo",
        desc="[Action Node] mayoclinic.org:F2:A3 - Locate and access Ginkgo in the herbal supplements list",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A4 - Switch tabs to view different information sections
    mayo_tabs_ok = (mayo_info.evidence_level_for_cognitive_improvement is not None and
                   len(mayo_info.side_effects_list) > 0)
    evaluator.add_custom_node(
        result=bool(mayo_tabs_ok),
        id="mayo_action_tabs",
        desc="[Action Node] mayoclinic.org:F2:A4 - Switch between tabs to view evidence level, side effects, and warnings",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F2:P2 - Extract side effects and warnings list
    side_effects_ok = len(mayo_info.side_effects_list) >= 2
    evaluator.add_custom_node(
        result=bool(side_effects_ok),
        id="mayo_perception_side_effects",
        desc="[Perception Node] mayoclinic.org:F2:P2 - Extract side effects and medication warnings",
        parent=mayo_node,
        critical=False
    )

    # Additional check: Evidence level for cognitive improvement mentioned
    evidence_level_ok = (mayo_info.evidence_level_for_cognitive_improvement is not None and
                        len(str(mayo_info.evidence_level_for_cognitive_improvement).strip()) > 0)
    evaluator.add_custom_node(
        result=bool(evidence_level_ok),
        id="mayo_evidence_level",
        desc="Evidence level assessment for cognitive improvement is provided",
        parent=mayo_node,
        critical=False
    )

    # 3.4 Drugs.com section
    drugscom_node = evaluator.add_sequential(
        id="drugscom_section",
        desc="Drugs.com drug interaction information for Ginkgo",
        parent=root,
        critical=False
    )

    # [Action Node] drugs.com:F3:A5 - Enter drug names for interaction check
    drugscom_input_ok = (has_any_ci(answer, ['drugs.com', 'drugs com']) and
                        has_any_ci(answer, ['ginkgo']) and
                        (has_any_ci(answer, ['warfarin']) or has_any_ci(answer, ['aspirin'])))
    evaluator.add_custom_node(
        result=bool(drugscom_input_ok),
        id="drugscom_action_input",
        desc="[Action Node] drugs.com:F3:A5 - Enter Ginkgo and common prescription drugs (warfarin, aspirin) for interaction check",
        parent=drugscom_node,
        critical=False
    )

    # [Action Node] drugs.com:F3:A6 - Add drugs to check list
    drugscom_check_ok = (drugs_info.interaction_page_link is not None and
                        looks_like_url(drugs_info.interaction_page_link, 'drugs.com'))
    evaluator.add_custom_node(
        result=bool(drugscom_check_ok),
        id="drugscom_action_add_check",
        desc="[Action Node] drugs.com:F3:A6 - Add drugs to the interaction check list and perform check",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P4 - Understand interaction report details
    interaction_details_ok = all(
        i.drug_name is not None and i.interaction_description is not None
        for i in drugs_info.interactions
    ) if len(drugs_info.interactions) > 0 else False
    evaluator.add_custom_node(
        result=bool(interaction_details_ok),
        id="drugscom_perception_details",
        desc="[Perception Node] drugs.com:F3:P4 - Extract interaction type, severity, and description from the report",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P10 - Identify severity labels (Major/Moderate)
    severity_labels_ok = all(
        is_severity_major_or_moderate(i.severity_level)
        for i in drugs_info.interactions
    ) if len(drugs_info.interactions) > 0 else False
    evaluator.add_custom_node(
        result=bool(severity_labels_ok),
        id="drugscom_perception_severity",
        desc="[Perception Node] drugs.com:F3:P10 - Identify and filter interactions by Major or Moderate severity",
        parent=drugscom_node,
        critical=False
    )

    # [Perception Node] drugs.com:F3:P20 - Identify interaction count statistics
    interaction_count_ok = len(drugs_info.interactions) > 0
    evaluator.add_custom_node(
        result=bool(interaction_count_ok),
        id="drugscom_perception_count",
        desc="[Perception Node] drugs.com:F3:P20 - Identify number of Major/Moderate interactions found",
        parent=drugscom_node,
        critical=False
    )

    # Additional check: Found interactions with common drugs mentioned in task
    common_drugs_ok = any(
        has_any_ci(i.drug_name, ['warfarin', 'aspirin'])
        for i in drugs_info.interactions
    ) if len(drugs_info.interactions) > 0 else False
    evaluator.add_custom_node(
        result=bool(common_drugs_ok),
        id="drugscom_common_drugs",
        desc="Found interactions with commonly prescribed drugs (warfarin, aspirin, etc.)",
        parent=drugscom_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
