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
TASK_ID = "task-e8c583"
TASK_DESCRIPTION = "I am researching education policy, and my advisor has asked me to find reliable government datasets.\n\nPlease help me identify several popular education-related datasets on data.gov. I'm specifically looking for those published at the federal level with high download counts.\n\nOnce you've identified these datasets, search Google Scholar to find highly-cited academic papers that have utilized them in their research. Please prioritize papers with over 100 citations.\n\nFollowing that, check GitHub for corresponding code implementations related to these papers. Focus on projects with a significant number of stars and recent updates.\n\nFinally, please provide a summary indicating which dataset offers both strong academic recognition and readily available code, allowing me to begin my work efficiently."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DataGovDatasets(BaseModel):
    """Education datasets extracted from data.gov"""
    dataset_names: Optional[List[str]] = Field(default_factory=list)
    federal_level_confirmed: Optional[bool] = None
    download_counts_mentioned: Optional[bool] = None


class ScholarPapers(BaseModel):
    """Academic papers extracted from Google Scholar"""
    paper_titles: Optional[List[str]] = Field(default_factory=list)
    citation_counts: Optional[List[str]] = Field(default_factory=list)
    over_100_citations: Optional[bool] = None


class GitHubRepos(BaseModel):
    """GitHub repositories extracted from the answer"""
    repo_names: Optional[List[str]] = Field(default_factory=list)
    star_counts_mentioned: Optional[bool] = None
    recent_updates_mentioned: Optional[bool] = None


class FinalSummary(BaseModel):
    """Final summary and recommendation"""
    recommended_dataset: Optional[str] = None
    has_academic_recognition: Optional[bool] = None
    has_available_code: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_datagov_datasets() -> str:
    return """
Extract information about education datasets from data.gov mentioned in the answer.

Return:
- dataset_names: list of dataset names identified (empty list if none)
- federal_level_confirmed: true if the answer confirms federal-level filtering was used
- download_counts_mentioned: true if the answer mentions download counts or popularity metrics

If any field cannot be determined, set it to null or empty list as appropriate.
"""


def prompt_extract_scholar_papers() -> str:
    return """
Extract information about academic papers from Google Scholar mentioned in the answer.

Return:
- paper_titles: list of paper titles mentioned (empty list if none)
- citation_counts: list of citation count values mentioned as strings (empty list if none)
- over_100_citations: true if the answer confirms filtering for papers with over 100 citations

If any field cannot be determined, set it to null or empty list as appropriate.
"""


def prompt_extract_github_repos() -> str:
    return """
Extract information about GitHub repositories mentioned in the answer.

Return:
- repo_names: list of repository names or projects mentioned (empty list if none)
- star_counts_mentioned: true if the answer mentions star counts or popularity
- recent_updates_mentioned: true if the answer mentions checking for recent updates

If any field cannot be determined, set it to null or empty list as appropriate.
"""


def prompt_extract_final_summary() -> str:
    return """
Extract the final recommendation from the answer.

Return:
- recommended_dataset: the name of the dataset recommended (if any)
- has_academic_recognition: true if the answer confirms the dataset has strong academic recognition
- has_available_code: true if the answer confirms readily available code exists

If any field cannot be determined, set it to null.
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


def extract_numbers(text: Optional[str]) -> List[int]:
    if not text:
        return []
    matches = re.findall(r'\d+', text)
    try:
        return [int(m) for m in matches]
    except Exception:
        return []


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
    datagov_info = await evaluator.extract(
        prompt=prompt_extract_datagov_datasets(),
        template_class=DataGovDatasets,
        extraction_name="datagov_datasets"
    )

    scholar_info = await evaluator.extract(
        prompt=prompt_extract_scholar_papers(),
        template_class=ScholarPapers,
        extraction_name="scholar_papers"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_repos(),
        template_class=GitHubRepos,
        extraction_name="github_repos"
    )

    final_summary = await evaluator.extract(
        prompt=prompt_extract_final_summary(),
        template_class=FinalSummary,
        extraction_name="final_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Data.gov section
    datagov_node = evaluator.add_sequential(
        id="datagov_section",
        desc="Data.gov education dataset identification with federal-level and download filtering",
        parent=root,
        critical=False
    )

    # [Action Node] data.gov:F2:A1 - Multi-select checkbox for federal organizations
    federal_filter_ok = (
        datagov_info and datagov_info.federal_level_confirmed or
        has_any_ci(answer, ['federal', 'organizations', 'filter'])
    )
    evaluator.add_custom_node(
        result=bool(federal_filter_ok),
        id="datagov_action_federal_filter",
        desc="[Action Node] data.gov:F2:A1 - Use Organizations filter to select federal-level agencies",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F2:P1 - Identify Federal labels and download metrics
    has_datasets = datagov_info and datagov_info.dataset_names and len(datagov_info.dataset_names) > 0
    federal_label_ok = has_any_ci(answer, ['federal'])
    download_metrics_ok = (
        datagov_info and datagov_info.download_counts_mentioned or
        has_any_ci(answer, ['download', 'downloads', 'popular', 'popularity'])
    )

    evaluator.add_custom_node(
        result=bool(has_datasets and federal_label_ok and download_metrics_ok),
        id="datagov_perception_federal_downloads",
        desc="[Perception Node] data.gov:F2:P1 - Identify Federal labels and download/access count information on dataset cards",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F8:A11 - Sort by download counts in Metrics page
    sort_by_downloads_ok = has_any_ci(answer, ['sort', 'download', 'most downloaded', 'popular'])
    evaluator.add_custom_node(
        result=bool(sort_by_downloads_ok),
        id="datagov_action_sort_downloads",
        desc="[Action Node] data.gov:F8:A11 - Sort datasets by download count to identify popular datasets",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F8:P6 - Understand download metrics table
    understands_metrics = has_any_ci(answer, ['download', 'metric', 'count', 'popular', 'most downloaded'])
    evaluator.add_custom_node(
        result=bool(understands_metrics and has_datasets),
        id="datagov_perception_metrics_table",
        desc="[Perception Node] data.gov:F8:P6 - Understand download counts in the Most Downloaded Files table",
        parent=datagov_node,
        critical=False
    )

    # 3.2 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="scholar_section",
        desc="Google Scholar search for highly-cited papers using identified datasets",
        parent=root,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Understand search results
    has_papers = scholar_info and scholar_info.paper_titles and len(scholar_info.paper_titles) > 0
    search_context_ok = has_any_ci(answer, ['google scholar', 'scholar', 'search', 'paper', 'academic'])
    evaluator.add_custom_node(
        result=bool(has_papers and search_context_ok),
        id="scholar_perception_search_results",
        desc="[Perception Node] scholar.google.com:F1:P1 - Understand paper titles, authors, and publication years in search results",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F4:P3 - Identify citation counts
    has_citations = scholar_info and scholar_info.citation_counts and len(scholar_info.citation_counts) > 0
    cited_by_ok = has_any_ci(answer, ['cited', 'citation', 'cite'])
    over_100_ok = (
        scholar_info and scholar_info.over_100_citations or
        has_any_ci(answer, ['100', 'over 100', 'more than 100', 'above 100'])
    )
    evaluator.add_custom_node(
        result=bool(has_citations and cited_by_ok and over_100_ok),
        id="scholar_perception_citation_counts",
        desc="[Perception Node] scholar.google.com:F4:P3 - Identify and filter papers with over 100 citations using 'Cited by' numbers",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A15 - Pagination to find more papers
    pagination_ok = has_any_ci(answer, ['page', 'next', 'more results', 'multiple', 'several'])
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="scholar_action_pagination",
        desc="[Action Node] scholar.google.com:F1:A15 - Navigate through multiple pages to find papers with sufficient citations",
        parent=scholar_node,
        critical=False
    )

    # 3.3 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub search for code implementations with high stars and recent updates",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by star count
    has_repos = github_info and github_info.repo_names and len(github_info.repo_names) > 0
    sort_stars_ok = has_any_ci(answer, ['sort', 'star', 'most stars', 'popular'])
    evaluator.add_custom_node(
        result=bool(sort_stars_ok),
        id="github_action_sort_stars",
        desc="[Action Node] github.com:F1:A7 - Use Sort dropdown to order repositories by Most stars",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Understand repo cards
    star_info_ok = (
        github_info and github_info.star_counts_mentioned or
        has_any_ci(answer, ['star', 'stars'])
    )
    update_info_ok = (
        github_info and github_info.recent_updates_mentioned or
        has_any_ci(answer, ['recent', 'update', 'updated', 'last updated', 'active'])
    )
    evaluator.add_custom_node(
        result=bool(has_repos and star_info_ok and update_info_ok),
        id="github_perception_repo_info",
        desc="[Perception Node] github.com:F1:P1 - Understand star counts and last update times on repository cards",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand directory tree to verify code
    expand_tree_ok = has_any_ci(answer, ['code', 'file', 'directory', 'folder', 'structure', 'implementation'])
    evaluator.add_custom_node(
        result=bool(expand_tree_ok),
        id="github_action_expand_tree",
        desc="[Action Node] github.com:F3:A17 - Expand project directory tree to examine code files",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Understand directory structure
    understands_structure = has_any_ci(answer, ['code', 'script', 'data', 'file', 'structure', 'repository'])
    dataset_match_ok = has_any_ci(answer, ['dataset', 'data', 'match', 'use', 'implement'])
    evaluator.add_custom_node(
        result=bool(understands_structure and dataset_match_ok),
        id="github_perception_directory_structure",
        desc="[Perception Node] github.com:F3:P12 - Understand file organization and verify dataset-code alignment",
        parent=github_node,
        critical=False
    )

    # 3.4 Final synthesis
    synthesis_node = evaluator.add_sequential(
        id="synthesis_section",
        desc="Final synthesis combining academic recognition and code availability",
        parent=root,
        critical=False
    )

    has_recommendation = final_summary and final_summary.recommended_dataset
    has_academic_value = (
        final_summary and final_summary.has_academic_recognition or
        has_any_ci(answer, ['academic', 'citation', 'research', 'scholar'])
    )
    has_code = (
        final_summary and final_summary.has_available_code or
        has_any_ci(answer, ['code', 'implementation', 'github', 'repository'])
    )

    evaluator.add_custom_node(
        result=bool(has_recommendation and has_academic_value and has_code),
        id="synthesis_recommendation",
        desc="Provides comprehensive recommendation identifying datasets with both strong academic recognition and readily available code",
        parent=synthesis_node,
        critical=False
    )

    # Information flow validation
    flow_ok = (
        has_datasets and  # Found datasets on data.gov
        has_papers and    # Found papers on Scholar
        has_repos and     # Found repos on GitHub
        has_recommendation  # Provided final synthesis
    )
    evaluator.add_custom_node(
        result=bool(flow_ok),
        id="information_flow_complete",
        desc="Complete information flow: data.gov → Google Scholar → GitHub → synthesis",
        parent=synthesis_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
