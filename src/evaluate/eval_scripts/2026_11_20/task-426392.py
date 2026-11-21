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
TASK_ID = "task-426392"
TASK_DESCRIPTION = 'I’m working on a climate data monitoring project and want to regularly and automatically pull the latest climate data from NOAA (National Oceanic and Atmospheric Administration). Please help me find a climate-related dataset on data.gov that has an API, has been updated within the last 3 months, and is available in CSV format. Check the API documentation for the endpoint URL and authentication method.\n\nThen, search GitHub to see whether anyone has written Python scripts for calling this type of federal data API. Find a project with as many stars as possible, still maintained within the past year (prefer non-archived), and review how the code handles API authentication and data parsing, including whether it implements pagination or rate-limiting logic.\n\nIf no dataset on data.gov meets all of these criteria (“updated within the last 3 months + CSV + NOAA + API”), relax the update window to the last 6 months first. If there is still no match, keep the NOAA and API requirements and record the actual available format.\n\nFinally, provide a structured summary of the dataset’s API endpoint, required authentication parameters, and the key code logic from the GitHub project, and clearly indicate whether each condition is satisfied.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DataGovDataset(BaseModel):
    """Dataset information extracted from data.gov"""
    dataset_name: Optional[str] = None
    organization: Optional[str] = None
    format_available: Optional[str] = None
    has_api: Optional[bool] = None
    last_update_timeframe: Optional[str] = None
    api_endpoint: Optional[str] = None
    authentication_method: Optional[str] = None


class GitHubProject(BaseModel):
    """GitHub project information extracted from the answer"""
    repository_name: Optional[str] = None
    stars_count: Optional[str] = None
    last_maintained: Optional[str] = None
    is_archived: Optional[bool] = None
    authentication_handling: Optional[str] = None
    data_parsing_approach: Optional[str] = None
    has_pagination_logic: Optional[bool] = None
    has_rate_limiting_logic: Optional[bool] = None


class ConditionsSummary(BaseModel):
    """Summary of whether conditions are satisfied"""
    csv_format_satisfied: Optional[bool] = None
    noaa_organization_satisfied: Optional[bool] = None
    api_available_satisfied: Optional[bool] = None
    update_within_3_months_satisfied: Optional[bool] = None
    relaxation_applied: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_datagov_dataset() -> str:
    return """
Extract the data.gov dataset information reported in the answer.

Return:
- dataset_name: the name of the dataset found
- organization: the organization/publisher (should be NOAA or related)
- format_available: the format(s) available (e.g., CSV, JSON, XML)
- has_api: whether an API is available (true/false)
- last_update_timeframe: when the dataset was last updated (e.g., "within 3 months", "within 6 months", "over 6 months ago")
- api_endpoint: the API endpoint URL if provided
- authentication_method: the authentication method required (e.g., "API key", "no authentication", "OAuth")

If any field is missing, set it to null.
"""


def prompt_extract_github_project() -> str:
    return """
Extract the GitHub project information reported in the answer.

Return:
- repository_name: the name of the GitHub repository
- stars_count: the number of stars (as text, e.g., "1.2k", "500")
- last_maintained: when the project was last maintained (e.g., "within 1 year", "2 months ago")
- is_archived: whether the project is archived (true/false)
- authentication_handling: how the code handles API authentication
- data_parsing_approach: how the code parses data
- has_pagination_logic: whether pagination logic is implemented (true/false)
- has_rate_limiting_logic: whether rate-limiting logic is implemented (true/false)

If any field is missing, set it to null.
"""


def prompt_extract_conditions_summary() -> str:
    return """
Extract the conditions satisfaction summary from the answer.

Return:
- csv_format_satisfied: whether CSV format requirement is satisfied (true/false)
- noaa_organization_satisfied: whether NOAA organization requirement is satisfied (true/false)
- api_available_satisfied: whether API availability requirement is satisfied (true/false)
- update_within_3_months_satisfied: whether the 3-month update requirement is satisfied (true/false)
- relaxation_applied: if relaxation was applied, describe what was relaxed (e.g., "relaxed to 6 months", "format relaxed to JSON", "no relaxation")

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


def mentions_datagov(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['data.gov', 'datagov', 'data gov'])


def mentions_noaa(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['noaa', 'national oceanic', 'atmospheric administration'])


def mentions_csv_format(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['csv', 'comma-separated'])


def mentions_api(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['api', 'endpoint', 'rest', 'web service'])


def mentions_github(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['github', 'github.com', 'repository', 'repo'])


def mentions_authentication(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['authentication', 'auth', 'api key', 'token', 'credential'])


def mentions_pagination(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['pagination', 'paging', 'page', 'offset', 'limit'])


def mentions_rate_limiting(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['rate limit', 'rate-limit', 'throttle', 'throttling'])


def mentions_stars_sorting(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['most stars', 'sort by stars', 'sorted by stars', 'stars sorting'])


def mentions_archived_filter(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return has_any_ci(answer, ['non-archived', 'not archived', 'exclude archived', 'archived'])


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text)) or has_any_ci(text, ['.gov', '.com', 'api/', 'endpoint'])


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
        prompt=prompt_extract_datagov_dataset(),
        template_class=DataGovDataset,
        extraction_name="datagov_dataset"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_project(),
        template_class=GitHubProject,
        extraction_name="github_project"
    )

    conditions_info = await evaluator.extract(
        prompt=prompt_extract_conditions_summary(),
        template_class=ConditionsSummary,
        extraction_name="conditions_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 data.gov section
    datagov_node = evaluator.add_sequential(
        id="datagov_section",
        desc="Finding and analyzing climate dataset on data.gov",
        parent=root,
        critical=False
    )

    # [Action Node] data.gov:F1:A4 - Pagination through search results
    datagov_pagination_ok = mentions_datagov(answer) and (has_any_ci(answer, ['search', 'results', 'page', 'found']) or bool(datagov_info.dataset_name))
    evaluator.add_custom_node(
        result=bool(datagov_pagination_ok),
        id="datagov_action_pagination",
        desc="[Action Node] data.gov:F1:A4 - Navigate through search results pages to find suitable climate datasets",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F1:P2 - Understanding dataset descriptions
    datagov_understanding_ok = bool(datagov_info.dataset_name) and (bool(datagov_info.has_api) or mentions_api(answer))
    evaluator.add_custom_node(
        result=bool(datagov_understanding_ok),
        id="datagov_perception_content",
        desc="[Perception Node] data.gov:F1:P2 - Understand dataset descriptions including climate relevance, update frequency, and API availability",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F2:A1 - Multi-select filters (NOAA + CSV)
    datagov_filters_ok = mentions_noaa(answer) and mentions_csv_format(answer)
    evaluator.add_custom_node(
        result=bool(datagov_filters_ok),
        id="datagov_action_filters",
        desc="[Action Node] data.gov:F2:A1 - Apply multi-select filters for NOAA organization and CSV format",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F2:A5 - Expand filter panels
    datagov_expand_ok = mentions_noaa(answer) and (has_any_ci(answer, ['organization', 'publisher', 'filter']) or bool(datagov_info.organization))
    evaluator.add_custom_node(
        result=bool(datagov_expand_ok),
        id="datagov_action_expand",
        desc="[Action Node] data.gov:F2:A5 - Expand Organizations filter panel to find NOAA",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F2:P1 - State awareness (Federal badge, format tags)
    datagov_state_ok = (has_any_ci(answer, ['federal', 'format', 'csv', 'tag', 'badge', 'label']) or
                        bool(datagov_info.format_available))
    evaluator.add_custom_node(
        result=bool(datagov_state_ok),
        id="datagov_perception_state",
        desc="[Perception Node] data.gov:F2:P1 - Recognize dataset state markers like Federal badge, CSV format tags, view counts",
        parent=datagov_node,
        critical=False
    )

    # [Action Node] data.gov:F4:A8 - Click to enter dataset details
    datagov_details_ok = bool(datagov_info.api_endpoint) or (mentions_api(answer) and has_any_ci(answer, ['detail', 'documentation', 'endpoint']))
    evaluator.add_custom_node(
        result=bool(datagov_details_ok),
        id="datagov_action_details",
        desc="[Action Node] data.gov:F4:A8 - Click dataset card to enter details page for API documentation",
        parent=datagov_node,
        critical=False
    )

    # [Perception Node] data.gov:F4:P4 - Data understanding (metadata fields)
    datagov_metadata_ok = bool(datagov_info.last_update_timeframe) or has_any_ci(answer, ['update', 'frequency', 'metadata', 'last modified', 'recent'])
    evaluator.add_custom_node(
        result=bool(datagov_metadata_ok),
        id="datagov_perception_metadata",
        desc="[Perception Node] data.gov:F4:P4 - Understand metadata fields like Frequency Of Update and Metadata Date to verify recent updates",
        parent=datagov_node,
        critical=False
    )

    # Extract API endpoint and authentication
    api_endpoint_ok = looks_like_url(datagov_info.api_endpoint)
    auth_method_ok = bool(datagov_info.authentication_method) or mentions_authentication(answer)
    evaluator.add_custom_node(
        result=bool(api_endpoint_ok and auth_method_ok),
        id="datagov_api_details",
        desc="Extract API endpoint URL and authentication method from documentation",
        parent=datagov_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="Finding and analyzing relevant Python projects on GitHub",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by most stars
    github_sort_ok = mentions_github(answer) and (mentions_stars_sorting(answer) or bool(github_info.stars_count))
    evaluator.add_custom_node(
        result=bool(github_sort_ok),
        id="github_action_sort_stars",
        desc="[Action Node] github.com:F1:A7 - Use Sort dropdown to select 'Most stars' sorting",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A15 - Filter out archived projects
    github_archived_filter_ok = mentions_archived_filter(answer) or (github_info.is_archived is False)
    evaluator.add_custom_node(
        result=bool(github_archived_filter_ok),
        id="github_action_filter_archived",
        desc="[Action Node] github.com:F1:A15 - Apply filter to exclude archived repositories",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Understanding repository information
    github_understanding_ok = bool(github_info.repository_name) and (bool(github_info.stars_count) or bool(github_info.last_maintained))
    evaluator.add_custom_node(
        result=bool(github_understanding_ok),
        id="github_perception_content",
        desc="[Perception Node] github.com:F1:P1 - Understand repository descriptions, star counts, and last update times",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P10 - Archived status awareness
    github_archived_awareness_ok = github_info.is_archived is not None or mentions_archived_filter(answer)
    evaluator.add_custom_node(
        result=bool(github_archived_awareness_ok),
        id="github_perception_archived",
        desc="[Perception Node] github.com:F1:P10 - Recognize Archived label to avoid selecting archived projects",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand file tree
    github_tree_expand_ok = has_any_ci(answer, ['file', 'code', 'directory', 'folder', 'tree', 'source']) or bool(github_info.authentication_handling)
    evaluator.add_custom_node(
        result=bool(github_tree_expand_ok),
        id="github_action_tree_expand",
        desc="[Action Node] github.com:F3:A17 - Expand file tree to locate API and data processing code",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A18 - Select code files
    github_file_select_ok = bool(github_info.authentication_handling) or bool(github_info.data_parsing_approach)
    evaluator.add_custom_node(
        result=bool(github_file_select_ok),
        id="github_action_file_select",
        desc="[Action Node] github.com:F3:A18 - Click to open and view relevant code files",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Hierarchical understanding
    github_hierarchy_ok = has_any_ci(answer, ['main', 'src', 'lib', 'api', 'client', 'parser', 'script']) or bool(github_info.authentication_handling)
    evaluator.add_custom_node(
        result=bool(github_hierarchy_ok),
        id="github_perception_hierarchy",
        desc="[Perception Node] github.com:F3:P12 - Understand project directory structure to locate main API and data processing logic",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F10:P6 - Commits and maintenance understanding
    github_commits_ok = bool(github_info.last_maintained) or has_any_ci(answer, ['commit', 'maintained', 'last update', 'recent', 'active'])
    evaluator.add_custom_node(
        result=bool(github_commits_ok),
        id="github_perception_commits",
        desc="[Perception Node] github.com:F10:P6 - Review commit history to understand project maintenance and activity",
        parent=github_node,
        critical=False
    )

    # Code analysis details
    auth_handling_ok = bool(github_info.authentication_handling)
    parsing_ok = bool(github_info.data_parsing_approach)
    pagination_ok = github_info.has_pagination_logic is True or mentions_pagination(answer)
    rate_limiting_ok = github_info.has_rate_limiting_logic is True or mentions_rate_limiting(answer)

    evaluator.add_custom_node(
        result=bool(auth_handling_ok and parsing_ok),
        id="github_code_analysis",
        desc="Extract API authentication and data parsing logic from code",
        parent=github_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(pagination_ok or rate_limiting_ok),
        id="github_advanced_features",
        desc="Identify pagination or rate-limiting implementation in code",
        parent=github_node,
        critical=False
    )

    # 3.3 Conditions and summary section
    summary_node = evaluator.add_sequential(
        id="summary_section",
        desc="Structured summary with condition satisfaction status",
        parent=root,
        critical=False
    )

    # Check all required conditions are addressed
    csv_condition_ok = conditions_info.csv_format_satisfied is not None or mentions_csv_format(answer)
    noaa_condition_ok = conditions_info.noaa_organization_satisfied is not None or mentions_noaa(answer)
    api_condition_ok = conditions_info.api_available_satisfied is not None or mentions_api(answer)
    update_condition_ok = conditions_info.update_within_3_months_satisfied is not None or has_any_ci(answer, ['3 months', 'three months', '6 months', 'six months', 'recent'])

    evaluator.add_custom_node(
        result=bool(csv_condition_ok and noaa_condition_ok and api_condition_ok and update_condition_ok),
        id="summary_conditions_addressed",
        desc="All conditions (CSV, NOAA, API, update timeframe) are clearly addressed",
        parent=summary_node,
        critical=False
    )

    # Check if relaxation logic was properly applied
    relaxation_ok = bool(conditions_info.relaxation_applied) or has_any_ci(answer, ['relax', 'fallback', '6 months', 'alternative'])
    evaluator.add_custom_node(
        result=bool(relaxation_ok),
        id="summary_relaxation_logic",
        desc="Fallback/relaxation logic properly documented if no perfect match found",
        parent=summary_node,
        critical=False
    )

    # Check summary structure completeness
    summary_complete = (
        bool(datagov_info.api_endpoint) and
        bool(datagov_info.authentication_method) and
        bool(github_info.authentication_handling)
    )
    evaluator.add_custom_node(
        result=bool(summary_complete),
        id="summary_structure_complete",
        desc="Provides structured summary with API endpoint, authentication parameters, and key code logic",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
