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
TASK_ID = "task-87f4ca"
TASK_DESCRIPTION = 'Our project uses the Python library **requests**, but one of its dependencies had a security incident in the past, so I want to conduct a thorough review this time.\n\nPlease first check **PyPI** for any known vulnerability records related to `requests`, especially those with high severity. Then review its **GitHub repository**, focusing on commit activity over the **past six months** and whether there are any unresolved Issues tagged with **security** (please check multiple pages, not just the first one).\n\nFinally, search **Google Scholar** for academic papers specifically studying the security of the `requests` library, especially those published in the **last two to three years**, and see whether any have identified architecture-level security risks. If there is no direct research on `requests`, then find studies on the security of Python HTTP client libraries as supplementary evidence and label them as **indirect evidence**.\n\nPlease summarize the findings for me:\n- Whether there are any severe vulnerabilities  \n- Whether maintenance is active  \n- Whether academia has identified potential risks'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PyPIVulnerabilityInfo(BaseModel):
    """Vulnerability information extracted from PyPI for requests package"""
    has_vulnerabilities: Optional[bool] = None
    severity_mentioned: Optional[str] = None
    vulnerability_details: Optional[str] = None


class GitHubActivityInfo(BaseModel):
    """GitHub activity information extracted for requests repository"""
    commit_activity_mentioned: Optional[bool] = None
    commit_timeframe: Optional[str] = None
    security_issues_found: Optional[bool] = None
    security_issues_status: Optional[str] = None
    multiple_pages_checked: Optional[bool] = None


class ScholarResearchInfo(BaseModel):
    """Google Scholar research findings extracted"""
    papers_found: Optional[bool] = None
    publication_years: Optional[str] = None
    architecture_risks_identified: Optional[bool] = None
    evidence_type: Optional[str] = None
    http_client_security_research: Optional[str] = None


class SecuritySummary(BaseModel):
    """Overall security assessment summary"""
    severe_vulnerabilities: Optional[str] = None
    maintenance_active: Optional[str] = None
    academic_risks: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_pypi_vulnerability() -> str:
    return """
Extract the PyPI vulnerability information for the 'requests' library from the answer.

Return:
- has_vulnerabilities: true if vulnerabilities are mentioned, false if explicitly stated there are none, null if not discussed
- severity_mentioned: any severity level mentioned (e.g., "high", "critical", "low"), null if not mentioned
- vulnerability_details: brief description of any vulnerabilities found, null if none mentioned

If the answer does not discuss PyPI vulnerabilities, set all fields to null.
"""


def prompt_extract_github_activity() -> str:
    return """
Extract GitHub repository activity information for 'requests' from the answer.

Return:
- commit_activity_mentioned: true if commit activity is discussed, false otherwise
- commit_timeframe: the timeframe mentioned (e.g., "past six months", "last 6 months"), null if not specified
- security_issues_found: true if security-tagged issues are mentioned, false if explicitly none, null if not discussed
- security_issues_status: status description (e.g., "unresolved", "open", "closed"), null if not mentioned
- multiple_pages_checked: true if the answer indicates checking multiple pages of issues, false otherwise

If GitHub activity is not discussed, set fields to null.
"""


def prompt_extract_scholar_research() -> str:
    return """
Extract Google Scholar research findings about 'requests' library security from the answer.

Return:
- papers_found: true if academic papers are mentioned, false if explicitly none found, null if not discussed
- publication_years: timeframe of papers mentioned (e.g., "2022-2024", "last 2-3 years"), null if not specified
- architecture_risks_identified: true if architecture-level risks are mentioned, false if none found, null if not discussed
- evidence_type: "direct" if papers specifically about requests, "indirect" if about Python HTTP clients generally, null if not specified
- http_client_security_research: brief description of findings, null if none

If Google Scholar search is not discussed, set all fields to null.
"""


def prompt_extract_security_summary() -> str:
    return """
Extract the overall security assessment summary from the answer.

Return:
- severe_vulnerabilities: summary of whether severe vulnerabilities exist (e.g., "yes", "no", "none found")
- maintenance_active: summary of whether maintenance is active (e.g., "yes", "active", "not very active")
- academic_risks: summary of academic risk findings (e.g., "yes", "no", "none identified")

If the answer does not provide a summary, set fields to null.
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


def mentions_pypi(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['pypi', 'python package index'])


def mentions_github(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['github'])


def mentions_google_scholar(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['google scholar', 'scholar.google'])


def mentions_requests_library(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['requests', '`requests`', '"requests"'])


def mentions_vulnerability(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['vulnerability', 'vulnerabilities', 'cve', 'security issue'])


def mentions_severity(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['severity', 'critical', 'high', 'medium', 'low', 'severe'])


def mentions_commits(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['commit', 'commits', 'commit activity'])


def mentions_six_months(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['six months', '6 months', 'past six months', 'last six months', 'half year'])


def mentions_security_tag(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['security tag', 'tagged with security', 'security label', 'security-tagged'])


def mentions_multiple_pages(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['multiple pages', 'several pages', 'more than one page', 'checked pages', 'page 2', 'second page'])


def mentions_academic_papers(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['paper', 'papers', 'academic', 'research', 'study', 'studies'])


def mentions_recent_years(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [r'202[2-4]', r'last.*years?', r'recent.*years?', r'past.*years?', r'2-3 years', r'two.*three years']
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def mentions_architecture_risks(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['architecture', 'architectural', 'design flaw', 'structural'])


def mentions_indirect_evidence(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['indirect', 'http client', 'python http'])


def has_summary_conclusion(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['summary', 'conclusion', 'findings', 'in summary', 'to summarize'])


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
    pypi_info = await evaluator.extract(
        prompt=prompt_extract_pypi_vulnerability(),
        template_class=PyPIVulnerabilityInfo,
        extraction_name="pypi_vulnerability_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_activity(),
        template_class=GitHubActivityInfo,
        extraction_name="github_activity_info"
    )

    scholar_info = await evaluator.extract(
        prompt=prompt_extract_scholar_research(),
        template_class=ScholarResearchInfo,
        extraction_name="scholar_research_info"
    )

    summary_info = await evaluator.extract(
        prompt=prompt_extract_security_summary(),
        template_class=SecuritySummary,
        extraction_name="security_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 PyPI section
    pypi_node = evaluator.add_sequential(
        id="pypi_section",
        desc="PyPI vulnerability check for requests library",
        parent=root,
        critical=False
    )

    # [Perception Node] pypi.org:F7:P10 - Identify vulnerabilities and severity
    vuln_mentioned = mentions_vulnerability(answer) or (pypi_info and pypi_info.has_vulnerabilities is not None)
    severity_mentioned = mentions_severity(answer) or (pypi_info and pypi_info.severity_mentioned is not None)
    pypi_accessed = mentions_pypi(answer) and mentions_requests_library(answer)

    evaluator.add_custom_node(
        result=bool(pypi_accessed and vuln_mentioned),
        id="pypi_vulnerability_check",
        desc="[Perception Node] pypi.org:F7:P10 - Identify known vulnerability records and severity levels for requests package",
        parent=pypi_node,
        critical=False
    )

    # [Perception Node] pypi.org:F3:P2 - Package status awareness
    package_status_ok = mentions_pypi(answer) and mentions_requests_library(answer)
    evaluator.add_custom_node(
        result=bool(package_status_ok),
        id="pypi_package_status",
        desc="[Perception Node] pypi.org:F3:P2 - Awareness of package maintenance status (e.g., archived, quarantined)",
        parent=pypi_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository review for requests library",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F4:A1 - Tab switching to Issues
    github_accessed = mentions_github(answer)
    issues_mentioned = has_any_ci(answer, ['issue', 'issues'])
    tab_switch_ok = github_accessed and issues_mentioned

    evaluator.add_custom_node(
        result=bool(tab_switch_ok),
        id="github_tab_switch",
        desc="[Action Node] github.com:F4:A1 - Navigate from Code tab to Issues tab",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A5 - Pagination through issues
    pagination_ok = mentions_multiple_pages(answer) or (github_info and github_info.multiple_pages_checked)

    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="github_pagination",
        desc="[Action Node] github.com:F4:A5 - Check multiple pages of issues, not just the first page",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A8 - Sort dropdown for recent updates
    sort_mentioned = has_any_ci(answer, ['sort', 'sorted', 'recently updated', 'recent'])
    six_months_mentioned = mentions_six_months(answer) or (github_info and github_info.commit_timeframe)

    evaluator.add_custom_node(
        result=bool(six_months_mentioned),
        id="github_sort_filter",
        desc="[Action Node] github.com:F4:A8 - Use Sort dropdown to filter by recently updated (past six months)",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - Understand issue tags and status
    security_tag_mentioned = mentions_security_tag(answer) or has_any_ci(answer, ['security issue', 'security-related'])
    issue_status_mentioned = has_any_ci(answer, ['unresolved', 'open', 'closed', 'status'])

    evaluator.add_custom_node(
        result=bool(github_accessed and security_tag_mentioned),
        id="github_issue_understanding",
        desc="[Perception Node] github.com:F4:P2 - Understand issue content including security tags and resolution status",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F10:P6 - Understand commit history
    commits_mentioned = mentions_commits(answer) or (github_info and github_info.commit_activity_mentioned)
    activity_assessed = has_any_ci(answer, ['active', 'activity', 'frequent', 'maintenance'])

    evaluator.add_custom_node(
        result=bool(commits_mentioned and six_months_mentioned),
        id="github_commit_history",
        desc="[Perception Node] github.com:F10:P6 - Understand commit history time distribution and frequency over past six months",
        parent=github_node,
        critical=False
    )

    # 3.3 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="scholar_section",
        desc="Google Scholar academic research on requests library security",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A1 - Academic search
    scholar_accessed = mentions_google_scholar(answer)
    search_performed = scholar_accessed and mentions_requests_library(answer)

    evaluator.add_custom_node(
        result=bool(search_performed),
        id="scholar_search",
        desc="[Action Node] scholar.google.com:F1:A1 - Perform academic search for requests library security research",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A6 - Time filtering
    year_filter_mentioned = mentions_recent_years(answer) or (scholar_info and scholar_info.publication_years)

    evaluator.add_custom_node(
        result=bool(scholar_accessed and year_filter_mentioned),
        id="scholar_time_filter",
        desc="[Action Node] scholar.google.com:F3:A6 - Use Year dropdown to filter papers from last 2-3 years (Since 2022 or Custom range)",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Deep understanding of paper content
    papers_mentioned = mentions_academic_papers(answer) or (scholar_info and scholar_info.papers_found)
    architecture_mentioned = mentions_architecture_risks(answer) or (scholar_info and scholar_info.architecture_risks_identified is not None)
    relevance_assessed = has_any_ci(answer, ['relevant', 'related', 'security', 'risk'])

    evaluator.add_custom_node(
        result=bool(papers_mentioned and architecture_mentioned),
        id="scholar_content_understanding",
        desc="[Perception Node] scholar.google.com:F1:P1 - Deep understanding of paper titles/abstracts for architecture-level security risks",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F4:P3 - Citation impact assessment
    citation_mentioned = has_any_ci(answer, ['cited', 'citation', 'citations', 'impact'])

    evaluator.add_custom_node(
        result=bool(scholar_accessed and citation_mentioned),
        id="scholar_citation_impact",
        desc="[Perception Node] scholar.google.com:F4:P3 - Understand citation counts to assess research authority and impact",
        parent=scholar_node,
        critical=False
    )

    # Indirect evidence handling
    indirect_mentioned = mentions_indirect_evidence(answer) or (scholar_info and scholar_info.evidence_type == "indirect")
    http_client_research = has_any_ci(answer, ['http client', 'python http', 'http library'])

    evaluator.add_custom_node(
        result=bool(indirect_mentioned and http_client_research),
        id="scholar_indirect_evidence",
        desc="Identifies and labels indirect evidence from Python HTTP client security research",
        parent=scholar_node,
        critical=False
    )

    # 3.4 Summary section
    summary_node = evaluator.add_sequential(
        id="summary_section",
        desc="Overall security assessment summary",
        parent=root,
        critical=False
    )

    has_summary = has_summary_conclusion(answer)
    severe_vuln_addressed = (summary_info and summary_info.severe_vulnerabilities is not None) or has_any_ci(answer, ['severe', 'critical'])
    maintenance_addressed = (summary_info and summary_info.maintenance_active is not None) or has_any_ci(answer, ['maintenance', 'active', 'maintained'])
    academic_addressed = (summary_info and summary_info.academic_risks is not None) or has_any_ci(answer, ['academic', 'research', 'risk'])

    evaluator.add_custom_node(
        result=bool(has_summary and severe_vuln_addressed),
        id="summary_vulnerabilities",
        desc="Summary addresses whether severe vulnerabilities exist",
        parent=summary_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_summary and maintenance_addressed),
        id="summary_maintenance",
        desc="Summary addresses whether maintenance is active",
        parent=summary_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_summary and academic_addressed),
        id="summary_academic_risks",
        desc="Summary addresses whether academia has identified potential risks",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
