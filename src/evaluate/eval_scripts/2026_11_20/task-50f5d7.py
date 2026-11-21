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
TASK_ID = "task-50f5d7"
TASK_DESCRIPTION = "Our project plans to introduce an NPM package named 'visx' for data visualization. However, given the recent surge in supply chain attacks, a security review is required before deployment.\n\nFirst, search for the 'visx' package on NPM. Record its weekly downloads, latest version, number of dependencies, and whether it has Provenance verification. Then, navigate to its GitHub repository.\n\nOn GitHub, check the Commit activity over the past 3 months (number of commits), the number of Contributors, and whether there are any security issues labeled 'security' or 'CVE' in the Issue list (if any, record the Issue title and number).\n\nNext, extract core algorithm keywords for the package (if specific visualization technology names are mentioned in the README or description). Search for related papers on Semantic Scholar, sort by citation count in descending order, and identify the top 3 most cited papers. Record the paper titles, citation counts, publication years, and whether a PDF is available.\n\nFinally, use the core keywords to search Google Patents for related patents (specifically those with a 'Granted' status). If any are found, record the patent title, applicant, and grant date.\n\n**Output:**\n*   **NPM Package Information:** (Weekly Downloads, Latest Version, Number of Dependencies, Provenance Status, NPM detail page link)\n*   **GitHub Health:** (Number of commits in the past 3 months, Number of Contributors, Presence of security issues and their IDs, GitHub repository link)\n*   **Academic Endorsement:** (Titles of the 3 most cited papers, Citation counts, Publication years, PDF availability, Semantic Scholar link)\n*   **Patent Risk:** (Existence of relevant granted patents; if yes, provide patent title, applicant, grant date, Google Patents link)\n*   **Overall Risk Rating:** (Low/Medium/High, based on a comprehensive assessment of security issues, maintenance activity, academic credibility, and patent risk)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NPMPackageInfo(BaseModel):
    """NPM package information extracted from the answer"""
    weekly_downloads: Optional[str] = None
    latest_version: Optional[str] = None
    dependencies_count: Optional[str] = None
    provenance_status: Optional[str] = None
    npm_link: Optional[str] = None


class GitHubHealthInfo(BaseModel):
    """GitHub repository health information extracted from the answer"""
    commits_3months: Optional[str] = None
    contributors_count: Optional[str] = None
    security_issues_present: Optional[str] = None
    security_issue_details: Optional[str] = None
    github_link: Optional[str] = None


class AcademicPaper(BaseModel):
    """A single academic paper entry"""
    title: Optional[str] = None
    citation_count: Optional[str] = None
    publication_year: Optional[str] = None
    pdf_available: Optional[str] = None
    link: Optional[str] = None


class AcademicEndorsement(BaseModel):
    """Academic papers extracted from the answer"""
    paper1: Optional[AcademicPaper] = None
    paper2: Optional[AcademicPaper] = None
    paper3: Optional[AcademicPaper] = None
    semantic_scholar_link: Optional[str] = None


class PatentRiskInfo(BaseModel):
    """Patent risk information extracted from the answer"""
    patents_exist: Optional[str] = None
    patent_title: Optional[str] = None
    patent_applicant: Optional[str] = None
    patent_grant_date: Optional[str] = None
    patent_link: Optional[str] = None


class OverallRiskRating(BaseModel):
    """Overall risk rating extracted from the answer"""
    risk_rating: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_npm_info() -> str:
    return """
Extract the NPM package information for 'visx' from the answer.

Return:
- weekly_downloads: the weekly downloads count exactly as stated
- latest_version: the latest version number exactly as stated
- dependencies_count: the number of dependencies exactly as stated
- provenance_status: whether Provenance verification is present (e.g., "Yes", "No", "Has Provenance", "No Provenance")
- npm_link: the NPM detail page URL if provided

If any field is missing in the answer, set it to null.
"""


def prompt_extract_github_health() -> str:
    return """
Extract the GitHub repository health information for visx from the answer.

Return:
- commits_3months: the number of commits in the past 3 months exactly as stated
- contributors_count: the number of contributors exactly as stated
- security_issues_present: whether security issues exist (e.g., "Yes", "No", "Present", "None found")
- security_issue_details: if security issues exist, their titles and numbers (e.g., "Issue #123: Security vulnerability")
- github_link: the GitHub repository URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_academic_endorsement() -> str:
    return """
Extract the top 3 most cited academic papers from Semantic Scholar from the answer.

Return:
- paper1, paper2, paper3: each containing:
  - title: the paper title
  - citation_count: the number of citations
  - publication_year: the publication year
  - pdf_available: whether PDF is available (e.g., "Yes", "No", "Available")
  - link: the Semantic Scholar link to the paper
- semantic_scholar_link: a general Semantic Scholar search results link if provided

If any field is missing, set it to null.
"""


def prompt_extract_patent_risk() -> str:
    return """
Extract the patent risk information from Google Patents from the answer.

Return:
- patents_exist: whether relevant granted patents exist (e.g., "Yes", "No", "Found", "None")
- patent_title: if patents exist, the patent title
- patent_applicant: if patents exist, the patent applicant/assignee
- patent_grant_date: if patents exist, the grant date
- patent_link: the Google Patents link if provided

If any field is missing, set it to null.
"""


def prompt_extract_overall_risk_rating() -> str:
    return """
Extract the overall risk rating from the answer.

Return:
- risk_rating: the risk rating (e.g., "Low", "Medium", "High")

If not present, set it to null.
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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove commas and extract first number
    cleaned = text.replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def mentions_visx(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'visx')


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
    npm_info = await evaluator.extract(
        prompt=prompt_extract_npm_info(),
        template_class=NPMPackageInfo,
        extraction_name="npm_package_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_health(),
        template_class=GitHubHealthInfo,
        extraction_name="github_health_info"
    )

    academic_info = await evaluator.extract(
        prompt=prompt_extract_academic_endorsement(),
        template_class=AcademicEndorsement,
        extraction_name="academic_endorsement"
    )

    patent_info = await evaluator.extract(
        prompt=prompt_extract_patent_risk(),
        template_class=PatentRiskInfo,
        extraction_name="patent_risk_info"
    )

    risk_rating = await evaluator.extract(
        prompt=prompt_extract_overall_risk_rating(),
        template_class=OverallRiskRating,
        extraction_name="overall_risk_rating"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 NPM Package Section
    npm_node = evaluator.add_sequential(
        id="npm_section",
        desc="NPM package information for visx",
        parent=root,
        critical=False
    )

    # [Action Node] npmjs.com:F1:A2 - Search box input
    npm_search_ok = mentions_visx(answer) and has_any_ci(answer, ['npm', 'npmjs'])
    evaluator.add_custom_node(
        result=bool(npm_search_ok),
        id="npm_search_action",
        desc="[Action Node] npmjs.com:F1:A2 - Input 'visx' in NPM search box",
        parent=npm_node,
        critical=False
    )

    # [Action Node] npmjs.com:F1:A7 - Click card to enter detail page
    npm_detail_page_ok = looks_like_url(npm_info.npm_link, 'npmjs.com')
    evaluator.add_custom_node(
        result=bool(npm_detail_page_ok),
        id="npm_detail_page_action",
        desc="[Action Node] npmjs.com:F1:A7 - Click search result to enter visx package detail page",
        parent=npm_node,
        critical=False
    )

    # [Perception Node] npmjs.com:F2:P7 - Identify package health indicators
    weekly_downloads_ok = contains_digits(npm_info.weekly_downloads)
    latest_version_ok = bool(npm_info.latest_version and npm_info.latest_version.strip())
    dependencies_count_ok = contains_digits(npm_info.dependencies_count)

    health_indicators_ok = weekly_downloads_ok and latest_version_ok and dependencies_count_ok
    evaluator.add_custom_node(
        result=bool(health_indicators_ok),
        id="npm_health_indicators_perception",
        desc="[Perception Node] npmjs.com:F2:P7 - Identify package health indicators (weekly downloads, version, dependencies)",
        parent=npm_node,
        critical=False
    )

    # [Perception Node] npmjs.com:F2:P8 - Identify Provenance status
    provenance_mentioned = bool(npm_info.provenance_status and npm_info.provenance_status.strip())
    evaluator.add_custom_node(
        result=bool(provenance_mentioned),
        id="npm_provenance_perception",
        desc="[Perception Node] npmjs.com:F2:P8 - Identify Provenance verification status",
        parent=npm_node,
        critical=False
    )

    # [Action Node] npmjs.com:F2:A4 - Tab switching to Dependencies
    dependencies_tab_ok = has_any_ci(answer, ['dependencies', 'dependency']) and dependencies_count_ok
    evaluator.add_custom_node(
        result=bool(dependencies_tab_ok),
        id="npm_dependencies_tab_action",
        desc="[Action Node] npmjs.com:F2:A4 - Switch to Dependencies tab to view dependency list",
        parent=npm_node,
        critical=False
    )

    # 3.2 GitHub Section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository health for visx",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand directory structure
    github_repo_ok = looks_like_url(github_info.github_link, 'github.com')
    evaluator.add_custom_node(
        result=bool(github_repo_ok),
        id="github_expand_structure_action",
        desc="[Action Node] github.com:F3:A17 - Navigate to GitHub and expand project file tree to view README",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F10:P6 - Understand commit history
    commits_ok = contains_digits(github_info.commits_3months)
    evaluator.add_custom_node(
        result=bool(commits_ok),
        id="github_commit_history_perception",
        desc="[Perception Node] github.com:F10:P6 - Understand commit history information (past 3 months)",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F10:A24 - Click to view Commit details
    commits_action_ok = commits_ok and has_any_ci(answer, ['commit', 'commits'])
    evaluator.add_custom_node(
        result=bool(commits_action_ok),
        id="github_commits_action",
        desc="[Action Node] github.com:F10:A24 - Browse Commits history list",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Understand Contributors count
    contributors_ok = contains_digits(github_info.contributors_count)
    evaluator.add_custom_node(
        result=bool(contributors_ok),
        id="github_contributors_perception",
        desc="[Perception Node] github.com:F3:P12 - Understand Contributors count",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A1 - Switch to Issues tab
    issues_tab_ok = has_any_ci(answer, ['issue', 'issues'])
    evaluator.add_custom_node(
        result=bool(issues_tab_ok),
        id="github_issues_tab_action",
        desc="[Action Node] github.com:F4:A1 - Switch to Issues tab",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A8 - Filter Issue sorting
    issues_filter_ok = issues_tab_ok and has_any_ci(answer, ['security', 'cve'])
    evaluator.add_custom_node(
        result=bool(issues_filter_ok),
        id="github_issues_filter_action",
        desc="[Action Node] github.com:F4:A8 - Filter or sort Issue list",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - Understand Issue list content
    security_issues_mentioned = bool(github_info.security_issues_present and github_info.security_issues_present.strip())
    evaluator.add_custom_node(
        result=bool(security_issues_mentioned),
        id="github_issues_perception",
        desc="[Perception Node] github.com:F4:P2 - Understand Issue list content (security or CVE labeled issues)",
        parent=github_node,
        critical=False
    )

    # 3.3 Semantic Scholar Section
    semantic_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Academic endorsement from Semantic Scholar",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A2 - Search box input
    semantic_search_ok = has_any_ci(answer, ['semantic scholar']) or (academic_info and academic_info.paper1 and academic_info.paper1.title)
    evaluator.add_custom_node(
        result=bool(semantic_search_ok),
        id="semantic_search_action",
        desc="[Action Node] semanticscholar.org:F1:A2 - Input keywords in Semantic Scholar search box",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Select sorting method
    citation_sort_ok = has_any_ci(answer, ['citation', 'most cited'])
    evaluator.add_custom_node(
        result=bool(citation_sort_ok),
        id="semantic_sort_action",
        desc="[Action Node] semanticscholar.org:F1:A4 - Select Sort by Citation Count",
        parent=semantic_node,
        critical=False
    )

    # Check if papers are in descending order by citation count
    p1_citations = extract_number(academic_info.paper1.citation_count) if academic_info.paper1 else None
    p2_citations = extract_number(academic_info.paper2.citation_count) if academic_info.paper2 else None
    p3_citations = extract_number(academic_info.paper3.citation_count) if academic_info.paper3 else None

    citations_descending = False
    if p1_citations is not None and p2_citations is not None and p3_citations is not None:
        citations_descending = p1_citations >= p2_citations >= p3_citations

    # [Perception Node] semanticscholar.org:F1:P2 - Identify paper special markers (PDF)
    pdf_status_ok = False
    if academic_info.paper1 and academic_info.paper1.pdf_available:
        pdf_status_ok = True
    evaluator.add_custom_node(
        result=bool(pdf_status_ok),
        id="semantic_pdf_perception",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Identify paper special markers (PDF icon or Open Access)",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P3 - Identify citation statistics
    citation_stats_ok = p1_citations is not None and p2_citations is not None and p3_citations is not None
    evaluator.add_custom_node(
        result=bool(citation_stats_ok),
        id="semantic_citation_stats_perception",
        desc="[Perception Node] semanticscholar.org:F2:P3 - Identify citation statistics for top 3 papers",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click card to enter details
    paper_links_ok = False
    if academic_info.paper1 and looks_like_url(academic_info.paper1.link, 'semanticscholar.org'):
        paper_links_ok = True
    evaluator.add_custom_node(
        result=bool(paper_links_ok),
        id="semantic_paper_detail_action",
        desc="[Action Node] semanticscholar.org:F1:A5 - Click paper card to view details",
        parent=semantic_node,
        critical=False
    )

    # 3.4 Google Patents Section
    patents_node = evaluator.add_sequential(
        id="google_patents_section",
        desc="Patent risk from Google Patents",
        parent=root,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A2 - Search input box
    patents_search_ok = has_any_ci(answer, ['google patents', 'patent'])
    evaluator.add_custom_node(
        result=bool(patents_search_ok),
        id="patents_search_action",
        desc="[Action Node] patents.google.com:F1:A2 - Input keywords in Google Patents search box",
        parent=patents_node,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A1 - Multi-condition filter configuration
    patents_filter_ok = has_any_ci(answer, ['grant', 'granted', 'status'])
    evaluator.add_custom_node(
        result=bool(patents_filter_ok),
        id="patents_filter_action",
        desc="[Action Node] patents.google.com:F1:A1 - Configure Status filter (Status=Grant)",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F1:P9 - Identify search result summary
    patents_exist_ok = bool(patent_info.patents_exist and patent_info.patents_exist.strip())
    evaluator.add_custom_node(
        result=bool(patents_exist_ok),
        id="patents_result_summary_perception",
        desc="[Perception Node] patents.google.com:F1:P9 - Identify search result summary (patent title, applicant, grant date)",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P2 - Identify patent legal status
    granted_status_ok = patents_exist_ok and has_any_ci(str(patent_info.patents_exist), ['yes', 'found', 'exist'])
    evaluator.add_custom_node(
        result=bool(granted_status_ok),
        id="patents_legal_status_perception",
        desc="[Perception Node] patents.google.com:F3:P2 - Identify patent legal status (Status=Grant)",
        parent=patents_node,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A12 - Click search result to enter details
    patent_link_ok = looks_like_url(patent_info.patent_link, 'patents.google.com')
    evaluator.add_custom_node(
        result=bool(patent_link_ok),
        id="patents_detail_action",
        desc="[Action Node] patents.google.com:F1:A12 - Click patent result to view detail page",
        parent=patents_node,
        critical=False
    )

    # 3.5 Overall Risk Rating
    risk_rating_node = evaluator.add_custom_node(
        result=bool(risk_rating.risk_rating and risk_rating.risk_rating.strip() and
                   has_any_ci(risk_rating.risk_rating, ['low', 'medium', 'high'])),
        id="overall_risk_rating",
        desc="Overall risk rating provided (Low/Medium/High)",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
