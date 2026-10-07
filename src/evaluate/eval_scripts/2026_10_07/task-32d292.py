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
TASK_ID = "task-32d292"
TASK_DESCRIPTION = 'I’m preparing for a Google interview and want to understand the latest research trends in the **Machine Learning** direction at their Mountain View headquarters.\n\nFirst, search LinkedIn for a **Research Scientist** role at Google in **Mountain View**, and find one that was posted recently (**Past Month**) or otherwise appears relatively new. Carefully review the job description and extract one core technical keyword (such as a specific large-model technique or algorithm name). If you encounter a LinkedIn login requirement or cannot view full job details, record that access limitation and extract the keyword from whatever LinkedIn-visible job summary is available; if the summary is insufficient, then extract one keyword from visible skills/requirements on the same page.\n\nAfter obtaining the keyword, go to Google Scholar and check Google’s publications in that area. Focus on papers from **2023 onward** and identify the top 3 by citation count (**Cited by**). For each of the three papers, provide the title, first author, and citation count, and also verify whether the author list appears to include Google-affiliated researchers (typically indicated by institution tags/abbreviations next to names). If Google Scholar triggers unusual traffic checks/CAPTCHA or cannot be accessed reliably, switch to Semantic Scholar and use the same keyword and year filter (**2023 onward**), then select the top 3 by citations and output the same fields (title, first author, citation count), plus the basis for Google-affiliation judgment (author institution or institution info on the paper page).'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LinkedInJobInfo(BaseModel):
    """LinkedIn job information extracted from the answer"""
    company_name: Optional[str] = None
    location: Optional[str] = None
    job_title: Optional[str] = None
    posting_time: Optional[str] = None
    technical_keyword: Optional[str] = None
    access_limitation_mentioned: Optional[bool] = None


class ScholarPaper(BaseModel):
    """Individual paper information"""
    title: Optional[str] = None
    first_author: Optional[str] = None
    citation_count: Optional[int] = None
    google_affiliation_mentioned: Optional[bool] = None


class ScholarInfo(BaseModel):
    """Google Scholar search results extracted from the answer"""
    keyword_used: Optional[str] = None
    year_filter_mentioned: Optional[bool] = None
    papers: Optional[List[ScholarPaper]] = Field(default_factory=list)
    scholar_alternative_used: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_linkedin_from_answer() -> str:
    return """
Extract the LinkedIn job search information from the answer:

- company_name: the company name (should be Google)
- location: the job location (should be Mountain View)
- job_title: the job title found (should be Research Scientist or similar)
- posting_time: any mention of when the job was posted (e.g., "Past Month", "2 weeks ago", "recently")
- technical_keyword: the core technical keyword extracted from the job description
- access_limitation_mentioned: whether the answer mentions encountering LinkedIn login requirements or access limitations (true/false)

If any field is missing, set it to null.
"""


def prompt_extract_scholar_from_answer() -> str:
    return """
Extract the Google Scholar (or Semantic Scholar) search information from the answer:

- keyword_used: the technical keyword used for the publication search
- year_filter_mentioned: whether the answer mentions filtering papers from 2023 onward (true/false)
- papers: a list of up to 3 papers, each containing:
  - title: paper title
  - first_author: first author name
  - citation_count: number of citations (as integer if possible)
  - google_affiliation_mentioned: whether the answer mentions Google affiliation for authors (true/false)
- scholar_alternative_used: if Semantic Scholar or another service was used instead of Google Scholar, record that name

If any field is missing, set it to null or empty list as appropriate.
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


def looks_like_recent_posting(text: Optional[str]) -> bool:
    if not text:
        return False
    recent_indicators = ['past month', 'recently', 'days ago', 'weeks ago', 'new', 'posted']
    return has_any_ci(text, recent_indicators)


def looks_like_year_2023_onwards(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    year_patterns = [r'2023\s+onward', r'since\s+2023', r'from\s+2023', r'after\s+2022', r'>=\s*2023', r'2023\+']
    return any(re.search(pattern, answer_text, re.IGNORECASE) for pattern in year_patterns) or ci_contains(answer_text, '2023')


def extract_citation_count(text: Optional[Any]) -> Optional[int]:
    if text is None:
        return None
    if isinstance(text, int):
        return text
    if isinstance(text, str):
        m = re.search(r'(\d+)', text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


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
        strategy=AggregationStrategy.SEQUENTIAL,
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
    linkedin_info = await evaluator.extract(
        prompt=prompt_extract_linkedin_from_answer(),
        template_class=LinkedInJobInfo,
        extraction_name="linkedin_job_info"
    )

    scholar_info = await evaluator.extract(
        prompt=prompt_extract_scholar_from_answer(),
        template_class=ScholarInfo,
        extraction_name="scholar_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 LinkedIn section
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="LinkedIn job search for Research Scientist at Google Mountain View",
        parent=root,
        critical=False
    )

    # Check basic search parameters
    google_ok = ci_contains(linkedin_info.company_name, 'google') or ci_contains(answer, 'google')
    mountain_view_ok = ci_contains(linkedin_info.location, 'mountain view') or ci_contains(answer, 'mountain view')
    research_scientist_ok = ci_contains(linkedin_info.job_title, 'research scientist') or ci_contains(answer, 'research scientist')
    linkedin_mentioned = ci_contains(answer, 'linkedin')

    evaluator.add_custom_node(
        result=bool(linkedin_mentioned and google_ok and mountain_view_ok and research_scientist_ok),
        id="linkedin_search_params",
        desc="Search LinkedIn for Research Scientist role at Google in Mountain View",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F1:P2 - Identify recent posting
    posting_recent = looks_like_recent_posting(linkedin_info.posting_time) or looks_like_recent_posting(answer)
    evaluator.add_custom_node(
        result=bool(posting_recent),
        id="linkedin_recent_posting",
        desc="[Perception Node] linkedin.com:F1:P2 - Identify a recently posted job (Past Month or appears relatively new)",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F3:P7 - Extract technical keyword from JD
    keyword_extracted = bool(linkedin_info.technical_keyword and linkedin_info.technical_keyword.strip())
    evaluator.add_custom_node(
        result=bool(keyword_extracted),
        id="linkedin_extract_keyword",
        desc="[Perception Node] linkedin.com:F3:P7 - Extract one core technical keyword from the job description",
        parent=linkedin_node,
        critical=False
    )

    # Check for access limitation handling
    access_limit_ok = linkedin_info.access_limitation_mentioned if linkedin_info.access_limitation_mentioned is not None else False
    access_limit_mention = ci_contains(answer, 'login') or ci_contains(answer, 'access') or ci_contains(answer, 'limitation')
    evaluator.add_custom_node(
        result=bool(access_limit_ok or access_limit_mention),
        id="linkedin_access_limitation",
        desc="Mentions handling LinkedIn login requirements or access limitations if encountered",
        parent=linkedin_node,
        critical=False
    )

    # 3.2 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="scholar_section",
        desc="Google Scholar (or Semantic Scholar) publication search",
        parent=root,
        critical=False
    )

    # Check keyword transfer from LinkedIn to Scholar
    keyword_transfer_ok = False
    if linkedin_info.technical_keyword and scholar_info.keyword_used:
        keyword_transfer_ok = ci_contains(scholar_info.keyword_used, linkedin_info.technical_keyword.strip()) or \
                             ci_contains(linkedin_info.technical_keyword, scholar_info.keyword_used.strip())
    elif scholar_info.keyword_used and scholar_info.keyword_used.strip():
        keyword_transfer_ok = True

    evaluator.add_custom_node(
        result=bool(keyword_transfer_ok),
        id="scholar_keyword_transfer",
        desc="Use the extracted technical keyword for Google Scholar search",
        parent=scholar_node,
        critical=False
    )

    # Check year filter 2023 onwards
    year_filter_ok = scholar_info.year_filter_mentioned if scholar_info.year_filter_mentioned else False
    year_in_answer = looks_like_year_2023_onwards(answer)
    evaluator.add_custom_node(
        result=bool(year_filter_ok or year_in_answer),
        id="scholar_year_filter",
        desc="Filter papers from 2023 onward",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F4:P3 - Identify top 3 by citation count
    papers = scholar_info.papers if scholar_info.papers else []
    has_three_papers = len(papers) >= 3

    citations_sorted = True
    if len(papers) >= 2:
        for i in range(len(papers) - 1):
            c1 = extract_citation_count(papers[i].citation_count)
            c2 = extract_citation_count(papers[i + 1].citation_count)
            if c1 is not None and c2 is not None and c1 < c2:
                citations_sorted = False
                break

    papers_have_citations = all(
        extract_citation_count(p.citation_count) is not None
        for p in papers
    ) if papers else False

    evaluator.add_custom_node(
        result=bool(has_three_papers and citations_sorted and papers_have_citations),
        id="scholar_top3_citations",
        desc="[Perception Node] scholar.google.com:F4:P3 - Identify top 3 papers by citation count (Cited by)",
        parent=scholar_node,
        critical=False
    )

    # Check paper details completeness
    papers_complete = all(
        bool(p.title and p.title.strip() and
             p.first_author and p.first_author.strip() and
             extract_citation_count(p.citation_count) is not None)
        for p in papers
    ) if papers else False

    evaluator.add_custom_node(
        result=bool(papers_complete and has_three_papers),
        id="scholar_paper_details",
        desc="Provide title, first author, and citation count for each of the 3 papers",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P15 - Verify Google affiliation
    affiliation_checked = all(
        p.google_affiliation_mentioned is not None
        for p in papers
    ) if papers else False

    google_affiliation_mentioned = ci_contains(answer, 'google') and (
        ci_contains(answer, 'affiliation') or
        ci_contains(answer, 'institution') or
        ci_contains(answer, 'author')
    )

    evaluator.add_custom_node(
        result=bool(affiliation_checked or google_affiliation_mentioned),
        id="scholar_google_affiliation",
        desc="[Perception Node] scholar.google.com:F1:P15 - Verify whether author list includes Google-affiliated researchers",
        parent=scholar_node,
        critical=False
    )

    # Check for alternative Scholar service if mentioned
    scholar_service_ok = ci_contains(answer, 'google scholar') or ci_contains(answer, 'semantic scholar') or \
                        (scholar_info.scholar_alternative_used and scholar_info.scholar_alternative_used.strip())
    evaluator.add_custom_node(
        result=bool(scholar_service_ok),
        id="scholar_service_used",
        desc="Uses Google Scholar or mentions switching to Semantic Scholar if needed",
        parent=scholar_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
