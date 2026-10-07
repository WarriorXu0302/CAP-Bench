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
TASK_ID = "task-6e79f1"
TASK_DESCRIPTION = 'We are preparing to hire a Senior Data Analyst in San Francisco and need to develop a competitive compensation package.\n\nFirst, go to LinkedIn and search for current job openings for "Senior Data Analyst" in the San Francisco Bay Area. Browse through several pages and select 3 technology companies that clearly state a salary range. Record their company names, salary ranges, and check if the job descriptions mention any special benefits (e.g., stock options or remote work).\n\nNext, for these 3 companies, go to Glassdoor and review genuine employee feedback. Remember to filter the reviews to only show "Current Employee" comments, and specifically summarize their key complaints or criticisms regarding "Management."\n\nFinally, go to BLS.gov and find the official data for the broader category "Data Scientists" (which typically includes analysts). I need to know the "Projected Growth" for the next ten years (2023-33) to determine if we are hiring for a growing profession.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LinkedInJobData(BaseModel):
    """LinkedIn job postings data extracted from the answer"""
    company_names: Optional[List[str]] = Field(default_factory=list)
    salary_ranges: Optional[List[str]] = Field(default_factory=list)
    benefits_mentions: Optional[List[str]] = Field(default_factory=list)


class GlassdoorReviewData(BaseModel):
    """Glassdoor review data extracted from the answer"""
    management_complaints: Optional[List[str]] = Field(default_factory=list)
    filter_applied: Optional[str] = None


class BLSData(BaseModel):
    """BLS.gov data extracted from the answer"""
    projected_growth_text: Optional[str] = None
    time_period: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_linkedin_jobs() -> str:
    return """
Extract the LinkedIn job posting information from the answer for Senior Data Analyst positions in San Francisco Bay Area.

Return:
- company_names: list of 3 technology company names mentioned (empty list if none found)
- salary_ranges: list of salary ranges stated for these companies (empty list if none found)
- benefits_mentions: list of special benefits mentioned like stock options or remote work (empty list if none found)

If any field is missing, use empty list or null as appropriate.
"""


def prompt_extract_glassdoor_reviews() -> str:
    return """
Extract the Glassdoor employee review information from the answer.

Return:
- management_complaints: list of key complaints or criticisms regarding "Management" from current employee reviews (empty list if none found)
- filter_applied: text indicating if the "Current Employee" filter was mentioned or applied (null if not mentioned)

If any field is missing, use empty list or null as appropriate.
"""


def prompt_extract_bls_data() -> str:
    return """
Extract the BLS.gov projected growth data for Data Scientists from the answer.

Return:
- projected_growth_text: the projected growth percentage or rate for the next ten years exactly as stated (include units if present)
- time_period: the time period mentioned (e.g., "2023-33", "2023-2033")

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


def looks_like_salary_range(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various formats: "$100k-$150k", "100000-150000", "$100K - $150K", etc.
    if not contains_digits(text):
        return False
    # Check for range indicators or currency symbols
    range_indicators = ['-', 'to', '–', '—']
    currency_symbols = ['$', 'k', 'usd', 'dollar']
    has_range = any(ind in text.lower() for ind in range_indicators)
    has_currency = any(ci_contains(text, sym) for sym in currency_symbols)
    return has_range or has_currency


def looks_like_growth_rate(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if contains number and percentage or growth-related terms
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['%', 'percent', 'growth', 'increase'])


def mentions_pages_browsing(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['page', 'pages', 'browse', 'browsed', 'several', 'multiple'])


def mentions_current_employee_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['current employee', 'filter', 'filtered'])


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
    Restrict evaluator.verify to at most one usage.
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
    linkedin_data = await evaluator.extract(
        prompt=prompt_extract_linkedin_jobs(),
        template_class=LinkedInJobData,
        extraction_name="linkedin_job_data"
    )

    glassdoor_data = await evaluator.extract(
        prompt=prompt_extract_glassdoor_reviews(),
        template_class=GlassdoorReviewData,
        extraction_name="glassdoor_review_data"
    )

    bls_data = await evaluator.extract(
        prompt=prompt_extract_bls_data(),
        template_class=BLSData,
        extraction_name="bls_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 LinkedIn part
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="LinkedIn job search for Senior Data Analyst in San Francisco Bay Area",
        parent=root,
        critical=False
    )

    # [Action Node] linkedin.com:F1:A11 - Browse multiple pages
    linkedin_mentions = has_any_ci(answer, ['linkedin'])
    pages_browsed = mentions_pages_browsing(answer)
    linkedin_paging_ok = linkedin_mentions and pages_browsed

    evaluator.add_custom_node(
        result=bool(linkedin_paging_ok),
        id="linkedin_action_paging",
        desc="[Action Node] linkedin.com:F1:A11 - Browse through several pages of job postings (not just the first page)",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F11:P18 - Identify jobs with clear salary ranges
    has_companies = linkedin_data.company_names and len(linkedin_data.company_names) >= 3
    has_salary_ranges = linkedin_data.salary_ranges and len(linkedin_data.salary_ranges) >= 3
    salary_ranges_look_valid = has_salary_ranges and all(looks_like_salary_range(sr) for sr in linkedin_data.salary_ranges if sr)

    linkedin_perception_ok = has_companies and has_salary_ranges and salary_ranges_look_valid

    evaluator.add_custom_node(
        result=bool(linkedin_perception_ok),
        id="linkedin_perception_salary_ranges",
        desc="[Perception Node] linkedin.com:F11:P18 - Identify and select 3 technology companies with clearly stated salary ranges",
        parent=linkedin_node,
        critical=False
    )

    # Non-prefixed: Check if benefits are mentioned
    benefits_mentioned = linkedin_data.benefits_mentions and len(linkedin_data.benefits_mentions) > 0
    evaluator.add_custom_node(
        result=bool(benefits_mentioned),
        id="linkedin_benefits_check",
        desc="Check for special benefits mentions (e.g., stock options, remote work) in job descriptions",
        parent=linkedin_node,
        critical=False
    )

    # 3.2 Glassdoor part
    glassdoor_node = evaluator.add_sequential(
        id="glassdoor_section",
        desc="Glassdoor employee reviews for the 3 selected companies",
        parent=root,
        critical=False
    )

    # [Action Node] glassdoor.com:F2:A6 - Apply "Current Employee" filter
    glassdoor_mentions = has_any_ci(answer, ['glassdoor'])
    current_employee_filter = mentions_current_employee_filter(answer) or (glassdoor_data.filter_applied and ci_contains(glassdoor_data.filter_applied, 'current employee'))

    glassdoor_filter_ok = glassdoor_mentions and current_employee_filter

    evaluator.add_custom_node(
        result=bool(glassdoor_filter_ok),
        id="glassdoor_action_filter",
        desc="[Action Node] glassdoor.com:F2:A6 - Apply filter to show only 'Current Employee' reviews",
        parent=glassdoor_node,
        critical=False
    )

    # Non-prefixed: Extract management complaints
    has_management_complaints = glassdoor_data.management_complaints and len(glassdoor_data.management_complaints) > 0
    mentions_management = has_any_ci(answer, ['management'])

    glassdoor_management_ok = has_management_complaints and mentions_management

    evaluator.add_custom_node(
        result=bool(glassdoor_management_ok),
        id="glassdoor_management_complaints",
        desc="Summarize key complaints or criticisms regarding 'Management' from current employee reviews",
        parent=glassdoor_node,
        critical=False
    )

    # Non-prefixed: Check if reviews are linked to the 3 companies
    company_context_ok = has_companies and glassdoor_mentions
    evaluator.add_custom_node(
        result=bool(company_context_ok),
        id="glassdoor_company_context",
        desc="Reviews are for the 3 companies identified from LinkedIn",
        parent=glassdoor_node,
        critical=False
    )

    # 3.3 BLS.gov part
    bls_node = evaluator.add_sequential(
        id="bls_section",
        desc="BLS.gov data for Data Scientists projected growth",
        parent=root,
        critical=False
    )

    # Non-prefixed: Navigate to BLS.gov
    bls_mentions = has_any_ci(answer, ['bls', 'bureau of labor statistics'])
    evaluator.add_custom_node(
        result=bool(bls_mentions),
        id="bls_action_navigate",
        desc="Navigate to BLS.gov to find Data Scientists employment data",
        parent=bls_node,
        critical=False
    )

    # [Perception Node] bls.gov:F3:P5 - Extract projected growth from table/quick facts
    growth_text_ok = bls_data.projected_growth_text and looks_like_growth_rate(bls_data.projected_growth_text)
    time_period_ok = bls_data.time_period and (ci_contains(bls_data.time_period, '2023') or ci_contains(bls_data.time_period, '2033'))

    bls_perception_ok = growth_text_ok and time_period_ok

    evaluator.add_custom_node(
        result=bool(bls_perception_ok),
        id="bls_perception_growth",
        desc="[Perception Node] bls.gov:F3:P5 - Extract the projected growth rate for Data Scientists (2023-33) from table or quick facts",
        parent=bls_node,
        critical=False
    )

    # Non-prefixed: Mentions Data Scientists category
    mentions_data_scientists = has_any_ci(answer, ['data scientist', 'data scientists'])
    evaluator.add_custom_node(
        result=bool(mentions_data_scientists),
        id="bls_category_mention",
        desc="References the 'Data Scientists' occupational category on BLS",
        parent=bls_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
