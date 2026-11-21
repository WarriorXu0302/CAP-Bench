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
TASK_ID = "task-7a01b5"
TASK_DESCRIPTION = 'I am a Senior Software Engineer in San Francisco, and in the current economic climate I care a lot about job stability. Please help me identify 3 job postings from publicly listed companies. First, search for “Senior Software Engineer” on Glassdoor, set the location to “San Francisco,” and use filters to show only roles with salaries above $170k. Try to select 3 positions that you can confirm are from publicly traded companies (if fewer than 3 are available, record the actual number found and continue with the remaining analysis). For each company, conduct the following risk review:\n\n1. Go to the SEC website (sec.gov), find the company’s latest quarterly report (10-Q), and locate the “Item 1A. Risk Factors” section. Check whether there are risk disclosures related to “restructuring,” “workforce reduction,” or “cost efficiency,” and extract relevant summaries.\n2. Return to Glassdoor and open the company’s Reviews page. Filter for “Current Employee” reviews and sort by “Lowest Rating.” Prioritize negative reviews from the most recent 6 months to identify what employees are mainly complaining about (e.g., management chaos, budget cuts, etc.); if there are not enough samples in the past 6 months, expand to the most recent 1 year and note that adjustment.\n\nIf you encounter a login wall or access restrictions during Glassdoor steps, record the blocked steps and reasons; for companies that remain accessible, continue completing the SEC review and other executable steps.\n\nOutput requirements: company name, job title, salary range, job link, SEC 10-Q filing link, SEC risk-factor summary, Glassdoor negative-review page link (if accessible), and summary of current-employee negative reviews (if accessible).'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JobPosting(BaseModel):
    """A single job posting extracted from the answer"""
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    salary_range: Optional[str] = None
    job_link: Optional[str] = None
    sec_10q_link: Optional[str] = None
    sec_risk_summary: Optional[str] = None
    glassdoor_review_link: Optional[str] = None
    negative_review_summary: Optional[str] = None


class JobPostings(BaseModel):
    """Collection of job postings extracted from the answer"""
    postings: List[JobPosting] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_job_postings() -> str:
    return """
Extract all job postings mentioned in the answer. For each posting, extract:

- company_name: the company name
- job_title: the job title
- salary_range: the salary range as stated
- job_link: the Glassdoor job posting URL if provided
- sec_10q_link: the SEC 10-Q filing URL if provided
- sec_risk_summary: summary of risk factors related to restructuring, workforce reduction, or cost efficiency
- glassdoor_review_link: the Glassdoor reviews page URL if provided
- negative_review_summary: summary of current employee negative reviews if provided

Also extract:
- total_count: the total number of positions found

If any field is missing for a posting, set it to null.
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


def extract_salary_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract numbers, handling k notation
    text_lower = text.lower()
    matches = re.findall(r'(\d+(?:,\d+)?(?:\.\d+)?)\s*k', text_lower)
    if matches:
        try:
            return float(matches[0].replace(',', '')) * 1000
        except Exception:
            pass
    matches = re.findall(r'\$?\s*(\d+(?:,\d+)?(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[0].replace(',', ''))
        except Exception:
            pass
    return None


def salary_above_170k(salary_text: Optional[str]) -> bool:
    if not salary_text:
        return False
    num = extract_salary_number(salary_text)
    if num is None:
        return False
    return num >= 170000


def looks_like_url(text: Optional[str], domain: Optional[str] = None) -> bool:
    if not text:
        return False
    is_url = text.startswith('http://') or text.startswith('https://')
    if domain:
        return is_url and domain.lower() in text.lower()
    return is_url


def is_sec_10q_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return looks_like_url(link, 'sec.gov') and ('10-q' in link.lower() or '10q' in link.lower())


def contains_risk_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['restructuring', 'workforce reduction', 'cost efficiency', 'layoff', 'headcount reduction', 'cost cutting']
    return has_any_ci(text, keywords)


def contains_risk_factors_section(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['risk factors', 'item 1a'])


def is_publicly_traded_company(company_name: Optional[str]) -> bool:
    if not company_name:
        return False
    # Known publicly traded tech companies (not exhaustive, lenient check)
    known_public = [
        'apple', 'google', 'alphabet', 'microsoft', 'amazon', 'meta', 'facebook',
        'netflix', 'tesla', 'nvidia', 'intel', 'amd', 'salesforce', 'oracle',
        'adobe', 'ibm', 'cisco', 'uber', 'lyft', 'airbnb', 'doordash', 'snowflake',
        'palantir', 'stripe', 'square', 'block', 'twitter', 'snap', 'pinterest',
        'zoom', 'slack', 'atlassian', 'servicenow', 'workday', 'splunk', 'twilio',
        'datadog', 'crowdstrike', 'okta', 'mongodb', 'elastic', 'confluent',
        'unity', 'roblox', 'shopify', 'paypal', 'ebay', 'booking', 'expedia'
    ]
    return any(pub.lower() in company_name.lower() for pub in known_public)


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
    job_postings = await evaluator.extract(
        prompt=prompt_extract_job_postings(),
        template_class=JobPostings,
        extraction_name="job_postings"
    )

    postings = job_postings.postings if job_postings and job_postings.postings else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Glassdoor job search section
    glassdoor_search_node = evaluator.add_sequential(
        id="glassdoor_job_search",
        desc="Glassdoor job search with filters for Senior Software Engineer in San Francisco",
        parent=root,
        critical=False
    )

    # [Action Node] glassdoor.com:F1:A2 - Salary filter above $170k
    mentions_glassdoor = has_any_ci(answer, ['glassdoor'])
    mentions_senior_swe = has_any_ci(answer, ['senior software engineer'])
    mentions_sf = has_any_ci(answer, ['san francisco'])
    salary_filter_mentioned = has_any_ci(answer, ['170k', '$170', '170,000', 'salary', 'filter'])

    at_least_one_salary_ok = any(salary_above_170k(p.salary_range) for p in postings)

    evaluator.add_custom_node(
        result=bool(mentions_glassdoor and mentions_senior_swe and mentions_sf and (salary_filter_mentioned or at_least_one_salary_ok)),
        id="glassdoor_salary_filter",
        desc="[Action Node] glassdoor.com:F1:A2 - Apply salary filter to show only positions above $170k",
        parent=glassdoor_search_node,
        critical=False
    )

    # [Perception Node] glassdoor.com:F1:P1 - Identify publicly traded companies
    public_company_count = sum(1 for p in postings if is_publicly_traded_company(p.company_name))
    mentions_public_status = has_any_ci(answer, ['publicly traded', 'publicly listed', 'public company', 'public companies'])

    evaluator.add_custom_node(
        result=bool(public_company_count > 0 or mentions_public_status),
        id="glassdoor_public_company_check",
        desc="[Perception Node] glassdoor.com:F1:P1 - Identify and confirm positions from publicly traded companies",
        parent=glassdoor_search_node,
        critical=False
    )

    # Check that at least one posting has salary >= 170k
    evaluator.add_custom_node(
        result=bool(at_least_one_salary_ok),
        id="verify_salary_170k",
        desc="At least one posting has salary >= $170k",
        parent=glassdoor_search_node,
        critical=False
    )

    # 3.2 SEC section
    sec_node = evaluator.add_sequential(
        id="sec_review",
        desc="SEC 10-Q review for risk factors",
        parent=root,
        critical=False
    )

    # [Action Node] sec.govedgar:F1:A1 - Search for company on SEC
    mentions_sec = has_any_ci(answer, ['sec.gov', 'sec website', 'edgar'])
    at_least_one_sec_link = any(p.sec_10q_link for p in postings)

    evaluator.add_custom_node(
        result=bool(mentions_sec or at_least_one_sec_link),
        id="sec_search_company",
        desc="[Action Node] sec.govedgar:F1:A1 - Search for company on SEC EDGAR system",
        parent=sec_node,
        critical=False
    )

    # [Action Node] sec.govedgar:F1:A19 - Filter for 10-Q filings
    mentions_10q = has_any_ci(answer, ['10-q', '10q', 'quarterly report'])
    at_least_one_valid_10q = any(is_sec_10q_link(p.sec_10q_link) for p in postings)

    evaluator.add_custom_node(
        result=bool(mentions_10q and (at_least_one_sec_link or at_least_one_valid_10q)),
        id="sec_filter_10q",
        desc="[Action Node] sec.govedgar:F1:A19 - Filter for Form 10-Q (quarterly report) filings",
        parent=sec_node,
        critical=False
    )

    # [Action Node] sec.govedgar:F4:A14 - Click to open document
    evaluator.add_custom_node(
        result=bool(at_least_one_valid_10q),
        id="sec_open_document",
        desc="[Action Node] sec.govedgar:F4:A14 - Click to open the 10-Q document details or full text",
        parent=sec_node,
        critical=False
    )

    # [Perception Node] sec.govedgar:F4:P11 - Locate Item 1A Risk Factors section
    at_least_one_risk_summary = any(p.sec_risk_summary and p.sec_risk_summary.strip() for p in postings)
    risk_factors_mentioned = any(contains_risk_factors_section(p.sec_risk_summary) for p in postings) or contains_risk_factors_section(answer)
    risk_keywords_present = any(contains_risk_keywords(p.sec_risk_summary) for p in postings) or contains_risk_keywords(answer)

    evaluator.add_custom_node(
        result=bool(at_least_one_risk_summary and (risk_factors_mentioned or risk_keywords_present)),
        id="sec_risk_factors_extraction",
        desc="[Perception Node] sec.govedgar:F4:P11 - Locate Item 1A Risk Factors section and extract relevant risk disclosures",
        parent=sec_node,
        critical=False
    )

    # Check for restructuring/workforce reduction keywords
    evaluator.add_custom_node(
        result=bool(risk_keywords_present),
        id="sec_risk_keywords",
        desc="Risk summary mentions restructuring, workforce reduction, or cost efficiency",
        parent=sec_node,
        critical=False
    )

    # 3.3 Glassdoor reviews section
    glassdoor_reviews_node = evaluator.add_sequential(
        id="glassdoor_reviews",
        desc="Glassdoor employee reviews filtering and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] glassdoor.com:F2:A6 - Filter for Current Employee reviews
    current_employee_mentioned = has_any_ci(answer, ['current employee', 'current employees'])
    at_least_one_review_summary = any(p.negative_review_summary and p.negative_review_summary.strip() for p in postings)

    evaluator.add_custom_node(
        result=bool(current_employee_mentioned or at_least_one_review_summary),
        id="glassdoor_filter_current_employee",
        desc="[Action Node] glassdoor.com:F2:A6 - Filter reviews to show only Current Employee reviews",
        parent=glassdoor_reviews_node,
        critical=False
    )

    # [Action Node] glassdoor.com:F2:A5 - Sort by Lowest Rating
    lowest_rating_mentioned = has_any_ci(answer, ['lowest rating', 'low rating', 'negative', 'sort'])

    evaluator.add_custom_node(
        result=bool(lowest_rating_mentioned or at_least_one_review_summary),
        id="glassdoor_sort_lowest_rating",
        desc="[Action Node] glassdoor.com:F2:A5 - Sort reviews by Lowest Rating to prioritize negative feedback",
        parent=glassdoor_reviews_node,
        critical=False
    )

    # [Perception Node] glassdoor.com:F2:P8 - Extract and summarize negative review content
    review_complaints_present = any(
        p.negative_review_summary and len(p.negative_review_summary.strip()) > 20
        for p in postings
    )

    evaluator.add_custom_node(
        result=bool(review_complaints_present),
        id="glassdoor_review_content_extraction",
        desc="[Perception Node] glassdoor.com:F2:P8 - Extract specific employee complaints from negative reviews",
        parent=glassdoor_reviews_node,
        critical=False
    )

    # 3.4 Output completeness checks
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Output completeness and format verification",
        parent=root,
        critical=False
    )

    # Check that at least one posting has all required fields
    complete_postings = sum(1 for p in postings if all([
        p.company_name,
        p.job_title,
        p.salary_range
    ]))

    evaluator.add_custom_node(
        result=bool(complete_postings > 0),
        id="output_basic_info",
        desc="At least one posting includes company name, job title, and salary range",
        parent=output_node,
        critical=False
    )

    # Check for SEC information
    evaluator.add_custom_node(
        result=bool(at_least_one_sec_link or at_least_one_risk_summary),
        id="output_sec_info",
        desc="At least one posting includes SEC 10-Q link or risk factor summary",
        parent=output_node,
        critical=False
    )

    # Check for Glassdoor review information
    at_least_one_review_link = any(p.glassdoor_review_link for p in postings)
    evaluator.add_custom_node(
        result=bool(at_least_one_review_link or at_least_one_review_summary),
        id="output_review_info",
        desc="At least one posting includes Glassdoor review link or negative review summary",
        parent=output_node,
        critical=False
    )

    # Handle access restrictions gracefully
    mentions_access_issues = has_any_ci(answer, ['login wall', 'access restriction', 'blocked', 'cannot access', 'login required'])
    if mentions_access_issues:
        evaluator.add_custom_node(
            result=True,
            id="access_restrictions_documented",
            desc="Access restrictions or login walls are documented in the answer",
            parent=output_node,
            critical=False
        )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
