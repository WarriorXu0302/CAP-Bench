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
TASK_ID = "task-194f6d"
TASK_DESCRIPTION = 'I found a data science industry summit in San Francisco on Eventbrite. I want to identify which companies associated with this event are currently hiring for data science-related roles. First, extract the company list from the event details page (possibly from sponsors, partners, organizers, agenda descriptions, or logo walls). If the event page does not explicitly provide an “exhibiting companies” field, extract all identifiable company/institution names on the page and label the section each one comes from.\n\nThen, search LinkedIn and Indeed company by company to check whether they are hiring for **Data Scientist** or **Machine Learning Engineer** roles, prioritizing positions in **San Francisco** or **Remote**, and prioritizing jobs posted in the **last 30 days**. If results are very limited under these strict filters, keep the same company and role keywords but relax the posting window to the **last 60 days**, and clearly mark this in the results.\n\nFor each role found, record:\n- Job title  \n- Company  \n- Location  \n- Salary range (if available)  \n- Key requirements (skills, years of experience, etc.)  \n- Job link  \n- Source platform\n\nIf you encounter LinkedIn login walls or access restrictions, record **“Restricted—unable to view details”**, continue with the remaining companies using whatever is visible, and use Indeed to supplement and cross-validate.\n\nFinally, compile everything into a checklist indicating:\n- Which companies have matching openings  \n- Which companies currently have no matching openings  \n\nThis will help me decide whom to prioritize networking with at the event and prepare targeted questions and resumes.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CompanyInfo(BaseModel):
    """Company information extracted from Eventbrite event page"""
    company_name: Optional[str] = None
    section_source: Optional[str] = None


class JobPosting(BaseModel):
    """Job posting details"""
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    key_requirements: Optional[str] = None
    job_link: Optional[str] = None
    source_platform: Optional[str] = None
    posting_date_window: Optional[str] = None


class EventbriteCompanies(BaseModel):
    """List of companies extracted from Eventbrite event page"""
    companies: List[CompanyInfo] = Field(default_factory=list)
    extraction_notes: Optional[str] = None


class JobSearchResults(BaseModel):
    """Job search results from LinkedIn and Indeed"""
    job_postings: List[JobPosting] = Field(default_factory=list)
    companies_with_openings: List[str] = Field(default_factory=list)
    companies_without_openings: List[str] = Field(default_factory=list)
    access_restrictions_noted: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eventbrite_companies() -> str:
    return """
From the answer, extract all companies mentioned from the Eventbrite event details page for the San Francisco data science industry summit.

For each company, provide:
- company_name: the exact name of the company/institution
- section_source: which section it came from (e.g., "sponsors", "partners", "organizers", "agenda", "logo wall", "description", etc.)

Also include extraction_notes if the answer mentions any special circumstances (e.g., no explicit exhibitor list, had to extract from various sections).

Return all companies found in the companies list.
"""


def prompt_extract_job_search_results() -> str:
    return """
From the answer, extract all job search results from LinkedIn and Indeed for data science roles (Data Scientist or Machine Learning Engineer).

For each job posting found, extract:
- job_title: exact job title
- company: company name
- location: location (San Francisco, Remote, or other)
- salary_range: salary information if available
- key_requirements: skills, years of experience, or other requirements mentioned
- job_link: URL to the job posting
- source_platform: "LinkedIn" or "Indeed"
- posting_date_window: "last 30 days", "last 60 days", or other time window mentioned

Also extract:
- companies_with_openings: list of company names that have matching job openings
- companies_without_openings: list of company names that currently have no matching openings
- access_restrictions_noted: any mentions of LinkedIn login walls, access restrictions, or "Restricted—unable to view details"

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


def mentions_eventbrite(answer: str) -> bool:
    return has_any_ci(answer, ['eventbrite'])


def mentions_san_francisco(answer: str) -> bool:
    return has_any_ci(answer, ['san francisco', 'sf'])


def mentions_data_science_event(answer: str) -> bool:
    return has_any_ci(answer, ['data science', 'data scientist', 'summit', 'conference', 'event'])


def mentions_company_extraction(answer: str) -> bool:
    keywords = ['company', 'companies', 'sponsor', 'partner', 'organizer', 'exhibitor', 'logo']
    return has_any_ci(answer, keywords)


def mentions_linkedin(answer: str) -> bool:
    return has_any_ci(answer, ['linkedin'])


def mentions_indeed(answer: str) -> bool:
    return has_any_ci(answer, ['indeed'])


def mentions_data_science_roles(answer: str) -> bool:
    return has_any_ci(answer, ['data scientist', 'machine learning engineer', 'ml engineer'])


def mentions_location_filters(answer: str) -> bool:
    return has_any_ci(answer, ['san francisco', 'remote', 'location'])


def mentions_time_filters(answer: str) -> bool:
    return has_any_ci(answer, ['30 days', '60 days', 'last 30', 'last 60', 'recent'])


def mentions_job_requirements(answer: str) -> bool:
    return has_any_ci(answer, ['requirement', 'skills', 'experience', 'years'])


def mentions_salary(answer: str) -> bool:
    return has_any_ci(answer, ['salary', 'compensation', 'pay', '$', 'usd'])


def mentions_access_restrictions(answer: str) -> bool:
    return has_any_ci(answer, ['restricted', 'login', 'access', 'unable to view'])


def has_hiring_checklist(answer: str) -> bool:
    hiring_keywords = has_any_ci(answer, ['hiring', 'openings', 'positions', 'jobs'])
    checklist_keywords = has_any_ci(answer, ['with openings', 'without openings', 'no openings', 'checklist', 'companies hiring'])
    return hiring_keywords and checklist_keywords


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
    eventbrite_companies = await evaluator.extract(
        prompt=prompt_extract_eventbrite_companies(),
        template_class=EventbriteCompanies,
        extraction_name="eventbrite_companies"
    )

    job_results = await evaluator.extract(
        prompt=prompt_extract_job_search_results(),
        template_class=JobSearchResults,
        extraction_name="job_search_results"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Eventbrite company extraction section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Extract company list from Eventbrite event details page",
        parent=root,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P6 - Visual feature extraction
    eventbrite_visual_ok = (
        mentions_eventbrite(answer) and
        mentions_company_extraction(answer) and
        has_any_ci(answer, ['sponsor', 'partner', 'organizer', 'agenda', 'logo', 'section'])
    )
    evaluator.add_custom_node(
        result=bool(eventbrite_visual_ok),
        id="eventbrite_visual_extraction",
        desc="[Perception Node] eventbrite.com:F3:P6 - Identify visual presentation locations of company information (sponsors, agenda, logos, etc.)",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P7 - Content understanding
    companies_extracted = bool(eventbrite_companies and len(eventbrite_companies.companies) > 0)
    sections_labeled = False
    if companies_extracted:
        sections_labeled = any(c.section_source for c in eventbrite_companies.companies)

    eventbrite_understanding_ok = companies_extracted and sections_labeled
    evaluator.add_custom_node(
        result=bool(eventbrite_understanding_ok),
        id="eventbrite_content_understanding",
        desc="[Perception Node] eventbrite.com:F3:P7 - Understand content meaning and identify which sections contain company lists with source labels",
        parent=eventbrite_node,
        critical=False
    )

    # Non-prefixed: mentions San Francisco and data science context
    evaluator.add_custom_node(
        result=bool(mentions_san_francisco(answer) and mentions_data_science_event(answer)),
        id="eventbrite_context",
        desc="Mentions San Francisco data science event context",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 LinkedIn job search section
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="LinkedIn job search for data science roles",
        parent=root,
        critical=False
    )

    # [Action Node] linkedin.com:F1:A12 - Location selection
    linkedin_location_ok = mentions_linkedin(answer) and mentions_location_filters(answer)
    evaluator.add_custom_node(
        result=bool(linkedin_location_ok),
        id="linkedin_location_filter",
        desc="[Action Node] linkedin.com:F1:A12 - Apply location filter for San Francisco or Remote",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F1:A30 - Scroll loading
    linkedin_scroll_ok = mentions_linkedin(answer) and has_any_ci(answer, ['job', 'position', 'role', 'opening'])
    evaluator.add_custom_node(
        result=bool(linkedin_scroll_ok),
        id="linkedin_scroll_load",
        desc="[Action Node] linkedin.com:F1:A30 - Scroll to load more search results",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A8 - Time filter
    linkedin_time_ok = mentions_linkedin(answer) and mentions_time_filters(answer)
    evaluator.add_custom_node(
        result=bool(linkedin_time_ok),
        id="linkedin_time_filter",
        desc="[Action Node] linkedin.com:F2:A8 - Apply Date Posted filter (last 30 days, or relaxed to 60 days)",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A9 - Job type filter
    linkedin_role_ok = mentions_linkedin(answer) and mentions_data_science_roles(answer)
    evaluator.add_custom_node(
        result=bool(linkedin_role_ok),
        id="linkedin_job_type_filter",
        desc="[Action Node] linkedin.com:F2:A9 - Identify and filter for Data Scientist or Machine Learning Engineer roles",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A10 - Remote mode filter
    linkedin_remote_ok = mentions_linkedin(answer) and has_any_ci(answer, ['remote'])
    evaluator.add_custom_node(
        result=bool(linkedin_remote_ok),
        id="linkedin_remote_filter",
        desc="[Action Node] linkedin.com:F2:A10 - Apply Remote filter",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F3:P7 - Job description understanding
    linkedin_desc_ok = False
    if job_results and job_results.job_postings:
        linkedin_jobs = [j for j in job_results.job_postings if j.source_platform and ci_contains(j.source_platform, 'linkedin')]
        if linkedin_jobs:
            linkedin_desc_ok = any(j.key_requirements for j in linkedin_jobs)

    evaluator.add_custom_node(
        result=bool(linkedin_desc_ok),
        id="linkedin_description_understanding",
        desc="[Perception Node] linkedin.com:F3:P7 - Deep understanding of job descriptions to extract key requirements",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F3:P8 - Skills list understanding
    linkedin_skills_ok = mentions_linkedin(answer) and mentions_job_requirements(answer)
    evaluator.add_custom_node(
        result=bool(linkedin_skills_ok),
        id="linkedin_skills_understanding",
        desc="[Perception Node] linkedin.com:F3:P8 - Identify and extract skill tags and requirements",
        parent=linkedin_node,
        critical=False
    )

    # Non-prefixed: Mentions access restrictions
    linkedin_restrictions = bool(job_results and job_results.access_restrictions_noted) or mentions_access_restrictions(answer)
    evaluator.add_custom_node(
        result=bool(linkedin_restrictions),
        id="linkedin_access_handling",
        desc="Acknowledges LinkedIn login walls or access restrictions",
        parent=linkedin_node,
        critical=False
    )

    # 3.3 Indeed job search section
    indeed_node = evaluator.add_sequential(
        id="indeed_section",
        desc="Indeed job search for data science roles",
        parent=root,
        critical=False
    )

    # [Action Node] indeed.com:F1:A1 - Search form submission
    indeed_search_ok = mentions_indeed(answer) and mentions_data_science_roles(answer)
    evaluator.add_custom_node(
        result=bool(indeed_search_ok),
        id="indeed_search_submit",
        desc="[Action Node] indeed.com:F1:A1 - Submit search form for Data Scientist or ML Engineer roles",
        parent=indeed_node,
        critical=False
    )

    # [Perception Node] indeed.com:F1:P3 - Job information understanding
    indeed_info_ok = False
    if job_results and job_results.job_postings:
        indeed_jobs = [j for j in job_results.job_postings if j.source_platform and ci_contains(j.source_platform, 'indeed')]
        if indeed_jobs:
            indeed_info_ok = any(
                j.job_title and j.company and j.location
                for j in indeed_jobs
            )

    evaluator.add_custom_node(
        result=bool(indeed_info_ok),
        id="indeed_job_info_extraction",
        desc="[Perception Node] indeed.com:F1:P3 - Extract job title, company, location, and salary information",
        parent=indeed_node,
        critical=False
    )

    # [Action Node] indeed.com:F2:A2 - Date filter
    indeed_date_ok = mentions_indeed(answer) and mentions_time_filters(answer)
    evaluator.add_custom_node(
        result=bool(indeed_date_ok),
        id="indeed_date_filter",
        desc="[Action Node] indeed.com:F2:A2 - Apply Date posted filter (last 30 days, or relaxed to 60 days)",
        parent=indeed_node,
        critical=False
    )

    # [Action Node] indeed.com:F2:A4 - Remote filter
    indeed_remote_ok = mentions_indeed(answer) and has_any_ci(answer, ['remote'])
    evaluator.add_custom_node(
        result=bool(indeed_remote_ok),
        id="indeed_remote_filter",
        desc="[Action Node] indeed.com:F2:A4 - Apply Remote checkbox filter",
        parent=indeed_node,
        critical=False
    )

    # [Perception Node] indeed.com:F1:P14 - Job description understanding
    indeed_desc_ok = False
    if job_results and job_results.job_postings:
        indeed_jobs = [j for j in job_results.job_postings if j.source_platform and ci_contains(j.source_platform, 'indeed')]
        if indeed_jobs:
            indeed_desc_ok = any(j.key_requirements for j in indeed_jobs)

    evaluator.add_custom_node(
        result=bool(indeed_desc_ok),
        id="indeed_description_understanding",
        desc="[Perception Node] indeed.com:F1:P14 - Understand detailed job description content to extract requirements",
        parent=indeed_node,
        critical=False
    )

    # [Perception Node] indeed.com:F1:P20 - Salary type recognition
    indeed_salary_ok = False
    if job_results and job_results.job_postings:
        indeed_jobs = [j for j in job_results.job_postings if j.source_platform and ci_contains(j.source_platform, 'indeed')]
        if indeed_jobs:
            indeed_salary_ok = any(j.salary_range for j in indeed_jobs)

    # Also check if answer mentions salary handling
    if not indeed_salary_ok:
        indeed_salary_ok = mentions_indeed(answer) and mentions_salary(answer)

    evaluator.add_custom_node(
        result=bool(indeed_salary_ok),
        id="indeed_salary_recognition",
        desc="[Perception Node] indeed.com:F1:P20 - Recognize and extract different salary expression formats",
        parent=indeed_node,
        critical=False
    )

    # 3.4 Final compilation and checklist
    checklist_node = evaluator.add_sequential(
        id="compilation_section",
        desc="Compile hiring checklist and final results",
        parent=root,
        critical=False
    )

    # Check for comprehensive job information
    has_job_postings = bool(job_results and len(job_results.job_postings) > 0)
    evaluator.add_custom_node(
        result=bool(has_job_postings),
        id="job_postings_extracted",
        desc="Extracted job postings with required fields (title, company, location, link, source)",
        parent=checklist_node,
        critical=False
    )

    # Check for hiring checklist
    has_checklist_data = bool(
        job_results and
        (len(job_results.companies_with_openings) > 0 or len(job_results.companies_without_openings) > 0)
    )
    checklist_mentioned = has_hiring_checklist(answer)

    evaluator.add_custom_node(
        result=bool(has_checklist_data and checklist_mentioned),
        id="hiring_checklist",
        desc="Compiled checklist indicating which companies have/don't have matching openings",
        parent=checklist_node,
        critical=False
    )

    # Check for cross-platform validation
    has_linkedin_jobs = False
    has_indeed_jobs = False
    if job_results and job_results.job_postings:
        has_linkedin_jobs = any(j.source_platform and ci_contains(j.source_platform, 'linkedin') for j in job_results.job_postings)
        has_indeed_jobs = any(j.source_platform and ci_contains(j.source_platform, 'indeed') for j in job_results.job_postings)

    cross_platform = has_linkedin_jobs and has_indeed_jobs
    evaluator.add_custom_node(
        result=bool(cross_platform),
        id="cross_platform_validation",
        desc="Used both LinkedIn and Indeed for cross-validation and supplementation",
        parent=checklist_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
