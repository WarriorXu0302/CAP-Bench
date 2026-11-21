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
TASK_ID = "task-957d1a"
TASK_DESCRIPTION = 'I’m transitioning from indie game development to a role at a major studio and plan to apply to game companies in Seattle. First, go to LinkedIn and search for Game Programmer or Engine Engineer positions in Seattle. Filter for jobs posted within the last 30 days, full-time, and Easy Apply. Identify 5 roles whose skill requirements explicitly mention Unreal Engine or Unity, and note the company names. If login requirements prevent access to full job details, record that limitation and continue completing the remaining steps with the visible information.\n\nThen, go to Glassdoor and check reviews and compensation for each of those 5 companies. Prioritize companies with an overall rating of 3.8 or higher, and find the salary range for the Game Programmer role (base salary, bonus, and total compensation) as well as the work-life balance rating. If ratings or salary details are not visible due to login restrictions, record “Restricted” and keep the company in scope rather than skipping it.\n\nNext, search these 5 company names on GitHub. For each company, find the open-source project with the highest star count, and review the README to identify the tech stack they use (programming languages, engines, frameworks).\n\nFinally, based on tech-stack keywords from those projects (e.g., Unreal Engine, C++, Multiplayer Networking), find matching high-quality courses on Udemy. Requirements: rating of at least 4.5, at least 1,000 students, and at least 6 hours of content. Review the course curriculum to confirm whether it includes hands-on projects.\n\nOutput for each company:\n- Company name  \n- LinkedIn job posting link  \n- Job skill requirements list  \n- Glassdoor overall rating  \n- Work-life balance rating  \n- Game Programmer salary range (base salary / bonus / total compensation)  \n- Glassdoor company page link  \n- GitHub open-source project name and star count  \n- Project tech stack  \n- GitHub project link  \n- Recommended Udemy course name  \n- Course rating  \n- Number of students  \n- Course duration  \n- Number of curriculum sections  \n- Udemy course link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CompanyInfo(BaseModel):
    """Information extracted from the answer for one company"""
    company_name: Optional[str] = None
    linkedin_job_link: Optional[str] = None
    job_skill_requirements: Optional[List[str]] = Field(default_factory=list)
    glassdoor_overall_rating: Optional[float] = None
    glassdoor_worklife_rating: Optional[float] = None
    salary_base: Optional[str] = None
    salary_bonus: Optional[str] = None
    salary_total: Optional[str] = None
    glassdoor_link: Optional[str] = None
    github_project_name: Optional[str] = None
    github_stars: Optional[int] = None
    tech_stack: Optional[List[str]] = Field(default_factory=list)
    github_link: Optional[str] = None
    udemy_course_name: Optional[str] = None
    udemy_rating: Optional[float] = None
    udemy_students: Optional[int] = None
    udemy_duration: Optional[str] = None
    udemy_sections: Optional[int] = None
    udemy_link: Optional[str] = None


class AllCompaniesData(BaseModel):
    """All companies data extracted from the answer"""
    companies: List[CompanyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_companies_from_answer() -> str:
    return """
Extract all company information from the answer. For each of the 5 companies mentioned, extract:

- company_name: the company name
- linkedin_job_link: the LinkedIn job posting URL
- job_skill_requirements: list of skill requirements mentioned (especially looking for Unreal Engine or Unity)
- glassdoor_overall_rating: the overall rating from Glassdoor (as a number)
- glassdoor_worklife_rating: the work-life balance rating from Glassdoor (as a number)
- salary_base: the base salary range text
- salary_bonus: the bonus range text
- salary_total: the total compensation range text
- glassdoor_link: the Glassdoor company page URL
- github_project_name: the name of the open-source project found on GitHub
- github_stars: the star count of that project (as an integer)
- tech_stack: list of technologies, languages, engines, or frameworks mentioned in the project
- github_link: the GitHub project URL
- udemy_course_name: the recommended Udemy course name
- udemy_rating: the course rating (as a number)
- udemy_students: the number of students enrolled (as an integer)
- udemy_duration: the course duration text
- udemy_sections: the number of curriculum sections (as an integer)
- udemy_link: the Udemy course URL

If any field is missing or marked as "Restricted", set it to null or empty list as appropriate.
Return all companies found in the companies list.
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


def looks_like_linkedin_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'linkedin.com') and ci_contains(url, 'job')


def looks_like_glassdoor_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'glassdoor.com')


def looks_like_github_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'github.com')


def looks_like_udemy_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'udemy.com')


def mentions_seattle(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return ci_contains(answer_text, 'seattle')


def mentions_30_days(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['30 days', 'last 30 days', 'within 30 days', 'past 30 days'])


def mentions_full_time(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['full-time', 'full time', 'fulltime'])


def mentions_easy_apply(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['easy apply'])


def has_unreal_or_unity(skills: Optional[List[str]]) -> bool:
    if not skills:
        return False
    skills_text = ' '.join(skills).lower()
    return 'unreal' in skills_text or 'unity' in skills_text


def rating_is_valid(rating: Optional[float], min_val: float = 0.0, max_val: float = 5.0) -> bool:
    if rating is None:
        return False
    return min_val <= rating <= max_val


def rating_meets_threshold(rating: Optional[float], threshold: float) -> bool:
    if rating is None:
        return False
    return rating >= threshold


def has_salary_components(base: Optional[str], bonus: Optional[str], total: Optional[str]) -> bool:
    return bool(base or bonus or total)


def star_count_valid(stars: Optional[int]) -> bool:
    if stars is None:
        return False
    return stars >= 0


def has_tech_stack(stack: Optional[List[str]]) -> bool:
    if not stack:
        return False
    return len(stack) > 0


def student_count_meets_threshold(students: Optional[int], threshold: int = 1000) -> bool:
    if students is None:
        return False
    return students >= threshold


def course_duration_valid(duration: Optional[str]) -> bool:
    if not duration:
        return False
    # Look for hour mentions, and try to extract numeric value >= 6
    if not re.search(r'\d', duration):
        return False
    # Extract first number
    match = re.search(r'(\d+(\.\d+)?)', duration)
    if match:
        try:
            hours = float(match.group(1))
            return hours >= 6.0
        except:
            return False
    return False


def sections_count_valid(sections: Optional[int]) -> bool:
    if sections is None:
        return False
    return sections > 0


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
    Restrict evaluator.verify to at most one usage (not used here).
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
    all_data = await evaluator.extract(
        prompt=prompt_extract_companies_from_answer(),
        template_class=AllCompaniesData,
        extraction_name="all_companies_data"
    )

    companies = all_data.companies if all_data and all_data.companies else []
    num_companies = len(companies)

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 LinkedIn section
    linkedin_node = evaluator.add_sequential(
        id="linkedin_section",
        desc="LinkedIn job search and filtering for Game Programmer/Engine Engineer positions in Seattle",
        parent=root,
        critical=False
    )

    # [Action Node] linkedin.com:F1:A12 - Location selection
    location_ok = mentions_seattle(answer)
    evaluator.add_custom_node(
        result=bool(location_ok),
        id="linkedin_location_filter",
        desc="[Action Node] linkedin.com:F1:A12 - Filter job search by Seattle location",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A8 - Time filter (last 30 days)
    time_filter_ok = mentions_30_days(answer)
    evaluator.add_custom_node(
        result=bool(time_filter_ok),
        id="linkedin_time_filter",
        desc="[Action Node] linkedin.com:F2:A8 - Filter for jobs posted within the last 30 days",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A9 - Job type filter (full-time)
    job_type_ok = mentions_full_time(answer)
    evaluator.add_custom_node(
        result=bool(job_type_ok),
        id="linkedin_jobtype_filter",
        desc="[Action Node] linkedin.com:F2:A9 - Filter for full-time positions",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F2:A18 - Easy Apply filter
    easy_apply_ok = mentions_easy_apply(answer)
    evaluator.add_custom_node(
        result=bool(easy_apply_ok),
        id="linkedin_easyapply_filter",
        desc="[Action Node] linkedin.com:F2:A18 - Filter for Easy Apply jobs",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F3:A23 - Click job cards to view details
    valid_linkedin_links = sum(1 for c in companies if looks_like_linkedin_url(c.linkedin_job_link))
    click_cards_ok = valid_linkedin_links >= 3  # Lenient: at least 3 valid links
    evaluator.add_custom_node(
        result=bool(click_cards_ok),
        id="linkedin_click_cards",
        desc="[Action Node] linkedin.com:F3:A23 - Click on job cards to view posting details",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F3:P7 - Understand job descriptions and skill requirements
    unreal_unity_count = sum(1 for c in companies if has_unreal_or_unity(c.job_skill_requirements))
    skill_understanding_ok = unreal_unity_count >= 3  # Lenient: at least 3 mention Unreal or Unity
    evaluator.add_custom_node(
        result=bool(skill_understanding_ok),
        id="linkedin_skill_requirements",
        desc="[Perception Node] linkedin.com:F3:P7 - Identify skill requirements mentioning Unreal Engine or Unity",
        parent=linkedin_node,
        critical=False
    )

    # [Perception Node] linkedin.com:F3:P1 - Identify Easy Apply labels
    evaluator.add_custom_node(
        result=bool(easy_apply_ok),  # Reuse the mention check
        id="linkedin_identify_easyapply",
        desc="[Perception Node] linkedin.com:F3:P1 - Recognize Easy Apply labels on job cards",
        parent=linkedin_node,
        critical=False
    )

    # [Action Node] linkedin.com:F1:A30 - Scroll to load more jobs
    found_five = num_companies >= 5
    evaluator.add_custom_node(
        result=bool(found_five),
        id="linkedin_scroll_load_more",
        desc="[Action Node] linkedin.com:F1:A30 - Scroll to load additional job postings to find 5 roles",
        parent=linkedin_node,
        critical=False
    )

    # 3.2 Glassdoor section
    glassdoor_node = evaluator.add_sequential(
        id="glassdoor_section",
        desc="Glassdoor reviews and salary information for identified companies",
        parent=root,
        critical=False
    )

    # [Action Node] glassdoor.com:F2:A8 - Switch to Reviews tab
    has_glassdoor_reviews = any(rating_is_valid(c.glassdoor_overall_rating) for c in companies)
    evaluator.add_custom_node(
        result=bool(has_glassdoor_reviews),
        id="glassdoor_reviews_tab",
        desc="[Action Node] glassdoor.com:F2:A8 - Navigate to Reviews tab for each company",
        parent=glassdoor_node,
        critical=False
    )

    # [Perception Node] glassdoor.com:F2:P4 - Read overall rating
    ratings_above_threshold = sum(1 for c in companies if rating_meets_threshold(c.glassdoor_overall_rating, 3.8))
    rating_check_ok = ratings_above_threshold >= 2  # Lenient: at least 2 companies meet threshold
    evaluator.add_custom_node(
        result=bool(rating_check_ok),
        id="glassdoor_overall_rating",
        desc="[Perception Node] glassdoor.com:F2:P4 - Extract overall ratings (prioritize 3.8+)",
        parent=glassdoor_node,
        critical=False
    )

    # Work-life balance rating extraction (non-prefixed check)
    has_worklife_ratings = any(rating_is_valid(c.glassdoor_worklife_rating) for c in companies)
    evaluator.add_custom_node(
        result=bool(has_worklife_ratings),
        id="glassdoor_worklife_rating",
        desc="Extract work-life balance ratings from Glassdoor reviews",
        parent=glassdoor_node,
        critical=False
    )

    # [Action Node] glassdoor.com:F3:A8 - Switch to Salaries tab
    has_salary_data = any(has_salary_components(c.salary_base, c.salary_bonus, c.salary_total) for c in companies)
    evaluator.add_custom_node(
        result=bool(has_salary_data),
        id="glassdoor_salaries_tab",
        desc="[Action Node] glassdoor.com:F3:A8 - Navigate to Salaries tab for each company",
        parent=glassdoor_node,
        critical=False
    )

    # [Action Node] glassdoor.com:F3:A9 - Filter salaries by job title
    game_programmer_mentions = ci_contains(answer, 'game programmer')
    evaluator.add_custom_node(
        result=bool(game_programmer_mentions),
        id="glassdoor_filter_jobtitle",
        desc="[Action Node] glassdoor.com:F3:A9 - Filter salary data by Game Programmer job title",
        parent=glassdoor_node,
        critical=False
    )

    # [Perception Node] glassdoor.com:F3:P9 - Read salary components
    complete_salary_count = sum(1 for c in companies if c.salary_base and c.salary_total)
    salary_components_ok = complete_salary_count >= 2  # Lenient: at least 2 companies have base and total
    evaluator.add_custom_node(
        result=bool(salary_components_ok),
        id="glassdoor_salary_components",
        desc="[Perception Node] glassdoor.com:F3:P9 - Extract base salary, bonus, and total compensation",
        parent=glassdoor_node,
        critical=False
    )

    # Valid Glassdoor links check
    valid_glassdoor_links = sum(1 for c in companies if looks_like_glassdoor_url(c.glassdoor_link))
    evaluator.add_custom_node(
        result=bool(valid_glassdoor_links >= 3),
        id="glassdoor_company_links",
        desc="Provide valid Glassdoor company page links",
        parent=glassdoor_node,
        critical=False
    )

    # 3.3 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub open-source project search and tech stack identification",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A10 - Submit search query for company names
    has_github_projects = any(c.github_project_name for c in companies)
    evaluator.add_custom_node(
        result=bool(has_github_projects),
        id="github_search_companies",
        desc="[Action Node] github.com:F1:A10 - Search GitHub for each company name",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by stars
    has_star_counts = any(star_count_valid(c.github_stars) for c in companies)
    evaluator.add_custom_node(
        result=bool(has_star_counts),
        id="github_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - Sort search results by star count (Most stars)",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A19 - Click into project details
    valid_github_links = sum(1 for c in companies if looks_like_github_url(c.github_link))
    click_projects_ok = valid_github_links >= 3
    evaluator.add_custom_node(
        result=bool(click_projects_ok),
        id="github_click_projects",
        desc="[Action Node] github.com:F1:A19 - Click on project cards to view repository details",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A1 - Switch to Code tab
    has_readme_mentions = ci_contains(answer, 'readme')
    evaluator.add_custom_node(
        result=bool(has_readme_mentions),
        id="github_code_tab",
        desc="[Action Node] github.com:F3:A1 - Navigate to Code tab to view README",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Understand project information
    project_info_ok = has_github_projects and has_star_counts
    evaluator.add_custom_node(
        result=bool(project_info_ok),
        id="github_project_info",
        desc="[Perception Node] github.com:F1:P1 - Extract project name and star count",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P21 - Understand language composition
    has_tech_stacks = sum(1 for c in companies if has_tech_stack(c.tech_stack))
    tech_stack_ok = has_tech_stacks >= 3
    evaluator.add_custom_node(
        result=bool(tech_stack_ok),
        id="github_tech_stack",
        desc="[Perception Node] github.com:F3:P21 - Identify tech stack (languages, engines, frameworks) from README",
        parent=github_node,
        critical=False
    )

    # 3.4 Udemy section
    udemy_node = evaluator.add_sequential(
        id="udemy_section",
        desc="Udemy course search based on identified tech stack keywords",
        parent=root,
        critical=False
    )

    # [Action Node] udemy.com:F1:A6 - Search courses based on tech stack
    has_udemy_courses = any(c.udemy_course_name for c in companies)
    evaluator.add_custom_node(
        result=bool(has_udemy_courses),
        id="udemy_search_courses",
        desc="[Action Node] udemy.com:F1:A6 - Search Udemy using tech stack keywords from GitHub projects",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A1 - Rating filter (4.5+)
    high_rated_courses = sum(1 for c in companies if rating_meets_threshold(c.udemy_rating, 4.5))
    rating_filter_ok = high_rated_courses >= 3
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),
        id="udemy_rating_filter",
        desc="[Action Node] udemy.com:F1:A1 - Filter courses with rating of at least 4.5",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F1:A2 - Duration filter (6+ hours)
    long_courses = sum(1 for c in companies if course_duration_valid(c.udemy_duration))
    duration_filter_ok = long_courses >= 3
    evaluator.add_custom_node(
        result=bool(duration_filter_ok),
        id="udemy_duration_filter",
        desc="[Action Node] udemy.com:F1:A2 - Filter courses with at least 6 hours of content",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P1 - Identify course quality tags
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),  # Reuse rating check
        id="udemy_quality_tags",
        desc="[Perception Node] udemy.com:F1:P1 - Recognize high-quality course indicators (Bestseller, Highest Rated)",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P3 - Understand course list information
    sufficient_students = sum(1 for c in companies if student_count_meets_threshold(c.udemy_students, 1000))
    student_count_ok = sufficient_students >= 3
    evaluator.add_custom_node(
        result=bool(student_count_ok),
        id="udemy_student_count",
        desc="[Perception Node] udemy.com:F1:P3 - Extract student enrollment count (at least 1,000 students)",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F2:A8 - Click course card
    valid_udemy_links = sum(1 for c in companies if looks_like_udemy_url(c.udemy_link))
    click_courses_ok = valid_udemy_links >= 3
    evaluator.add_custom_node(
        result=bool(click_courses_ok),
        id="udemy_click_courses",
        desc="[Action Node] udemy.com:F2:A8 - Click on course cards to view details",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F2:A9 - Switch to Course content tab
    has_curriculum = any(sections_count_valid(c.udemy_sections) for c in companies)
    evaluator.add_custom_node(
        result=bool(has_curriculum),
        id="udemy_course_content_tab",
        desc="[Action Node] udemy.com:F2:A9 - Navigate to Course content tab to view curriculum",
        parent=udemy_node,
        critical=False
    )

    # [Action Node] udemy.com:F2:A10 - Expand course sections
    hands_on_mentions = has_any_ci(answer, ['hands-on', 'hands on', 'project', 'practical'])
    evaluator.add_custom_node(
        result=bool(hands_on_mentions),
        id="udemy_expand_sections",
        desc="[Action Node] udemy.com:F2:A10 - Expand curriculum sections to check for hands-on projects",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F2:P5 - Understand course outline structure
    evaluator.add_custom_node(
        result=bool(has_curriculum),
        id="udemy_outline_structure",
        desc="[Perception Node] udemy.com:F2:P5 - Understand course section and lecture hierarchy",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F2:P6 - Understand lecture details
    evaluator.add_custom_node(
        result=bool(hands_on_mentions),
        id="udemy_lecture_details",
        desc="[Perception Node] udemy.com:F2:P6 - Identify lecture types and confirm presence of hands-on projects",
        parent=udemy_node,
        critical=False
    )

    # 3.5 Overall completeness checks (non-prefixed)
    evaluator.add_custom_node(
        result=bool(num_companies >= 5),
        id="completeness_five_companies",
        desc="Successfully identified and processed 5 companies",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(num_companies >= 4),  # Lenient: at least 4
        id="completeness_output_structure",
        desc="Provided complete output structure for each company with all requested fields",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
