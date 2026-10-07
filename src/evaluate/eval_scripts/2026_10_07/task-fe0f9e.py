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
TASK_ID = "task-fe0f9e"
TASK_DESCRIPTION = "I am writing an in-depth article on the 2025 status of autonomous driving deployment, focusing on the Mercedes-Benz Drive Pilot system.\n\nFirst, navigate to Car and Driver and locate an in-depth review article about Drive Pilot. From this article, extract the specific hard limitations (maximum speed, applicable road types, weather requirements) for the system's activation.\n\nNext, consult the 'SAE J3016' entry on Wikipedia to identify the core definitional differences between SAE Level 3 and Level 2 regarding 'fallback performance of dynamic driving task'.\n\nFinally, go to Google Patents and search for a granted U.S. invention patent filed by Mercedes-Benz Group AG or Daimler AG. The patent should be related to either 'driver handover' or 'driver monitoring', and its application date must be after 2021.\n\n**Output:**\n*   System limitations mentioned in the Car and Driver article and its link.\n*   Original definition text from Wikipedia regarding L3/L2 fallback performance differences and its link.\n*   Found patent's title, patent number, application date, applicant, and a link to the patent's details page."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CarAndDriverInfo(BaseModel):
    """Information extracted from Car and Driver article about Drive Pilot"""
    maximum_speed: Optional[str] = None
    road_types: Optional[str] = None
    weather_requirements: Optional[str] = None
    article_link: Optional[str] = None


class WikipediaInfo(BaseModel):
    """Information extracted from Wikipedia about SAE J3016 L3/L2 differences"""
    fallback_definition_text: Optional[str] = None
    wikipedia_link: Optional[str] = None


class PatentInfo(BaseModel):
    """Information extracted about the Mercedes-Benz patent"""
    patent_title: Optional[str] = None
    patent_number: Optional[str] = None
    application_date: Optional[str] = None
    applicant: Optional[str] = None
    patent_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_caranddriver_from_answer() -> str:
    return """
Extract the Car and Driver article information about Mercedes-Benz Drive Pilot from the answer.

Return:
- maximum_speed: the maximum speed limitation mentioned for Drive Pilot activation (include units).
- road_types: the applicable road types mentioned for Drive Pilot.
- weather_requirements: the weather requirements mentioned for Drive Pilot.
- article_link: the URL or link to the Car and Driver article.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_wikipedia_from_answer() -> str:
    return """
Extract the Wikipedia SAE J3016 information from the answer.

Return:
- fallback_definition_text: the original definition text about fallback performance differences between Level 3 and Level 2.
- wikipedia_link: the URL or link to the Wikipedia page.

If any field is missing, set it to null.
"""


def prompt_extract_patent_from_answer() -> str:
    return """
Extract the patent information from the answer.

Return:
- patent_title: the title of the patent.
- patent_number: the patent number (e.g., US1234567B2).
- application_date: the application date of the patent.
- applicant: the applicant name (should be Mercedes-Benz or Daimler related).
- patent_link: the URL or link to the patent details page.

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


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.strip().startswith('http://') or text.strip().startswith('https://')


def looks_like_caranddriver_url(text: Optional[str]) -> bool:
    if not looks_like_url(text):
        return False
    return ci_contains(text, 'caranddriver.com')


def looks_like_wikipedia_url(text: Optional[str]) -> bool:
    if not looks_like_url(text):
        return False
    return ci_contains(text, 'wikipedia.org')


def looks_like_patent_url(text: Optional[str]) -> bool:
    if not looks_like_url(text):
        return False
    return has_any_ci(text, ['patents.google.com', 'patent'])


def looks_like_speed(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['mph', 'km/h', 'kph', 'mi/h', 'speed'])


def looks_like_patent_number(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'US\s*\d+', text, re.IGNORECASE))


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats: YYYY-MM-DD, MM/DD/YYYY, Month Day Year, etc.
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in date_patterns)


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\b(20\d{2})\b', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
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
    Restrict evaluator.verify to at most one usage (we'll not use it here).
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
    caranddriver_info = await evaluator.extract(
        prompt=prompt_extract_caranddriver_from_answer(),
        template_class=CarAndDriverInfo,
        extraction_name="caranddriver_info"
    )

    wikipedia_info = await evaluator.extract(
        prompt=prompt_extract_wikipedia_from_answer(),
        template_class=WikipediaInfo,
        extraction_name="wikipedia_info"
    )

    patent_info = await evaluator.extract(
        prompt=prompt_extract_patent_from_answer(),
        template_class=PatentInfo,
        extraction_name="patent_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver: Drive Pilot review and system limitations",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F10:A3 - Navigate to Car and Driver and locate Drive Pilot article
    carandriver_nav_ok = (has_any_ci(answer, ['car and driver']) and
                          has_any_ci(answer, ['drive pilot']))
    evaluator.add_custom_node(
        result=bool(carandriver_nav_ok),
        id="caranddriver_navigate",
        desc="[Action Node] caranddriver.com:F10:A3 - Navigate to Car and Driver and locate Drive Pilot article",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F10:P2 - Extract specific hard limitations
    speed_ok = looks_like_speed(caranddriver_info.maximum_speed)
    road_ok = bool(caranddriver_info.road_types and caranddriver_info.road_types.strip())
    weather_ok = bool(caranddriver_info.weather_requirements and caranddriver_info.weather_requirements.strip())

    evaluator.add_custom_node(
        result=bool(speed_ok and road_ok and weather_ok),
        id="caranddriver_extract_limitations",
        desc="[Perception Node] caranddriver.com:F10:P2 - Extract maximum speed, road types, and weather requirements",
        parent=caranddriver_node,
        critical=False
    )

    # Article link check (output o3)
    article_link_ok = looks_like_caranddriver_url(caranddriver_info.article_link)
    evaluator.add_custom_node(
        result=bool(article_link_ok),
        id="caranddriver_article_link",
        desc="Provide valid Car and Driver article link (output o3)",
        parent=caranddriver_node,
        critical=False
    )

    # 3.2 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia: SAE J3016 Level 3 vs Level 2 fallback performance differences",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Navigate to SAE J3016 entry
    wiki_nav_ok = (has_any_ci(answer, ['wikipedia']) and
                   has_any_ci(answer, ['sae j3016']))
    evaluator.add_custom_node(
        result=bool(wiki_nav_ok),
        id="wikipedia_navigate",
        desc="[Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia SAE J3016 entry",
        parent=wikipedia_node,
        critical=False
    )

    # [Action Node] wikipedia.org:F4:A19 - Use table of contents or jump to relevant section
    toc_usage_ok = has_any_ci(answer, ['level 3', 'level 2', 'fallback'])
    evaluator.add_custom_node(
        result=bool(toc_usage_ok),
        id="wikipedia_toc_jump",
        desc="[Action Node] wikipedia.org:F4:A19 - Use table of contents to navigate to level definitions section",
        parent=wikipedia_node,
        critical=False
    )

    # [Perception Node] Extract fallback definition differences (output o4)
    fallback_text_ok = bool(wikipedia_info.fallback_definition_text and
                           len(wikipedia_info.fallback_definition_text.strip()) > 20)
    fallback_keywords_ok = (fallback_text_ok and
                           has_any_ci(wikipedia_info.fallback_definition_text, ['fallback']) and
                           has_any_ci(wikipedia_info.fallback_definition_text, ['level 3', 'level 2', 'l3', 'l2']))

    evaluator.add_custom_node(
        result=bool(fallback_keywords_ok),
        id="wikipedia_extract_fallback",
        desc="Extract original definition text about L3/L2 fallback performance differences (output o4)",
        parent=wikipedia_node,
        critical=False
    )

    # Wikipedia link check (output o5)
    wiki_link_ok = looks_like_wikipedia_url(wikipedia_info.wikipedia_link)
    evaluator.add_custom_node(
        result=bool(wiki_link_ok),
        id="wikipedia_link",
        desc="Provide valid Wikipedia link (output o5)",
        parent=wikipedia_node,
        critical=False
    )

    # 3.3 Google Patents section
    patents_node = evaluator.add_sequential(
        id="patents_section",
        desc="Google Patents: Mercedes-Benz/Daimler patent on driver handover/monitoring",
        parent=root,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A2 - Keyword combination search
    keyword_search_ok = (has_any_ci(answer, ['google patents']) and
                        has_any_ci(answer, ['driver handover', 'driver monitoring', 'handover', 'monitoring']))
    evaluator.add_custom_node(
        result=bool(keyword_search_ok),
        id="patents_keyword_search",
        desc="[Action Node] patents.google.com:F1:A2 - Search with driver handover or monitoring keywords",
        parent=patents_node,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A1 - Multi-criteria filtering (applicant/date/status)
    applicant_filter_ok = has_any_ci(answer, ['mercedes-benz', 'daimler', 'mercedes benz'])
    date_filter_ok = has_any_ci(answer, ['2021', '2022', '2023', '2024', 'after 2021'])
    status_filter_ok = has_any_ci(answer, ['granted', 'grant', 'issued'])

    evaluator.add_custom_node(
        result=bool(applicant_filter_ok and date_filter_ok and status_filter_ok),
        id="patents_filtering",
        desc="[Action Node] patents.google.com:F1:A1 - Filter by applicant (Mercedes/Daimler), date (after 2021), and granted status",
        parent=patents_node,
        critical=False
    )

    # [Action Node] patents.google.com:F3:A12 - Click into patent details
    patent_details_ok = bool(patent_info.patent_title or patent_info.patent_number)
    evaluator.add_custom_node(
        result=bool(patent_details_ok),
        id="patents_click_details",
        desc="[Action Node] patents.google.com:F3:A12 - Click into patent details page",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P1 - Extract metadata (title, number, date, applicant)
    title_ok = bool(patent_info.patent_title and patent_info.patent_title.strip())
    title_relevant_ok = (title_ok and
                        has_any_ci(patent_info.patent_title, ['driver', 'handover', 'monitoring', 'takeover', 'attention']))

    number_ok = looks_like_patent_number(patent_info.patent_number)

    date_ok = looks_like_date(patent_info.application_date)
    year = extract_year(patent_info.application_date)
    date_after_2021_ok = year is not None and year > 2021

    applicant_ok = has_any_ci(patent_info.applicant, ['mercedes', 'daimler'])

    evaluator.add_custom_node(
        result=bool(title_relevant_ok and number_ok and date_ok and date_after_2021_ok and applicant_ok),
        id="patents_extract_metadata",
        desc="[Perception Node] patents.google.com:F3:P1 - Extract patent title (output o6), number (output o7), application date (output o8, after 2021), and applicant (output o9)",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P2 - Identify legal status as granted
    status_granted_ok = has_any_ci(answer, ['granted', 'grant', 'active', 'issued'])
    evaluator.add_custom_node(
        result=bool(status_granted_ok),
        id="patents_legal_status",
        desc="[Perception Node] patents.google.com:F3:P2 - Confirm patent is granted/active (output o10)",
        parent=patents_node,
        critical=False
    )

    # Patent link check (output o11)
    patent_link_ok = looks_like_patent_url(patent_info.patent_link)
    evaluator.add_custom_node(
        result=bool(patent_link_ok),
        id="patents_link",
        desc="Provide valid patent details page link (output o11)",
        parent=patents_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
