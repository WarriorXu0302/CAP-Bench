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
TASK_ID = "task-91ef36"
TASK_DESCRIPTION = 'I am a high school history teacher preparing a lesson on the “impact of the Industrial Revolution on global living standards” (with a focus on the United Kingdom).  \n\nFirst, check Wikipedia for “Industrial Revolution” and confirm its approximate start and end years, as well as its place of origin. Then go to Gapminder Tools, select the United Kingdom, and in **Trends** (line chart) mode set the y-axis to **“Income per person (GDP/capita, inflation-adjusted)”**, with time as the default x-axis. Identify the approximate year when the UK’s income per person clearly takes off from **Level 1** (about the $1,000–$2,000 range) and surpasses the **$3,000** threshold, and record the exact value for the most recent complete year.  \n\nNext, still in Gapminder, switch the y-axis to **“Life expectancy”** and find the UK’s life expectancy values in **1850** and **1950**.  \n\nFinally, go to Khan Academy, search for “Industrial Revolution,” and find a video or article suitable for an AP history course that includes content related to **“Social impact”** or **“Living standards.”** Confirm the name of the unit it belongs to.  \n\nPlease output: the Industrial Revolution’s start and end years, place of origin, the year the UK’s income per person in Gapminder surpassed $3,000, the UK’s income-per-person value for the most recent complete year, UK life expectancy in 1850 and 1950, the title of the selected Khan Academy content, its unit name, and links to each page.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IndustrialRevolutionInfo(BaseModel):
    """Industrial Revolution information extracted from Wikipedia"""
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    place_of_origin: Optional[str] = None
    wikipedia_link: Optional[str] = None


class GapminderIncomeData(BaseModel):
    """Gapminder income data for UK extracted from the answer"""
    year_surpassed_3000: Optional[str] = None
    most_recent_year_value: Optional[str] = None
    gapminder_link: Optional[str] = None


class GapmapperLifeExpectancyData(BaseModel):
    """Gapminder life expectancy data for UK extracted from the answer"""
    life_expectancy_1850: Optional[str] = None
    life_expectancy_1950: Optional[str] = None


class KhanAcademyContent(BaseModel):
    """Khan Academy content extracted from the answer"""
    content_title: Optional[str] = None
    unit_name: Optional[str] = None
    khan_academy_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_industrial_revolution_info() -> str:
    return """
Extract the Industrial Revolution information from the answer that the user obtained from Wikipedia.

Return:
- start_year: the approximate start year of the Industrial Revolution (e.g., "1760", "mid-18th century"). If not present, set null.
- end_year: the approximate end year of the Industrial Revolution (e.g., "1840", "mid-19th century"). If not present, set null.
- place_of_origin: the place of origin (e.g., "United Kingdom", "Britain", "England"). If not present, set null.
- wikipedia_link: the Wikipedia URL mentioned in the answer. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_gapminder_income_data() -> str:
    return """
From the answer, extract the Gapminder income data for the United Kingdom:

- year_surpassed_3000: the approximate year when UK's income per person surpassed $3,000 (e.g., "1865", "around 1870"). If not present, set null.
- most_recent_year_value: the UK's income per person value for the most recent complete year, exactly as stated (include units if present). If not present, set null.
- gapminder_link: the Gapminder Tools URL mentioned in the answer. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_gapminder_life_expectancy_data() -> str:
    return """
From the answer, extract the Gapminder life expectancy data for the United Kingdom:

- life_expectancy_1850: the UK's life expectancy in 1850 exactly as stated (include units if present). If not present, set null.
- life_expectancy_1950: the UK's life expectancy in 1950 exactly as stated (include units if present). If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_khan_academy_content() -> str:
    return """
From the answer, extract the Khan Academy content information:

- content_title: the title of the video or article found on Khan Academy related to Industrial Revolution. If not present, set null.
- unit_name: the name of the unit this content belongs to (e.g., "AP World History", "World History"). If not present, set null.
- khan_academy_link: the Khan Academy URL mentioned in the answer. If not present, set null.

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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r'\b(1[5-9]\d{2}|20\d{2})\b', text)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def looks_like_year_range(start: Optional[str], end: Optional[str]) -> bool:
    start_year = extract_year(start)
    end_year = extract_year(end)
    if start_year is None or end_year is None:
        return False
    return 1700 <= start_year <= 1800 and 1800 <= end_year <= 1900


def looks_like_income_takeoff_year(text: Optional[str]) -> bool:
    year = extract_year(text)
    if year is None:
        return False
    return 1850 <= year <= 1900


def looks_like_life_expectancy(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 20 <= num <= 90


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
    ir_info = await evaluator.extract(
        prompt=prompt_extract_industrial_revolution_info(),
        template_class=IndustrialRevolutionInfo,
        extraction_name="industrial_revolution_info"
    )

    income_info = await evaluator.extract(
        prompt=prompt_extract_gapminder_income_data(),
        template_class=GapminderIncomeData,
        extraction_name="gapminder_income_data"
    )

    life_expectancy_info = await evaluator.extract(
        prompt=prompt_extract_gapminder_life_expectancy_data(),
        template_class=GapmapperLifeExpectancyData,
        extraction_name="gapminder_life_expectancy_data"
    )

    khan_info = await evaluator.extract(
        prompt=prompt_extract_khan_academy_content(),
        template_class=KhanAcademyContent,
        extraction_name="khan_academy_content"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia Industrial Revolution information",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia Industrial Revolution page
    wikipedia_action_ok = (has_any_ci(answer, ['wikipedia']) and
                          has_any_ci(answer, ['industrial revolution']))
    evaluator.add_custom_node(
        result=bool(wikipedia_action_ok),
        id="wikipedia_action_navigate",
        desc="[Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia Industrial Revolution article",
        parent=wikipedia_node,
        critical=False
    )

    # Check if start/end years are present and reasonable
    years_ok = looks_like_year_range(ir_info.start_year, ir_info.end_year)
    evaluator.add_custom_node(
        result=bool(years_ok),
        id="wikipedia_years_extracted",
        desc="Extract start and end years of Industrial Revolution with reasonable values",
        parent=wikipedia_node,
        critical=False
    )

    # Check if place of origin is present
    place_ok = bool(ir_info.place_of_origin and ir_info.place_of_origin.strip())
    place_uk_ok = place_ok and has_any_ci(ir_info.place_of_origin, ['britain', 'united kingdom', 'england', 'uk'])
    evaluator.add_custom_node(
        result=bool(place_uk_ok),
        id="wikipedia_place_extracted",
        desc="Extract place of origin (Britain/UK)",
        parent=wikipedia_node,
        critical=False
    )

    # Check if Wikipedia link is provided (output o3)
    wiki_link_ok = looks_like_url(ir_info.wikipedia_link, 'wikipedia') or looks_like_url(answer, 'wikipedia')
    evaluator.add_custom_node(
        result=bool(wiki_link_ok),
        id="wikipedia_link_provided",
        desc="Provide Wikipedia link (o3)",
        parent=wikipedia_node,
        critical=False
    )

    # 3.2 Gapminder section
    gapminder_node = evaluator.add_sequential(
        id="gapminder_section",
        desc="Gapminder Tools data for United Kingdom",
        parent=root,
        critical=False
    )

    # [Action Node] gapminder.org:F4:A3 - Select United Kingdom
    gapminder_select_uk_ok = has_any_ci(answer, ['united kingdom', 'uk']) and has_any_ci(answer, ['gapminder'])
    evaluator.add_custom_node(
        result=bool(gapminder_select_uk_ok),
        id="gapminder_action_select_uk",
        desc="[Action Node] gapminder.org:F4:A3 - Select United Kingdom in Gapminder",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F1:A2 - Set y-axis to Income per person
    gapminder_income_axis_ok = has_any_ci(answer, ['income per person', 'gdp/capita', 'gdp per capita'])
    evaluator.add_custom_node(
        result=bool(gapminder_income_axis_ok),
        id="gapminder_action_income_axis",
        desc="[Action Node] gapminder.org:F1:A2 - Set y-axis to Income per person (GDP/capita)",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F1:A7 - Switch to Trends (line chart) mode
    gapminder_trends_ok = has_any_ci(answer, ['trends', 'line chart', 'line graph'])
    evaluator.add_custom_node(
        result=bool(gapminder_trends_ok),
        id="gapminder_action_trends_mode",
        desc="[Action Node] gapminder.org:F1:A7 - Switch to Trends (line chart) mode",
        parent=gapminder_node,
        critical=False
    )

    # [Perception Node] gapminder.org:F4:P1 - Identify year when income surpassed $3,000 (output o4)
    year_3k_ok = looks_like_income_takeoff_year(income_info.year_surpassed_3000)
    evaluator.add_custom_node(
        result=bool(year_3k_ok),
        id="gapminder_perception_year_3k",
        desc="[Perception Node] gapminder.org:F4:P1 - Identify year when UK income surpassed $3,000 (o4)",
        parent=gapminder_node,
        critical=False
    )

    # Extract most recent year income value (output o5)
    recent_income_ok = bool(income_info.most_recent_year_value and contains_digits(income_info.most_recent_year_value))
    evaluator.add_custom_node(
        result=bool(recent_income_ok),
        id="gapminder_recent_income_value",
        desc="Extract UK income per person for most recent complete year (o5)",
        parent=gapminder_node,
        critical=False
    )

    # Switch y-axis to Life expectancy
    gapminder_life_axis_ok = has_any_ci(answer, ['life expectancy'])
    evaluator.add_custom_node(
        result=bool(gapminder_life_axis_ok),
        id="gapminder_life_expectancy_axis",
        desc="Switch y-axis to Life expectancy",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F1:A6 - Hover to view specific year values
    gapminder_hover_ok = (has_any_ci(answer, ['1850']) and has_any_ci(answer, ['1950']))
    evaluator.add_custom_node(
        result=bool(gapminder_hover_ok),
        id="gapminder_action_hover",
        desc="[Action Node] gapminder.org:F1:A6 - Hover to view life expectancy values for 1850 and 1950",
        parent=gapminder_node,
        critical=False
    )

    # Extract life expectancy 1850 (output o6)
    life_1850_ok = looks_like_life_expectancy(life_expectancy_info.life_expectancy_1850)
    evaluator.add_custom_node(
        result=bool(life_1850_ok),
        id="gapminder_life_1850",
        desc="Extract UK life expectancy in 1850 (o6)",
        parent=gapminder_node,
        critical=False
    )

    # Extract life expectancy 1950 (output o7)
    life_1950_ok = looks_like_life_expectancy(life_expectancy_info.life_expectancy_1950)
    evaluator.add_custom_node(
        result=bool(life_1950_ok),
        id="gapminder_life_1950",
        desc="Extract UK life expectancy in 1950 (o7)",
        parent=gapminder_node,
        critical=False
    )

    # Check if Gapminder link is provided
    gapminder_link_ok = looks_like_url(income_info.gapminder_link, 'gapminder') or looks_like_url(answer, 'gapminder')
    evaluator.add_custom_node(
        result=bool(gapminder_link_ok),
        id="gapminder_link_provided",
        desc="Provide Gapminder Tools link",
        parent=gapminder_node,
        critical=False
    )

    # 3.3 Khan Academy section
    khan_node = evaluator.add_sequential(
        id="khan_academy_section",
        desc="Khan Academy Industrial Revolution content",
        parent=root,
        critical=False
    )

    # Search for Industrial Revolution
    khan_search_ok = has_any_ci(answer, ['khan academy']) and has_any_ci(answer, ['industrial revolution'])
    evaluator.add_custom_node(
        result=bool(khan_search_ok),
        id="khan_search",
        desc="Search for Industrial Revolution on Khan Academy",
        parent=khan_node,
        critical=False
    )

    # [Action Node] khanacademy.org:F1:A6 - Use filter/dropdown to find suitable content
    khan_filter_ok = (has_any_ci(answer, ['video', 'article']) or
                     has_any_ci(answer, ['ap', 'world history']))
    evaluator.add_custom_node(
        result=bool(khan_filter_ok),
        id="khan_action_filter",
        desc="[Action Node] khanacademy.org:F1:A6 - Use filter to find video or article",
        parent=khan_node,
        critical=False
    )

    # Check if content relates to social impact or living standards
    khan_social_ok = has_any_ci(answer, ['social impact', 'living standard', 'social', 'living'])
    evaluator.add_custom_node(
        result=bool(khan_social_ok),
        id="khan_social_living_content",
        desc="Content relates to social impact or living standards",
        parent=khan_node,
        critical=False
    )

    # [Perception Node] khanacademy.org:F1:P20 - Understand and extract unit name (output o9)
    unit_name_ok = bool(khan_info.unit_name and khan_info.unit_name.strip())
    evaluator.add_custom_node(
        result=bool(unit_name_ok),
        id="khan_perception_unit_name",
        desc="[Perception Node] khanacademy.org:F1:P20 - Confirm unit name (o9)",
        parent=khan_node,
        critical=False
    )

    # [Action Node] khanacademy.org:F3:A21 - Click into content item to view details
    khan_click_ok = bool(khan_info.content_title and khan_info.content_title.strip())
    evaluator.add_custom_node(
        result=bool(khan_click_ok),
        id="khan_action_click_content",
        desc="[Action Node] khanacademy.org:F3:A21 - Click into selected content (o8 title extracted)",
        parent=khan_node,
        critical=False
    )

    # Check if Khan Academy link is provided (output o10)
    khan_link_ok = looks_like_url(khan_info.khan_academy_link, 'khanacademy') or looks_like_url(answer, 'khanacademy')
    evaluator.add_custom_node(
        result=bool(khan_link_ok),
        id="khan_link_provided",
        desc="Provide Khan Academy content link (o10)",
        parent=khan_node,
        critical=False
    )

    # Check if content is suitable for AP History
    khan_ap_ok = has_any_ci(answer, ['ap', 'advanced placement']) or (unit_name_ok and has_any_ci(khan_info.unit_name, ['ap', 'world history']))
    evaluator.add_custom_node(
        result=bool(khan_ap_ok),
        id="khan_ap_suitable",
        desc="Content suitable for AP History course",
        parent=khan_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
