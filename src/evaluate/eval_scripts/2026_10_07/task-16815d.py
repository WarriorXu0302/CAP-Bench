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
TASK_ID = "task-16815d"
TASK_DESCRIPTION = 'I am a film student in the directing department, currently researching the visual style of Film Noir. Please help me compile a shot-by-shot analysis research index.\n\nFirst, go to the official Criterion website, filter for films categorized as "Film Noir," and ensure they have a Blu-ray version. From these, select 3 classic representative works released between the 1940s and 1950s (preferably by renowned directors).\n\nThen, go to IMDb to find the technical specifications for these 3 films. Specifically, I need their "Aspect Ratio" and "Negative Format."\n\nFinally, go to Bilibili and search for in-depth shot-by-shot analysis videos of these 3 films. Find high-quality videos with over 10,000 views and titles that include "拉片" (shot-by-shot analysis) or "解析" (analysis/breakdown).\n\nOutput: Film Title, Director, Year, Criterion Link, IMDb Aspect Ratio, IMDb Negative Format, Bilibili Analysis Video Title, Views, and Link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmEntry(BaseModel):
    """Single film entry with all required information"""
    title: Optional[str] = None
    director: Optional[str] = None
    year: Optional[str] = None
    criterion_link: Optional[str] = None
    aspect_ratio: Optional[str] = None
    negative_format: Optional[str] = None
    bilibili_title: Optional[str] = None
    bilibili_views: Optional[str] = None
    bilibili_link: Optional[str] = None


class FilmsCollection(BaseModel):
    """Collection of all three films"""
    film1: Optional[FilmEntry] = None
    film2: Optional[FilmEntry] = None
    film3: Optional[FilmEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_films_from_answer() -> str:
    return """
Extract information about the 3 Film Noir films from the answer.

For each film, extract:
- title: the film title
- director: the director name
- year: the release year
- criterion_link: the Criterion website URL for this film
- aspect_ratio: the aspect ratio from IMDb Technical Specs
- negative_format: the negative format from IMDb Technical Specs
- bilibili_title: the title of the Bilibili analysis video
- bilibili_views: the view count of the Bilibili video
- bilibili_link: the URL to the Bilibili video

Return three film entries as film1, film2, and film3. If any field is missing, set it to null.
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
    m = re.search(r'(19[4-5]\d)', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_1940s_1950s(year_text: Optional[str]) -> bool:
    year = extract_year(year_text)
    if year is None:
        return False
    return 1940 <= year <= 1959


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def extract_views_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Handle Chinese number formats like "1.5万" or English like "15000"
    m = re.search(r'(\d+\.?\d*)\s*万', text)
    if m:
        try:
            return int(float(m.group(1)) * 10000)
        except Exception:
            pass
    m = re.search(r'(\d+)', text.replace(',', ''))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def views_over_10k(views_text: Optional[str]) -> bool:
    views = extract_views_number(views_text)
    if views is None:
        return False
    return views > 10000


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
    films = await evaluator.extract(
        prompt=prompt_extract_films_from_answer(),
        template_class=FilmsCollection,
        extraction_name="films_collection"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Criterion.com section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion website film selection with filters",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A1 - Select Blu-ray format
    bluray_mention = has_any_ci(answer, ['blu-ray', 'blu ray', 'bluray'])
    evaluator.add_custom_node(
        result=bool(bluray_mention),
        id="criterion_action_bluray",
        desc="[Action Node] criterion.com:F1:A1 - Filter for Blu-ray format availability",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A2 - Filter by Film Noir genre
    noir_mention = has_any_ci(answer, ['film noir', 'noir'])
    evaluator.add_custom_node(
        result=bool(noir_mention),
        id="criterion_action_noir",
        desc="[Action Node] criterion.com:F1:A2 - Filter for Film Noir genre",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A5 - Expand decades panel
    decades_mention = has_any_ci(answer, ['1940', '1950', 'decade', '40s', '50s'])
    evaluator.add_custom_node(
        result=bool(decades_mention),
        id="criterion_action_decades_expand",
        desc="[Action Node] criterion.com:F1:A5 - Expand the Decades filter panel",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A8 - Select 1940s and 1950s decades
    all_films = [films.film1, films.film2, films.film3]
    years_ok = sum(1 for f in all_films if f and is_1940s_1950s(f.year)) >= 3
    evaluator.add_custom_node(
        result=bool(years_ok),
        id="criterion_action_decades_select",
        desc="[Action Node] criterion.com:F1:A8 - Select 1940s and 1950s decade filters",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F1:P2 - Identify Blu-ray format tags
    criterion_links_ok = sum(1 for f in all_films if f and f.criterion_link and looks_like_url(f.criterion_link) and 'criterion' in f.criterion_link.lower()) >= 3
    evaluator.add_custom_node(
        result=bool(criterion_links_ok),
        id="criterion_perception_bluray",
        desc="[Perception Node] criterion.com:F1:P2 - Verify Blu-ray availability for selected films",
        parent=criterion_node,
        critical=False
    )

    # 3.2 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb technical specifications lookup",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Use search dropdown to find titles
    imdb_mention = has_any_ci(answer, ['imdb'])
    titles_present = sum(1 for f in all_films if f and f.title) >= 3
    evaluator.add_custom_node(
        result=bool(imdb_mention and titles_present),
        id="imdb_action_search",
        desc="[Action Node] imdb.com:F1:A1 - Search for film titles using IMDb search (preferably Titles category)",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A34 - Expand/navigate to Technical Specs
    tech_specs_mention = has_any_ci(answer, ['technical spec', 'tech spec', 'aspect ratio', 'negative format'])
    evaluator.add_custom_node(
        result=bool(tech_specs_mention),
        id="imdb_action_tech_specs",
        desc="[Action Node] imdb.com:F3:A34 - Navigate to or expand Technical Specs section",
        parent=imdb_node,
        critical=False
    )

    # Check aspect ratio extraction
    aspect_ratios_ok = sum(1 for f in all_films if f and f.aspect_ratio and contains_digits(f.aspect_ratio)) >= 3
    evaluator.add_custom_node(
        result=bool(aspect_ratios_ok),
        id="imdb_extract_aspect_ratio",
        desc="Extract Aspect Ratio from IMDb Technical Specs for all 3 films",
        parent=imdb_node,
        critical=False
    )

    # Check negative format extraction
    negative_formats_ok = sum(1 for f in all_films if f and f.negative_format and len(f.negative_format.strip()) > 0) >= 3
    evaluator.add_custom_node(
        result=bool(negative_formats_ok),
        id="imdb_extract_negative_format",
        desc="Extract Negative Format from IMDb Technical Specs for all 3 films",
        parent=imdb_node,
        critical=False
    )

    # 3.3 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili shot-by-shot analysis video search",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by view count
    sort_mention = has_any_ci(answer, ['最多播放', 'sort', 'view', '播放'])
    evaluator.add_custom_node(
        result=bool(sort_mention),
        id="bilibili_action_sort",
        desc="[Action Node] bilibili.com:F1:A5 - Sort search results by view count (最多播放)",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P13 - Identify video titles with keywords
    titles_with_keywords = 0
    for f in all_films:
        if f and f.bilibili_title:
            if '拉片' in f.bilibili_title or '解析' in f.bilibili_title:
                titles_with_keywords += 1

    evaluator.add_custom_node(
        result=bool(titles_with_keywords >= 3),
        id="bilibili_perception_title_keywords",
        desc="[Perception Node] bilibili.com:F1:P13 - Verify video titles contain '拉片' or '解析' keywords",
        parent=bilibili_node,
        critical=False
    )

    # Check view count > 10,000
    views_ok = sum(1 for f in all_films if f and views_over_10k(f.bilibili_views)) >= 3
    evaluator.add_custom_node(
        result=bool(views_ok),
        id="bilibili_verify_views",
        desc="Verify all selected videos have over 10,000 views",
        parent=bilibili_node,
        critical=False
    )

    # Check Bilibili links
    bilibili_links_ok = sum(1 for f in all_films if f and f.bilibili_link and looks_like_url(f.bilibili_link) and 'bilibili' in f.bilibili_link.lower()) >= 3
    evaluator.add_custom_node(
        result=bool(bilibili_links_ok),
        id="bilibili_verify_links",
        desc="Verify Bilibili video links are provided for all 3 films",
        parent=bilibili_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall output completeness verification",
        parent=root,
        critical=False
    )

    # All 3 films have titles
    all_titles = sum(1 for f in all_films if f and f.title) >= 3
    evaluator.add_custom_node(
        result=bool(all_titles),
        id="completeness_titles",
        desc="All 3 films have titles provided",
        parent=completeness_node,
        critical=False
    )

    # All 3 films have directors
    all_directors = sum(1 for f in all_films if f and f.director) >= 3
    evaluator.add_custom_node(
        result=bool(all_directors),
        id="completeness_directors",
        desc="All 3 films have directors provided",
        parent=completeness_node,
        critical=False
    )

    # All 3 films have years in correct range
    evaluator.add_custom_node(
        result=bool(years_ok),
        id="completeness_years",
        desc="All 3 films have years in 1940s-1950s range",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
