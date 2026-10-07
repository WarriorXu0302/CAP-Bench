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
TASK_ID = "task-26914d"
TASK_DESCRIPTION = "I've recently developed a keen interest in mythological motifs on ancient Greek pottery and would like to study them systematically.\n\nFirst, please go to the Getty Museum website and find three examples of 5th-4th century BCE Greek red-figure pottery. Ensure that the description for each piece explicitly mentions specific Greek mythological figures (e.g., Athena, Heracles, Achilles, etc.). Record the name of the pottery, the mythological figure(s) depicted, the date, and the link to the Getty detail page.\n\nNext, use the names of these three mythological figures to search on Google Scholar. For each figure, find two English academic papers published since 2020 with more than 10 citations, whose themes are related to Greek pottery art or mythological iconography.\n\nFinally, go to YouTube and search for explanatory videos using the English name of each mythological figure. Filter for educational content between 10-30 minutes in length with over 5,000 views. Find the single most relevant video for each figure, and expand its description to confirm whether it mentions ancient Greek art or pottery.\n\n**Output:**\n*   **For each piece of pottery:** name, date, mythological figure(s) depicted, Getty link.\n*   **For each paper:** title, author(s), publication year, citation count, Scholar link.\n*   **For each video:** title, duration, view count, channel name, a summary of the video description (highlighting whether it mentions pottery or art), and YouTube link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PotteryItem(BaseModel):
    """Single pottery piece from Getty Museum"""
    name: Optional[str] = None
    date: Optional[str] = None
    mythological_figures: Optional[List[str]] = Field(default_factory=list)
    getty_link: Optional[str] = None


class GettyPotteryCollection(BaseModel):
    """Collection of three pottery pieces from Getty Museum"""
    pottery_items: List[PotteryItem] = Field(default_factory=list)


class ScholarPaper(BaseModel):
    """Single academic paper from Google Scholar"""
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    scholar_link: Optional[str] = None
    related_figure: Optional[str] = None


class ScholarPapersCollection(BaseModel):
    """Collection of academic papers (2 per figure, 6 total)"""
    papers: List[ScholarPaper] = Field(default_factory=list)


class YouTubeVideo(BaseModel):
    """Single YouTube video"""
    title: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None
    channel_name: Optional[str] = None
    description_summary: Optional[str] = None
    youtube_link: Optional[str] = None
    related_figure: Optional[str] = None


class YouTubeVideosCollection(BaseModel):
    """Collection of YouTube videos (1 per figure, 3 total)"""
    videos: List[YouTubeVideo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_getty_pottery() -> str:
    return """
Extract all Getty Museum pottery pieces mentioned in the answer.

For each pottery piece, extract:
- name: the name of the pottery piece
- date: the date or date range (e.g., "5th century BCE", "450-400 BCE")
- mythological_figures: list of specific Greek mythological figures mentioned (e.g., ["Athena", "Heracles"])
- getty_link: the Getty Museum detail page URL

Return a list of pottery items. If fewer than 3 are found, return what is available.
"""


def prompt_extract_scholar_papers() -> str:
    return """
Extract all Google Scholar academic papers mentioned in the answer.

For each paper, extract:
- title: the paper title
- authors: author name(s)
- year: publication year as an integer
- citations: citation count as an integer
- scholar_link: the Google Scholar link
- related_figure: which mythological figure this paper is associated with

Return a list of papers. If fewer than 6 are found, return what is available.
"""


def prompt_extract_youtube_videos() -> str:
    return """
Extract all YouTube videos mentioned in the answer.

For each video, extract:
- title: video title
- duration: video duration (e.g., "15:30", "20 minutes")
- view_count: view count (e.g., "10,000 views", "15K")
- channel_name: channel name
- description_summary: summary of the video description, particularly noting whether it mentions ancient Greek art or pottery
- youtube_link: the YouTube video URL
- related_figure: which mythological figure this video is associated with

Return a list of videos. If fewer than 3 are found, return what is available.
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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(\d+)', str(text))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def looks_like_greek_pottery_type(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['red-figure', 'red figure', 'krater', 'amphora', 'kylix', 'hydria', 'lekythos', 'pelike']
    return has_any_ci(text, keywords)


def looks_like_5th_4th_century_bce(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'5th\s+century\s+bce',
        r'4th\s+century\s+bce',
        r'5th-4th\s+century',
        r'[45]\d{2}\s*-?\s*[45]?\d{2}\s+bce',
        r'circa\s+[45]\d{2}'
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def looks_like_greek_mythological_figure(text: Optional[str]) -> bool:
    if not text:
        return False
    common_figures = ['athena', 'heracles', 'hercules', 'achilles', 'apollo', 'artemis', 'zeus',
                      'hera', 'poseidon', 'aphrodite', 'dionysus', 'hades', 'persephone',
                      'odysseus', 'theseus', 'perseus', 'medusa', 'ajax', 'hector']
    return has_any_ci(text, common_figures)


def looks_like_pottery_art_theme(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['pottery', 'ceramic', 'vase', 'iconography', 'greek art', 'attic', 'red-figure', 'black-figure']
    return has_any_ci(text, keywords)


def is_year_since_2020(year: Optional[int]) -> bool:
    if year is None:
        return False
    return year >= 2020


def is_citations_over_10(citations: Optional[int]) -> bool:
    if citations is None:
        return False
    return citations > 10


def parse_duration_minutes(duration_text: Optional[str]) -> Optional[int]:
    if not duration_text:
        return None
    # Try to extract minutes from formats like "15:30", "20 minutes", "25min"
    m = re.search(r'(\d+):(\d+)', duration_text)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*min', duration_text.lower())
    if m:
        return int(m.group(1))
    return None


def is_duration_10_to_30_min(duration_text: Optional[str]) -> bool:
    minutes = parse_duration_minutes(duration_text)
    if minutes is None:
        return False
    return 10 <= minutes <= 30


def parse_view_count(view_text: Optional[str]) -> Optional[int]:
    if not view_text:
        return None
    text = view_text.lower().replace(',', '').replace(' ', '')
    # Handle formats like "10K", "5.5K", "10,000"
    if 'k' in text:
        m = re.search(r'(\d+\.?\d*)\s*k', text)
        if m:
            return int(float(m.group(1)) * 1000)
    m = re.search(r'(\d+)', text)
    if m:
        return int(m.group(1))
    return None


def is_view_count_over_5000(view_text: Optional[str]) -> bool:
    count = parse_view_count(view_text)
    if count is None:
        return False
    return count > 5000


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
    getty_pottery = await evaluator.extract(
        prompt=prompt_extract_getty_pottery(),
        template_class=GettyPotteryCollection,
        extraction_name="getty_pottery_collection"
    )

    scholar_papers = await evaluator.extract(
        prompt=prompt_extract_scholar_papers(),
        template_class=ScholarPapersCollection,
        extraction_name="scholar_papers_collection"
    )

    youtube_videos = await evaluator.extract(
        prompt=prompt_extract_youtube_videos(),
        template_class=YouTubeVideosCollection,
        extraction_name="youtube_videos_collection"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Getty Museum section
    getty_node = evaluator.add_sequential(
        id="getty_museum_section",
        desc="Getty Museum pottery collection - 3 pieces of 5th-4th century BCE Greek red-figure pottery",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F1:A2 - Hover to expand submenu
    getty_navigation_ok = has_any_ci(answer, ['getty', 'getty museum']) and len(getty_pottery.pottery_items) > 0
    evaluator.add_custom_node(
        result=bool(getty_navigation_ok),
        id="getty_hover_submenu",
        desc="[Action Node] getty.edu:F1:A2 - Navigate Getty Museum website and access collection browsing functionality",
        parent=getty_node,
        critical=False
    )

    # Check if we have 3 pottery items
    has_three_pottery = len(getty_pottery.pottery_items) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_pottery),
        id="getty_three_items",
        desc="Found three pottery pieces as requested",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P1 - Object recognition (red-figure pottery type)
    pottery_type_ok = False
    if getty_pottery.pottery_items:
        pottery_type_ok = any(
            looks_like_greek_pottery_type(item.name) or looks_like_greek_pottery_type(str(item.date))
            for item in getty_pottery.pottery_items
        )
    evaluator.add_custom_node(
        result=bool(pottery_type_ok),
        id="getty_object_recognition",
        desc="[Perception Node] getty.edu:F1:P1 - Identify Greek red-figure pottery type from images and descriptions",
        parent=getty_node,
        critical=False
    )

    # Check date range (5th-4th century BCE)
    date_range_ok = False
    if getty_pottery.pottery_items:
        date_range_ok = any(
            looks_like_5th_4th_century_bce(item.date)
            for item in getty_pottery.pottery_items
        )
    evaluator.add_custom_node(
        result=bool(date_range_ok),
        id="getty_date_range",
        desc="Pottery pieces are from 5th-4th century BCE period",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P3 - Feature extraction (mythological figures)
    mythological_figures_ok = False
    extracted_figures = []
    if getty_pottery.pottery_items:
        for item in getty_pottery.pottery_items:
            if item.mythological_figures:
                for fig in item.mythological_figures:
                    if looks_like_greek_mythological_figure(fig):
                        extracted_figures.append(fig)
                        mythological_figures_ok = True

    evaluator.add_custom_node(
        result=bool(mythological_figures_ok),
        id="getty_feature_extraction",
        desc="[Perception Node] getty.edu:F1:P3 - Extract specific Greek mythological figures from pottery descriptions",
        parent=getty_node,
        critical=False
    )

    # Check Getty links
    getty_links_ok = False
    if getty_pottery.pottery_items:
        getty_links_ok = any(
            looks_like_url(item.getty_link, 'getty.edu')
            for item in getty_pottery.pottery_items
        )
    evaluator.add_custom_node(
        result=bool(getty_links_ok),
        id="getty_links_present",
        desc="Getty Museum detail page links are provided",
        parent=getty_node,
        critical=False
    )

    # 3.2 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="google_scholar_section",
        desc="Google Scholar papers - 2 papers per mythological figure (6 total)",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A1 - Search form filling
    scholar_search_ok = has_any_ci(answer, ['google scholar', 'scholar']) and len(scholar_papers.papers) > 0
    evaluator.add_custom_node(
        result=bool(scholar_search_ok),
        id="scholar_search_form",
        desc="[Action Node] scholar.google.com:F1:A1 - Use Google Scholar search with mythological figure names",
        parent=scholar_node,
        critical=False
    )

    # Check if we have 6 papers (2 per figure)
    has_six_papers = len(scholar_papers.papers) >= 6
    evaluator.add_custom_node(
        result=bool(has_six_papers),
        id="scholar_six_papers",
        desc="Found six papers as requested (2 per mythological figure)",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Content understanding (pottery art or iconography theme)
    pottery_theme_ok = False
    if scholar_papers.papers:
        pottery_theme_ok = any(
            looks_like_pottery_art_theme(paper.title)
            for paper in scholar_papers.papers
        )
    evaluator.add_custom_node(
        result=bool(pottery_theme_ok),
        id="scholar_content_understanding",
        desc="[Perception Node] scholar.google.com:F1:P1 - Papers are related to Greek pottery art or mythological iconography",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A6 - Time filter dropdown (since 2020)
    year_filter_ok = False
    if scholar_papers.papers:
        year_filter_ok = all(
            is_year_since_2020(paper.year)
            for paper in scholar_papers.papers if paper.year is not None
        )
    evaluator.add_custom_node(
        result=bool(year_filter_ok),
        id="scholar_time_filter",
        desc="[Action Node] scholar.google.com:F3:A6 - Papers are published since 2020 (time filter applied)",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A7 - Sort by citations (over 10)
    citation_filter_ok = False
    if scholar_papers.papers:
        citation_filter_ok = all(
            is_citations_over_10(paper.citations)
            for paper in scholar_papers.papers if paper.citations is not None
        )
    evaluator.add_custom_node(
        result=bool(citation_filter_ok),
        id="scholar_citation_sort",
        desc="[Action Node] scholar.google.com:F3:A7 - Papers have more than 10 citations (citation sorting/filtering applied)",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A15 - Page browsing
    multiple_searches_ok = len(scholar_papers.papers) >= 2
    evaluator.add_custom_node(
        result=bool(multiple_searches_ok),
        id="scholar_page_browsing",
        desc="[Action Node] scholar.google.com:F3:A15 - Browse multiple search result pages to find suitable papers",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F10:A8 - Click to enter detail page
    scholar_links_ok = False
    if scholar_papers.papers:
        scholar_links_ok = any(
            looks_like_url(paper.scholar_link, 'scholar.google')
            for paper in scholar_papers.papers
        )
    evaluator.add_custom_node(
        result=bool(scholar_links_ok),
        id="scholar_detail_page",
        desc="[Action Node] scholar.google.com:F10:A8 - Access paper detail pages to get complete information and Scholar links",
        parent=scholar_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube videos - 1 educational video per mythological figure (3 total)",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video card
    youtube_search_ok = has_any_ci(answer, ['youtube']) and len(youtube_videos.videos) > 0
    evaluator.add_custom_node(
        result=bool(youtube_search_ok),
        id="youtube_click_video",
        desc="[Action Node] youtube.com:F1:A22 - Search YouTube with mythological figure names and click video cards to access playback pages",
        parent=youtube_node,
        critical=False
    )

    # Check if we have 3 videos (1 per figure)
    has_three_videos = len(youtube_videos.videos) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_videos),
        id="youtube_three_videos",
        desc="Found three videos as requested (1 per mythological figure)",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more results
    # Inferred from finding videos that meet specific criteria (duration, views)
    scroll_load_ok = len(youtube_videos.videos) > 0
    evaluator.add_custom_node(
        result=bool(scroll_load_ok),
        id="youtube_scroll_load",
        desc="[Action Node] youtube.com:F1:A69 - Scroll to load additional search results to find videos meeting criteria",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Image understanding (thumbnail relevance)
    video_relevance_ok = False
    if youtube_videos.videos:
        video_relevance_ok = any(
            looks_like_greek_mythological_figure(video.title) or
            (video.related_figure and looks_like_greek_mythological_figure(video.related_figure))
            for video in youtube_videos.videos
        )
    evaluator.add_custom_node(
        result=bool(video_relevance_ok),
        id="youtube_image_understanding",
        desc="[Perception Node] youtube.com:F1:P4 - Identify video thumbnails and titles relevant to mythological figures",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Filter dropdown (duration 10-30 minutes)
    duration_filter_ok = False
    if youtube_videos.videos:
        duration_filter_ok = all(
            is_duration_10_to_30_min(video.duration)
            for video in youtube_videos.videos if video.duration
        )
    evaluator.add_custom_node(
        result=bool(duration_filter_ok),
        id="youtube_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Apply duration filter for videos between 10-30 minutes",
        parent=youtube_node,
        critical=False
    )

    # Check view count (over 5,000)
    view_count_ok = False
    if youtube_videos.videos:
        view_count_ok = all(
            is_view_count_over_5000(video.view_count)
            for video in youtube_videos.videos if video.view_count
        )
    evaluator.add_custom_node(
        result=bool(view_count_ok),
        id="youtube_view_count",
        desc="Videos have over 5,000 views as required",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A45 - Expand video description
    description_expanded_ok = False
    if youtube_videos.videos:
        description_expanded_ok = any(
            video.description_summary and len(video.description_summary) > 20
            for video in youtube_videos.videos
        )
    evaluator.add_custom_node(
        result=bool(description_expanded_ok),
        id="youtube_expand_description",
        desc="[Action Node] youtube.com:F2:A45 - Expand video descriptions to view full content",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F2:P20 - State awareness (description expanded/collapsed)
    # Check if descriptions mention pottery or Greek art
    description_content_ok = False
    if youtube_videos.videos:
        description_content_ok = any(
            video.description_summary and (
                has_any_ci(video.description_summary, ['pottery', 'greek art', 'ancient art', 'vase', 'ceramic'])
            )
            for video in youtube_videos.videos
        )
    evaluator.add_custom_node(
        result=bool(description_content_ok),
        id="youtube_state_awareness",
        desc="[Perception Node] youtube.com:F2:P20 - Verify expanded descriptions and confirm mentions of ancient Greek art or pottery",
        parent=youtube_node,
        critical=False
    )

    # Check YouTube links
    youtube_links_ok = False
    if youtube_videos.videos:
        youtube_links_ok = any(
            looks_like_url(video.youtube_link, 'youtube.com') or looks_like_url(video.youtube_link, 'youtu.be')
            for video in youtube_videos.videos
        )
    evaluator.add_custom_node(
        result=bool(youtube_links_ok),
        id="youtube_links_present",
        desc="YouTube video links are provided",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
