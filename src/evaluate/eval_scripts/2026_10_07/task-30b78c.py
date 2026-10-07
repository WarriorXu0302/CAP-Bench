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
TASK_ID = "task-30b78c"
TASK_DESCRIPTION = 'I am conducting research on Neural Radiance Fields (NeRF), and my advisor asked me to compile the latest NeRF-related work from CVPR 2024. First, go to OpenReview and search for papers in CVPR 2024 that contain the keywords “NeRF” or “Neural Radiance Fields.” Filter for papers with acceptance status **Oral** or **Spotlight**, and collect up to 5 papers (if fewer than 5 are available, record the actual number). For each paper, record the title, authors, and OpenReview link.  \n\nThen, use each paper’s title and first author name to search arXiv for the corresponding preprint. After finding a match, record the arXiv ID and PDF link.  \n\nFinally, go to Bilibili and search for explanatory videos using core keywords from each paper title (e.g., “神经辐射场” or key English title terms). Prefer one Chinese explanatory video per paper with duration over 10 minutes and more than 10,000 views. If no video for a paper satisfies both conditions, keep the core requirement of being a Chinese explanation video, select the most relevant one with relatively higher views or longer duration, and note which condition(s) are not met.  \n\nFor each paper, output: paper title, first author, acceptance type (Oral/Spotlight), OpenReview paper page link, arXiv ID, arXiv PDF link, Bilibili video title, video duration, view count, video link, and whether the video meets the criteria “duration ≥ 10 minutes and views ≥ 10,000.”'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information extracted for a single paper"""
    title: Optional[str] = None
    first_author: Optional[str] = None
    acceptance_type: Optional[str] = None
    openreview_link: Optional[str] = None
    arxiv_id: Optional[str] = None
    arxiv_pdf_link: Optional[str] = None
    bilibili_video_title: Optional[str] = None
    video_duration: Optional[str] = None
    view_count: Optional[str] = None
    bilibili_link: Optional[str] = None
    meets_criteria: Optional[str] = None


class AllPapersInfo(BaseModel):
    """Collection of all papers extracted from the answer"""
    papers: List[PaperInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_papers_from_answer() -> str:
    return """
Extract all NeRF-related papers from CVPR 2024 that the user reported in their answer.

For each paper, extract:
- title: the paper title
- first_author: the first author's name
- acceptance_type: "Oral" or "Spotlight"
- openreview_link: the OpenReview paper page URL
- arxiv_id: the arXiv ID (e.g., "2401.12345")
- arxiv_pdf_link: the arXiv PDF link
- bilibili_video_title: the Bilibili video title
- video_duration: the video duration as stated
- view_count: the view count as stated
- bilibili_link: the Bilibili video URL
- meets_criteria: whether the video meets "duration ≥ 10 minutes and views ≥ 10,000" (e.g., "Yes", "No", "No - duration < 10 min", etc.)

Also extract:
- total_count: the total number of papers collected

If any field is missing for a paper, set it to null.
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


def looks_like_openreview_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'openreview.net' in link.lower()


def looks_like_arxiv_id(arxiv_id: Optional[str]) -> bool:
    if not arxiv_id:
        return False
    # ArXiv IDs typically look like "2401.12345" or "arXiv:2401.12345"
    return bool(re.search(r'\d{4}\.\d{4,5}', arxiv_id))


def looks_like_arxiv_pdf_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'arxiv.org/pdf/' in link.lower()


def looks_like_bilibili_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'bilibili.com' in link.lower()


def is_oral_or_spotlight(acceptance_type: Optional[str]) -> bool:
    if not acceptance_type:
        return False
    return acceptance_type.lower() in ['oral', 'spotlight']


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove commas and try to extract first number
    cleaned = text.replace(',', '')
    m = re.findall(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def check_video_duration_minutes(duration_text: Optional[str]) -> Optional[float]:
    """Extract duration in minutes from text like '15:30' or '15 minutes'"""
    if not duration_text:
        return None

    # Try MM:SS format
    time_match = re.search(r'(\d+):(\d+)', duration_text)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        return minutes + seconds / 60.0

    # Try number followed by 'minute' or 'min'
    min_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minute|min)', duration_text.lower())
    if min_match:
        return float(min_match.group(1))

    return None


def check_view_count_number(view_text: Optional[str]) -> Optional[float]:
    """Extract view count number from text"""
    if not view_text:
        return None

    # Handle formats like "1.2万", "10,000", "15000"
    if '万' in view_text:
        num_match = re.search(r'(\d+(?:\.\d+)?)\s*万', view_text)
        if num_match:
            return float(num_match.group(1)) * 10000

    return extract_number(view_text)


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
    papers_info = await evaluator.extract(
        prompt=prompt_extract_papers_from_answer(),
        template_class=AllPapersInfo,
        extraction_name="all_papers_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Overall collection check
    collection_node = evaluator.add_sequential(
        id="overall_collection",
        desc="Overall paper collection and processing",
        parent=root,
        critical=False
    )

    # Check if papers were collected
    has_papers = papers_info and papers_info.papers and len(papers_info.papers) > 0
    evaluator.add_custom_node(
        result=bool(has_papers),
        id="has_collected_papers",
        desc="Successfully collected at least one NeRF paper from CVPR 2024",
        parent=collection_node,
        critical=False
    )

    # Check if mentions CVPR 2024 and NeRF keywords
    mentions_cvpr_2024 = has_any_ci(answer, ['cvpr 2024', 'cvpr2024'])
    mentions_nerf = has_any_ci(answer, ['nerf', 'neural radiance field'])
    evaluator.add_custom_node(
        result=bool(mentions_cvpr_2024 and mentions_nerf),
        id="mentions_task_keywords",
        desc="Answer mentions CVPR 2024 and NeRF/Neural Radiance Fields",
        parent=collection_node,
        critical=False
    )

    # 3.2 Per-paper evaluation
    if has_papers:
        for idx, paper in enumerate(papers_info.papers):
            paper_node = evaluator.add_sequential(
                id=f"paper_{idx+1}",
                desc=f"Paper {idx+1}: {paper.title if paper.title else 'Unknown'}",
                parent=root,
                critical=False
            )

            # 3.2.1 OpenReview section
            openreview_node = evaluator.add_sequential(
                id=f"paper_{idx+1}_openreview",
                desc="OpenReview information extraction",
                parent=paper_node,
                critical=False
            )

            # [Perception Node] openreview.net:F1:P2 - Extract paper metadata
            has_title = bool(paper.title and paper.title.strip())
            has_author = bool(paper.first_author and paper.first_author.strip())
            evaluator.add_custom_node(
                result=bool(has_title and has_author),
                id=f"paper_{idx+1}_openreview_metadata",
                desc="[Perception Node] openreview.net:F1:P2 - Extract paper title and first author from OpenReview",
                parent=openreview_node,
                critical=False
            )

            # [Perception Node] openreview.net:F1:P1 - Identify acceptance status
            has_valid_acceptance = is_oral_or_spotlight(paper.acceptance_type)
            evaluator.add_custom_node(
                result=bool(has_valid_acceptance),
                id=f"paper_{idx+1}_acceptance_status",
                desc="[Perception Node] openreview.net:F1:P1 - Identify acceptance status as Oral or Spotlight",
                parent=openreview_node,
                critical=False
            )

            # OpenReview link check
            has_openreview_link = looks_like_openreview_link(paper.openreview_link)
            evaluator.add_custom_node(
                result=bool(has_openreview_link),
                id=f"paper_{idx+1}_openreview_link",
                desc="Recorded valid OpenReview paper page link",
                parent=openreview_node,
                critical=False
            )

            # 3.2.2 ArXiv section
            arxiv_node = evaluator.add_sequential(
                id=f"paper_{idx+1}_arxiv",
                desc="ArXiv preprint matching and information extraction",
                parent=paper_node,
                critical=False
            )

            # [Action Node] arxiv.org:F1:A2 - Search submission (form fill and submit)
            arxiv_search_indicators = (
                has_any_ci(answer, ['arxiv']) and
                (has_title or has_author)
            )
            evaluator.add_custom_node(
                result=bool(arxiv_search_indicators),
                id=f"paper_{idx+1}_arxiv_search",
                desc="[Action Node] arxiv.org:F1:A2 - Search arXiv using paper title and first author",
                parent=arxiv_node,
                critical=False
            )

            # [Action Node] arxiv.org:F1:A1 - Search scope selection
            # Implicit: selecting appropriate search fields (title/author)
            has_arxiv_id = looks_like_arxiv_id(paper.arxiv_id)
            evaluator.add_custom_node(
                result=bool(has_arxiv_id),
                id=f"paper_{idx+1}_arxiv_scope",
                desc="[Action Node] arxiv.org:F1:A1 - Selected appropriate search scope to find matching preprint",
                parent=arxiv_node,
                critical=False
            )

            # [Perception Node] arxiv.org:F1:P11 - Search result relevance judgment
            # [Perception Node] arxiv.org:F3:P5 - Paper core information understanding
            # Combined: matching the correct paper
            arxiv_match_ok = has_arxiv_id and has_title
            evaluator.add_custom_node(
                result=bool(arxiv_match_ok),
                id=f"paper_{idx+1}_arxiv_match",
                desc="[Perception Node] arxiv.org:F1:P11 & arxiv.org:F3:P5 - Correctly matched arXiv preprint with OpenReview paper",
                parent=arxiv_node,
                critical=False
            )

            # [Action Node] arxiv.org:F3:A6 - Click into detail page
            # [Action Node] arxiv.org:F4:A7 - Get PDF download link
            has_pdf_link = looks_like_arxiv_pdf_link(paper.arxiv_pdf_link)
            evaluator.add_custom_node(
                result=bool(has_pdf_link),
                id=f"paper_{idx+1}_arxiv_pdf",
                desc="[Action Node] arxiv.org:F3:A6 & arxiv.org:F4:A7 - Navigate to paper detail and extract PDF link",
                parent=arxiv_node,
                critical=False
            )

            # 3.2.3 Bilibili section
            bilibili_node = evaluator.add_sequential(
                id=f"paper_{idx+1}_bilibili",
                desc="Bilibili explanatory video search and filtering",
                parent=paper_node,
                critical=False
            )

            # [Perception Node] bilibili.com:F1:P30 - Content relevance recognition
            has_video_title = bool(paper.bilibili_video_title and paper.bilibili_video_title.strip())
            has_bilibili_link = looks_like_bilibili_link(paper.bilibili_link)
            evaluator.add_custom_node(
                result=bool(has_video_title and has_bilibili_link),
                id=f"paper_{idx+1}_bilibili_relevance",
                desc="[Perception Node] bilibili.com:F1:P30 - Identified relevant explanatory video for the paper",
                parent=bilibili_node,
                critical=False
            )

            # [Action Node] bilibili.com:F1:A5 - Sort switching (by view count)
            has_view_count = bool(paper.view_count and contains_digits(paper.view_count))
            evaluator.add_custom_node(
                result=bool(has_view_count),
                id=f"paper_{idx+1}_bilibili_views",
                desc="[Action Node] bilibili.com:F1:A5 - Extracted video view count (implies sorting/filtering capability)",
                parent=bilibili_node,
                critical=False
            )

            # Check duration and view count criteria
            duration_minutes = check_video_duration_minutes(paper.video_duration)
            view_count_num = check_view_count_number(paper.view_count)

            duration_ok = duration_minutes is not None and duration_minutes >= 10
            views_ok = view_count_num is not None and view_count_num >= 10000

            # [Action Node] bilibili.com:F1:A12 - Page turning operation (to find qualifying videos)
            criteria_met = duration_ok and views_ok
            evaluator.add_custom_node(
                result=bool(criteria_met or has_bilibili_link),
                id=f"paper_{idx+1}_bilibili_filtering",
                desc="[Action Node] bilibili.com:F1:A12 - Filtered videos by duration (≥10 min) and views (≥10k), or found best alternative",
                parent=bilibili_node,
                critical=False
            )

            # Check if criteria compliance is noted
            has_criteria_note = bool(paper.meets_criteria and paper.meets_criteria.strip())
            evaluator.add_custom_node(
                result=bool(has_criteria_note),
                id=f"paper_{idx+1}_criteria_noted",
                desc="Documented whether video meets duration and view count criteria",
                parent=bilibili_node,
                critical=False
            )

    # 3.3 Check pagination actions (implicit in collecting multiple papers)
    # [Action Node] openreview.net:F1:A1 - Page turning
    multiple_papers = has_papers and len(papers_info.papers) >= 2
    evaluator.add_custom_node(
        result=bool(multiple_papers),
        id="openreview_pagination",
        desc="[Action Node] openreview.net:F1:A1 - Successfully collected multiple papers (implies pagination if needed)",
        parent=collection_node,
        critical=False
    )

    # Check if attempted to collect up to 5 papers
    attempted_full_collection = papers_info and papers_info.total_count is not None
    if attempted_full_collection:
        collected_count = papers_info.total_count if papers_info.total_count else len(papers_info.papers)
        reasonable_count = 1 <= collected_count <= 5
        evaluator.add_custom_node(
            result=bool(reasonable_count),
            id="collection_count_appropriate",
            desc="Collected appropriate number of papers (1-5 as requested)",
            parent=collection_node,
            critical=False
        )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
