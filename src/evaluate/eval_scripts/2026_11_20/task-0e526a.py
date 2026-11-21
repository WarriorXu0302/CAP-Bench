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
TASK_ID = "task-0e526a"
TASK_DESCRIPTION = 'I want to reproduce image segmentation experiments. First, go to Semantic Scholar and search for image segmentation papers published since 2024 in this field. Identify up to 5 papers with relatively high citation counts, and keep only those with downloadable PDFs. Prioritize papers whose PDFs are directly accessible on a public webpage; if fewer than 5 meet this condition, proceed with however many are actually found.\n\nThen, pick one of these papers and go to GitHub. Search by the paper title or the first author’s name to find a related code repository (official or community implementation). Prefer repositories with more than 500 stars. Verify that the README cites the paper and that the repository includes a `requirements.txt` file (if not, move to the next paper until one is found).\n\nNext, go to Hugging Face and search for the paper title. Find up to 3 pretrained models with relatively high download counts, and check each Model Card to confirm the model actually corresponds to that paper; if fewer than 3 are available, record the actual number found.\n\nFinally, go to YouTube and search by the paper title or first author. Find in-depth explainer videos longer than 10 minutes with more than 5,000 views. Expand the video description and confirm it mentions the paper title or author. If too few videos strictly meet all conditions, keep videos with duration ≥10 minutes and prioritize those with higher view counts and descriptions that clearly mention paper-related information.\n\nOutput:\n- **Paper**: title, first author, publication year, Semantic Scholar citation count, whether PDF is available, paper link  \n- **GitHub**: repository name, star count, whether `requirements.txt` is present, repository link  \n- **Hugging Face**: model name, download count, Model Card summary, model link  \n- **YouTube**: video title, duration, view count, video description summary, video link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Paper information extracted from the answer"""
    title: Optional[str] = None
    first_author: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: Optional[int] = None
    pdf_available: Optional[bool] = None
    paper_link: Optional[str] = None


class GitHubRepoInfo(BaseModel):
    """GitHub repository information extracted from the answer"""
    repository_name: Optional[str] = None
    star_count: Optional[int] = None
    has_requirements_txt: Optional[bool] = None
    repository_link: Optional[str] = None


class HuggingFaceModelInfo(BaseModel):
    """Hugging Face model information extracted from the answer"""
    models: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class YouTubeVideoInfo(BaseModel):
    """YouTube video information extracted from the answer"""
    videos: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
Extract the paper information from Semantic Scholar that the user selected from the answer.

Return:
- title: the full paper title exactly as stated
- first_author: the first author's name
- publication_year: the year of publication as an integer
- citation_count: the citation count as an integer
- pdf_available: true if PDF is available, false otherwise
- paper_link: the Semantic Scholar link to the paper

If any field is missing, set it to null.
"""


def prompt_extract_github_repo_info() -> str:
    return """
Extract the GitHub repository information from the answer.

Return:
- repository_name: the repository name (e.g., "username/repo-name")
- star_count: the number of stars as an integer
- has_requirements_txt: true if requirements.txt is present, false otherwise
- repository_link: the GitHub link to the repository

If any field is missing, set it to null.
"""


def prompt_extract_huggingface_models() -> str:
    return """
Extract all Hugging Face model information from the answer.

Return:
- models: a list of dictionaries, each containing:
  - model_name: the model name/identifier
  - download_count: the download count (as integer or string)
  - model_card_summary: a brief summary of what the Model Card says
  - model_link: the Hugging Face link to the model

If no models are found, return an empty list.
"""


def prompt_extract_youtube_videos() -> str:
    return """
Extract all YouTube video information from the answer.

Return:
- videos: a list of dictionaries, each containing:
  - video_title: the video title
  - duration: the video duration/length
  - view_count: the view count (as integer or string)
  - video_description_summary: a summary of the video description
  - video_link: the YouTube link to the video

If no videos are found, return an empty list.
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


def extract_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r'\d+', value.replace(',', ''))
        if m:
            try:
                return int(m.group())
            except Exception:
                return None
    return None


def is_valid_year_2024_or_later(year: Optional[int]) -> bool:
    if year is None:
        return False
    return year >= 2024


def looks_like_high_citation_count(count: Optional[int]) -> bool:
    if count is None:
        return False
    return count > 0


def parse_duration_to_minutes(duration_str: Optional[str]) -> Optional[int]:
    if not duration_str:
        return None
    # Try to parse formats like "10:30", "15 minutes", "1:20:00", etc.
    # Pattern for HH:MM:SS or MM:SS
    time_pattern = re.search(r'(\d+):(\d+)(?::(\d+))?', duration_str)
    if time_pattern:
        groups = time_pattern.groups()
        if groups[2] is not None:  # HH:MM:SS
            hours = int(groups[0])
            minutes = int(groups[1])
            return hours * 60 + minutes
        else:  # MM:SS
            minutes = int(groups[0])
            return minutes

    # Pattern for "X minutes" or "X mins"
    minutes_pattern = re.search(r'(\d+)\s*(?:minute|min)', duration_str, re.IGNORECASE)
    if minutes_pattern:
        return int(minutes_pattern.group(1))

    return None


def is_duration_10_minutes_or_longer(duration_str: Optional[str]) -> bool:
    minutes = parse_duration_to_minutes(duration_str)
    if minutes is None:
        return False
    return minutes >= 10


def extract_view_count(view_str: Any) -> Optional[int]:
    if view_str is None:
        return None
    if isinstance(view_str, int):
        return view_str
    if isinstance(view_str, str):
        # Remove commas and 'k', 'K', 'M', etc.
        s = view_str.lower().replace(',', '')
        if 'k' in s:
            num = re.search(r'([\d.]+)k', s)
            if num:
                try:
                    return int(float(num.group(1)) * 1000)
                except Exception:
                    return None
        if 'm' in s:
            num = re.search(r'([\d.]+)m', s)
            if num:
                try:
                    return int(float(num.group(1)) * 1000000)
                except Exception:
                    return None
        # Try plain number
        num = re.search(r'\d+', s)
        if num:
            try:
                return int(num.group())
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
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_repo_info(),
        template_class=GitHubRepoInfo,
        extraction_name="github_repo_info"
    )

    hf_models = await evaluator.extract(
        prompt=prompt_extract_huggingface_models(),
        template_class=HuggingFaceModelInfo,
        extraction_name="huggingface_models"
    )

    yt_videos = await evaluator.extract(
        prompt=prompt_extract_youtube_videos(),
        template_class=YouTubeVideoInfo,
        extraction_name="youtube_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Semantic Scholar section
    semanticscholar_node = evaluator.add_sequential(
        id="semanticscholar_section",
        desc="Semantic Scholar paper search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A3 - Date range filtering (2024+)
    year_ok = is_valid_year_2024_or_later(paper_info.publication_year)
    evaluator.add_custom_node(
        result=bool(year_ok),
        id="semanticscholar_date_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Filter papers published since 2024",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citation count
    citation_ok = looks_like_high_citation_count(paper_info.citation_count)
    evaluator.add_custom_node(
        result=bool(citation_ok),
        id="semanticscholar_sort_citations",
        desc="[Action Node] semanticscholar.org:F1:A4 - Sort by citation count (descending)",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P2 - Identify PDF availability
    pdf_ok = paper_info.pdf_available is True
    evaluator.add_custom_node(
        result=bool(pdf_ok),
        id="semanticscholar_pdf_status",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Identify papers with downloadable PDFs",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F2:A5 - Click into paper details
    paper_title_ok = bool(paper_info.title and len(paper_info.title.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(paper_title_ok),
        id="semanticscholar_paper_details",
        desc="[Action Node] semanticscholar.org:F2:A5 - Navigate to paper detail page to get complete information",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P3 - Understand citation statistics
    citation_data_ok = paper_info.citation_count is not None
    evaluator.add_custom_node(
        result=bool(citation_data_ok),
        id="semanticscholar_citation_understanding",
        desc="[Perception Node] semanticscholar.org:F2:P3 - Extract accurate citation count from paper detail page",
        parent=semanticscholar_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository search and verification",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A10 - Search form submission
    github_search_ok = has_any_ci(answer, ['github']) and (
        (paper_info.title and ci_contains(answer, paper_info.title[:20])) or
        (paper_info.first_author and ci_contains(answer, paper_info.first_author))
    )
    evaluator.add_custom_node(
        result=bool(github_search_ok),
        id="github_search_submission",
        desc="[Action Node] github.com:F1:A10 - Submit search query with paper title or first author name",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by stars
    star_count = extract_int(github_info.star_count)
    star_ok = star_count is not None and star_count > 500
    evaluator.add_custom_node(
        result=bool(star_ok),
        id="github_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - Sort repositories by star count to find popular ones (>500 stars)",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Extract star count from search results
    star_extracted_ok = star_count is not None
    evaluator.add_custom_node(
        result=bool(star_extracted_ok),
        id="github_extract_stars",
        desc="[Perception Node] github.com:F1:P1 - Extract star count from repository search result card",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand file tree
    requirements_ok = github_info.has_requirements_txt is True
    evaluator.add_custom_node(
        result=bool(requirements_ok),
        id="github_file_tree_expand",
        desc="[Action Node] github.com:F3:A17 - Expand directory/file tree to locate requirements.txt",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A18 - View README file
    readme_checked_ok = github_info.repository_name is not None
    evaluator.add_custom_node(
        result=bool(readme_checked_ok),
        id="github_view_readme",
        desc="[Action Node] github.com:F3:A18 - Open and read README file to verify paper citation",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Understand directory structure
    dir_understanding_ok = github_info.has_requirements_txt is not None
    evaluator.add_custom_node(
        result=bool(dir_understanding_ok),
        id="github_directory_understanding",
        desc="[Perception Node] github.com:F3:P12 - Understand project root structure to locate requirements.txt",
        parent=github_node,
        critical=False
    )

    # 3.3 Hugging Face section
    huggingface_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="Hugging Face model search and verification",
        parent=root,
        critical=False
    )

    # [Action Node] huggingface.co:F1:A9 - Search and filter
    hf_search_ok = has_any_ci(answer, ['hugging face', 'huggingface']) and (
        paper_info.title and ci_contains(answer, paper_info.title[:20])
    )
    evaluator.add_custom_node(
        result=bool(hf_search_ok),
        id="huggingface_search_filter",
        desc="[Action Node] huggingface.co:F1:A9 - Search Hugging Face with paper title",
        parent=huggingface_node,
        critical=False
    )

    # [Action Node] huggingface.co:F1:A4 - Browse multiple pages for download counts
    models_list = hf_models.models if hf_models and hf_models.models else []
    has_download_counts = any(
        'download_count' in m and m.get('download_count') is not None
        for m in models_list
    )
    evaluator.add_custom_node(
        result=bool(has_download_counts),
        id="huggingface_browse_pages",
        desc="[Action Node] huggingface.co:F1:A4 - Browse multiple pages to compare download counts",
        parent=huggingface_node,
        critical=False
    )

    # [Action Node] huggingface.co:F3:A3 - Switch to Model Card tab
    has_model_card_summaries = any(
        'model_card_summary' in m and m.get('model_card_summary')
        for m in models_list
    )
    evaluator.add_custom_node(
        result=bool(has_model_card_summaries),
        id="huggingface_model_card_tab",
        desc="[Action Node] huggingface.co:F3:A3 - Switch to Model Card tab to view details",
        parent=huggingface_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F3:P10 - Understand Model Card content
    model_card_verified = any(
        'model_card_summary' in m and m.get('model_card_summary') and
        (paper_info.title and ci_contains(str(m.get('model_card_summary')), paper_info.title[:15]))
        for m in models_list
    )
    evaluator.add_custom_node(
        result=bool(model_card_verified),
        id="huggingface_model_card_understanding",
        desc="[Perception Node] huggingface.co:F3:P10 - Verify Model Card confirms correspondence to the paper",
        parent=huggingface_node,
        critical=False
    )

    # 3.4 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube video search and verification",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click search result
    videos_list = yt_videos.videos if yt_videos and yt_videos.videos else []
    has_videos = len(videos_list) > 0
    evaluator.add_custom_node(
        result=bool(has_videos),
        id="youtube_click_result",
        desc="[Action Node] youtube.com:F1:A22 - Click on search result videos to view details",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Apply duration filter (10+ minutes)
    has_long_videos = any(
        is_duration_10_minutes_or_longer(v.get('duration'))
        for v in videos_list
    )
    evaluator.add_custom_node(
        result=bool(has_long_videos),
        id="youtube_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Filter videos by duration (10+ minutes)",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify relevant content (5000+ views)
    has_popular_videos = any(
        extract_view_count(v.get('view_count', 0)) and extract_view_count(v.get('view_count', 0)) > 5000
        for v in videos_list
    )
    evaluator.add_custom_node(
        result=bool(has_popular_videos),
        id="youtube_content_relevance",
        desc="[Perception Node] youtube.com:F1:P4 - Identify in-depth explainer videos with 5000+ views",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A46 - Expand video description
    has_descriptions = any(
        'video_description_summary' in v and v.get('video_description_summary')
        for v in videos_list
    )
    evaluator.add_custom_node(
        result=bool(has_descriptions),
        id="youtube_expand_description",
        desc="[Action Node] youtube.com:F2:A46 - Expand video description to verify paper mention",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F2:P21 - Understand video content type
    descriptions_mention_paper = any(
        'video_description_summary' in v and v.get('video_description_summary') and
        ((paper_info.title and ci_contains(str(v.get('video_description_summary')), paper_info.title[:15])) or
         (paper_info.first_author and ci_contains(str(v.get('video_description_summary')), paper_info.first_author)))
        for v in videos_list
    )
    evaluator.add_custom_node(
        result=bool(descriptions_mention_paper),
        id="youtube_content_understanding",
        desc="[Perception Node] youtube.com:F2:P21 - Verify video description mentions paper title or author",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
