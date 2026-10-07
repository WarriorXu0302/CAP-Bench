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
TASK_ID = "task-73c4b7"
TASK_DESCRIPTION = 'Our team plans to implement a speech cloning system in early next year and currently needs to conduct a technology selection. Please conduct this research as of now.\n\nFirst, search for "Text-to-Speech" models on HuggingFace. Using the left-hand filter bar, only consider models that support the "PyTorch" framework and have an "Apache-2.0" License (to ensure commercial viability). Sort by download count and select the top 3.\n\nNext, find the corresponding GitHub repositories for these 3 models. Navigate to their Issues page, count the number of Open and Closed Issues for each, and calculate the Open/Closed ratio.\n\nFinally, for the model with the lowest Open/Closed ratio (i.e., highest resolution efficiency), search on Bilibili for currently available, detailed Chinese tutorial videos longer than 10 minutes (with keywords including "fine-tuning" or "deployment"). Find 3 videos that meet these criteria.\n\nOutput: The names of the top 3 models, their HuggingFace download count, HuggingFace update date, GitHub link, and GitHub Open/Closed issue counts; and for the finally selected model, the titles, durations, publication dates, and links of the Bilibili tutorial videos.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HuggingFaceModel(BaseModel):
    """Single HuggingFace model information"""
    name: Optional[str] = None
    download_count: Optional[str] = None
    update_date: Optional[str] = None
    github_link: Optional[str] = None
    open_issues: Optional[str] = None
    closed_issues: Optional[str] = None


class HuggingFaceModels(BaseModel):
    """Top 3 HuggingFace models extracted from the answer"""
    model_1: Optional[HuggingFaceModel] = None
    model_2: Optional[HuggingFaceModel] = None
    model_3: Optional[HuggingFaceModel] = None


class BilibiliVideo(BaseModel):
    """Single Bilibili video information"""
    title: Optional[str] = None
    duration: Optional[str] = None
    publication_date: Optional[str] = None
    link: Optional[str] = None


class BilibiliVideos(BaseModel):
    """Bilibili tutorial videos extracted from the answer"""
    video_1: Optional[BilibiliVideo] = None
    video_2: Optional[BilibiliVideo] = None
    video_3: Optional[BilibiliVideo] = None
    selected_model_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_huggingface_models() -> str:
    return """
Extract the top 3 HuggingFace Text-to-Speech models from the answer.

For each model, return:
- name: the model name exactly as stated
- download_count: the download count exactly as written (include units if present)
- update_date: the update/last modified date exactly as stated
- github_link: the GitHub repository URL
- open_issues: the number of open issues
- closed_issues: the number of closed issues

If any field is missing for a model, set it to null.
"""


def prompt_extract_bilibili_videos() -> str:
    return """
Extract the 3 Bilibili tutorial videos from the answer for the selected model.

Also extract:
- selected_model_name: the name of the model for which these videos were found

For each video, return:
- title: the video title
- duration: the video duration exactly as stated
- publication_date: the publication date
- link: the video URL

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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove common separators and extract number
    cleaned = re.sub(r'[,\s]+', '', text)
    m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_duration_to_minutes(duration: Optional[str]) -> Optional[float]:
    """Parse duration string to minutes. Handle formats like '12:34', '1:23:45', '15分钟'"""
    if not duration:
        return None

    # Handle Chinese format like "15分钟" or "1小时20分"
    if '分' in duration or '小时' in duration:
        total_minutes = 0.0
        hour_match = re.search(r'(\d+)\s*小时', duration)
        min_match = re.search(r'(\d+)\s*分', duration)
        if hour_match:
            total_minutes += float(hour_match.group(1)) * 60
        if min_match:
            total_minutes += float(min_match.group(1))
        return total_minutes if total_minutes > 0 else None

    # Handle time format like "12:34" or "1:23:45"
    time_match = re.search(r'(\d+):(\d+)(?::(\d+))?', duration)
    if time_match:
        parts = [p for p in time_match.groups() if p is not None]
        if len(parts) == 2:  # MM:SS
            return float(parts[0]) + float(parts[1]) / 60.0
        elif len(parts) == 3:  # HH:MM:SS
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60.0

    return None


def looks_like_year_2025(text: Optional[str]) -> bool:
    if not text:
        return False
    return '2025' in text


def has_model_count(models: HuggingFaceModels, count: int) -> bool:
    """Check if at least 'count' models are present"""
    present = 0
    if models.model_1 and models.model_1.name:
        present += 1
    if models.model_2 and models.model_2.name:
        present += 1
    if models.model_3 and models.model_3.name:
        present += 1
    return present >= count


def download_counts_descending(models: HuggingFaceModels) -> bool:
    """Check if download counts are in descending order"""
    counts = []
    for model in [models.model_1, models.model_2, models.model_3]:
        if model and model.download_count:
            num = extract_number(model.download_count)
            if num is not None:
                counts.append(num)

    if len(counts) < 2:
        return False

    for i in range(len(counts) - 1):
        if counts[i] < counts[i + 1]:
            return False
    return True


def has_video_count(videos: BilibiliVideos, count: int) -> bool:
    """Check if at least 'count' videos are present"""
    present = 0
    if videos.video_1 and videos.video_1.title:
        present += 1
    if videos.video_2 and videos.video_2.title:
        present += 1
    if videos.video_3 and videos.video_3.title:
        present += 1
    return present >= count


def all_videos_over_10_minutes(videos: BilibiliVideos) -> bool:
    """Check if all present videos are over 10 minutes"""
    for video in [videos.video_1, videos.video_2, videos.video_3]:
        if video and video.title:
            duration_minutes = parse_duration_to_minutes(video.duration)
            if duration_minutes is None or duration_minutes <= 10:
                return False
    return True


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
    hf_models = await evaluator.extract(
        prompt=prompt_extract_huggingface_models(),
        template_class=HuggingFaceModels,
        extraction_name="huggingface_models"
    )

    bilibili_videos = await evaluator.extract(
        prompt=prompt_extract_bilibili_videos(),
        template_class=BilibiliVideos,
        extraction_name="bilibili_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 HuggingFace section
    hf_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="HuggingFace Text-to-Speech model search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A18 - Expand filter bar
    filter_bar_used = has_any_ci(answer, ['filter', 'left', 'sidebar', 'pytorch', 'apache'])
    evaluator.add_custom_node(
        result=bool(filter_bar_used),
        id="hf_action_filter_bar",
        desc="[Action Node] huggingface.co:F2:A18 - Use the left-hand filter bar (expand if folded)",
        parent=hf_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A7 - Filter by PyTorch framework
    pytorch_filter = has_any_ci(answer, ['pytorch']) and has_model_count(hf_models, 3)
    evaluator.add_custom_node(
        result=bool(pytorch_filter),
        id="hf_action_pytorch_filter",
        desc="[Action Node] huggingface.co:F2:A7 - Filter models by PyTorch framework support",
        parent=hf_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A22 - Filter by Apache-2.0 License
    license_filter = has_any_ci(answer, ['apache', 'license']) and has_model_count(hf_models, 3)
    evaluator.add_custom_node(
        result=bool(license_filter),
        id="hf_action_license_filter",
        desc="[Action Node] huggingface.co:F2:A22 - Filter models by Apache-2.0 License",
        parent=hf_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A5 - Sort by download count
    sort_action = has_any_ci(answer, ['download', 'sort']) or download_counts_descending(hf_models)
    evaluator.add_custom_node(
        result=bool(sort_action),
        id="hf_action_sort_downloads",
        desc="[Action Node] huggingface.co:F2:A5 - Sort models by download count (descending)",
        parent=hf_node,
        critical=False
    )

    # Check if top 3 models are present with required info
    top3_present = has_model_count(hf_models, 3)
    evaluator.add_custom_node(
        result=bool(top3_present),
        id="hf_top3_models_present",
        desc="Top 3 models are identified with names",
        parent=hf_node,
        critical=False
    )

    # Check download counts are in descending order
    evaluator.add_custom_node(
        result=bool(download_counts_descending(hf_models)),
        id="hf_downloads_descending",
        desc="Download counts are in descending order (indicating proper sorting)",
        parent=hf_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository analysis for issue statistics",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F4:A1 - Navigate to Issues tab
    issues_tab_used = has_any_ci(answer, ['issue', 'github'])
    evaluator.add_custom_node(
        result=bool(issues_tab_used),
        id="github_action_issues_tab",
        desc="[Action Node] github.com:F4:A1 - Navigate to Issues page/tab for each repository",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - Extract Open and Closed issue counts
    has_open_closed = False
    if hf_models.model_1 and hf_models.model_1.open_issues and hf_models.model_1.closed_issues:
        has_open_closed = True
    if hf_models.model_2 and hf_models.model_2.open_issues and hf_models.model_2.closed_issues:
        has_open_closed = True
    if hf_models.model_3 and hf_models.model_3.open_issues and hf_models.model_3.closed_issues:
        has_open_closed = True

    evaluator.add_custom_node(
        result=bool(has_open_closed),
        id="github_perception_issue_counts",
        desc="[Perception Node] github.com:F4:P2 - Extract Open and Closed issue counts for each model",
        parent=github_node,
        critical=False
    )

    # Check if all 3 models have GitHub links
    all_have_github = (
        hf_models.model_1 and hf_models.model_1.github_link and
        hf_models.model_2 and hf_models.model_2.github_link and
        hf_models.model_3 and hf_models.model_3.github_link
    )
    evaluator.add_custom_node(
        result=bool(all_have_github),
        id="github_all_links_present",
        desc="All 3 models have GitHub repository links",
        parent=github_node,
        critical=False
    )

    # Check if Open/Closed ratio calculation is mentioned
    ratio_mentioned = has_any_ci(answer, ['ratio', 'open/closed', 'open closed', 'efficiency'])
    evaluator.add_custom_node(
        result=bool(ratio_mentioned),
        id="github_ratio_calculation",
        desc="Open/Closed issue ratio is calculated or mentioned",
        parent=github_node,
        critical=False
    )

    # 3.3 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili tutorial video search with specific criteria",
        parent=root,
        critical=False
    )

    # Check if a model was selected based on lowest Open/Closed ratio
    model_selected = bool(bilibili_videos.selected_model_name)
    evaluator.add_custom_node(
        result=bool(model_selected),
        id="bilibili_model_selection",
        desc="Selected the model with lowest Open/Closed ratio for Bilibili search",
        parent=bilibili_node,
        critical=False
    )

    # Check if search includes required keywords
    search_keywords = has_any_ci(answer, ['fine-tuning', 'deployment', '微调', '部署'])
    evaluator.add_custom_node(
        result=bool(search_keywords),
        id="bilibili_search_keywords",
        desc="Search includes keywords like 'fine-tuning' or 'deployment'",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A48 - Use duration filter (>10 minutes)
    duration_filter_used = has_any_ci(answer, ['10分钟', '10 minutes', 'duration', '时长'])
    evaluator.add_custom_node(
        result=bool(duration_filter_used),
        id="bilibili_action_duration_filter",
        desc="[Action Node] bilibili.com:F1:A48 - Use advanced filter for duration >10 minutes",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P13 - Extract/recognize video duration
    has_valid_durations = has_video_count(bilibili_videos, 1)
    for video in [bilibili_videos.video_1, bilibili_videos.video_2, bilibili_videos.video_3]:
        if video and video.title and not video.duration:
            has_valid_durations = False
            break

    evaluator.add_custom_node(
        result=bool(has_valid_durations),
        id="bilibili_perception_duration",
        desc="[Perception Node] bilibili.com:F1:P13 - Extract or recognize video durations from results",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by publication date
    date_sort_used = has_any_ci(answer, ['sort', 'date', 'publication', '发布', '时间'])
    evaluator.add_custom_node(
        result=bool(date_sort_used),
        id="bilibili_action_date_sort",
        desc="[Action Node] bilibili.com:F1:A5 - Sort by publication date to find 2025 videos",
        parent=bilibili_node,
        critical=False
    )

    # Check if 3 videos are present
    three_videos_present = has_video_count(bilibili_videos, 3)
    evaluator.add_custom_node(
        result=bool(three_videos_present),
        id="bilibili_three_videos_present",
        desc="3 Bilibili tutorial videos are identified",
        parent=bilibili_node,
        critical=False
    )

    # Check if all videos are over 10 minutes
    all_over_10min = False
    if three_videos_present:
        all_over_10min = all_videos_over_10_minutes(bilibili_videos)

    evaluator.add_custom_node(
        result=bool(all_over_10min),
        id="bilibili_all_over_10min",
        desc="All 3 videos have duration longer than 10 minutes",
        parent=bilibili_node,
        critical=False
    )

    # Check if videos are from 2025
    videos_from_2025 = False
    if bilibili_videos.video_1 and bilibili_videos.video_1.publication_date:
        if looks_like_year_2025(bilibili_videos.video_1.publication_date):
            videos_from_2025 = True
    if bilibili_videos.video_2 and bilibili_videos.video_2.publication_date:
        if looks_like_year_2025(bilibili_videos.video_2.publication_date):
            videos_from_2025 = True
    if bilibili_videos.video_3 and bilibili_videos.video_3.publication_date:
        if looks_like_year_2025(bilibili_videos.video_3.publication_date):
            videos_from_2025 = True

    evaluator.add_custom_node(
        result=bool(videos_from_2025),
        id="bilibili_videos_2025",
        desc="At least one video is from 2025 (currently available)",
        parent=bilibili_node,
        critical=False
    )

    # Check if video links are provided
    has_video_links = False
    if bilibili_videos.video_1 and bilibili_videos.video_1.link:
        has_video_links = True
    if bilibili_videos.video_2 and bilibili_videos.video_2.link:
        has_video_links = True
    if bilibili_videos.video_3 and bilibili_videos.video_3.link:
        has_video_links = True

    evaluator.add_custom_node(
        result=bool(has_video_links),
        id="bilibili_video_links_present",
        desc="Video links are provided for Bilibili tutorials",
        parent=bilibili_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
