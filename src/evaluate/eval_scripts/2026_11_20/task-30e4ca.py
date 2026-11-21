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
TASK_ID = "task-30e4ca"
TASK_DESCRIPTION = 'I’m preparing a submission to CVPR and want a comprehensive understanding of the conference. First, please check DBLP to analyze the topic distribution of CVPR papers from the past three years, with a focus on identifying major trending areas. Then, on arXiv, find several highly influential papers in those trending areas (judged using both visible paper-page signals and external citation-metric pages), and review their writing style and experimental design. Next, go to GitHub to locate the corresponding open-source code repositories for those papers, and evaluate their star counts and code completeness to understand current experimental standards. Finally, search Bilibili for CVPR-related peer-review experience sharing or conference talk videos with high view counts and durations over 20 minutes, and summarize the key reviewing points that are most frequently emphasized.  \n\nAt the end, provide me with a structured summary of: what directions this conference currently values most, what experiments the reference papers typically include, expected code quality standards, and what reviewers pay special attention to.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DBLPTopicAnalysis(BaseModel):
    """Topic distribution analysis from DBLP for CVPR papers"""
    mentioned_dblp: Optional[bool] = False
    past_three_years_mentioned: Optional[bool] = False
    trending_areas: Optional[List[str]] = Field(default_factory=list)
    topic_distribution_described: Optional[bool] = False


class ArXivPaperAnalysis(BaseModel):
    """Analysis of influential papers from arXiv"""
    mentioned_arxiv: Optional[bool] = False
    paper_titles: Optional[List[str]] = Field(default_factory=list)
    citation_metrics_mentioned: Optional[bool] = False
    writing_style_described: Optional[bool] = False
    experimental_design_described: Optional[bool] = False


class GitHubCodeAnalysis(BaseModel):
    """GitHub repository analysis"""
    mentioned_github: Optional[bool] = False
    repository_names: Optional[List[str]] = Field(default_factory=list)
    star_counts_mentioned: Optional[bool] = False
    code_completeness_evaluated: Optional[bool] = False


class BilibiliVideoAnalysis(BaseModel):
    """Bilibili video analysis"""
    mentioned_bilibili: Optional[bool] = False
    video_titles: Optional[List[str]] = Field(default_factory=list)
    view_counts_mentioned: Optional[bool] = False
    duration_filter_mentioned: Optional[bool] = False
    review_points_summarized: Optional[bool] = False


class FinalSummary(BaseModel):
    """Final structured summary"""
    valued_directions: Optional[List[str]] = Field(default_factory=list)
    typical_experiments: Optional[List[str]] = Field(default_factory=list)
    code_quality_standards: Optional[str] = None
    reviewer_attention_points: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_dblp_analysis() -> str:
    return """
Extract information about DBLP analysis from the answer:

- mentioned_dblp: true if DBLP is mentioned in the answer, false otherwise
- past_three_years_mentioned: true if the answer mentions analyzing past three years
- trending_areas: list of trending topic areas identified (e.g., ["transformer", "diffusion models", "3D vision"])
- topic_distribution_described: true if the answer describes topic distribution or patterns

If any field cannot be determined, use the default value.
"""


def prompt_extract_arxiv_analysis() -> str:
    return """
Extract information about arXiv paper analysis from the answer:

- mentioned_arxiv: true if arXiv is mentioned
- paper_titles: list of specific paper titles mentioned (if any)
- citation_metrics_mentioned: true if citation counts, h-index, or other metrics are mentioned
- writing_style_described: true if the answer describes writing style characteristics
- experimental_design_described: true if the answer describes experimental design patterns

If any field cannot be determined, use the default value.
"""


def prompt_extract_github_analysis() -> str:
    return """
Extract information about GitHub repository analysis from the answer:

- mentioned_github: true if GitHub is mentioned
- repository_names: list of specific repository names mentioned (if any)
- star_counts_mentioned: true if star counts are mentioned
- code_completeness_evaluated: true if code completeness, structure, or quality is discussed

If any field cannot be determined, use the default value.
"""


def prompt_extract_bilibili_analysis() -> str:
    return """
Extract information about Bilibili video analysis from the answer:

- mentioned_bilibili: true if Bilibili is mentioned
- video_titles: list of specific video titles mentioned (if any)
- view_counts_mentioned: true if view counts or popularity metrics are mentioned
- duration_filter_mentioned: true if 20-minute duration filter is mentioned
- review_points_summarized: true if key review points are summarized

If any field cannot be determined, use the default value.
"""


def prompt_extract_final_summary() -> str:
    return """
Extract the final structured summary from the answer:

- valued_directions: list of directions/topics CVPR currently values
- typical_experiments: list of typical experiments mentioned in reference papers
- code_quality_standards: description of expected code quality standards
- reviewer_attention_points: list of points reviewers pay special attention to

If any field cannot be determined, use the default value.
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


def list_not_empty(lst: Optional[List]) -> bool:
    return bool(lst and len(lst) > 0)


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
    dblp_info = await evaluator.extract(
        prompt=prompt_extract_dblp_analysis(),
        template_class=DBLPTopicAnalysis,
        extraction_name="dblp_analysis"
    )

    arxiv_info = await evaluator.extract(
        prompt=prompt_extract_arxiv_analysis(),
        template_class=ArXivPaperAnalysis,
        extraction_name="arxiv_analysis"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_analysis(),
        template_class=GitHubCodeAnalysis,
        extraction_name="github_analysis"
    )

    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_analysis(),
        template_class=BilibiliVideoAnalysis,
        extraction_name="bilibili_analysis"
    )

    final_summary = await evaluator.extract(
        prompt=prompt_extract_final_summary(),
        template_class=FinalSummary,
        extraction_name="final_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 DBLP section
    dblp_node = evaluator.add_sequential(
        id="dblp_section",
        desc="DBLP analysis of CVPR papers from past three years",
        parent=root,
        critical=False
    )

    # [Action Node] dblp.org:F4:A8 - Navigate to CVPR conference page
    dblp_navigation_ok = (dblp_info and dblp_info.mentioned_dblp) or has_any_ci(answer, ['dblp'])
    evaluator.add_custom_node(
        result=bool(dblp_navigation_ok),
        id="dblp_navigation",
        desc="[Action Node] dblp.org:F4:A8 - Navigate to CVPR conference details page on DBLP",
        parent=dblp_node,
        critical=False
    )

    # [Perception Node] dblp.org:F4:P2 - Understand conference structure and papers
    dblp_understanding_ok = (
        (dblp_info and dblp_info.past_three_years_mentioned) or
        has_any_ci(answer, ['past three years', 'last three years', '近三年', '过去三年'])
    )
    evaluator.add_custom_node(
        result=bool(dblp_understanding_ok),
        id="dblp_conference_understanding",
        desc="[Perception Node] dblp.org:F4:P2 - Understand conference structure and identify papers from past three years",
        parent=dblp_node,
        critical=False
    )

    # Check trending areas identified
    trending_areas_ok = (dblp_info and list_not_empty(dblp_info.trending_areas)) or has_any_ci(answer, ['trending', 'hot', '热门', 'popular', 'major areas'])
    evaluator.add_custom_node(
        result=bool(trending_areas_ok),
        id="dblp_trending_areas",
        desc="Identify major trending topic areas from CVPR papers",
        parent=dblp_node,
        critical=False
    )

    # 3.2 arXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="arXiv search for highly influential papers in trending areas",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A16 - Browse multiple pages
    arxiv_browse_ok = (arxiv_info and arxiv_info.mentioned_arxiv) or has_any_ci(answer, ['arxiv'])
    evaluator.add_custom_node(
        result=bool(arxiv_browse_ok),
        id="arxiv_browse",
        desc="[Action Node] arxiv.org:F1:A16 - Browse and paginate through arXiv search results",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F1:P11 - Understand content relevance
    arxiv_relevance_ok = (
        (arxiv_info and arxiv_info.citation_metrics_mentioned) or
        has_any_ci(answer, ['citation', 'influence', 'highly cited', '引用', 'h-index', 'impact'])
    )
    evaluator.add_custom_node(
        result=bool(arxiv_relevance_ok),
        id="arxiv_relevance_understanding",
        desc="[Perception Node] arxiv.org:F1:P11 - Understand content relevance and match papers to trending areas",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A20 - Expand abstract panel
    arxiv_expand_ok = (
        (arxiv_info and arxiv_info.writing_style_described) or
        has_any_ci(answer, ['abstract', 'writing', 'style', '摘要', '写作'])
    )
    evaluator.add_custom_node(
        result=bool(arxiv_expand_ok),
        id="arxiv_expand_abstract",
        desc="[Action Node] arxiv.org:F3:A20 - Expand abstract panel to view full content",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P5 - Understand paper content
    arxiv_content_ok = (
        (arxiv_info and arxiv_info.writing_style_described and arxiv_info.experimental_design_described) or
        has_any_ci(answer, ['experimental design', 'methodology', 'approach', '实验设计', '方法'])
    )
    evaluator.add_custom_node(
        result=bool(arxiv_content_ok),
        id="arxiv_content_understanding",
        desc="[Perception Node] arxiv.org:F3:P5 - Understand paper content including writing style and experimental design",
        parent=arxiv_node,
        critical=False
    )

    # 3.3 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository search and evaluation",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by stars
    github_sort_ok = (
        (github_info and github_info.star_counts_mentioned) or
        has_any_ci(answer, ['star', 'github', 'repository', '仓库'])
    )
    evaluator.add_custom_node(
        result=bool(github_sort_ok),
        id="github_sort_stars",
        desc="[Action Node] github.com:F1:A7 - Sort repositories by star count",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand directory structure
    github_expand_ok = (
        (github_info and github_info.code_completeness_evaluated) or
        has_any_ci(answer, ['code structure', 'directory', 'completeness', '代码结构', '完整'])
    )
    evaluator.add_custom_node(
        result=bool(github_expand_ok),
        id="github_expand_directory",
        desc="[Action Node] github.com:F3:A17 - Expand directory structure to view project organization",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Understand hierarchical structure
    github_structure_ok = (
        (github_info and github_info.code_completeness_evaluated) or
        has_any_ci(answer, ['module', 'organization', 'structure', '模块', '组织'])
    )
    evaluator.add_custom_node(
        result=bool(github_structure_ok),
        id="github_structure_understanding",
        desc="[Perception Node] github.com:F3:P12 - Understand code hierarchical structure and module organization",
        parent=github_node,
        critical=False
    )

    # 3.4 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili search for CVPR review experience videos",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Switch to sort by view count
    bilibili_sort_ok = (
        (bilibili_info and bilibili_info.view_counts_mentioned) or
        has_any_ci(answer, ['bilibili', 'view', 'popular', '播放', 'b站'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_sort_ok),
        id="bilibili_sort_views",
        desc="[Action Node] bilibili.com:F1:A5 - Switch sorting to view count/popularity",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Visual content recognition
    bilibili_visual_ok = (
        (bilibili_info and bilibili_info.mentioned_bilibili) or
        has_any_ci(answer, ['cvpr', 'review', 'conference', '审稿', '会议'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_visual_ok),
        id="bilibili_visual_recognition",
        desc="[Perception Node] bilibili.com:F1:P30 - Recognize CVPR-related content through thumbnails and titles",
        parent=bilibili_node,
        critical=False
    )

    # Duration filter check
    bilibili_duration_ok = (
        (bilibili_info and bilibili_info.duration_filter_mentioned) or
        has_any_ci(answer, ['20 min', 'duration', 'over 20', '20分钟', '时长'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_duration_ok),
        id="bilibili_duration_filter",
        desc="Filter videos by duration over 20 minutes",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F2:A68 - Progress seek/skip
    bilibili_seek_ok = has_any_ci(answer, ['key point', 'summary', 'review point', '要点', '审稿'])
    evaluator.add_custom_node(
        result=bool(bilibili_seek_ok),
        id="bilibili_seek_progress",
        desc="[Action Node] bilibili.com:F2:A68 - Seek through video to find key review points",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F2:P23 - Understand video content
    bilibili_content_ok = (
        (bilibili_info and bilibili_info.review_points_summarized) or
        has_any_ci(answer, ['review point', 'reviewer', 'attention', '审稿要点', '评审'])
    )
    evaluator.add_custom_node(
        result=bool(bilibili_content_ok),
        id="bilibili_content_understanding",
        desc="[Perception Node] bilibili.com:F2:P23 - Understand video content and extract review points",
        parent=bilibili_node,
        critical=False
    )

    # 3.5 Final summary section
    summary_node = evaluator.add_parallel(
        id="final_summary_section",
        desc="Structured summary of CVPR submission preparation insights",
        parent=root,
        critical=False
    )

    # Check valued directions
    valued_directions_ok = (final_summary and list_not_empty(final_summary.valued_directions)) or has_any_ci(answer, ['direction', 'value', 'focus', '方向', '重视'])
    evaluator.add_custom_node(
        result=bool(valued_directions_ok),
        id="summary_valued_directions",
        desc="Summary includes directions/topics CVPR currently values most",
        parent=summary_node,
        critical=False
    )

    # Check typical experiments
    typical_experiments_ok = (final_summary and list_not_empty(final_summary.typical_experiments)) or has_any_ci(answer, ['experiment', 'typical', 'include', '实验', '通常'])
    evaluator.add_custom_node(
        result=bool(typical_experiments_ok),
        id="summary_typical_experiments",
        desc="Summary includes typical experiments in reference papers",
        parent=summary_node,
        critical=False
    )

    # Check code quality standards
    code_quality_ok = (final_summary and final_summary.code_quality_standards) or has_any_ci(answer, ['code quality', 'standard', 'expected', '代码质量', '标准'])
    evaluator.add_custom_node(
        result=bool(code_quality_ok),
        id="summary_code_quality",
        desc="Summary includes expected code quality standards",
        parent=summary_node,
        critical=False
    )

    # Check reviewer attention points
    reviewer_points_ok = (final_summary and list_not_empty(final_summary.reviewer_attention_points)) or has_any_ci(answer, ['reviewer', 'attention', 'pay attention', '评审', '关注'])
    evaluator.add_custom_node(
        result=bool(reviewer_points_ok),
        id="summary_reviewer_attention",
        desc="Summary includes points reviewers pay special attention to",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
