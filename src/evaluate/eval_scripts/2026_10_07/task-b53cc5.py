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
TASK_ID = "task-b53cc5"
TASK_DESCRIPTION = "I am conducting research in computer vision and intend to reproduce recent popular image segmentation models.\n\nHelp me find 3 image segmentation-related papers published on arXiv since 2024. Requirements: each paper must have at least 50 citations and an available PDF download. Record the paper title, authors, publication year, citation count, and arXiv link for each.\n\nNext, use each paper's title or the main author's name to search for the corresponding official code implementation on GitHub. Filter repositories with over 500 stars. Record the repository name, star count, last updated date, and GitHub link.\n\nThen, use the paper title to search for relevant technical discussions on Stack Overflow. Find at least 2 highly upvoted question threads. Identify common issues encountered during reproduction (e.g., environment setup, dependency conflicts, pre-trained weight loading) and extract the problem descriptions and key takeaways from the highly upvoted answers.\n\nFinally, search for the titles of these papers or model names on Bilibili. Find in-depth explanation videos with over 10,000 views and a duration of more than 15 minutes. Record the video title, uploader, view count, duration, and Bilibili link.\n\nFor each paper, output a complete resource package including: paper title, authors, publication year, citation count, arXiv link, GitHub repository name, star count, last updated date, GitHub link, Stack Overflow relevant question links and a summary of key issues/solutions (at least 2 questions), Bilibili explanation video title, uploader, view count, duration, and Bilibili link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information for a single paper extracted from the answer"""
    title: Optional[str] = None
    authors: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: Optional[int] = None
    arxiv_link: Optional[str] = None


class GitHubRepoInfo(BaseModel):
    """GitHub repository information for a single paper extracted from the answer"""
    repo_name: Optional[str] = None
    star_count: Optional[int] = None
    last_updated: Optional[str] = None
    github_link: Optional[str] = None


class StackOverflowInfo(BaseModel):
    """Stack Overflow discussion information for a single paper extracted from the answer"""
    question_links: Optional[List[str]] = Field(default_factory=list)
    issues_summary: Optional[str] = None


class BilibiliVideoInfo(BaseModel):
    """Bilibili video information for a single paper extracted from the answer"""
    video_title: Optional[str] = None
    uploader: Optional[str] = None
    view_count: Optional[int] = None
    duration: Optional[str] = None
    bilibili_link: Optional[str] = None


class CompleteResourcePackage(BaseModel):
    """Complete resource package for all papers extracted from the answer"""
    papers: Optional[List[PaperInfo]] = Field(default_factory=list)
    github_repos: Optional[List[GitHubRepoInfo]] = Field(default_factory=list)
    stackoverflow_discussions: Optional[List[StackOverflowInfo]] = Field(default_factory=list)
    bilibili_videos: Optional[List[BilibiliVideoInfo]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_papers_from_answer() -> str:
    return """
Extract all image segmentation papers mentioned in the answer that were found on arXiv.

For each paper, extract:
- title: the paper title exactly as stated
- authors: the authors list or string exactly as stated
- publication_year: the publication year as an integer (e.g., 2024)
- citation_count: the number of citations as an integer
- arxiv_link: the arXiv link URL

Return a list of papers. If any field is missing for a paper, set it to null.
"""


def prompt_extract_github_repos_from_answer() -> str:
    return """
Extract all GitHub repository information mentioned in the answer for the papers.

For each repository, extract:
- repo_name: the repository name exactly as stated
- star_count: the number of stars as an integer
- last_updated: the last updated date exactly as stated
- github_link: the GitHub repository URL

Return a list of repositories. If any field is missing, set it to null.
"""


def prompt_extract_stackoverflow_from_answer() -> str:
    return """
Extract all Stack Overflow discussion information mentioned in the answer for the papers.

For each paper's Stack Overflow discussions, extract:
- question_links: a list of Stack Overflow question URLs
- issues_summary: a summary of common issues and solutions mentioned

Return a list of Stack Overflow discussion entries. If any field is missing, set it to null or empty list.
"""


def prompt_extract_bilibili_videos_from_answer() -> str:
    return """
Extract all Bilibili video information mentioned in the answer for the papers.

For each video, extract:
- video_title: the video title exactly as stated
- uploader: the uploader name exactly as stated
- view_count: the view count as an integer
- duration: the video duration exactly as stated
- bilibili_link: the Bilibili video URL

Return a list of videos. If any field is missing, set it to null.
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


def contains_url_pattern(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower()


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\b(202[4-9])\b', text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def count_items_in_list(items: Optional[List]) -> int:
    if not items:
        return 0
    return len(items)


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
        template_class=CompleteResourcePackage,
        extraction_name="papers_from_arxiv"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_repos_from_answer(),
        template_class=CompleteResourcePackage,
        extraction_name="github_repositories"
    )

    stackoverflow_info = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_from_answer(),
        template_class=CompleteResourcePackage,
        extraction_name="stackoverflow_discussions"
    )

    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_videos_from_answer(),
        template_class=CompleteResourcePackage,
        extraction_name="bilibili_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 arXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="arXiv paper search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A2 - Search form submission
    arxiv_search_ok = (has_any_ci(answer, ['arxiv']) and
                       has_any_ci(answer, ['image segmentation', 'segmentation']))
    evaluator.add_custom_node(
        result=bool(arxiv_search_ok),
        id="arxiv_search_action",
        desc="[Action Node] arxiv.org:F1:A2 - Submit search for image segmentation papers on arXiv",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F1:P11 - Understanding search result relevance
    papers_list = papers_info.papers if papers_info and papers_info.papers else []
    has_segmentation_papers = any(
        paper and paper.title and has_any_ci(paper.title, ['segment', 'segmentation'])
        for paper in papers_list
    )
    evaluator.add_custom_node(
        result=bool(has_segmentation_papers),
        id="arxiv_relevance_perception",
        desc="[Perception Node] arxiv.org:F1:P11 - Identify papers relevant to image segmentation",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A6 - Click into paper details
    has_complete_paper_info = any(
        paper and paper.title and paper.authors
        for paper in papers_list
    )
    evaluator.add_custom_node(
        result=bool(has_complete_paper_info),
        id="arxiv_details_action",
        desc="[Action Node] arxiv.org:F3:A6 - Navigate to paper detail pages to extract complete information",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P4 - Recognize author information
    has_authors = any(
        paper and paper.authors and len(str(paper.authors).strip()) > 0
        for paper in papers_list
    )
    evaluator.add_custom_node(
        result=bool(has_authors),
        id="arxiv_authors_perception",
        desc="[Perception Node] arxiv.org:F3:P4 - Extract author information from paper details",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P7 - Recognize publication year and version history
    papers_since_2024 = [
        paper for paper in papers_list
        if paper and paper.publication_year and paper.publication_year >= 2024
    ]
    evaluator.add_custom_node(
        result=bool(len(papers_since_2024) >= 1),
        id="arxiv_year_perception",
        desc="[Perception Node] arxiv.org:F3:P7 - Identify papers published since 2024",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P24 - Understanding citation impact
    papers_with_citations = [
        paper for paper in papers_list
        if paper and paper.citation_count and paper.citation_count >= 50
    ]
    evaluator.add_custom_node(
        result=bool(len(papers_with_citations) >= 1),
        id="arxiv_citations_perception",
        desc="[Perception Node] arxiv.org:F3:P24 - Identify papers with at least 50 citations",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F4:A7 - Download PDF
    has_pdf_links = any(
        paper and paper.arxiv_link and contains_url_pattern(paper.arxiv_link, 'arxiv')
        for paper in papers_list
    )
    evaluator.add_custom_node(
        result=bool(has_pdf_links),
        id="arxiv_pdf_action",
        desc="[Action Node] arxiv.org:F4:A7 - Verify PDF download availability via arXiv links",
        parent=arxiv_node,
        critical=False
    )

    # Check for 3 complete papers
    complete_papers_count = len([
        p for p in papers_list
        if p and p.title and p.authors and p.publication_year and p.citation_count and p.arxiv_link
    ])
    evaluator.add_custom_node(
        result=bool(complete_papers_count >= 3),
        id="arxiv_three_papers",
        desc="Found 3 complete papers with all required information",
        parent=arxiv_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository search and information extraction",
        parent=root,
        critical=False
    )

    github_repos = github_info.github_repos if github_info and github_info.github_repos else []

    # Search action
    has_github_search = has_any_ci(answer, ['github']) and len(github_repos) > 0
    evaluator.add_custom_node(
        result=bool(has_github_search),
        id="github_search_action",
        desc="Search GitHub for official code implementations using paper titles or author names",
        parent=github_node,
        critical=False
    )

    # Filter repositories with over 500 stars
    repos_over_500_stars = [
        repo for repo in github_repos
        if repo and repo.star_count and repo.star_count > 500
    ]
    evaluator.add_custom_node(
        result=bool(len(repos_over_500_stars) >= 1),
        id="github_star_filter",
        desc="Filter and identify repositories with over 500 stars",
        parent=github_node,
        critical=False
    )

    # Check for complete repository information
    complete_repos = [
        repo for repo in github_repos
        if repo and repo.repo_name and repo.star_count and repo.last_updated and repo.github_link
    ]
    evaluator.add_custom_node(
        result=bool(len(complete_repos) >= 3),
        id="github_complete_info",
        desc="Extract complete repository information (name, stars, last updated, link) for 3 papers",
        parent=github_node,
        critical=False
    )

    # 3.3 Stack Overflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="Stack Overflow technical discussion search and extraction",
        parent=root,
        critical=False
    )

    so_discussions = stackoverflow_info.stackoverflow_discussions if stackoverflow_info and stackoverflow_info.stackoverflow_discussions else []

    # [Action Node] stackoverflow.com:F1:A2 - Search form submission
    has_so_search = has_any_ci(answer, ['stack overflow', 'stackoverflow'])
    evaluator.add_custom_node(
        result=bool(has_so_search),
        id="stackoverflow_search_action",
        desc="[Action Node] stackoverflow.com:F1:A2 - Search Stack Overflow using paper titles",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F4:A21 - Navigate into question details
    total_question_links = sum(
        count_items_in_list(disc.question_links) for disc in so_discussions if disc
    )
    evaluator.add_custom_node(
        result=bool(total_question_links >= 2),
        id="stackoverflow_question_details_action",
        desc="[Action Node] stackoverflow.com:F4:A21 - Navigate into at least 2 highly upvoted question threads",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F4:P5 - Identify accepted answers
    has_solutions_summary = any(
        disc and disc.issues_summary and len(disc.issues_summary.strip()) > 0
        for disc in so_discussions
    )
    evaluator.add_custom_node(
        result=bool(has_solutions_summary),
        id="stackoverflow_answers_perception",
        desc="[Perception Node] stackoverflow.com:F4:P5 - Extract key takeaways from highly upvoted answers",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F4:P14 - Understanding code issues
    reproduction_issues_mentioned = has_any_ci(answer, [
        'environment', 'dependency', 'dependencies', 'pre-trained', 'pretrained',
        'weight', 'weights', 'configuration', 'setup', 'conflict'
    ])
    evaluator.add_custom_node(
        result=bool(reproduction_issues_mentioned),
        id="stackoverflow_code_issues_perception",
        desc="[Perception Node] stackoverflow.com:F4:P14 - Identify reproduction issues like environment setup, dependencies, or weight loading",
        parent=stackoverflow_node,
        critical=False
    )

    # 3.4 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili video search and information extraction",
        parent=root,
        critical=False
    )

    bilibili_videos = bilibili_info.bilibili_videos if bilibili_info and bilibili_info.bilibili_videos else []

    # [Action Node] bilibili.com:F1:A5 - Sort by view count
    has_bilibili_search = has_any_ci(answer, ['bilibili', 'b站'])
    evaluator.add_custom_node(
        result=bool(has_bilibili_search),
        id="bilibili_search_sort_action",
        desc="[Action Node] bilibili.com:F1:A5 - Search and sort videos by view count to find popular content",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Identify content relevance
    relevant_videos = [
        video for video in bilibili_videos
        if video and video.video_title
    ]
    evaluator.add_custom_node(
        result=bool(len(relevant_videos) >= 1),
        id="bilibili_relevance_perception",
        desc="[Perception Node] bilibili.com:F1:P30 - Identify videos relevant to paper titles or model names",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F3:P1 - Identify in-depth explanation videos
    indepth_videos = [
        video for video in bilibili_videos
        if video and video.video_title and has_any_ci(video.video_title, ['讲解', '解析', '详解', '教程'])
    ]
    evaluator.add_custom_node(
        result=bool(len(indepth_videos) >= 1),
        id="bilibili_indepth_perception",
        desc="[Perception Node] bilibili.com:F3:P1 - Identify in-depth explanation videos from thumbnails and titles",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F3:A33 - Click into video details
    complete_videos = [
        video for video in bilibili_videos
        if video and video.video_title and video.uploader and video.view_count and video.duration and video.bilibili_link
    ]
    evaluator.add_custom_node(
        result=bool(len(complete_videos) >= 1),
        id="bilibili_details_action",
        desc="[Action Node] bilibili.com:F3:A33 - Navigate to video pages to extract complete information",
        parent=bilibili_node,
        critical=False
    )

    # Check for videos over 10,000 views
    popular_videos = [
        video for video in bilibili_videos
        if video and video.view_count and video.view_count >= 10000
    ]
    evaluator.add_custom_node(
        result=bool(len(popular_videos) >= 1),
        id="bilibili_view_count",
        desc="Filter videos with over 10,000 views",
        parent=bilibili_node,
        critical=False
    )

    # Check for videos over 15 minutes (lenient duration check)
    duration_ok_videos = [
        video for video in bilibili_videos
        if video and video.duration and (
            has_any_ci(video.duration, ['15:', '16:', '17:', '18:', '19:', '20:', '21:', '22:', '23:', '24:', '25:', '26:', '27:', '28:', '29:']) or
            re.search(r'[3-9]\d:', str(video.duration)) or
            has_any_ci(video.duration, ['hour', '小时', 'h'])
        )
    ]
    evaluator.add_custom_node(
        result=bool(len(duration_ok_videos) >= 1),
        id="bilibili_duration",
        desc="Filter videos with duration over 15 minutes",
        parent=bilibili_node,
        critical=False
    )

    # 3.5 Overall completeness check
    overall_node = evaluator.add_parallel(
        id="overall_completeness",
        desc="Overall resource package completeness for all papers",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(complete_papers_count >= 3),
        id="overall_papers_complete",
        desc="All 3 papers have complete arXiv information",
        parent=overall_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(len(complete_repos) >= 3),
        id="overall_github_complete",
        desc="All 3 papers have corresponding GitHub repositories",
        parent=overall_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(total_question_links >= 6),
        id="overall_stackoverflow_complete",
        desc="All 3 papers have at least 2 Stack Overflow questions each",
        parent=overall_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(len(complete_videos) >= 3),
        id="overall_bilibili_complete",
        desc="All 3 papers have corresponding Bilibili explanation videos",
        parent=overall_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
