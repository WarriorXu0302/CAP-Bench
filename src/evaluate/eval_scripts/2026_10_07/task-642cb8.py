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
TASK_ID = "task-642cb8"
TASK_DESCRIPTION = 'I’ve recently started learning machine learning and want to find a high-quality hands-on project tutorial on Bilibili to follow. Please search using keywords like “machine learning project practice” or “deep learning project,” sort by view count, and identify videos with at least 500,000 views.\n\nFor each candidate video, check whether the description includes a GitHub repository link. If the description is unclear, expand it to view the full text, and also check the pinned comment in case the uploader added the code link there.\n\nAfter finding a GitHub link, verify whether the repository actually exists and whether the code appears complete. Check for files such as a README and requirements.txt, and see whether the repository has been updated recently.\n\nIf a video does not provide a GitHub link or the link is invalid, move on to the next video that meets the view-count requirement and continue until you find at least one accessible repository. If none can be found in the end, clearly state the scope of what was checked and why no valid result was found.\n\nFinally, tell me whether the video is worth following for learning and whether the code can be used directly.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BilibiliSearchInfo(BaseModel):
    """Information about Bilibili search and video selection"""
    search_keywords_used: Optional[str] = None
    sorted_by_views: Optional[bool] = None
    video_titles_found: Optional[List[str]] = None
    view_counts_mentioned: Optional[List[str]] = None


class VideoDescriptionInfo(BaseModel):
    """Information about video description and GitHub links"""
    description_checked: Optional[bool] = None
    description_expanded: Optional[bool] = None
    github_link_in_description: Optional[str] = None
    pinned_comment_checked: Optional[bool] = None
    github_link_in_comment: Optional[str] = None


class GitHubVerificationInfo(BaseModel):
    """Information about GitHub repository verification"""
    repository_url: Optional[str] = None
    repository_exists: Optional[bool] = None
    has_readme: Optional[bool] = None
    has_requirements: Optional[bool] = None
    recently_updated: Optional[bool] = None
    update_time_mentioned: Optional[str] = None


class RecommendationInfo(BaseModel):
    """Final recommendation from the answer"""
    worth_following: Optional[bool] = None
    code_usable_directly: Optional[bool] = None
    recommendation_reasoning: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bilibili_search() -> str:
    return """
Extract information about the Bilibili search performed from the answer.

Return:
- search_keywords_used: the keywords mentioned as used for search (e.g., "machine learning project practice", "deep learning project")
- sorted_by_views: whether the answer mentions sorting by view count (true/false)
- video_titles_found: list of video titles mentioned in the answer
- view_counts_mentioned: list of view count numbers or descriptions mentioned

If any field is missing, set it to null or empty list.
"""


def prompt_extract_video_description() -> str:
    return """
Extract information about video description checking from the answer.

Return:
- description_checked: whether the answer mentions checking video descriptions
- description_expanded: whether the answer mentions expanding descriptions to see full text
- github_link_in_description: any GitHub link found in description (full URL if present)
- pinned_comment_checked: whether the answer mentions checking pinned comments
- github_link_in_comment: any GitHub link found in pinned comment (full URL if present)

If any field is missing, set it to null.
"""


def prompt_extract_github_verification() -> str:
    return """
Extract information about GitHub repository verification from the answer.

Return:
- repository_url: the GitHub repository URL mentioned
- repository_exists: whether the answer confirms the repository exists
- has_readme: whether the answer mentions finding a README file
- has_requirements: whether the answer mentions finding requirements.txt or similar dependency file
- recently_updated: whether the answer indicates the repository was recently updated
- update_time_mentioned: any specific update time or date mentioned

If any field is missing, set it to null.
"""


def prompt_extract_recommendation() -> str:
    return """
Extract the final recommendation from the answer.

Return:
- worth_following: whether the answer recommends the video is worth following for learning
- code_usable_directly: whether the answer indicates the code can be used directly
- recommendation_reasoning: the reasoning provided for the recommendation

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
    # Handle Chinese numbers and formats like "50万" (500,000)
    if '万' in text:
        match = re.search(r'(\d+(?:\.\d+)?)\s*万', text)
        if match:
            return float(match.group(1)) * 10000
    # Handle regular numbers with commas or without
    match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', text.replace(',', ''))
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except Exception:
            return None
    return None


def looks_like_github_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'github\.com', text.lower()))


def mentions_view_threshold(answer: str) -> bool:
    # Check for 500,000 or 50万
    if '500' in answer and ('000' in answer or 'k' in answer.lower() or 'thousand' in answer.lower()):
        return True
    if '50' in answer and '万' in answer:
        return True
    return False


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
    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_search(),
        template_class=BilibiliSearchInfo,
        extraction_name="bilibili_search_info"
    )

    description_info = await evaluator.extract(
        prompt=prompt_extract_video_description(),
        template_class=VideoDescriptionInfo,
        extraction_name="video_description_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_verification(),
        template_class=GitHubVerificationInfo,
        extraction_name="github_verification_info"
    )

    recommendation_info = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=RecommendationInfo,
        extraction_name="recommendation_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Bilibili search and sorting section
    bilibili_search_node = evaluator.add_sequential(
        id="bilibili_search_section",
        desc="Bilibili search with keyword filtering and view count sorting",
        parent=root,
        critical=False
    )

    # Check if appropriate keywords were used
    keywords_ok = has_any_ci(answer, ['machine learning', '机器学习', 'deep learning', '深度学习', 'project', '项目', 'practice', '实战'])
    evaluator.add_custom_node(
        result=bool(keywords_ok),
        id="bilibili_search_keywords",
        desc="Used appropriate search keywords related to machine learning or deep learning projects",
        parent=bilibili_search_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by view count
    sort_by_views = (bilibili_info and bilibili_info.sorted_by_views) or has_any_ci(answer, ['sort', '排序', 'view count', '播放量', 'views'])
    evaluator.add_custom_node(
        result=bool(sort_by_views),
        id="bilibili_sort_by_views",
        desc="[Action Node] bilibili.com:F1:A5 - Sort search results by view count using dropdown menu",
        parent=bilibili_search_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Visual content recognition for video quality
    has_video_titles = bilibili_info and bilibili_info.video_titles_found and len(bilibili_info.video_titles_found) > 0
    mentions_thumbnails_or_titles = has_any_ci(answer, ['video', '视频', 'title', '标题', 'thumbnail', '缩略图'])
    evaluator.add_custom_node(
        result=bool(has_video_titles or mentions_thumbnails_or_titles),
        id="bilibili_visual_recognition",
        desc="[Perception Node] bilibili.com:F1:P30 - Recognize video content relevance from thumbnails and titles",
        parent=bilibili_search_node,
        critical=False
    )

    # Check 500,000 view threshold
    threshold_mentioned = mentions_view_threshold(answer)
    evaluator.add_custom_node(
        result=bool(threshold_mentioned),
        id="bilibili_view_threshold",
        desc="Filter videos with at least 500,000 views (50万播放量)",
        parent=bilibili_search_node,
        critical=False
    )

    # 3.2 Video description and comment checking section
    video_detail_node = evaluator.add_sequential(
        id="video_detail_section",
        desc="Check video description and comments for GitHub repository links",
        parent=root,
        critical=False
    )

    # Check description was examined
    description_checked = (description_info and description_info.description_checked) or has_any_ci(answer, ['description', '简介', 'desc'])
    evaluator.add_custom_node(
        result=bool(description_checked),
        id="video_description_checked",
        desc="Check video description for GitHub repository link",
        parent=video_detail_node,
        critical=False
    )

    # [Action Node] bilibili.com:F2:A46 - Expand description if needed
    description_expanded = (description_info and description_info.description_expanded) or has_any_ci(answer, ['expand', '展开', 'full text', '完整'])
    evaluator.add_custom_node(
        result=bool(description_expanded),
        id="bilibili_expand_description",
        desc="[Action Node] bilibili.com:F2:A46 - Expand description to view full text when unclear",
        parent=video_detail_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F2:P17 - State awareness for expanded content
    state_awareness = has_any_ci(answer, ['expanded', '已展开', 'collapse', '折叠', 'full', '完整'])
    evaluator.add_custom_node(
        result=bool(state_awareness),
        id="bilibili_state_awareness",
        desc="[Perception Node] bilibili.com:F2:P17 - Aware of description expansion state",
        parent=video_detail_node,
        critical=False
    )

    # [Action Node] bilibili.com:F5:A8 - Switch to comments tab
    comments_checked = (description_info and description_info.pinned_comment_checked) or has_any_ci(answer, ['comment', '评论', 'pinned', '置顶'])
    evaluator.add_custom_node(
        result=bool(comments_checked),
        id="bilibili_check_comments",
        desc="[Action Node] bilibili.com:F5:A8 - Switch to comments tab to check pinned comment",
        parent=video_detail_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F5:P19 - Content understanding in comments
    github_link_found = (
        (description_info and looks_like_github_url(description_info.github_link_in_description)) or
        (description_info and looks_like_github_url(description_info.github_link_in_comment)) or
        (github_info and looks_like_github_url(github_info.repository_url)) or
        has_any_ci(answer, ['github.com'])
    )
    evaluator.add_custom_node(
        result=bool(github_link_found),
        id="bilibili_extract_github_link",
        desc="[Perception Node] bilibili.com:F5:P19 - Extract GitHub repository link from description or comments",
        parent=video_detail_node,
        critical=False
    )

    # 3.3 GitHub repository verification section
    github_verify_node = evaluator.add_sequential(
        id="github_verification_section",
        desc="Verify GitHub repository exists and code completeness",
        parent=root,
        critical=False
    )

    # Repository exists
    repo_exists = (github_info and github_info.repository_exists) or has_any_ci(answer, ['repository exists', '仓库存在', 'repo found', 'accessible'])
    evaluator.add_custom_node(
        result=bool(repo_exists),
        id="github_repo_exists",
        desc="Verify GitHub repository exists and is accessible",
        parent=github_verify_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Expand directory tree
    tree_expanded = has_any_ci(answer, ['directory', '目录', 'folder', 'file structure', '文件结构', 'tree', 'expand'])
    evaluator.add_custom_node(
        result=bool(tree_expanded),
        id="github_expand_tree",
        desc="[Action Node] github.com:F3:A17 - Expand directory tree to view project structure",
        parent=github_verify_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Hierarchical understanding
    hierarchy_understood = has_any_ci(answer, ['structure', '结构', 'organized', '组织', 'layout', 'hierarchy'])
    evaluator.add_custom_node(
        result=bool(hierarchy_understood),
        id="github_hierarchy_understanding",
        desc="[Perception Node] github.com:F3:P12 - Understand project directory hierarchy and organization",
        parent=github_verify_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P13 - File type identification
    has_readme = (github_info and github_info.has_readme) or has_any_ci(answer, ['readme', 'README'])
    has_requirements = (github_info and github_info.has_requirements) or has_any_ci(answer, ['requirements', 'dependencies', 'setup.py', 'package.json', 'requirements.txt'])
    evaluator.add_custom_node(
        result=bool(has_readme and has_requirements),
        id="github_key_files_identified",
        desc="[Perception Node] github.com:F3:P13 - Identify key files like README and requirements.txt",
        parent=github_verify_node,
        critical=False
    )

    # Check recent updates
    recently_updated = (github_info and github_info.recently_updated) or has_any_ci(answer, ['updated', '更新', 'recent', '最近', 'commit', 'active'])
    evaluator.add_custom_node(
        result=bool(recently_updated),
        id="github_recent_updates",
        desc="Check whether repository has been updated recently",
        parent=github_verify_node,
        critical=False
    )

    # Code completeness assessment
    code_complete = has_any_ci(answer, ['complete', '完整', 'comprehensive', 'full code', 'working'])
    evaluator.add_custom_node(
        result=bool(code_complete),
        id="github_code_completeness",
        desc="Assess code completeness based on file structure and content",
        parent=github_verify_node,
        critical=False
    )

    # 3.4 Final recommendation section
    recommendation_node = evaluator.add_parallel(
        id="recommendation_section",
        desc="Final recommendation on video learning value and code usability",
        parent=root,
        critical=False
    )

    # Worth following for learning
    worth_following = (recommendation_info and recommendation_info.worth_following is not None) or has_any_ci(answer, ['worth', '值得', 'recommend', '推荐', 'suitable', '适合'])
    evaluator.add_custom_node(
        result=bool(worth_following),
        id="recommendation_worth_following",
        desc="Provide assessment on whether video is worth following for learning",
        parent=recommendation_node,
        critical=False
    )

    # Code usability
    code_usable = (recommendation_info and recommendation_info.code_usable_directly is not None) or has_any_ci(answer, ['usable', '可用', 'directly', '直接', 'ready to use'])
    evaluator.add_custom_node(
        result=bool(code_usable),
        id="recommendation_code_usable",
        desc="Provide assessment on whether code can be used directly",
        parent=recommendation_node,
        critical=False
    )

    # Has reasoning
    has_reasoning = (recommendation_info and recommendation_info.recommendation_reasoning) or len(answer) > 100
    evaluator.add_custom_node(
        result=bool(has_reasoning),
        id="recommendation_has_reasoning",
        desc="Provide reasoning for the recommendation",
        parent=recommendation_node,
        critical=False
    )

    # 3.5 Iterative fallback handling
    fallback_handling = has_any_ci(answer, ['next video', '下一个', 'another', 'continue', '继续', 'alternative', 'if not found', '如果没有'])
    evaluator.add_custom_node(
        result=bool(fallback_handling),
        id="iterative_fallback",
        desc="Handle cases where video lacks valid GitHub link by moving to next candidate",
        parent=root,
        critical=False
    )

    # Clear communication if no result found
    no_result_communication = has_any_ci(answer, ['no valid', '没有找到', 'none found', 'could not find', 'checked', '检查了', 'scope'])
    evaluator.add_custom_node(
        result=bool(no_result_communication),
        id="no_result_clarity",
        desc="Clearly state scope checked and reason if no valid result found",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
