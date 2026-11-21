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
TASK_ID = "task-035e06"
TASK_DESCRIPTION = 'I’m writing a sci-fi script about AI ethics and want to systematically gather creative source material. First, go to IMDb and search for sci-fi films with theme tags including “artificial intelligence” or “dystopia,” with ratings above 7.5. Select 5 classic works and record the title, director, IMDb rating, and detail-page link for each.\n\nThen, search these 5 films on Letterboxd. For each film, check the community rating and top user reviews (prioritize highly liked reviews; if the page cannot be reliably sorted by likes, select the 3 most relevant visible reviews and state your selection criteria), and extract keywords describing commonly discussed narrative techniques (e.g., “multi-thread narrative,” “time loop,” “first-person perspective”).\n\nNext, on Goodreads, try to find the original novels corresponding to these films or books from the same IP/franchise. Prioritize books with ratings above 4.0 and more than 1,000 reviews, and aim to find 3 books. If fewer than 3 are available, keep the results found and clearly state the gap. Record the title, author, Goodreads rating, philosophical topics mentioned in popular reviews (e.g., “free will,” “the nature of consciousness”), and link.\n\nThen, go to Semantic Scholar and search for “AI ethics” or “machine consciousness.” Sort by citation count and find 5 papers with more than 100 citations (if fewer than 5 exist, record the actual number found and explain). Record the title, authors, citation count, publication year, abstract, and paper link.\n\nFinally, on YouTube, find in-depth analysis videos for any 2 of the 5 films (e.g., “[Film] review,” “[Film] analysis”), with each video longer than 15 minutes and having more than 100,000 views; find 1 video per film. If a specific film has insufficient results, you may switch to another film among the 5 and expand search keywords (e.g., “ending explained,” “deep dive”). Record the video title, channel name, view count, duration, and link.\n\nOutput format:\n- IMDb films (title, director, rating, link)\n- Letterboxd info (community rating, narrative-technique keywords)\n- Goodreads books (title, author, rating, philosophical topics, link)\n- Papers (title, authors, citation count, year, abstract, link)\n- YouTube videos (title, channel, views, duration, link)\n\nIf any step encounters a login wall or regional restriction, record the limitation and continue with the remaining steps.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IMDbFilm(BaseModel):
    """Single IMDb film entry"""
    title: Optional[str] = None
    director: Optional[str] = None
    rating: Optional[float] = None
    link: Optional[str] = None


class IMDbFilms(BaseModel):
    """All IMDb films extracted from the answer"""
    films: List[IMDbFilm] = Field(default_factory=list)


class LetterboxdFilm(BaseModel):
    """Letterboxd info for a single film"""
    title: Optional[str] = None
    community_rating: Optional[float] = None
    narrative_keywords: List[str] = Field(default_factory=list)


class LetterboxdFilms(BaseModel):
    """All Letterboxd films extracted from the answer"""
    films: List[LetterboxdFilm] = Field(default_factory=list)


class GoodreadsBook(BaseModel):
    """Single Goodreads book entry"""
    title: Optional[str] = None
    author: Optional[str] = None
    rating: Optional[float] = None
    philosophical_topics: List[str] = Field(default_factory=list)
    link: Optional[str] = None


class GoodreadsBooks(BaseModel):
    """All Goodreads books extracted from the answer"""
    books: List[GoodreadsBook] = Field(default_factory=list)


class Paper(BaseModel):
    """Single paper entry"""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    citation_count: Optional[int] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    link: Optional[str] = None


class Papers(BaseModel):
    """All papers extracted from the answer"""
    papers: List[Paper] = Field(default_factory=list)


class YouTubeVideo(BaseModel):
    """Single YouTube video entry"""
    title: Optional[str] = None
    channel: Optional[str] = None
    views: Optional[int] = None
    duration_minutes: Optional[int] = None
    link: Optional[str] = None


class YouTubeVideos(BaseModel):
    """All YouTube videos extracted from the answer"""
    videos: List[YouTubeVideo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_imdb_films() -> str:
    return """
Extract all IMDb films mentioned in the answer.

For each film, return:
- title: the film title
- director: the director name
- rating: the IMDb rating as a float (e.g., 7.5, 8.2)
- link: the IMDb detail page link

If any field is missing, set it to null. Return all films found in a list.
"""


def prompt_extract_letterboxd_films() -> str:
    return """
Extract all Letterboxd film information mentioned in the answer.

For each film, return:
- title: the film title
- community_rating: the Letterboxd community rating as a float
- narrative_keywords: list of narrative technique keywords extracted from reviews (e.g., ["multi-thread narrative", "time loop"])

If any field is missing, set it to null or empty list. Return all films found.
"""


def prompt_extract_goodreads_books() -> str:
    return """
Extract all Goodreads books mentioned in the answer.

For each book, return:
- title: the book title
- author: the author name
- rating: the Goodreads rating as a float
- philosophical_topics: list of philosophical topics mentioned in reviews (e.g., ["free will", "consciousness"])
- link: the Goodreads book link

If any field is missing, set it to null or empty list. Return all books found.
"""


def prompt_extract_papers() -> str:
    return """
Extract all academic papers mentioned in the answer.

For each paper, return:
- title: the paper title
- authors: list of author names
- citation_count: the citation count as an integer
- year: the publication year as an integer
- abstract: the paper abstract text
- link: the paper link (arXiv or Semantic Scholar)

If any field is missing, set it to null or empty list. Return all papers found.
"""


def prompt_extract_youtube_videos() -> str:
    return """
Extract all YouTube videos mentioned in the answer.

For each video, return:
- title: the video title
- channel: the channel name
- views: the view count as an integer
- duration_minutes: the duration in minutes as an integer
- link: the YouTube video link

If any field is missing, set it to null. Return all videos found.
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


def count_non_null(items: List, attr: str) -> int:
    return sum(1 for item in items if getattr(item, attr, None) is not None)


def all_have_attr(items: List, attr: str) -> bool:
    return all(getattr(item, attr, None) is not None for item in items)


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
    imdb_films = await evaluator.extract(
        prompt=prompt_extract_imdb_films(),
        template_class=IMDbFilms,
        extraction_name="imdb_films"
    )

    letterboxd_films = await evaluator.extract(
        prompt=prompt_extract_letterboxd_films(),
        template_class=LetterboxdFilms,
        extraction_name="letterboxd_films"
    )

    goodreads_books = await evaluator.extract(
        prompt=prompt_extract_goodreads_books(),
        template_class=GoodreadsBooks,
        extraction_name="goodreads_books"
    )

    papers = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=Papers,
        extraction_name="papers"
    )

    youtube_videos = await evaluator.extract(
        prompt=prompt_extract_youtube_videos(),
        template_class=YouTubeVideos,
        extraction_name="youtube_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb AI/dystopia sci-fi films with ratings above 7.5",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Search type selection
    imdb_search_action = has_any_ci(answer, ['imdb']) and has_any_ci(answer, ['artificial intelligence', 'dystopia', 'sci-fi', 'science fiction'])
    evaluator.add_custom_node(
        result=bool(imdb_search_action),
        id="imdb_action_search_type",
        desc="[Action Node] imdb.com:F1:A1 - Search for films with AI/dystopia theme tags on IMDb",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F9:A11 - Genre filter (sci-fi)
    scifi_filter = has_any_ci(answer, ['sci-fi', 'science fiction'])
    evaluator.add_custom_node(
        result=bool(scifi_filter),
        id="imdb_action_genre_filter",
        desc="[Action Node] imdb.com:F9:A11 - Filter by sci-fi genre",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F9:A19 - Rating filter (>7.5)
    films = imdb_films.films if imdb_films else []
    all_ratings_above_75 = all(f.rating and f.rating >= 7.5 for f in films if f.rating is not None)
    evaluator.add_custom_node(
        result=bool(len(films) > 0 and all_ratings_above_75),
        id="imdb_action_rating_filter",
        desc="[Action Node] imdb.com:F9:A19 - Filter by rating above 7.5",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F1:A10 - Search results pagination
    has_5_films = len(films) == 5
    evaluator.add_custom_node(
        result=bool(has_5_films),
        id="imdb_action_pagination",
        desc="[Action Node] imdb.com:F1:A10 - Browse search results to find 5 films",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A26 - Click into detail pages
    has_directors = count_non_null(films, 'director') >= 4
    has_links = count_non_null(films, 'link') >= 4
    evaluator.add_custom_node(
        result=bool(has_directors and has_links),
        id="imdb_action_detail_pages",
        desc="[Action Node] imdb.com:F3:A26 - Click into film detail pages to get director and links",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P17 - Extract director from cast info
    directors_extracted = count_non_null(films, 'director') >= 4
    evaluator.add_custom_node(
        result=bool(directors_extracted),
        id="imdb_perception_directors",
        desc="[Perception Node] imdb.com:F3:P17 - Extract director names from film detail pages",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Letterboxd section
    letterboxd_node = evaluator.add_sequential(
        id="letterboxd_section",
        desc="Letterboxd community ratings and narrative technique keywords",
        parent=root,
        critical=False
    )

    lb_films = letterboxd_films.films if letterboxd_films else []

    # [Action Node] letterboxd.com:F1:A5 - Click film cards
    letterboxd_search = has_any_ci(answer, ['letterboxd']) and len(lb_films) >= 4
    evaluator.add_custom_node(
        result=bool(letterboxd_search),
        id="letterboxd_action_click_cards",
        desc="[Action Node] letterboxd.com:F1:A5 - Search and click film cards on Letterboxd",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A8 - Switch to reviews tab
    has_keywords = any(len(f.narrative_keywords) > 0 for f in lb_films)
    evaluator.add_custom_node(
        result=bool(has_keywords),
        id="letterboxd_action_reviews_tab",
        desc="[Action Node] letterboxd.com:F3:A8 - Switch to Community Reviews tab",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A16 - Browse review list
    evaluator.add_custom_node(
        result=bool(has_keywords),
        id="letterboxd_action_browse_reviews",
        desc="[Action Node] letterboxd.com:F3:A16 - Browse review list to find top reviews",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P6 - Community ratings
    has_lb_ratings = count_non_null(lb_films, 'community_rating') >= 4
    evaluator.add_custom_node(
        result=bool(has_lb_ratings),
        id="letterboxd_perception_ratings",
        desc="[Perception Node] letterboxd.com:F3:P6 - Extract community ratings",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P7 - Review content and narrative keywords
    evaluator.add_custom_node(
        result=bool(has_keywords),
        id="letterboxd_perception_keywords",
        desc="[Perception Node] letterboxd.com:F3:P7 - Extract narrative technique keywords from reviews",
        parent=letterboxd_node,
        critical=False
    )

    # 3.3 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads books related to the films with ratings and philosophical topics",
        parent=root,
        critical=False
    )

    gr_books = goodreads_books.books if goodreads_books else []

    # [Action Node] goodreads.com:F1:A5 - Click book cards
    goodreads_search = has_any_ci(answer, ['goodreads']) and len(gr_books) >= 1
    evaluator.add_custom_node(
        result=bool(goodreads_search),
        id="goodreads_action_click_cards",
        desc="[Action Node] goodreads.com:F1:A5 - Search and click book cards on Goodreads",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F2:A2 - Switch to reviews tab
    has_topics = any(len(b.philosophical_topics) > 0 for b in gr_books)
    evaluator.add_custom_node(
        result=bool(has_topics),
        id="goodreads_action_reviews_tab",
        desc="[Action Node] goodreads.com:F2:A2 - Switch to Community Reviews tab",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P5 - Book metadata (rating, review count)
    ratings_above_4 = all(b.rating and b.rating >= 4.0 for b in gr_books if b.rating is not None)
    evaluator.add_custom_node(
        result=bool(len(gr_books) > 0 and ratings_above_4),
        id="goodreads_perception_metadata",
        desc="[Perception Node] goodreads.com:F2:P5 - Extract book ratings above 4.0 and review counts",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P6 - Hover for precise rating
    has_precise_ratings = count_non_null(gr_books, 'rating') >= 1
    evaluator.add_custom_node(
        result=bool(has_precise_ratings),
        id="goodreads_perception_precise_rating",
        desc="[Perception Node] goodreads.com:F2:P6 - Extract precise ratings from book pages",
        parent=goodreads_node,
        critical=False
    )

    # 3.4 arXiv/Semantic Scholar section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="Academic papers on AI ethics with high citation counts",
        parent=root,
        critical=False
    )

    paper_list = papers.papers if papers else []

    # [Action Node] arxiv.org:F1:A1 - Select search field
    arxiv_search = (has_any_ci(answer, ['arxiv', 'semantic scholar']) and
                    has_any_ci(answer, ['ai ethics', 'machine consciousness']))
    evaluator.add_custom_node(
        result=bool(arxiv_search),
        id="arxiv_action_search_field",
        desc="[Action Node] arxiv.org:F1:A1 - Select search field and query AI ethics topics",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A2 - Submit search
    evaluator.add_custom_node(
        result=bool(len(paper_list) >= 4),
        id="arxiv_action_submit_search",
        desc="[Action Node] arxiv.org:F1:A2 - Submit search query and get results",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A16 - Pagination for high-citation papers
    citations_above_100 = all(p.citation_count and p.citation_count >= 100 for p in paper_list if p.citation_count is not None)
    evaluator.add_custom_node(
        result=bool(len(paper_list) >= 4 and citations_above_100),
        id="arxiv_action_pagination",
        desc="[Action Node] arxiv.org:F1:A16 - Browse multiple pages to find papers with 100+ citations",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A6 - Click into paper details
    has_abstracts = count_non_null(paper_list, 'abstract') >= 4
    evaluator.add_custom_node(
        result=bool(has_abstracts),
        id="arxiv_action_paper_details",
        desc="[Action Node] arxiv.org:F3:A6 - Click into paper detail pages",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A20 - Expand abstract if collapsed
    evaluator.add_custom_node(
        result=bool(has_abstracts),
        id="arxiv_action_expand_abstract",
        desc="[Action Node] arxiv.org:F3:A20 - Expand collapsed abstracts to view full text",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P5 - Extract abstract content
    evaluator.add_custom_node(
        result=bool(has_abstracts),
        id="arxiv_perception_abstracts",
        desc="[Perception Node] arxiv.org:F3:P5 - Extract paper abstracts and understand research content",
        parent=arxiv_node,
        critical=False
    )

    # 3.5 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube in-depth analysis videos for 2 of the 5 films",
        parent=root,
        critical=False
    )

    yt_videos = youtube_videos.videos if youtube_videos else []

    # [Action Node] youtube.com:F1:A22 - Click video cards
    youtube_search = has_any_ci(answer, ['youtube']) and len(yt_videos) >= 1
    evaluator.add_custom_node(
        result=bool(youtube_search),
        id="youtube_action_click_cards",
        desc="[Action Node] youtube.com:F1:A22 - Search and click video cards for film analysis",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more results
    views_above_100k = all(v.views and v.views >= 100000 for v in yt_videos if v.views is not None)
    duration_above_15 = all(v.duration_minutes and v.duration_minutes >= 15 for v in yt_videos if v.duration_minutes is not None)
    evaluator.add_custom_node(
        result=bool(len(yt_videos) >= 2 and views_above_100k and duration_above_15),
        id="youtube_action_scroll_results",
        desc="[Action Node] youtube.com:F1:A69 - Scroll search results to find videos with 100k+ views and 15+ minutes duration",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Video relevance
    titles_relevant = all(v.title and (has_any_ci(v.title, ['analysis', 'review', 'explained', 'deep dive'])) for v in yt_videos if v.title)
    evaluator.add_custom_node(
        result=bool(len(yt_videos) >= 2 and titles_relevant),
        id="youtube_perception_relevance",
        desc="[Perception Node] youtube.com:F1:P4 - Identify video relevance to film analysis from titles and thumbnails",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P5 - Filter out ads
    evaluator.add_custom_node(
        result=bool(len(yt_videos) >= 2),
        id="youtube_perception_filter_ads",
        desc="[Perception Node] youtube.com:F1:P5 - Filter out sponsored/ad content from organic results",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
