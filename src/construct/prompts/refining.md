# Task Refinement Guide

You are instantiating a task proposal into a concrete, executable, and verifiable benchmark task.

## INPUT

- A single task proposal with user scenario and information flow
- Site cards with functions F, execution items A and perception items P for involved websites

## GOAL-ORIENTED ELICITATION

Express requirements through user goals, NOT UI operations. This ensures complex interactions are triggered implicitly while keeping task descriptions natural.

**WRONG**: "Use the duration filter to select videos under 10 minutes"  
**RIGHT**: "Find tutorial videos under 10 minutes"

**WRONG**: "Scroll down to load more comments, then click expand"  
**RIGHT**: "Check what users are discussing about this product"

**WRONG**: "Click the expand button to show the full description"  
**RIGHT**: "Verify if the video description mentions the original paper"

**WRONG**: "Hover over the seller name to view reputation score"  
**RIGHT**: "Find products from sellers with reputation above 4.8"

## OUTPUT FIELD DESIGN

Every output field must serve one of three verification roles:

**1. User Value**: Core information the user actually needs  
Examples: product name, course title, paper abstract

**2. Condition Verification**: Fields to validate filter criteria used in the task  
CRITICAL: If task says "rating >= 4.5", output MUST include the rating  
If task says "published within 2 years", output MUST include date  
If task says "distance under 5 miles", output MUST include distance

**3. Provenance**: URLs for source verification to prevent hallucination  
Every claimed fact should be traceable to a source URL

## ROBUSTNESS GUIDELINES

**1. Prefer Range Filtering over Superlatives**  
WRONG: "Find the most downloaded model" (sorting may not be supported)  
RIGHT: "Find 5 models with downloads exceeding 10,000"  
Reason: Superlatives require specific sorting features; range filters are more robust

**2. Prefer Relative Time over Absolute Dates**  
WRONG: "Find events on June 20, 2026" (task expires)  
RIGHT: "Find events next weekend"  
Reason: Absolute dates make tasks time-sensitive and eventually invalid

**3. Include Fallback Instructions for Dynamic Content**  
"If this weekend has no events, check next weekend"  
"If the course is unavailable, find alternatives from the same instructor"  
Reason: Web content changes; fallbacks ensure task remains completable

**4. Specify Quantities Explicitly**  
WRONG: "Find some highly-rated restaurants"  
RIGHT: "Find 5 restaurants with rating above 4.5"  
Reason: Explicit quantities enable objective evaluation

**5. Consolidate Output Requirements**  
Place all output requirements at the END of task description, not scattered throughout  
WRONG: "Find books on Goodreads, note title and author. Then check reviews on StoryGraph, record the mood tags. Finally..."  
RIGHT: "Find books on Goodreads... check reviews on StoryGraph... Output for each book: title, author, Goodreads rating, StoryGraph mood tags, and URLs for both platforms."

## TASK DESCRIPTION STYLE

- First person ("I want...", "Help me...")
- Natural conversational tone, as if talking to an AI assistant
- 100-250 words
- Include context that implicitly triggers complex operations

---

# Full Task Refinement Guide (Extended)

## Task Robustness Design

### Core Principle: Range Filtering over Superlatives

**Problem with superlatives**: Queries relying on "most", "hottest", "latest", "ranked first" often require specific sorting features that many websites don't support or produce unstable results.

**Solution**: Use range filtering instead of superlative queries

| Superlative Query [WRONG] | Range Filter [RIGHT] |
|--------------|--------------|
| "Find the 5 courses with most students" | "Find 5 courses with rating above 4.5 and duration over 17 hours" |
| "Find the video with highest views" | "Find 5 videos with over 100,000 views" |
| "Find the Issue with most comments" | "Find 5 Issues with over 50 comments" |
| "Find the best-selling product" | "Find 3 products with over 1,000 sales and rating above 4.5" |

**Exception**: If sorting is core to the task and the website explicitly supports it (e.g., YouTube's "Sort by view count"), it can be used, but should output the specific sorting field values for verification.

### Built-in Fallback Strategies

When tasks involve **features or content with uncertain availability**, naturally embed fallback options in the task description without overshadowing the main goal:

```
[NATURAL] Natural fallback: "Check Weather Underground for historical weather data in Seattle for mid-June"
[LIGHT] Light alternative: "If the course is unavailable, choose another course from the same instructor"
[TIME] Time fallback: "Check next Saturday's event schedule; if not yet published, check this Saturday's"

[WRONG] Over-reliance on fallbacks: "First try site A, if A doesn't work go to B, if B also fails go to C..." (overshadows main goal)
```

### High-Risk Scenario Alerts

Pay special attention when designing tasks in these scenarios:

| Scenario | Risk | Recommendation |
|----------|------|----------------|
| Historical weather data | Some weather sites only have forecasts | Explicitly use Weather Underground, NOAA, etc. |
| Sorting by specific fields | Website may not support it | Use range filtering instead |
| Niche category product filtering | Insufficient product count | Choose popular categories or relax conditions |
| Existence of specific content | May not exist | Choose confirmed existing content, or add light fallback |

## Generalized SQL Perspective: Task Construction Framework

This benchmark tests whether AI browsers ("generalized SQL executors") can correctly execute task instructions ("generalized SQL commands").

| Database Concept | Benchmark Correspondence |
|-----------------|-------------------------|
| Database/Table | Website |
| Field | Information points on the website |
| WHERE condition | Filtering requirements (source of complexity) |
| SELECT fields | Output requirements |
| JOIN | Cross-site information association |
| ORDER BY + LIMIT | Sorting + quantity limits |
| Query result | Text + URLs returned by AI |

**Target SQL pattern**: Multi-condition WHERE + Cross-table JOIN + Clear SELECT + ORDER BY + LIMIT  
**Avoid**: `SELECT * FROM website WHERE keyword='xxx' LIMIT 10`

### Conditional Queries over Existence Checks

| Existence Check [WRONG] | Conditional Query [RIGHT] |
|----------------|--------------|
| "Confirm if it includes XX module" | "Find 5 courses, list their respective modules" |
| "Check if Bestseller tag exists" | "Find 3 products with Bestseller tag" |
| "Verify if views exceed 100,000" | "Find the 5 videos with highest views, provide view counts" |

**Reason**: Existence checks produce only 1 bit of information; conditional queries produce N rows × M columns of data points, making evaluation easier.

### Task Description Coherence Requirements

**Theme Continuity Principle**: Task descriptions should always let readers know "what object is being operated on currently".

| Problem | Example | Fix |
|---------|---------|-----|
| Theme break mid-task | "Want to collect album A... filter Near Mint items" | Use pronouns: "Filter Near Mint items for [that album]" |
| Existence check without follow-up | "Confirm if X exists" | Clarify: "If X exists then...; if not then..." |
| Distance constraint without verification | "Find hotels within 1 mile" | Introduce maps.google.com to verify distance |
| **Abrupt introduction of new elements** | "Finally search 'Gapminder Hans Rosling'" | New elements must have clear connection to preceding context, or be铺垫 in opening persona |

**New Element Introduction Rules**:
- Any **specific names, terms, keywords** appearing in the latter half of the task must:
  - (A) Be dynamically obtained from previous steps, OR
  - (B) Be mentioned in the opening persona/background
- Prohibit suddenly introducing search terms with no logical connection to preceding context mid-task

**Existence checks must specify follow-up actions**:
```
[WRONG] "Confirm if requirements.txt exists"
[RIGHT] "Confirm requirements.txt exists; if not, move to next model until finding one that has it"
[RIGHT] "Output whether requirements.txt exists (yes/no); if not, explain deployment considerations"
```

### Evaluation Logic

Verify whether WHERE condition fields are satisfied, SELECT fields are complete, and use URLs for hallucination prevention.

### Output Field Design Principles

**Core rule: Fields used in WHERE conditions must be required as output items**

| Field Type | Description | Example |
|-----------|-------------|---------|
| User value fields | Information users actually need | Course name, link, instructor |
| WHERE verification fields | For validating filter conditions | Rating, duration, price, date |
| URL verification fields | For hallucination prevention | Detail page links |

**WHERE condition → Output field mapping**:

| WHERE Condition | Must Output |
|----------------|-------------|
| Rating ≥ 4.5 / Price $5-$15 / Duration ≥ 17 hours | Specific values |
| Published within 1 year | Publication date |
| Has certain tag | Whether tag exists |

**Example**:
- [WRONG] Find courses with rating 4.5+ and duration 17+. Output: course name, instructor, price
- [RIGHT] Find courses with rating 4.5+ and duration 17+. Output: course name, instructor, price, **rating, duration**, course link

### Complete Mapping between Outputs and Capability Points

**Core constraint: Every covered_point must have a corresponding core_output for verification.**

| Omission Scenario | Fix Method |
|------------------|------------|
| Expand description but don't require output | Add "description summary" output item |
| Tab switch but don't require output | Add key information output after switching |
| Scroll loading but don't require output | Add "comment viewpoints and count" output item |
| Filter operation but don't require output | Add filter condition value fields |

**Fix example**:
```
[BEFORE FIX]:
Task: Check video description to confirm if it mentions the paper...
Output: Video title, view count, link
Problem: youtube.com:F5:A30 (expand/collapse) → no corresponding output

[AFTER FIX]:
Task: Check video description to confirm if it mentions the paper, briefly explain key information in the description...
Output: Video title, view count, link, **video description summary**
Mapping: youtube.com:F5:A30 → output_ref: o9 (description summary)
```

## Benchmark Positioning Requirements

This benchmark focuses on evaluating **complex operation and complex perception** capabilities. Tasks must include at least one of:
- **Complex UI operations**: Expand/collapse, scroll loading, popup multi-select, drag-and-drop sorting, etc.
- **Data/state perception**: Understanding table data meanings, recognizing UI state changes

**Judgment criterion**: If a task can be completed solely through search engine + LLM, it is not suitable for this benchmark.

**Evaluator limitations**:
- Evaluators can only access text results output by agents; task success must be determinable through text or URLs
- **Avoid image/video perception**: Do not design tasks requiring viewing images/videos, recognizing visual features, or performing visual comparisons
- **Avoid cross-modal matching**: Do not design tasks that convert visual information to search conditions then visually verify

**Ground Truth friendliness**:
- Make task answers relatively closed and fixed for easier GT annotation
- Avoid time-varying results (newest, hottest non-persistent content)
- Focus on relatively stable features, ensure outputs include such features

## Refinement Strategies

### Implicit Guidance Strategy

Adjust task requirements so AI **naturally** encounters complex operation/perception challenges, rather than explicitly requiring them.

**Core distinction**:
- **To be implicit**: UI operation methods (clicking, expanding, using filters)
- **To be explicit**: Specific values of filter conditions (WHERE conditions must be precise for evaluation)

| Target Capability | Explicit Requirement [WRONG] | Implicit Guidance [RIGHT] |
|------------------|-----------------------------|--------------------------|
| Duration filtering | "Use filter function to select under 4 minutes" | "Videos under 4 minutes" |
| Distance constraint | "Measure driving distance on map" | "Driving distance within 15 miles" |
| Scroll loading | "Please scroll page to load more" | "See if anyone in comments mentions XX issue" |
| Expand/collapse | "Please click expand button" | "See how people replied to this hot comment" |
| Popup multi-select | "Please check multiple options in popup" | "Save to 'Study Materials' and 'To Organize' collections" |
| Table understanding | "Please understand meaning of each column" | "Find the 3 journals with highest h5 index" |
| State recognition | "Please determine if button was clicked" | "If not yet saved, help me save it" |
| Multi-source comparison | "Please compare information across three platforms" | "See which platform has the best price" |
| Rule constraints | "Please filter according to rules" | "Control each segment distance to 300-500 miles" |
| Route planning | "Please plan route satisfying constraints" | "Find charging stations along the route meeting range requirements" |

### Increase Localization Difficulty

Avoid point queries (directly giving precise targets), change to conditional queries:

| Point Query [WRONG] | Conditional Query [RIGHT] | Difficulty Source |
|--------------------|--------------------------|------------------|
| Find Google's UX course | Find 5 UX courses, ensure including Google | Requires filtering and comparison |
| Find collection number 90.PA.20 | Find Van Gogh's "Irises" | Requires search and localization |
| Find a specific hospital | Find top 5 hospitals in US News ranking | Requires understanding rankings |

### Sorting Task Requirements

**Four musts**:
1. **Output sorting field values**: Facilitates self-proof of sorting correctness
2. **Specify ascending/descending**: Task description must state sort direction
3. **Appropriate N amplification**: Recommend N≥3, more data points easier to verify
4. **Extra output of M sorting fields**: Besides N complete results, additionally output sorting field values for next M items (no other details needed)

**Example**:
- [WRONG] Find the video with highest views, output title and link
- [RIGHT] Find the 5 videos with highest views, output titles, view counts, and links in descending order by views. Additionally list view counts for ranks 6-8 for reference

### Factual Reasonableness Handling

**Core principle**: Ensure tasks are completable and verifiable. Confirm element existence before using specific descriptions as much as possible.

| Evidence Type | Handling Method |
|--------------|----------------|
| Explicitly included in function point list | Use specific description directly |
| Well-known platform standard features (e.g., LinkedIn's Easy Apply) | Use specific description directly |
| Widely known facts | Use specific description directly |
| Based only on speculation | Fuzzy handling or verify before use |

**Use fuzzy handling ONLY in these cases**:

- Element existence truly cannot be confirmed
- Content is dynamically changing (e.g., hot comments, real-time rankings)
- Multiple valid answers exist requiring AI selection

### Domain Common Sense Verification

When refining tasks, verify whether tasks conform to domain common sense:

| Check Item | Wrong Example | Problem |
|-----------|--------------|---------|
| Content type matching | Go to Genius to find lyrics for "River Flows in You" | Pure instrumental music has no lyrics |
| Platform function matching | Go to GitHub to check paper citation count | GitHub is not an academic platform |
| Entity attribute matching | Check "latest updates" for deceased person | Logical error |

**Verification methods**:
- For unfamiliar domains, **first use search tools to confirm basic facts**
- For each "go to XX to find YY" in the task, confirm XX platform indeed has YY-type content
- When selecting novel entities, first search to confirm they can be found on target websites
- If uncertain, add fallback instructions in the task

### Function Point Selection and Selective Coverage

Upstream `function_points` are potential possibilities, **not all need to be used**. Analyze which are essential to task主线, which may trigger but are non-core, and which are irrelevant and can be ignored.

**Select 8-20 operation/perception points to cover**:

| Priority | Type | Target Count |
|----------|------|-------------|
| High | Complex UI operations (multi-select, expand/collapse, scroll loading, map interaction) | ≥3 |
| High | Data/state perception (table understanding, state recognition, hierarchical navigation) | ≥3 |
| Medium | Pagination browsing, form interaction, sorting/filtering | ≥2 |
| Low | Simple clicks, basic navigation | As needed |

**Coverage density principle**:
- Each core website should cover **at least 1-2** operation or perception points
- Official website additions have no complex items (only simple information retrieval)
- Each hop in the information chain should have **at least 1** complex interaction point
- Total: action_points ≥ 5, perception_points ≥ 4

**Abandonment principle**: If an operation/perception item cannot be naturally integrated into the task, abandon coverage rather than forcing inclusion.

**Coverage recording specification**:
- Each covered operation point (A) and perception point (P) must be listed separately in `covered_points`
- `point_id` format must be precise to `website.com:Fx:Ay` or `website.com:Fx:Pz`
- Prohibit using vague identifiers only to function point (F) level like `website.com:F3`
- Number of `covered_points` entries must strictly equal `action_point_count + perception_point_count`

### Expression Standards

Use **goal-oriented** rather than path-oriented expressions:

| Expression Type | Wrong Example | Right Example |
|----------------|--------------|---------------|
| Operation description | "Click favorite button to open popup then check" | "Save to your Study Materials collection" |
| Content search | "Scroll comment section to load 20 comments" | "See what everyone is discussing in comments" |

## User Instruction Writing Standards

### Style Requirements

| Element | Requirement |
|---------|-------------|
| Person | First person ("I want...", "Help me...") |
| Tone | Natural conversational, like real user talking to AI assistant |
| Length | 100-250 words |
| Key points | Include expressions that implicitly trigger complex operations/perceptions, maintain factual reasonableness |

### Expression Clarity Requirements

| Element | Requirement | Example |
|---------|-------------|---------|
| Task boundaries | Clarify what to do and what not to do | "Select two sessions—one morning, one afternoon" |
| Output format | Explain what information each item needs | "Provide departure time, ticket price, trip duration" |
| Special cases | Preset possible situations and handling methods | "If not yet announced, please clearly state" |
| Verification requirements | Explain how to verify results | "Provide direct link to official website search results page" |
| Fallback instructions | Preset output when not found | "If not found, explain reason and recommend alternatives" |

### Output Requirements Specification

**Output requirements uniformly placed at end of task description**, following three-layer structure:

| Layer | Description | Necessity |
|-------|-------------|-----------|
| User value fields | Core information users actually need | Required |
| WHERE verification fields | Numerical/status fields involved in filter conditions | Required (needed for eval) |
| URL verification fields | Direct links to corresponding pages | Required (hallucination prevention) |

**⚠️ Mandatory requirements**:
- **Do NOT scatter output items after each website operation in task description**
- All output requirements **must be unified at the end of task description**
- Intermediate steps only describe operation goals, do not list "note XX, provide YY"

**Rewrite example**:
- [WRONG] First go to Goodreads to find books, **note title, author and ISBN**. Then go to StoryGraph to check mood tags, **note tag distribution**. Finally go to Amazon to check prices, **provide price and link**.
- [RIGHT] First go to Goodreads to find 5 contemporary literature new books published after September 2024 with rating above 4.2. Then go to StoryGraph to check mood tag distribution and pace evaluation for each book, filter those with emotional/reflective占比 over 60%. Finally check wholesale prices on Amazon. **Output: book title, author, ISBN, Goodreads rating, StoryGraph mood tag percentages, Amazon wholesale price, detail page links for each platform.**

### Positive Examples

**Example 1 (Cross-platform association + conditional filtering)**:

> I'm planning to move with my family to San Antonio, Texas. First find 5 single-family homes for sale on Zillow priced $250-400K, then check elementary school ratings for these 5 properties' school districts on GreatSchools, keeping only those with ratings 8 or above. Output for each property: Zillow link, address, asking price, corresponding elementary school name, elementary school rating, GreatSchools link.

**Analysis**:
- Core websites明确: Zillow + GreatSchools
- WHERE conditions: Price $250-400K, school rating ≥ 8
- **WHERE verification fields**: Asking price, elementary school rating (included in output for verifying filter conditions)
- SELECT fields complete: Property info + school info + dual-platform links

**Example 2 (Implicit triggering + secondary filtering)**:

> I'm looking for Data Scientist jobs in San Francisco and want to make decisions based on both job information and company reputation. First search for "Data Scientist" positions on LinkedIn, select San Francisco location, find 5 positions with highest salary ranges supporting Easy Apply, note position title, company name and salary range. Then check overall ratings and work-life balance scores for these 5 companies on Glassdoor, keeping only companies with Glassdoor overall rating above 4.0. Provide LinkedIn link, position title, company name, salary range, whether Easy Apply is supported for each position, plus the company's Glassdoor overall rating, work-life balance score and Glassdoor company page link.

**Analysis**:
- "Supporting Easy Apply": Implicitly triggers state recognition (not explicitly requiring "recognize label")
- "5 positions with highest salary ranges": Implicitly triggers pagination browsing and sorting
- "Glassdoor overall rating above 4.0": Clear secondary filtering WHERE condition
- Cross-site JOIN: LinkedIn company name → Glassdoor company evaluation
- SELECT fields complete: Job info + company ratings + dual-platform links

**Example 3 (Rule-driven itinerary planning)**:

> I plan to attend some art activities in Los Angeles this weekend and want to arrange a fulfilling but not rushed itinerary. First check Getty's official website for exhibitions or events this weekend (lectures, workshops, guided tours are all fine), select one of interest and note it down. Then search for art activities in Los Angeles this Saturday-Sunday on Eventbrite, find 2-3 well-reviewed activities under Art, Museums or Culture categories. Arrange Getty's activity and Eventbrite-found activities in chronological order, use Google Maps to calculate driving times between adjacent activity locations one by one. Help me filter a feasible combination: requiring sufficient time between each activity (previous activity end time + travel time + at least 30 minutes buffer ≤ next activity start time), finally retaining 3-4 non-conflicting activities. Output for each activity: name, start time, end time (or duration), location address, activity brief, activity page link; and between adjacent activities: Google Maps route link, driving distance, estimated travel time.

**Analysis**:
- Rules clear: Time constraints (previous end + travel + 30 min buffer ≤ next start)
- Information flow clear: Getty activity → Eventbrite activity → Google Maps feasibility verification
- Complex execution: AI must calculate times and travel one by one, cannot take shortcuts
- Output complete and unified at end

### Negative Examples

**Example 1 (Exposing UI operations)**:
> Help me click the pagination button on LinkedIn to load more positions, then identify the Easy Apply label status on each position card...

**Problem**: Explicitly requires UI operations, exposes testing intent, should change to goal-oriented expression.

**Example 2 (Existence check)**:
> Go to Coursera to find Google's UX Design certificate course, confirm whether it includes "Building Wireframes" module...

**Problem**: Existence checks produce only yes/no results with low information content. Should change to conditional query: "Find 5 UX Design courses with rating above 4.5, list their core modules and links".

**Example 3 (Superlative query without fallback)**:
> Find the 5 data analysis courses with most students on Coursera...

**Problem**: Relies on "most students" sorting, but Coursera may not support this sorting. Should change to range filtering:
> Find 5 data analysis courses on Coursera with rating above 4.5 and duration between 10-30 hours...

**Example 4 (Absolute time expression)**:
> I plan to hold outdoor activities at Discovery Park on June 20, 2026...

**Problem**: Absolute dates cause tasks to become invalid over time. Should change to relative time:
> I plan to hold outdoor activities at Discovery Park on a Saturday in mid-next month...

## Verification Scheme Design

### Evaluation Logic from Generalized SQL Perspective

The essence of evaluation is: Based on the "query results" (in text form) returned by the "generalized SQL executor", judge execution effectiveness.

| Complexity Type | Verification Method |
|----------------|-------------------|
| WHERE-related | Check if output fields used as filter conditions satisfy WHERE requirements |
| SELECT-related | Check if required fields are output, format types correct, content reasonable |
| Hallucination prevention | Use LLM Judge to execute "generalized validation SQL", take URLs to verify claims |

**Source field types and verification**:

| Source Field Type | Extraction Method | Verification Method |
|------------------|------------------|-------------------|
| Numerical/text | Direct HTML extraction | Directly verify if numerical/text satisfies WHERE conditions |

### Task Construction Support for Evaluation

Task construction must ensure:
1. **WHERE verification fields must be output**: Fields used in filter conditions must be output items, otherwise cannot verify WHERE execution correctness
2. **Key URLs must be output**: Used for LLM Judge hallucination prevention verification

### Correspondence Principle between Verification and Output

| Task Requirement Type | Output Form | Verification Method |
|---------------------|-------------|-------------------|
| Numerical filtering (price, rating) | Specific values | Check if values satisfy WHERE conditions |
| Text filtering (tags, categories) | Text content | Check if text matches filtering requirements |
| Rule constraints (distance intervals) | Specific value for each item | Check if each item satisfies constraint conditions |
| Result quantity requirements | List | Check if quantity meets requirement |

**Key principles**:
- All outputs should ultimately transform into **verifiable text or URLs**
- URLs are key means for hallucination prevention verification
- Rule-driven tasks should output sufficient information to verify rule satisfaction

## Output Format
```json
{
  "task_id": "TASK-XXX",
  "source_proposal": "PROP-XXX",
  "task_description": "Complete user instruction, 100-250 words, natural conversational, no line breaks",
  "information_flow_summary": "Briefly describe how information flows from first website to last website",
  
  "core_outputs": [
    {
      "id": "o1",
      "name": "Output item name",
      "type": "text|number|boolean|url|list",
      "source": "Which website it comes from",
      "description": "Specific description of this output",
      "output_role": "user_value|where_verification|url_verification"
    },
    {
      "id": "o2",
      "name": "Another output item",
      ...
    }
  ],
  
  "covered_points": [
    {
      "point_id": "semanticscholar.org:F1:A3",
      "ability_type": "Date range filtering",
      "trigger_method": "Implicitly triggered through 'search for papers published since 2023'",
      "verification": {
        "output_ref": "o2",
        "verify_method": "Check if o2 (publication year) >= 2023"
      }
    },
    {
      "point_id": "youtube.com:F5:A30",
      "ability_type": "Expand/collapse",
      "trigger_method": "Implicitly triggered through 'check video description to confirm if it mentions the paper or author' (description collapsed by default requires expansion)",
      "verification": {
        "output_ref": "o9",
        "verify_method": "Check if o9 contains video description summary, and whether it includes paper title or author name"
      }
    }
  ],
  
  "uncovered_points": [
    {
      "point_id": "linkedin.com:F3:A27",
      "reason": "Share function unrelated to task主线, forcing inclusion would破坏 naturalness"
    }
  ],
  
  "deep_research_differentiation": "Explain visual/operational capabilities this task must rely on, and why pure search + LLM cannot complete it",
  
  "cross_site_necessity_check": {
    "skip_test_result": "Explain what happens if skipping a certain website",
    "unique_info_dependency": "Explain what unique information from which website the task depends on",
    "information_handoff_points": ["Information transfer nodes, e.g., 'LinkedIn company name → Indeed comparison verification'"]
  },
  
  "factual_reliability_check": {
    "verified_elements": ["Confirmed existing elements"],
    "dynamic_elements": ["Elements that may change"],
    "fallback_strategies": ["Built-in fault tolerance strategies in task"]
  },
  
  "groundtruth": [
    {
      "checkpoint": "Checkpoint description",
      "verification_type": "visual_match|data_extraction|state_change|content_coverage|path_completion|judgment_quality",
      "verification_method": "Specific verification method, including how to handle dynamic content",
      "expected_result": "Pass criteria",
      "is_required": true
    }
  ],
  
  "edge_cases": {
    "content_not_found": "Handling method when content meeting conditions not found",
    "dynamic_content": "Handling method for dynamic content"
  },
  
  "metadata": {
    "clusters_used": ["functional cluster 1", "functional cluster 2"],
    "websites_involved": ["xxx.com", "yyy.com"],
    "official_websites": ["brand official site 1", "company official site 2"],
    "function_points_from_proposal": ["xxx.com:F1", "xxx.com:F2", "yyy.com:F1"],
    "function_points_used": ["xxx.com:F1", "yyy.com:F1"],
    "action_points_used": ["xxx.com:F1:A3", "yyy.com:F1:A8"],
    "perception_points_used": ["xxx.com:F1:P2", "yyy.com:F1:P9"],
    "cluster_count": "Number of functional clusters",
    "website_count": "Total website count (core + additional official sites)",
    "core_website_count": "Number of core websites",
    "official_website_count": "Number of additional official websites",
    "function_point_count": "Count of function_points_used",
    "action_point_count": "Count of action_points_used (target ≥ 5)",
    "perception_point_count": "Count of perception_points_used (target ≥ 4)",
    "cross_site_handoffs": "Number of cross-site information transfers (target ≥ 2)",
    "complexity_density": "Total complex points / core website count (target ≥ 2.0)",
    "estimated_time_minutes": "Estimated completion time (minutes)",
    "test_barriers": "None|Login required|Payment required|Regional restrictions",
    "is_rule_driven_planning": "Whether it's a rule-driven complex planning task"
  }
}
```

### Key Format Constraints

**point_id format**:
- Must be precise to operation point (A) or perception point (P): `website.com:Fx:Ay` or `website.com:Fx:Pz`
- [WRONG] Error: `allrecipes.com:F3` (only to function point)
- [RIGHT] Correct: `allrecipes.com:F3:A5`

**Quantity consistency**:
- Number of `covered_points` elements = `action_point_count` + `perception_point_count`
- Each point in `action_points_used` and `perception_points_used` must have corresponding entry in `covered_points`

**trigger_method format**:
```
Implicitly triggered through '[original text fragment from task description]'
```

### Complete Example
```json
{
  "core_outputs": [
    {"id": "o1", "name": "Paper title", "type": "text", "source": "semanticscholar.org", "output_role": "user_value"},
    {"id": "o2", "name": "Publication year", "type": "number", "source": "semanticscholar.org", "output_role": "where_verification"},
    {"id": "o3", "name": "Citation count", "type": "number", "source": "semanticscholar.org", "output_role": "where_verification"},
    {"id": "o4", "name": "Has PDF", "type": "boolean", "source": "semanticscholar.org", "output_role": "where_verification"},
    {"id": "o5", "name": "Semantic Scholar link", "type": "url", "source": "semanticscholar.org", "output_role": "url_verification"},
    {"id": "o6", "name": "YouTube video title", "type": "text", "source": "youtube.com", "output_role": "user_value"},
    {"id": "o7", "name": "Video view count", "type": "number", "source": "youtube.com", "output_role": "where_verification"},
    {"id": "o8", "name": "Video link", "type": "url", "source": "youtube.com", "output_role": "url_verification"},
    {"id": "o9", "name": "Video description summary", "type": "text", "source": "youtube.com", "output_role": "where_verification"}
  ],
  
  "covered_points": [
    {
      "point_id": "semanticscholar.org:F1:A3",
      "ability_type": "Date range filtering",
      "trigger_method": "Implicitly triggered through 'search for papers published since 2023'",
      "verification": {"output_ref": "o2", "verify_method": "Check o2 >= 2023"}
    },
    {
      "point_id": "semanticscholar.org:F1:A4",
      "ability_type": "Sorting operation",
      "trigger_method": "Implicitly triggered through 'high citation ranking'",
      "verification": {"output_ref": "o3", "verify_method": "Check o3 value relatively high (e.g., > 100)"}
    },
    {
      "point_id": "semanticscholar.org:F1:P2",
      "ability_type": "State perception",
      "trigger_method": "Implicitly triggered through 'must have PDF download link'",
      "verification": {"output_ref": "o4", "verify_method": "Check o4 == true"}
    },
    {
      "point_id": "youtube.com:F1:P4",
      "ability_type": "Content matching",
      "trigger_method": "Implicitly triggered through 'find corresponding demo video'",
      "verification": {"output_ref": "o6", "verify_method": "Check relevance between o6 and o1"}
    },
    {
      "point_id": "youtube.com:F5:A30",
      "ability_type": "Expand/collapse",
      "trigger_method": "Implicitly triggered through 'check video description to confirm if it mentions paper or author'",
      "verification": {"output_ref": "o9", "verify_method": "Check if o9 exists and contains paper/author related information"}
    },
    {
      "point_id": "youtube.com:F1:P7",
      "ability_type": "Data perception",
      "trigger_method": "Implicitly triggered through 'video views must exceed 500'",
      "verification": {"output_ref": "o7", "verify_method": "Check o7 > 500"}
    }
  ]
}
```

### Field Descriptions

**core_outputs fields**:

| Field | Type | Description |
|-------|------|-------------|
| id | String | Output item unique identifier (o1, o2...) |
| name | String | Output item name |
| type | String | text|number|boolean|url|list |
| source | String | Which website it comes from |
| description | String | Specific description of this output |
| output_role | String | user_value|where_verification|url_verification |

**covered_points fields**:

| Field | Type | Description |
|-------|------|-------------|
| point_id | String | Capability point ID (website.com:Fx:Ay or Pz) |
| ability_type | String | Ability type description |
| trigger_method | String | Original text fragment in task description triggering this capability point |
| verification.output_ref | String | Corresponding core_output id |
| verification.verify_method | String | Verification method without GT |

**groundtruth fields**:

| Field | Type | Description |
|-------|------|-------------|
| checkpoint | String | Checkpoint description |
| gt_availability | String | full (complete GT)|partial (partial GT)|none (no GT) |
| verification_method | String | Matching method with GT, or verification logic without GT |
| expected_result | String | Pass criteria |
| is_required | Boolean | Whether it's a mandatory checkpoint |

### covered_points Logical Constraints

**Core principle: trigger_method must be strong logic, not speculation or indirect association**

| Logic Type | Example | Allowed |
|-----------|---------|---------|
| Strong logic | "Implicitly triggered through 'filter sellers with reputation above 4.8' to hover and view seller info" | ✓ |
| Weak logic/speculation | "Implicitly triggered through 'compare wholesale prices', need to hover seller name to view seller reputation to assess procurement risk" | ✗ |

**Judgment methods**:
1. Does task description **explicitly require** outputting information produced by this capability point?
2. If this capability point is not executed, will the task **inevitably fail**?
3. Is there a **direct causal relationship** between this capability point and trigger_method?

**Handling methods**:
- If wanting to cover a capability point, **must add corresponding output requirements in task description**
- If unable to naturally add output requirements, then **abandon covering that capability point**
- Prohibit using speculative trigger_methods like "need XX in order to YY"

**Example corrections**:
- [WRONG] trigger_method: "Implicitly triggered through 'compare wholesale prices', need to hover seller name to view seller reputation to assess procurement risk"
- [RIGHT] Option A: Add "filter products from sellers with reputation above 4.8, output seller name and reputation score" to task description
- [RIGHT] Option B: Delete this covered_point, because task doesn't require seller information

## Self-Check Checklist

### Cross-Site Necessity
- [ ] Would skipping any website make the task impossible to complete?
- [ ] Does information_flow_summary clearly describe the information chain?
- [ ] Is cross_site_necessity_check completely filled?
- A natural, reasonable single-site task is better than a contrived two-site task
- A natural, reasonable two-site task is better than a deliberately chained three-site task
- If cross-site makes the task feel unnatural, prefer fewer sites

Do not cross sites just for the sake of crossing sites. Cross-site is a means, not an end.

### Information Flow Authenticity (Highest priority, reject if fails)
- [ ] Are transfer objects **specific entities** (names, work titles, values), not theme keywords?
- [ ] Are transfer objects **dynamically obtained**, not preset at task start?
- [ ] Does task use words like "assume", "suppose" to connect different websites? (If yes, directly reject)
- [ ] **Skip test**: After skipping website A, do you still know what to search on website B? (If yes, directly reject)
- [ ] Does each core_output have logical connection with other parts of task? (If no, directly reject)
- [ ] Does task description clarify **quantities** to find at each step? (e.g., "find 5 papers" not "find papers")

### Core Outputs (Generalized SQL SELECT)
- [ ] Is each output specific and verifiable?
- [ ] **Does each WHERE condition have corresponding output verification field**?
- [ ] **Are output requirements unified at end of task description**?
- [ ] **Does it include URL fields for hallucination prevention**?

### Output and Capability Point Mapping
- [ ] **Does each covered_point have verification.output_ref pointing to some core_output**?
- [ ] **Does each covered_point's verify_method explain no-GT verification logic**?
- [ ] **Are there capability points triggered but unverifiable through outputs**? If yes, need to add corresponding output items
- [ ] **Does task_description include output requirements for all core_outputs**?

### Factual Reasonableness
- [ ] Do key elements have reliable existence basis?
- [ ] Are confirmed existing elements described with sufficient specificity?
- [ ] Do dynamic contents have fault tolerance strategies?
- [ ] **Is reliance on uncertain sorting/filtering functions avoided**?
- [ ] **Do time expressions use relative time, avoiding absolute dates**?
- [ ] **Do time-related tasks have time window expiration fault tolerance**?

### Benchmark Positioning
- [ ] Can the task NOT be completed solely through search engine + LLM?
- [ ] Is task endpoint a specific object or operation, not an analysis report?

### Complexity Compliance
- [ ] action_points_used ≥ 5
- [ ] perception_points_used ≥ 4
- [ ] Each core website has at least 1 operation or perception point
- [ ] cross_site_handoffs ≥ 2
- [ ] All complex points triggered through implicit guidance (not explicit requirements)
- [ ] Official website additions naturally integrated
- [ ] Total websites (core + official) ≥ 3

**Non-compliance handling**:
1. Mine more naturally triggered complex points in existing steps
2. Expand "search → click" to "filter → compare → select"
3. Add "verify" or "confirm" steps, introducing state perception
4. Consider adding official website additions
5. Consider designing as rule-driven complex planning task

### Naturalness
- [ ] Do user instructions conform to real user expression habits?
- [ ] Is exposure of "testing XX function" intent avoided?
- [ ] Are operation/perception points naturally triggered through goal descriptions?

### Metadata Format
- [ ] Are statistical indicators in metadata accurate?
- [ ] **Is each point_id in covered_points precise to A or P level**?
- [ ] **Does number of covered_points elements = action_point_count + perception_point_count**?


### Five-Dimensional Quality Check (Final Review)

After completing task design, must conduct final check from these five dimensions:

| Dimension | Check Questions | If Fails |
|-----------|----------------|----------|
| **Reasonableness** | Is task intent natural and reasonable? Is cross-site logic coherent? Is function usage forced? | Redesign task scenario |
| **Complexity** | Is it challenging for agent? Can it be completed with just search + LLM? | Add complex operation/perception points |
| **Executability** | Can humans smoothly execute and annotate GT? Are website functions usable? | Adjust task or change websites |
| **Safety** | Does it involve sensitive topics like politics, pornography, violence, discrimination? | Change task theme |
| **Evaluability** | Can outputs be verified through text/URL? Do visual tasks have GT support? | Adjust output requirements or abandon visual tasks |

**Recommendation**: Before submitting task, try executing it yourself according to task description, confirming each step can be smoothly completed.

### Quality Rejection Criteria

The following situations should **reject and redesign**:

| Rejection Condition | Description |
|--------------------|-------------|
| Common sense errors | Task based on wrong assumptions (e.g., pure instrumental music has lyrics) |
| Unverifiable | Core outputs cannot be verified through text or URLs |
| Severely insufficient complexity | action_points < 4 or perception_points < 3 |
| Unclear websites | Using vague expressions like "some platform", "related website" |
| Overly ambitious | Task too grand, not completable in single session |
| Non-standard format | point_id only to F level, or quantity inconsistency |
| Missing WHERE fields | Filter conditions not output as fields, cannot verify |
| Capability points without output mapping | covered_point cannot be verified through any core_output |
| **False information flow** | Using "assume" connections, preset keywords, theme parallel without entity transfer, isolated output items |
| **Superlative queries without fallback** | Relying on "most", "hottest" sorting without range filter alternatives |
| **Absolute time causing invalidation** | Using fixed dates making task unexecutable over time |

**Handling process**:
1. Self-check after refinement whether rejection conditions triggered
2. Tasks triggering rejection conditions should be returned for re-proposal or re-refinement
3. Borderline cases can be fixed by adjusting task description

### Time Expression Standards

**Core principle**: Use relative time expressions, avoid absolute dates, ensure long-term task executability.

| Expression Type | Not Recommended [WRONG] | Recommended [RIGHT] |
|----------------|------------------------|-------------------|
| Future plans | "June 20, 2026" | "Next Saturday", "end of this month", "next summer" |
| Activities/events | "Spring 2026 semester" | "Next semester", "upcoming semester" |
| Historical data | "Since 2023" | "Past two years", "within past 12 months" |
| Periodic content | "2025 annual report" | "Latest annual report", "previous fiscal year report" |

**Reasonable matching between time periods and task types**:

Different task types have their natural time cycles; time expressions should match task nature:

| Task Type | Reasonable Period | Example |
|-----------|------------------|---------|
| Hotel/restaurant booking | Days ~ weeks | "Next weekend", "a Saturday this month" |
| Flight/train tickets | Weeks ~ 2 months | "Mid-next month", "around National Day holiday" |
| Activity/exhibition queries | Current week ~ next month | "This weekend", "next week", "within this month" |
| Course/semester planning | Current season ~ next season | "Next semester", "fall semester" |
| Annual planning/procurement | End of month/quarter/year | "Before year end", "during Q4", "end of month" |
| Travel planning | 1~3 months | "During summer vacation", "around Spring Festival" |

```
[WRONG] Unreasonable: "Book hotel room for next summer" (period too long, hotels usually don't accept)
[RIGHT] Reasonable: "Book hotel room for next weekend"

[WRONG] Unreasonable: "Check flights for this afternoon" (period too short, cannot actually purchase tickets)
[RIGHT] Reasonable: "Check flights for next Wednesday"

[WRONG] Unreasonable: "Plan transnational self-driving tour for day after tomorrow" (period too short, unrealistic)
[RIGHT] Reasonable: "Plan weekend self-driving tour for next month"
```

**Fault tolerance design for time-related tasks**:

For tasks dependent on specific times, consider possibility that time window may have passed:
```
[RIGHT] "Check this weekend's event schedule; if no events this weekend, check next weekend's"
[RIGHT] "Find flights for next month; if no suitable dates remaining in current month, extend to following month"
[RIGHT] "Check upcoming marathon events; if current season ended, check first event of next season"
```

**Avoid overly distant time references**:
- Avoid citing data from 2023 or earlier as "latest"
- For academic papers, suggest using "published in past two years" rather than specific years
- For products/versions, use "latest version", "current version" rather than specific version numbers
