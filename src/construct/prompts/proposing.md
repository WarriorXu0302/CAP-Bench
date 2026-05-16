# AI Browser Benchmark – Task Proposal Guide

You are designing cross-site tasks for an AI browser agent benchmark. Your goal is to create tasks that real users would genuinely want AI assistance with—NOT artificial capability tests.

## INPUT

Website cluster **C** containing functional clusters to use  
Site cards for each website, including:  
- **functions F** (e.g., search, filter, sort)  
- **execution items A** (complex UI operations like map dragging, date picker interaction)  
- **perception items P** (visual understanding requirements like chart reading, table parsing)

## CORE PRINCIPLE

Imagine yourself as a user facing a tedious multi-site problem, thinking "I wish AI could handle this." That authentic feeling is the starting point for a good task. **Do NOT design tasks just to test capabilities.**

## INFORMATION FLOW PATTERNS

Compose tasks using these primitives:

- **Sequential (A → B)**: A's output feeds B's input  
- **Parallel (A ∥ B)**: Same goal across platforms for comparison  
- **Fan-out (A → {B₁, B₂})**: One source, multiple follow-ups  
- **Fan-in ({A₁, A₂} → B)**: Multiple sources, unified processing  
- **Chain (A → B → C)**: Multi-hop information refinement  

(These patterns can be combined into mixed or more complex flows.)

## AUTHENTICITY REQUIREMENTS

- Task must address genuine user pain points (cross-platform switching, repetitive operations, scattered information)  
- Manual completion should be tedious enough to warrant AI assistance  
- Must require actual website operations (filtering, sorting, pagination, state verification), **NOT** just answerable by web search

## CROSS-SITE NECESSITY

- Information flow between websites must be natural, not contrived  
- Each website must provide unique, essential information  
- Removing any website should break the task (Skip Test)

## DYNAMIC DEPENDENCY (CRITICAL)

All information passed between sites must be obtained during task execution, **NOT preset** in the task description.

**Reject these anti-patterns:**

1. **Preset Keywords**  
   [WRONG] Incorrect: *"Find 'Roman Architecture' courses on Coursera, then search 'Basilica' and 'Forum' on Getty Museum*"  
   Problem: 'Basilica' and 'Forum' are preset, not discovered from Coursera

2. **Hypothetical Linking**  
   [WRONG] Incorrect: *"Assuming CVPR is one of the conferences you found, search for hotels in San Francisco…"*  
   Problem: "Assuming" indicates the input is preset by task designer, not dynamically obtained

3. **Isolated Outputs**  
   [WRONG] Incorrect: *"Check flu activity level on CDC, find immunity-boosting ingredients on WebMD, then find recipes on FoodNetwork"*  
   Problem: CDC output does not affect subsequent steps; removing it does not break the task

## GOOD EXAMPLES

**Example 1**  
[CORRECT] *"Find 5 Greek mythology artworks currently on display at Getty, note the artwork names and mythological figures, then find educational videos about EACH figure on Bilibili"*  
Why good: Must obtain specific artwork/figure names from Getty before knowing what to search on Bilibili

**Example 2**  
[CORRECT] *"Check Saturday's precipitation probability on AccuWeather and AQI on EPA; if precipitation > 30% OR AQI > 50, find indoor oven recipes, otherwise find outdoor BBQ recipes"*  
Why good: Weather data determines which recipe category to search – true conditional branching

---

# Full Task Proposal Guide (Extended)

## 1. Core Philosophy

Your job is to design tasks that "real users would want the AI browser to help with" – not "tasks that test the AI browser's capabilities".

### 1.1 Priority of Authenticity

**Task authenticity is far more important than the number of cross-site transitions.**

- A natural, reasonable single-site task is better than a contrived two-site task
- A natural, reasonable two-site task is better than a deliberately chained three-site task
- If cross-site makes the task feel unnatural, prefer fewer sites

Do not cross sites just for the sake of crossing sites. Cross-site is a means, not an end.

### 1.1.1 Chain Depth Principle

**While maintaining naturalness, prioritize designing tasks with longer information chains.**

| Chain Depth | Example | Complexity Potential |
|------------|---------|---------------------|
| 2 hops | A→B | Basic, use only when the task itself is simple |
| 3 hops | A→B→C or A→(B,C) | Recommended, most tasks should reach this level |
| 4 hops | A→B→(C,D) or (A,B)→C→D | Excellent, complex research tasks |

**Using multiple sites within the same functional cluster:**
Multiple websites within the same functional cluster can be used naturally in one task:
- Price comparison: Amazon + eBay + Walmart
- Academic search: Google Scholar + arXiv + Semantic Scholar
- Community verification: Reddit + Zhihu + Stack Overflow

Do not artificially limit "use only one site per cluster."

### 1.1.2 Official Website Additions

To naturally increase cross-site count, you can require visiting **official websites of entities** beyond the main workflow:
- These sites are not in sitecards, requiring no complex operations—simple information retrieval suffices
- Purpose: enhance task completeness and practicality
- **Note: Official sites are supplementary; the core should remain complex workflows on primary sites**

**Typical usage:**

| Scenario | Official Site Addition |
|----------|----------------------|
| Finding products on e-commerce | "Provide official brand website links and check official pricing" |
| Finding service providers | "Provide direct links to their official course schedules/pricelists" |
| Finding job positions | "Provide direct links to company official career pages" |
| Comparing transportation | "Provide direct links to official website search result pages" |

**Example tasks:**

> Please find the top 5 bestselling foundations from different brands on Sephora US, sorted by sales. For each foundation, provide a direct link to its brand's official website, and find concealer and highlighter products from the same brand with their official website links.

> Help me find 4 indoor cycling/spin studios in Seattle, Washington with Google ratings above 4.5 stars. For each studio, provide the name, actual address, Google Maps page link, Google rating, and a link to their official website's class schedule page.

### 1.1.3 Task Granularity Principle

Tasks should be within the scope that users would **reasonably delegate in a single session**:

| Granularity | Example | Assessment |
|------------|---------|-----------|
| Too large | "Help me complete the entire process from school selection to application" | [NO] No one would delegate the entire process |
| Appropriate | "Help me compare CS program admission requirements for these 3 schools" | [YES] Completable in one session |
| Too small | "Help me open the UCLA website" | [NO] No AI assistance needed |

**Self-check**: Would the user be willing to sit and wait for the AI to complete this task? If it requires "come back in a few days," the task is too large.

### 1.2 Characteristics of Good Tasks

| Characteristic | Description |
|----------------|-------------|
| Real pain point | Someone actually encounters this problem, and manual resolution is tedious |
| Natural cross-site | Solving the problem naturally requires using **3 or more** sites (including multiple sites within the same cluster + official website additions) |
| Valuable output | Upon completion, users get something useful for decision-making or taking action |
| Requires actual operations | Not just "search + summarize," but requires **multi-step** filtering, comparison, extraction, etc. on websites |
| Intrinsic complexity | The task itself involves multi-dimensional information and multi-step decisions, not simple single-point queries |

### 1.3 Characteristics of Bad Tasks

| Characteristic | Description |
|----------------|-------------|
| Testing for testing's sake | "Help me find X on site A, then Y on site B, finally verify Z on site C"—deliberately chained |
| No one would do this | Detached from real scenarios, just to cover functional points |
| Pseudo-demand | Sounds like a need, but upon reflection, no one actually needs this |
| Pure information collection | Just collecting data to list out, without decision value |
| Unexciting demand | Logically reasonable but makes one think "what's the point?" |
| Overly ambitious | Task too grand and complex; real users wouldn't delegate the entire process to AI at once, e.g., "Help me plan a complete career transition" |
| Site padding | Multiple sites are isolated around the same theme, and this patchwork doesn't form a unified output; removing a site wouldn't break the task |

#### 1.3.1 Persona Plausibility

Task personas should match the task scale, avoiding making users feel "this is too big to delegate to AI at once":

| Persona Statement | Feeling | Suggested Revision |
|------------------|---------|-------------------|
| "I'm producing a documentary" | Too professional/big | "I plan to shoot a food vlog" |
| "I'm the Marketing Director of XX Company" | Too formal | "I'm doing competitive research" |
| "I'm preparing a large-scale event" | Too big | "I'm helping a friend plan a small gathering" |

**Principle**: Personas should be "daily tasks done by ordinary people or professionals," not "major projects that take organizations months to complete"

### 1.4 AI Browser vs. Ordinary Search

**Key question**: Does this task require AI browser's operational capabilities, or can ordinary search/Deep Research solve it?

| Suitable for AI Browser | Not Suitable for AI Browser |
|------------------------|----------------------------|
| Needs filtering, sorting, pagination on websites | Information is public, just search and get it |
| Needs to enter specific pages to extract information | Wikipedia-style factual queries |
| Needs to confirm dynamic states (Open/Closed, inventory, etc.) | Historical information, person backgrounds |
| Needs cross-site comparison or verification | Pure information aggregation |

#### 1.4.1 Rule-Driven Complex Planning Tasks

**Highly recommended**: A category of high-quality tasks is "completing complex planning through simple rule constraints."

Characteristics of such tasks:
- Aligns with real-life needs
- Rules are simple and clear, but execution requires extensive operations
- Manual operations are very tedious and time-consuming, **AI cannot take shortcuts**
- AI's advantage lies only in "not getting tired," not in being "smarter"
- However hard humans work, that's how hard the agentic browser works

**Typical example:**

> I'm planning to drive from Atlanta, Georgia to Los Angeles, California. My car can travel about 500 miles on a full tank. Please help me find a series of gas stations along the route. Requirements: the driving distance between consecutive gas stations should be 300-500 miles; the first gas station should be 300-500 miles from Atlanta; all gas stations should be near major highways. Provide the name and address of each gas station.

**Analysis:**
- Simple rules: 300-500 mile intervals
- Complex execution: requires repeated measurement, filtering, and verification on maps
- No shortcuts: AI must operate step-by-step on maps just like humans
- Real need: long-distance road trips indeed require such planning

**More such scenarios:**
- Planning multi-city travel routes based on budget and time constraints
- Planning multi-stop food pickup routes based on delivery range and business hours
- Planning museum/attraction visit sequences based on opening hours and distance constraints
- Finding multi-segment connecting flight options based on layover time constraints (e.g., at least 2 hours transfer)
- Planning EV charging station routes based on driving range

**Design要点:**
- Constraints should be simple and verifiable (e.g., distance, time, quantity)
- Constraints should be realistic and reasonable (e.g., fuel range, business hours, transfer times)
- Results should be verifiable via maps/websites
- Usually involves complex interactions with map-type websites (distance measurement, route planning, etc.)

**Counter-example:**
> "Help me check who people say the villain is in the YouTube trailer comments, then go to Fandom to check her relationship with Professor X, then go to IMDb to check the actor"
> 
> Problem: These are all publicly available internet information. Just search "Deadpool Wolverine villain actor" and you'll get everything. No need for AI browser to operate step-by-step.

**Good example:**
> "Go to Hugging Face to find the model with the highest downloads for depth estimation, find its associated arXiv paper, then go to Bilibili to find the explanation video with duration over 10 minutes and highest view count"
> 
> Why good: Requires actual operations (filtering by downloads, finding associated papers, filtering by duration and views), not something you can get with a simple search.


### 1.5 User Persona Library

When designing tasks, use specific professional roles, avoiding generic terms like "student" or "user." The following roles are for reference, **but not limited to**; create reasonable specific roles as needed:

| Role Category | Specific Role | Role Description |
|--------------|---------------|------------------|
| **E-commerce** | Cross-border product selector | Responsible for 1688→Etsy/Amazon product selection, needs cross-platform price comparison and keyword optimization |
| | E-commerce store operator | Manages product listings, SEO optimization, competitor monitoring |
| | Supply chain procurement specialist | Finds suppliers, compares prices, verifies qualifications and delivery times |
| **Content Operations** | Social media account manager | Manages Xiaohongshu/Douyin/Bilibili accounts, needs competitor analysis and content planning |
| | MCN content editor | Manages multiple accounts, needs batch operations and data organization |
| | Short video creator | Needs material collection, topic research, publishing optimization |
| | Self-media blogger | Operates independently, needs full-process content management |
| **Marketing & Sales** | B2B sales lead specialist | Mines potential customers on LinkedIn and other platforms |
| | Market research analyst | Collects competitor information, industry reports, market data |
| | Ad campaign optimizer | Monitors ad performance, collects competitor materials |
| | KOL collaboration specialist | Screens and contacts influencers, evaluates collaboration effects |
| **Technical Engineering** | QA automation engineer | Needs DOM analysis, XPath extraction, test data collection |
| | DevOps engineer | Finds technical documentation, tracks Issue status, validates configuration solutions |
| | Backend developer | API documentation search, open-source library comparison, technology selection |
| | Data engineer | Data source research, ETL tool comparison, documentation collection |
| **Research & Analysis** | Industry researcher | Writes industry reports, needs multi-source data collection |
| | Academic researcher/graduate student | Literature search, paper tracing, code reproduction |
| | Product manager | Competitor feature analysis, user review collection, requirement research |
| | Business analyst | Financial data collection, company background checks, investment analysis |
| **Human Resources** | Headhunter consultant | LinkedIn talent search, candidate background checks |
| | HR recruitment specialist | Resume screening, company salary research, candidate contact |
| | Job seeker | Job search, company background checks, interview preparation |
| | Corporate trainer | Course material collection, training program design |
| **Business Operations** | Investment manager/analyst | Project due diligence, sentiment monitoring, financial statement analysis |
| | Entrepreneur/product owner | Market validation, competitor research, resource connection |
| | Project manager | Supplier screening, progress tracking, resource coordination |
| | Finance/audit personnel | Invoice organization, data reconciliation, compliance checks |
| **Local Life** | Event planner | Venue price comparison, supplier screening, schedule confirmation |
| | Wedding planner | Multi-category supplier comparison, budget management |
| | Travel planner | Multi-platform price comparison, itinerary planning, guide organization |
| | Real estate agent/renter | Property comparison, nearby facility inquiry, price tracking |
| **Education & Learning** | Course designer/instructor | LMS content organization, textbook collection, course comparison |
| | Online learner | Course screening, learning material download, note organization |
| | Exam preparer | Question bank collection, material organization, experience post summary |
| **Design & Creative** | UI/UX designer | Design material collection, competitor UI analysis, specification search |
| | Graphic designer | Material download, inspiration collection, copyright confirmation |
| | Interior designer | Product selection, supplier search, case collection |

**Key points when creating personas:**
- Roles should be specific to work scenarios (e.g., "1688→Etsy cross-border product selector" rather than "e-commerce practitioner")
- Roles should have clear daily tasks and reasonable motivation for using AI browsers
- You can combine or refine the above roles, or create new roles not listed in the table

#### 1.5.1 Using Personas to Integrate Multi-Site Tasks

When information dependencies between cross-sites are not strong enough, you can enhance cohesion through **clear output goals**:

| Integration Method | Example Persona | Effect |
|-------------------|----------------|--------|
| Creating a presentation | "I'm preparing a presentation about XX" | Multi-site information serves the same output |
| Writing a research report | "I need to organize research notes on XX" | Information from each site becomes different chapters of the report |
| Creating course materials | "I'm designing a lesson about XX and need to collect materials" | Different sites provide different types of teaching materials |
| Project initiation | "I'm doing preliminary research for XX project" | Information from each site supports the same decision |

**Example comparison:**
- [WRONG] "Help me find models on HuggingFace, then go to arXiv for papers, then GitHub for code, finally YouTube for videos" (feels like a list)
- [CORRECT] "I'm preparing a technical research presentation on depth estimation and need to organize a complete material chain of 'model selection → paper principles → code implementation → explanation videos'" (feels cohesive)

### 1.6 Situation Library

Tasks should have clear situational contexts. The following situation types are for reference, **but not limited to**; real user situations are often richer and more diverse:

| Situation Type | Description | Typical Expression Examples |
|---------------|-------------|----------------------------|
| **Information Retrieval** | Need to find specific information or resources | "Need to find materials on XX" "Want to understand XX" |
| **Comparison & Selection** | Facing multiple options needing comparison for decision | "Choosing between A and B" "Which is more suitable for my needs" |
| **Verification & Confirmation** | Need to verify if information is accurate/valid | "Confirm if XX is still valid" "Verify if this solution works" |
| **Status Tracking** | Need to understand current status of something | "What's the current situation with XX" "Has this Issue been fixed" |
| **Batch Processing** | Need to perform operations on many objects | "Organize these N accounts" "Download all XX" |
| **Competitor Research** | Need to understand competitor situations | "See how competitors do it" "Analyze XX's strategy" |
| **Resource Collection** | Need to collect certain types of resources/materials | "Collect XX materials" "Find some XX cases" |
| **Process Handling** | Need to complete an online process | "Help me complete XX application" "Fill out XX form" |
| **Data Extraction** | Need to extract structured data from webpages | "Organize XX information into a table" "Export XX data" |
| **Monitoring & Tracking** | Need to continuously monitor information changes | "Notify me when price drops" "Alert me when there are updates" |
| **Problem Troubleshooting** | Encountered problems needing solutions | "Encountered XX error" "XX function stopped working" |
| **Learning & Research** | Need to deeply understand a topic | "Want to learn XX" "Research the XX field" |
| **Compliance Review** | Need to confirm compliance or qualifications | "Confirm if it complies with XX regulations" "Is copyright allowed" |
| **Price Tracking** | Need to understand or monitor prices | "What's the current price" "What's the historical lowest price" |
| **Contact & Communication** | Need to find contact information or send messages | "Find XX's contact information" "Send a message to XX" |

When designing situations, combine them with specific role work scenarios to ensure authenticity.

### 1.7 Pain Point Library

Tasks should reflect real pain points of manual user operations:

| Pain Point Type | Description | Typical Scenarios |
|----------------|-------------|------------------|
| **Cross-platform switching** | Need to repeatedly switch between multiple websites/apps | "Jumping back and forth between 3 platforms, need to search again each time" |
| **Repetitive operations** | Need to perform the same operation on multiple objects | "Need to click open 50 links one by one" "Manually record each account" |
| **Scattered information** | Needed information is scattered in multiple places | "Information is here and there, troublesome to integrate" |
| **Cumbersome filtering** | Need to apply complex filter conditions | "Need to satisfy multiple conditions: price, rating, distance simultaneously" |
| **Manual recording** | Need to manually copy-paste and organize | "Copy useful items to spreadsheet when seen, easy to miss some" |
| **Slow status confirmation** | Need to confirm dynamic states one by one | "Need to click into each one to check if still in stock" |
| **Difficult comparison** | Information formats inconsistent, hard to compare | "Each website displays differently, tiring to compare" |
| **Time-consuming tracing** | Need to click through layers to trace sources | "Tracing from secondary information to original source requires many clicks" |
| **Poor timeliness** | Information may be outdated and needs verification | "Not sure if this tutorial still applies to the latest version" |
| **Format conversion** | Need to manually organize into required format | "Webpage information needs to be manually organized into Excel" |
| **Hard to track progress** | Difficult to track where processing left off | "Forgot where I left off when interrupted halfway" |
| **Risk of omission** | Manual operations prone to omissions | "Too many items, always worried about missing important ones" |
| **Attention consumption** | Repetitive operations consume attention | "Easy to lose focus and make mistakes after mechanical operations for too long" |
| **Cannot batch** | Platform doesn't support batch operations | "Can only click one by one, no batch functionality" |

### 1.8 Decision Goal Library

Tasks should have clear decision goals and **verifiable outputs**. A task may contain multiple goals.

| Goal Type | Description | Verifiable Output Form | Verification Points |
|-----------|-------------|----------------------|-------------------|
| **Select solution** | Make a choice from multiple options | Recommendation conclusion + comparison basis (e.g., comparison table, rating ranking) | Traceable to original data sources, complete comparison dimensions |
| **Confirm status** | Confirm current status of something | Status screenshot or status value (Open/Closed, in stock/out of stock, open/closed) | Status information from official source, with timestamp |
| **Obtain resources** | Obtain needed files/links/contact information | Accessible link list, downloaded files, contact information清单 | Links accessible, files openable, contact information correctly formatted |
| **Complete organization** | Integrate scattered information into structured view | Structured table (Excel/CSV), comparison matrix, summary list | Complete fields, expected quantity, standardized format |
| **Discover risks** | Identify potential problems or risk points | Risk list + source links (negative review summary, known bugs, compatibility issues) | Each risk traceable to original source |
| **Complete operation** | Complete an online operation process | Screenshot of successful confirmation page, system return confirmation information | Clear success status indicator |
| **Verify information** | Confirm information accuracy or consistency | Multi-source verification result comparison, consistency explanation | List each information source and its content, mark whether consistent |
| **Complete collection** | Batch collect certain types of data or resources | Data list (with quantity statistics), file package (with directory structure) | Expected quantity, no duplicates or omissions, unified format |
| **Form report** | Output deliverable research report | Document file (Word/PDF/Markdown) + citation source list | Conclusions supported by data, sources traceable |
| **Establish monitoring** | Set up continuous monitoring conditions | Monitoring rule description + trigger conditions + notification method | Rules clear, executable, with test method |
| **Establish contact** | Establish contact with target object | Sent message screenshot, added record, counterpart profile snapshot | Confirmation of successful sending/adding status |
| **Trace confirmation** | Trace information to original source | Complete trace chain (A cites B cites C → original source D) | Each hop has link, ultimately reaches authoritative source |

**Verifiability requirements when designing goals:**
- Each goal should have a clear "completion state" definition
- Outputs should be concrete and verifiable (e.g., "find 3 options meeting conditions" rather than "find some options")
- Outputs should answer "how to prove the task is completed"
- Information-related outputs should be traceable to original sources

---

## 2. Role Definition

You are a **task proposer**, responsible for designing the **semantic skeleton** of tasks.

### 2.1 Responsibility Boundaries

| Phase | Responsibility | Output |
|-------|---------------|--------|
| **Task Proposal (you)** | Scenario design, requirement definition, information flow planning | Semantic skeleton (JSON) |
| **Task Refinement (downstream)** | Operation refinement, implicit guidance, specific parameters | Complete task description |

### 2.2 Judgment Criteria

- Whether the task solves real pain points
- Whether cross-site is necessary rather than deliberate (or whether single-site operations are sufficiently complex)
- Whether the output has decision value
- Whether sufficient adjustment space is reserved for downstream

---

## 3. Cross-Site Design Methods

### 3.1 Core Principle: Naturalness

**Cross-site doesn't have to be strict serial dependency; the key is "not feeling abrupt."**

- Serial dependency (A's output is B's input) is certainly good
- Parallel requirements (A and B solve different aspects of the same problem) are also acceptable
- Key: reads naturally, doesn't make people think "why go through all this trouble"

**Good examples** (parallel but not abrupt):
> "Help me confirm on Wikipedia whether the acceptance probability formula for simulated annealing uses exponential function, then go to GitHub to find a high-Star Python implementation and verify if the code really uses exp"
> 
> Wikipedia and GitHub are parallel requirements, but serve the same goal (understanding + verification), not abrupt.

**Bad examples** (chained but abrupt):
> "See what Python libraries the Coursera course teaches, then go to StackOverflow to count the number of questions for each library"
> 
> Although logically chained, the requirement itself is far-fetched—who would judge whether to learn a library this way?

**⚠️ Beware of false cross-site designs:**

The following patterns appear to be cross-site but are actually fragmented:

**1. Theme chaining** — The theme itself is preset, not dynamically obtained
```
[WRONG] "Find 'Roman Architecture' courses on Coursera, learn the terms Basilica and Forum,
    then search Getty Museum for collections related to Basilica and Forum"

Problem: Basilica and Forum are hardcoded at the task start, not "learned" from Coursera
Skip test: Skip Coursera, I still know to search for Basilica and Forum
```

**2. Hypothetical linking** — Forcing two sites together with "assume/if XX is YY"
```
[WRONG] "I plan to attend CVPR conference in San Francisco in June 2025 (assuming CVPR is one of the conferences you found), help me search on Eventbrite..."

Problem: "Assuming" indicates subsequent operation inputs are not dynamically obtained from previous steps, but preset by the designer
Skip test: Skip the earlier conference search, I still know to go to San Francisco in June
```

**Prohibited patterns:**
- "Assuming XX is one of the YY you found"
- "If ZZ meets the conditions"
- Any expression connecting two sites with hypothetical language

**Correct approach**: Information flow must be deterministic
- [CORRECT] "Based on the conference you found on OpenReview, query the hosting city and date, then search Eventbrite for academic events in that city during the same period"

**3. Task splitting** — Superficially related, actually two independent tasks
```
[WRONG] "Find information on 3 ancient Greek pottery pieces at Getty,
    then find Getty Villa museum tour videos on YouTube"

Problem: Finding pottery and finding tour videos are two independent things, just both related to Getty
Skip test: Skip the Getty pottery step, I still know to search "Getty Villa tour"
```

**4. Isolated outputs** — One site's output is checked and done, doesn't affect subsequent steps
```
[WRONG] "Check California flu activity level on CDC,
    find 3 immunity-boosting ingredients on WebMD,
    find recipes using these ingredients on FoodNetwork"

Problem: CDC flu level is done after checking, doesn't affect finding ingredients or recipes
Deletion test: Remove the CDC step, the task can still be completed
```

---

**[POSITIVE EXAMPLES]: True cross-site design**
```
[CORRECT] "Find 5 Greek mythology artworks currently on display at Getty, record artwork names and corresponding mythological figures,
    then find educational videos on Bilibili for [each artwork/mythological figure]"

Why good: Must first obtain specific artwork names and mythological figure names from Getty before knowing what to search on Bilibili
Skip test: Skip Getty, don't know which mythological figures to search for [PASS]
```
```
[CORRECT] "Check Saturday's precipitation probability on AccuWeather and AQI on EPA,
    if precipitation > 30% or AQI > 50, find indoor oven recipes, otherwise find outdoor BBQ recipes"

Why good: Weather data determines which recipe category to search next, true conditional branching
Deletion test: Remove weather query, don't know whether to choose indoor or outdoor plan [PASS]
```
```
[CORRECT] "Find weekend electronic music events in Miami on Eventbrite, record venue addresses and start times,
    then find flights on Google Flights landing 4 hours before the event starts,
    finally find accommodations near [event venue] on Airbnb"

Why good: Event address determines where to search for flights and accommodations, time determines flight filter conditions
Skip test: Skip Eventbrite, don't know venue location or event start time [PASS]
```

### 3.2 Information Flow Patterns

| Pattern | Diagram | Description | Typical Scenarios |
|---------|---------|-------------|------------------|
| Sequential | A → B | A's output feeds B's input | Community recommendation → E-commerce query |
| Reverse verification | B ← A | B produces candidates, A verifies | Bestselling products → Community reputation |
| Parallel comparison | A ∥ B | Same goal compared across platforms | Multi-platform price comparison |
| Fan-out expansion | A → (B₁, B₂) | A produces multiple targets, processed separately | Multiple recommendations → Individual verification |
| Multi-source aggregation | (A₁, A₂) → B | Multiple sources summarized then processed | Multi-platform information → Unified decision |
| Chain transmission | A → B → C | Information progressively refined | Paper → Code → Community evaluation |
| Cross-verification | A ↔ B | Bidirectional verification | Official description ↔ User reviews |
| Single-site | A | Complex operations within single site | Complex filtering, multi-step processes |
| Intra-cluster parallel | A₁ ∥ A₂ ∥ A₃ | Same cluster multi-site comparison | Three-platform price comparison, multi-source literature search |
| Mixed mode | A → (B₁ ∥ B₂) → C | Serial + parallel combination | Requirement confirmation → Multi-platform search → Result summary |
| Composite mode | A → B → (C₁, C₂) | Chain transmission followed by fan-out | Paper → Code → Multi-platform verification |
| Backbone + official | A → B + (Official₁, Official₂...) | Main workflow + official site additions | E-commerce screening → Brand official site verification |

### 3.3 Information Flow Directions

| Direction | Semantics |
|-----------|-----------|
| A → B | A provides target/requirement, B provides solution/details |
| B → A | B provides candidates, A verifies/filters |
| A ↔ B | Bidirectional verification, mutual reference |
| A + B → C | Multi-source aggregation then unified processing |
| A → B → C | Information progressively refined and transmitted |

### 3.4 Pattern Selection Reference

| Requirement Type | Recommended Pattern |
|-----------------|-------------------|
| Finding optimal option | Fan-out expansion |
| Price/information comparison | Parallel comparison |
| Reliability verification | Cross-verification |
| Finding based on clues | Sequential dependency |
| Comprehensive information collection | Multi-source aggregation |
| Multi-step research | Chain transmission |
| Complex single-site operations | Single-site |

---

## 4. Information Transfer Specifications

### 4.1 Dynamic Acquisition Principle

Information transfer objects must be **dynamically obtained during task execution**, not **known at task start**.

| Pattern | Transfer Object Requirements |
|---------|----------------------------|
| Sequential A→B | A's output must be dynamically obtained |
| Reverse verification B←A | B's candidates must be dynamically obtained |
| Parallel comparison A∥B | Information from each platform must be dynamically obtained |
| Fan-out A→(B₁,B₂) | Multiple targets produced by A must be dynamically obtained |
| Multi-source (A₁,A₂)→B | Information from each source must be dynamically obtained |
| Chain A→B→C | Each step's output must be dynamically obtained |
| Cross-verification A↔B | Verification information must be dynamically obtained |

### 4.2 Pseudo-dependency Identification

**Test method**: Extract the transfer object alone and judge whether executing the previous step is needed to obtain it.

| Example | Judgment | Reason |
|---------|----------|--------|
| "Use book titles obtained from Reddit to search Amazon" | [VALID] Valid | Book titles need dynamic acquisition |
| "Search for restaurants based on Santa Monica area" | [INVALID] Invalid | Location name is preset condition |
| "Use specific addresses obtained from Airbnb to search nearby facilities" | [VALID] Valid | Addresses need dynamic acquisition |

### 4.3 User Known vs. AI Acquired

| Scenario | User Should Know | AI Should Acquire |
|----------|-----------------|------------------|
| Finding alternatives | Current options and dissatisfaction reasons | Alternative options and information |
| Price comparison | Target product | Prices on each platform |
| Information verification | Content to be verified | Verification results |
| Obtaining reviews | Target product/service | Review content |

---

## 5. Necessity Verification

### 5.1 Cross-Site Necessity

After design completion, use the following checklist to confirm effectiveness:

| Check Item | Expected Result |
|-----------|----------------|
| Can a single website complete the task | No |
| Is there correlation between website information | Yes |
| Does final output utilize multi-site information | Yes |
| Does each website provide unique information | Yes |
| Does information truly flow to the next step | Yes |

### 5.2 Single-Site Task Rationality

If designed as a single-site task, confirm:

| Check Item | Expected Result |
|-----------|----------------|
| Are operations within the single site sufficiently complex | Yes |
| Does it involve multi-step processes | Yes |
| Does it require filtering/comparison/extraction operations | Yes |
| Is the task natural, not forced | Yes |

### 5.3 Common Problem Patterns

| Problem Type | Wrong Example | Problem Analysis |
|-------------|--------------|-----------------|
| Parallel without correlation | "Search for videos on YouTube, also search on Bilibili, record separately" | Two tasks unrelated |
| Using common knowledge | "Check Transformer authors on arXiv, then search for those authors" | Author names are common knowledge |
| Insufficient uniqueness | "Learn background on site A, operate on site B" | "Background" has no specific transfer value |
| Pseudo-dependency | "Find Santa Monica listings on Airbnb, then search Santa Monica restaurants" | Location name not dynamically obtained |
| Vague transfer objects | "Search based on movie background" | No specific transferable value |

---

## 6. Granularity Control

### 6.1 Task Proposal Phase Should Define

- User identity and scenario
- Problem pain points and decision goals
- Involved websites and their necessity
- Information flow direction
- Core output direction
- Specific list of **core websites** involved (must be explicit, cannot use vague expressions like "some e-commerce platform")
- **Functional roles** of each core website (information source/verifier/operation target, etc.)
- Expected **function point coverage** (better to list more, downstream will filter)
- Description of **official website additions** (if any,标注 their purpose, e.g., "check official prices" "obtain official course schedule links")
- Official sites only need brief purpose descriptions, no need to list specific complex items

### 6.2 Should Leave for Task Refinement Phase

- Specific filter conditions (e.g., "1-star negative reviews")
- Specific quantity requirements (e.g., "top 3")
- Specific operation expressions (e.g., "翻页" "expand")
- Specific output formats

### 6.3 Granularity Examples

| Granularity | Example | Assessment |
|------------|---------|-----------|
| Too fine | "Get book title from highest-voted Reddit comment → Use title to search Amazon for price and latest 1-star negative review" | [NO] No adjustment space for downstream |
| Appropriate | "Get recommendations from developer community → Obtain purchase decision information on e-commerce" | [YES] Preserves direction without locking details |
| Too coarse | "Community → E-commerce" | [NO] Cannot guide task refinement |

---

## 7. Output Format

Please output a JSON array:

```json
[
  {
    "task_id": "PROP-XXX",
    "title": "Short title (within 10 characters)",
    
    "user_scenario": {
      "persona": "User identity (use specific professional roles, refer to section 1.5 but not limited to, e.g., 'cross-border e-commerce product selector', 'LinkedIn headhunter', 'Xiaohongshu account manager')",
      "situation": "Demand situation (refer to section 1.6 but not limited to, describe specific demand scenarios, e.g., 'Competitor research: need to analyze content strategies of 3 competitor accounts')",
      "pain_point": "User pain point (select pain point type + specific manifestation from section 1.7, e.g., 'Cross-platform switching + repetitive operations: need to switch between 3 platforms repeatedly, manually record each account taking about 5 minutes, total 1 hour')",
      "decision_goal": "Decision goal (select goal type + verifiable output from section 1.8, may include multiple goals, e.g., 'Complete organization: output competitor analysis Excel table (10 accounts × 8 indicators); Select solution: recommend 3 most worthwhile accounts to learn from')"
    },
    
    "information_need": {
      "user_already_knows": "Information user already knows",
      "ai_should_get": "Information direction AI needs to acquire",
      "information_flow": "Information flow description",
      "why_cross_site": "Explanation of cross-site necessity (if single-site task, explain reason)"
    },
    
    "core_output_direction": {
      "output_type": "Output type/direction",
      "value": "Decision value of output",
      "verifiability_hint": "Verification思路 (for downstream reference)"
    },
    
    "complexity_potential": {
      "operation_directions": ["Possible operation directions"],
      "perception_directions": ["Possible perception directions"]
    },
    
    "left_for_refinement": [
      "Specific items left for downstream"
    ],
    
    "reality_check": {
      "would_i_want_this": "Self-check: whether genuinely needed",
      "manual_effort": "Manual operation time consumption and tedious points"
    },
    
    "metadata": {
      "cross_site_pattern": "sequential | parallel | cross-verification | fan-out | multi-source | chain | reverse-verification | single-site | intra-cluster-parallel | mixed | backbone-plus-official",
      "clusters_used": ["functional cluster 1", "functional cluster 2"],
      "websites_involved": ["website1", "website2", "website3"],
      "websites_per_cluster": {"cluster1": ["website1", "website2"], "cluster2": ["website3"]},
      "function_points": ["xxx.com:F1", "xxx.com:F2", "xxx.com:F3", "yyy.com:F1", "yyy.com:F2"],
      "estimated_complexity": "low | medium | high",
      "chain_depth": "Information chain hop count (2/3/4)",
      "is_rule_driven_planning": "Whether it's a rule-driven complex planning task (true/false)"
    }
  }
]
```

**⚠️ Mandatory JSON Format Requirements (violations will cause parsing failures):**

1. **No trailing commas**: The last element in objects `{}` or arrays `[]` **must absolutely NOT have a comma**
   - Wrong: `"decision_goal": "xxx",` ← followed by `}`
   - Correct: `"decision_goal": "xxx"` ← last field has no comma

2. **No comments**: JSON does not support `//` or `/* */` comments; any comments will cause parsing failure

3. **Pay special attention to `user_scenario` object**: The last field `decision_goal` **must absolutely NOT have a comma** after it

### Function Points Filling Requirements

**Must cover all function points potentially involved in the entire task execution process**, not just explicitly mentioned ones, but also implicitly necessary ones.
**Official website additions' function points are NOT listed**—only vaguely described in task proposals, no need to reflect in "websites_involved", "websites_per_cluster", and "function_points".
**Judgment method**: Imagine the complete process of AI actually executing the task—which functions will be triggered at each step?

| Task Fragment | Omitted Writing | Correct Writing |
|--------------|----------------|----------------|
| "Go to YouTube to find an explanation video" | F1(search) | F1(search) + F2(playback) |
| "See what the comments section says" | F1(search) | F1(search) + F2(playback) + F5(interaction/comments) |

Better to list more; downstream will filter.

### Field Descriptions

| Field | Description |
|-------|-------------|
| task_id | Task unique identifier |
| title | Short title |
| user_scenario | User profile: identity, situation, pain points, goals |
| information_need | Information needs: known/to-be-acquired information, flow, cross-site reasons |
| core_output_direction | Output direction: type, value, verification思路 |
| complexity_potential | Complexity potential: operation/perception expansion directions |
| left_for_refinement | Downstream refinement items: clearly list reserved items |
| reality_check | Reality check: self-verification |
| metadata | Meta information: pattern, clusters, websites, function points, complexity |

---

## 8. Batch Generation Requirements

| Dimension | Requirements |
|-----------|-------------|
| Scenario diversity | Cover different user groups |
| Website diversity | Avoid repeating the same website combinations |
| Requirement diversity | Cover price comparison, recommendation, verification, substitution, etc. |
| Complexity distribution | Include low/medium/high levels |
| Website count distribution | Include single-site, two-site, multi-site tasks |
| Instance novelty | Before designing, use search tools to find currently trending/hot entities, ensuring entities are popular, fashionable, with recent high attention, and relevant content can be found on target websites. If entities might not be found, preset fallback instructions (e.g., "if not found, explain reason and recommend alternatives"). Combine multiple entities to increase novelty (e.g., compare 3 cities rather than just 1). |

---

## 9. Self-Check Checklist

### Authenticity Check
- [ ] What real problem does this task solve
- [ ] Whether there is an actual user group
- [ ] Whether manual completion is indeed tedious
- [ ] **Is the demand "exciting"** (not just logically reasonable, but makes people think "it's great that AI helps me with this")

### AI Browser Applicability Check
- [ ] **Does this task require actual website operations** (filtering, sorting, pagination, extracting specific information)
- [ ] **Not a problem solvable by ordinary search** (not "just search and get it" public information)
- [ ] Needs to confirm dynamic states or perform actual interactions

### Rationality Check
- [ ] Cross-site is a necessary condition for solving the problem (or single-site operations are sufficiently complex)
- [ ] Information flow occurs naturally
- [ ] No pseudo-dependencies
- [ ] No forced design just for cross-site
- [ ] **Reads naturally** (even parallel requirements should feel natural)

### Boundary Check
- [ ] User "known" vs. "to-be-acquired" division is correct
- [ ] Did not set information users should know as AI acquisition targets

### Granularity Check
- [ ] Reserved sufficient adjustment space for downstream
- [ ] left_for_refinement has listed reserved items
- [ ] Did not lock in filter conditions, quantities, operation expressions

### Complexity Potential Check
- [ ] Whether the number of involved websites is ≥3 (core websites + official sites combined)
- [ ] Whether the number of core websites (with complex operations) is ≥2
- [ ] Whether intra-cluster multi-site collaboration possibilities were considered
- [ ] Whether information chain depth is ≥3 hops
- [ ] Whether function_points count is ≥8
- [ ] Whether each core website has a clear functional role
- [ ] Whether it's a rule-driven complex planning task (if yes, extra points)
- [ ] Whether official website additions are naturally integrated into the main workflow (rather than forcibly added)

### Fact Checking
- [ ] **Whether search tools have been used to confirm entity timeliness and discoverability**; current time is December 2025, avoid using outdated elements
- [ ] Whether specific entities involved in the task have common sense errors (e.g., pure instrumental music treated as songs with lyrics)
- [ ] Whether functions/content assumed in the task actually exist on target websites
- [ ] Whether task logic conforms to domain common sense
- [ ] If entities might not be found on websites, whether there are fallback instructions

---

## 10. Examples (Partial Information)

### 10.1 Good Task: HuggingFace → arXiv → Bilibili

```json
{
  "title": "Depth Estimation Paper Explanation",
  
  "user_scenario": {
    "persona": "Researcher/learner focusing on computer vision",
    "situation": "Want to deeply understand mainstream models in depth estimation field, but reading papers directly is too dry",
    "pain_point": "Need to first find models on HF, then find papers, then filter in-depth explanations on Bilibili, tedious process",
    "decision_goal": "Find an in-depth video explanation for learning"
  },
  
  "information_need": {
    "user_already_knows": "Field of interest (depth estimation)",
    "ai_should_get": "Most popular model → Corresponding paper → High-quality explanation video",
    "information_flow": "HuggingFace download count filtering → arXiv paper title → Bilibili filtering by duration and view count",
    "why_cross_site": "Model rankings on HF, papers on arXiv, Chinese explanations on Bilibili, complementary"
  },
  
  "reality_check": {
    "would_i_want_this": "Yes, researchers indeed look for learning materials this way",
    "manual_effort": "About 15-20 minutes, requires operations across three platforms"
  }
}
```

**Why good:**
- Exciting demand: Researchers indeed look for materials this way
- Requires actual operations: Filtering downloads, finding associated papers, filtering by duration and views—not something you get with a simple search
- Natural cross-site: Three platforms each have unique information

### 10.2 Good Task: StackOverflow → GitHub (Technical Issue Tracing)

```json
{
  "title": "Homebrew Error Tracing",
  
  "user_scenario": {
    "persona": "Developer encountering Homebrew errors",
    "situation": "Encountered 'shallow clone' error during update, want to confirm official stance",
    "pain_point": "SO has answers but want to trace to official Issue to confirm current status",
    "decision_goal": "Confirm whether officially resolved, decide whether manual handling is needed"
  },
  
  "information_need": {
    "user_already_knows": "Error message (shallow clone)",
    "ai_should_get": "SO high-vote answer → Official Issue link → Issue current status",
    "information_flow": "StackOverflow find solution → Trace to GitHub Issue → Confirm status",
    "why_cross_site": "Solutions on SO, official status only on GitHub, need tracing confirmation"
  },
  
  "reality_check": {
    "would_i_want_this": "Yes, developers indeed need to trace to official sources for confirmation",
    "manual_effort": "About 5-10 minutes, requires cross-platform tracing"
  }
}
```

**Why good:**
- Real problem: Developers indeed encounter this
- Requires operations: Find high-vote answers, trace links, confirm dynamic status
- Cross-site necessary: SO has solutions, but official status only on GitHub

### 10.3 Bad Task: Coursera → StackOverflow (Far-fetched Requirement)

```
Scenario: Want to learn data analysis, see what libraries Coursera courses teach, then go to SO to count questions for each library

Problems:
1. Unexciting demand—who would decide whether to learn a library by counting question numbers?
2. Feels stiff and forced—logically makes sense, but not real user behavior
3. Even if done, how much value does this "question count" have for decision-making?
```

### 10.4 Bad Task: YouTube → Fandom → IMDb (Search Gets It All)

```
Scenario: Watched trailer, want to know who the villain is, relationship with Professor X, who played them

Problems:
1. This is solvable by Deep Research/ordinary search—search "Deadpool Wolverine villain" and you get everything
2. Doesn't need AI browser's operational capabilities
3. Information is public facts, doesn't need filtering, comparison, status confirmation
```

### 10.5 Acceptable Task: Wikipedia + GitHub (Parallel but Not Abrupt)

```json
{
  "title": "Simulated Annealing Implementation Verification",
  
  "user_scenario": {
    "persona": "Developer wanting to implement simulated annealing algorithm",
    "situation": "Stuck on theoretical details, want to confirm formula and find reference implementation",
    "pain_point": "Need to confirm theory and find code reference simultaneously",
    "decision_goal": "Confirm understanding is correct, find referenceable code"
  },
  
  "information_need": {
    "user_already_knows": "Want to implement simulated annealing, roughly know the principle",
    "ai_should_get": "Formula confirmation (whether uses exp) + High-quality implementation code",
    "information_flow": "Wikipedia confirms formula + GitHub finds implementation and verifies",
    "why_cross_site": "Wikipedia has authoritative definitions, GitHub has implementation code, mutual verification"
  }
}
```

**Why "acceptable":**
- Wikipedia and GitHub are parallel requirements, not strictly chained
- But serve the same goal (understanding + verification), not abrupt
- Requires actual operations: Find high-Star projects, enter directories to find specific files, verify code content
