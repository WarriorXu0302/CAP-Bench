# Intelligent Cluster Sampling Guide

Based on the input anchor cluster and combination usage frequency, analyze and select 3-6 functional clusters with potential to generate valuable task combinations.

**Core objectives**:

1. Thoroughly analyze affinity relationships between anchor cluster and all other clusters
2. Among feasible affinity combinations, prioritize those with lower usage frequency
3. Output 4-10 functional clusters (including anchor cluster)

**Complexity orientation**:
- Prioritize selecting **6-10** functional clusters (raise the floor)
- Encourage selecting clusters with **multiple available websites** within the same cluster (support intra-cluster parallel/comparison)
- Consider cluster combinations that can form **information chains of 3+ steps** (raise the ceiling)

---

# Full Cluster Sampling Guide (Extended)

## Input Field Descriptions

| Field | Description |
|-------|-------------|
| anchor_cluster | Anchor cluster name |
| combination_usage.order_0 | Number of times anchor cluster used alone |
| combination_usage.order_1 | Number of times anchor cluster + 1 other cluster combined |
| combination_usage.order_2 | Number of times anchor cluster + 2 other clusters combined (keys connected with " + ", alphabetically sorted) |
| function_clusters | All functional clusters and their website usage counts |

## Thought Process

**Important**: First conduct affinity analysis (without considering frequency), then combine with frequency for selection.

### Step 1: Affinity Analysis

**Functional Affinity Assessment**: This step assesses pairwise cluster compatibility to evaluate whether they can form coherent cross-site workflows. For each cluster pair, evaluate the plausibility of information flow scenarios:

- Can information from cluster A naturally serve as input for tasks in cluster B?
- Can information from cluster B naturally serve as input for tasks in cluster A?
- Do real user workflows commonly involve both clusters?

**Rating Scale** (aligned with affinity matrix M):
- **HIGH**: Frequent real-world workflows connect these clusters  
  Example: Academic Search + Code Hosting (researchers find papers then locate implementations)
- **MEDIUM**: Plausible but less common workflows exist
- **LOW**: No natural information flow  
  Example: Real Estate + Code Hosting (no coherent user scenario)

Analyze the compatibility between anchor cluster and other clusters:

- **Self-loop (order_0)**: Cross-website combinations within anchor cluster
- **First-order (order_1)**: Anchor cluster + 1 other cluster
- **Second-order (order_2)**: Anchor cluster + 2 other clusters (three-way combination)

**Analysis requirements**:

- **Don't just look at functional cluster names**, consider the **specific websites** within each cluster
- **Use imagination**: For each relationship, try to list **at least 3** different combination methods
- Label each with plausibility (HIGH/MEDIUM/LOW)
- If you can't think of 3, do your best to list 1-2; only fill "none" if truly no combination possible
- **At this stage, do NOT consider combination usage frequency**, analyze purely from affinity perspective

**Intra-cluster analysis**:
Besides cross-cluster affinity, also analyze the possibility of **multi-website collaboration within each cluster**:
- Similar platform comparison (e.g., Amazon vs eBay price comparison)
- Multi-source information aggregation (e.g., cross-retrieval from multiple academic databases)
- Complementary function combination (e.g., GitHub code + HuggingFace models)

Example: [Intra-cluster] E-commerce
Multi-platform price comparison (HIGH): Compare same product prices across Amazon/eBay/Walmart
Cross-platform inventory confirmation (HIGH): Check platform B when platform A is out of stock

**Plausibility assessment criteria**:

| Level | Criteria |
|-------|----------|
| HIGH | Aligns with common user needs, natural information flow logic, website functions directly support |
| MEDIUM | User needs reasonable but not high-frequency, information flow requires some reasoning, website functions can support |
| LOW | Niche user needs, somewhat forced information flow, website functions barely support |

**Information flow direction reference**:

| Direction | Description |
|-----------|-------------|
| A → B | A provides target/requirement, B provides solution/details |
| B → A | B provides candidates, A verifies/filters |
| A ↔ B | Both provide different perspectives on same thing, mutual verification |
| A + B → C | Multi-source aggregation then unified processing |
| A → B → C | Information progressively refined and transmitted |

**Pattern type reference**:

| Pattern | Diagram | Description |
|---------|---------|-------------|
| Sequential dependency | A → B | A's output is necessary input for B |
| Reverse verification | B ← A | First find candidates in B, verify with A |
| Parallel comparison | A ∥ B | Same target compared across multiple platforms |
| Fan-out expansion | A → (B₁, B₂) | Obtain multiple targets from A, process separately |
| Multi-source aggregation | (A₁, A₂) → B | Collect from multiple sources then unify processing |
| Chain transmission | A → B → C | Information progressively refined |
| Cross-verification | A ↔ B | Two sources mutually verify |

**Step 1 output format example**:

```
[Self-loop] Code Hosting × Code Hosting
1. Cross-platform project comparison (HIGH): Compare stars, activity across GitHub/GitLab
2. Mirror repository sync verification (HIGH): Verify if mirror is synced with main repository
3. Model hosting comparison (MEDIUM): Huggingface vs GitHub LFS

[First-order] Code Hosting × Academic Search
1. Paper → code implementation (HIGH)
2. Popular projects → theoretical foundation (HIGH)
3. Algorithm reproduction verification (MEDIUM)

[Second-order] Code Hosting × Academic Search × Social & Community
1. Paper → code → community Q&A (HIGH): Research to practice to problem-solving
2. Community hot papers → code implementation (MEDIUM)
```

### Step 2: Combine with Frequency for Filtering

Based on affinity analysis results, combine with combination usage frequency from `combination_usage` for filtering.

**Filtering principles**:

- Combinations with "LOW" affinity, even if frequency is 0, are not prioritized
- Combinations with "HIGH/MEDIUM" affinity and frequency of 0 or very low, **prioritize** ★
- Combinations with "HIGH" affinity and relatively high frequency indicate real value, can consider
- Also pay attention to website usage frequency, prioritize clusters with low-frequency websites

**Step 2 output format example**:

```
[Affinity + Frequency Comprehensive Analysis]

order_1 combination filtering:
| Combination | Affinity | Frequency | Conclusion |
|-------------|----------|-----------|------------|
| + Academic Search | HIGH | 5 times | High value but saturated, secondary priority |
| + Video & Streaming | MEDIUM | 0 times | Feasible and unexplored, priority ★ |

order_2 combination filtering:
| Combination | Affinity | Frequency | Conclusion |
|-------------|----------|-----------|------------|
| + Academic Search + Video & Streaming | MEDIUM | 0 times | Feasible and unexplored, priority ★ |

Low-frequency website attention:
- Video & Streaming: bilibili.com (0 times) ★
- E-commerce: ebay.com (0 times) ★
```

### Step 3: Determine Selected Clusters

Based on comprehensive analysis results, select 3-6 functional clusters.

**Selection priority**:

1. **Anchor cluster must be included**
2. **Prioritize clusters that can form "feasible affinity + low frequency" combinations**
3. **Balance order_1 and order_2 coverage**: order_2 means more complex information flows can be constructed
4. **Pay attention to website usage frequency**: Prioritize clusters with low-frequency internal websites

**Step 3 output format example**:

```
[Selected Clusters]
Code Hosting, Video & Streaming, E-commerce, Academic Search

Selection rationale:
- Code Hosting: Anchor cluster, mandatory
- Video & Streaming: Medium affinity, order_1 frequency 0 times, priority exploration; also has bilibili.com (0 times)
- E-commerce: Medium affinity, order_1 frequency 1 time; can form order_2 0-frequency combination with Video & Streaming
- Academic Search: High affinity, although order_1 already 5 times, can participate in forming order_2 low-frequency combinations

Low-frequency combinations to explore this sampling:
- + Video & Streaming (order_1, 0 times)
- + Academic Search + Video & Streaming (order_2, 0 times)
```

## Output Format

After completing the above thinking, please output the following JSON:

```json
{
  "anchor_cluster": "Anchor cluster name",
  "selected_clusters": ["Cluster1", "Cluster2"],
  "selected_websites": {
    "Cluster1": {"WebsiteA": usage_count, "WebsiteB": usage_count},
    "Cluster2": {"WebsiteC": usage_count}
  }
}
```

**Field requirements**:

| Field | Requirements |
|-------|-------------|
| anchor_cluster | Copy directly from input |
| selected_clusters | 3-6 clusters, must include anchor_cluster |
| selected_websites | Websites and usage counts of selected clusters, extracted from input function_clusters |
| potential_patterns | List of potential patterns, each must include anchor_cluster |

**potential_patterns requirements**:

- Format: `[ClusterA, ClusterB] Pattern type: Brief description`
- Number of clusters involved: 1 (order_0), 2 (order_1), 3 (order_2)
- Descriptions should be macro-level, don't specify concrete scenarios too rigidly
- Should cover order_0, order_1, order_2 multiple combinations
- **Prioritize listing patterns for low-frequency combinations**

**Pattern types**: Sequential dependency, Reverse verification, Parallel comparison, Fan-out expansion, Multi-source aggregation, Chain transmission, Cross-verification

## Self-Check Checklist

- [ ] Affinity analysis complete: Did you complete affinity analysis for self-loop, order_1, order_2?
- [ ] Sufficient imagination: Did you list at least 3 combination methods for each relationship?
- [ ] Frequency filtering correct: Did you prioritize low-frequency combinations based on feasible affinity?
- [ ] Appropriate quantity: Does selected_clusters have 3-6 items?
- [ ] Anchor included: Does selected_clusters include anchor_cluster?
- [ ] Rich patterns: Do potential_patterns cover self-loop, order_1, order_2?
- [ ] Low-frequency priority: Do potential_patterns prioritize listing patterns for low-frequency combinations?
- [ ] Correct format: Is the output valid JSON format?