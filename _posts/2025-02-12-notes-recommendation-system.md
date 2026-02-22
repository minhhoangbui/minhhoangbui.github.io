---
layout: post
title: A Practical Recommendation System Pipeline
author: hoangbm
---

In a recommendation system, there are typically three stages:

1. **Candidate generation (retrieval)**
2. **Ranking**
3. **Re-ranking**

## Requirements

1. **Low latency**: the system should respond as fast as possible. This often implies heavy use of caching and a design that can repopulate caches quickly.
2. **Relevance**: recommendations should be personalized and aligned with product objectives.

## 1. Candidate generation (retrieval)

From the entire item catalog, we first generate a short list of items that a user is likely to engage with. This step must be extremely efficient because the catalog can be huge. The goal is to be **broad/exhaustive** enough to avoid missing good candidates, while staying fast.

Common retrieval strategies include:

- Fetch recent items from channels the user is subscribed to (rule-based, no ML required).
- Retrieve items similar to what the user has watched (item-to-item similarity; often easiest because it does not require a user representation).
- Retrieve items that similar users watched (collaborative filtering).

The last two approaches often rely on **embeddings** for users and/or items. A trained model plus a vector index (ANN search) is a common implementation.

## 2. Two-tower retrieval and why you still need a re-ranker (cross-encoder)

A popular ML architecture for candidate generation is the **two-tower model** (a.k.a. dual encoder):

- A **user tower** encodes user context (history, profile, session signals) into a user embedding.
- An **item tower** encodes item features into an item embedding.
- At serving time, you score candidates with a simple similarity function (e.g., dot product or cosine similarity) and use ANN search to retrieve the top-K items.

Two-tower retrieval is powerful because it is fast: item embeddings can be precomputed and indexed, and user embeddings can be computed online.

However, you often still need a **re-ranker** because two-tower scoring has inherent limitations:

- **Limited interaction modeling**: the score is typically a dot product between embeddings. This is efficient, but it cannot model fine-grained interactions between the user and item the way a heavier model can.
- **Information bottleneck**: the user and item towers compress many features into fixed-size vectors. Some subtle signals are lost.
- **Approximate search trade-offs**: ANN retrieval is approximate. It is optimized for speed, not perfect ordering.

This is where a **cross-encoder re-ranker** is useful:

- A cross-encoder takes the *pair* (user context, candidate item) together and computes a score with full attention/interactions (e.g., BERT-style or a feature-interaction network).
- It is much more accurate, but too expensive to run over the whole catalog.
- Therefore, you run it only on a **small candidate set** (e.g., top 200–2,000 from retrieval) to produce a better final ordering.

In short:

> Two-tower models are excellent for **fast retrieval**. Cross-encoders are excellent for **accurate ranking** on a small set.

## 3. Ranking

After retrieval, scoring becomes feasible because the candidate set is small. It can be tempting to use the retrieval similarity as the final score, but that is usually not ideal:

- Multiple candidate generators often run in parallel (subscriptions, similarity, collaborative filtering, etc.). Their raw scores are not directly comparable.
- Retrieval is optimized for speed and recall, so it typically sacrifices accuracy.
- In ranking, you may optimize for objectives beyond similarity (e.g., clicks, watch time, retention, revenue), and you can incorporate richer context.

## 4. Re-ranking

After ranking, you can apply business logic and additional constraints to produce the final recommendation list.

Common considerations include:

1. **Freshness**: run parts of the pipeline more frequently; incrementally index new items; or boost recent items.
2. **Diversity**: combine candidates from multiple categories/strategies and sample/allocate slots to avoid repetition.