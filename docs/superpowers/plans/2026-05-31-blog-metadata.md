# Blog Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tags, subtitles, author, and social-share front matter to all 28 posts, and publish the untracked LLM architectures post.

**Architecture:** Each task edits a batch of posts by adding missing YAML front matter fields. No content is changed — only the `---` block at the top of each file. The untracked post + images are committed in the final task.

**Tech Stack:** Jekyll front matter (YAML), git

---

## Front Matter Pattern

Every post should end up with this structure (add only the missing fields, don't duplicate existing ones):

```yaml
---
layout: post
title: "..."
subtitle: "..."
author: hoangbm
tags: [tag1, tag2]
social-share: true
---
```

Posts that already have `author: hoangbm` don't need it added again. Check before editing.

---

## Task 1: 2018 posts (14 files)

**Files:** All `_posts/2018-*.md`

- [ ] **Step 1: Add front matter to each 2018 post**

  For each file below, open it and add the missing fields to the `---` block. Do not change any content below the front matter.

  | File | Add `subtitle:` | Add `tags:` | Notes |
  |---|---|---|---|
  | `2018-01-04-first-post.md` | `Why I started writing about machine learning` | _(none)_ | No tags for meta post |
  | `2018-01-08-convnet-element.md` | `The building blocks of convolutional neural networks` | `[deep-learning]` | |
  | `2018-01-12-convnet-architecture.md` | `A tour of landmark CNN architectures from AlexNet to ResNet` | `[deep-learning, computer-vision]` | |
  | `2018-01-16-challenges.md` | `Common pitfalls when training deep neural networks` | `[deep-learning]` | |
  | `2018-01-18-distributed-tensorflow.md` | `Setting up distributed training with TensorFlow` | `[deep-learning, infra]` | |
  | `2018-02-25-improvement-distributed-training.md` | `Techniques for faster and more stable distributed training` | `[deep-learning, infra]` | |
  | `2018-03-20-image-similarity.md` | `Learning visual representations for similarity search` | `[computer-vision, deep-learning]` | |
  | `2018-06-01-sagemaker-tut.md` | `Training and deploying models with Amazon SageMaker` | `[aws, systems]` | |
  | `2018-10-20-multi-armed-bandits.md` | `Balancing exploration and exploitation in recommendation` | `[recsys, statistics]` | |
  | `2018-10-28-randomization.md` | `Why randomization matters for valid experiment design` | `[statistics]` | |
  | `2018-10-30-apache-hadoop-introduction.md` | `A practical introduction to the Hadoop ecosystem` | `[infra]` | |
  | `2018-10-8-ner-tagger.md` | `Building a named entity recognition system from scratch` | `[nlp, deep-learning]` | |
  | `2018-11-07-hdfs-1.md` | `How the Hadoop Distributed File System stores data` | `[infra]` | |
  | `2018-11-10-hdfs-2.md` | `HDFS architecture: NameNode, DataNode, and replication` | `[infra]` | |

  Also add `author: hoangbm` and `social-share: true` to every file that is missing them.

  Example — `2018-01-08-convnet-element.md` before:
  ```yaml
  ---
  layout: post
  title: ConvNet and its principal elements
  ---
  ```

  After:
  ```yaml
  ---
  layout: post
  title: ConvNet and its principal elements
  subtitle: The building blocks of convolutional neural networks
  author: hoangbm
  tags: [deep-learning]
  social-share: true
  ---
  ```

- [ ] **Step 2: Build to verify no front matter errors**

  ```bash
  cd /Users/harbui/axon/axon-ai/minhhoangbui.github.io
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: `done in X seconds.` — no YAML errors.

- [ ] **Step 3: Commit**

  ```bash
  git add _posts/2018-*.md
  git commit -m "content: add tags, subtitle, author to 2018 posts"
  ```

---

## Task 2: 2019 posts (6 files)

**Files:** All `_posts/2019-*.md`

- [ ] **Step 1: Add front matter to each 2019 post**

  | File | `subtitle:` | `tags:` |
  |---|---|---|
  | `2019-01-12-amazon-personalize.md` | `Using Amazon Personalize for production recommendation` | `[aws, recsys]` |
  | `2019-01-16-retrieval.md` | `Candidate retrieval in large-scale recommendation systems` | `[nlp, recsys]` |
  | `2019-02-22-clustering.md` | `From K-Means to DBSCAN: clustering algorithms compared` | `[statistics, deep-learning]` |
  | `2019-03-04-mixture-models.md` | `Gaussian mixture models and the EM algorithm` | `[statistics]` |
  | `2019-05-19-triplet.md` | `Learning embeddings with triplet loss` | `[deep-learning, computer-vision]` |
  | `2019-09-06-pose-extraction.md` | `Estimating human pose from images` | `[computer-vision, deep-learning]` |

  Add `author: hoangbm` and `social-share: true` to any missing them.

- [ ] **Step 2: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 3: Commit**

  ```bash
  git add _posts/2019-*.md
  git commit -m "content: add tags, subtitle, author to 2019 posts"
  ```

---

## Task 3: 2021–2025 tracked posts (7 files)

**Files:** `_posts/2021-*.md`, `_posts/2022-*.md`, `_posts/2023-*.md`, `_posts/2025-*.md`

- [ ] **Step 1: Add front matter**

  | File | `subtitle:` | `tags:` |
  |---|---|---|
  | `2021-02-06-yolact.md` | `Real-time instance segmentation with YOLACT` | `[computer-vision, deep-learning]` |
  | `2021-02-22-solo.md` | `Instance segmentation without bounding boxes` | `[computer-vision, deep-learning]` |
  | `2022-12-24-object-detection.md` | `A personal guide to modern object detection` | `[computer-vision, deep-learning]` |
  | `2023-01-01-attention.md` | `Understanding the attention mechanism in transformers` | `[nlp, deep-learning]` |
  | `2023-11-4-tts.md` | `How modern text-to-speech systems work` | `[nlp, deep-learning]` |
  | `2025-02-12-notes-recommendation-system.md` | `A practical three-stage recommendation system pipeline` | `[recsys, systems]` |
  | `2025-02-18-notes-k8s.md` | `Kubernetes concepts for ML engineers` | `[infra, systems]` |

  Note: `2025-02-12-notes-recommendation-system.md` already has `author: hoangbm` — skip that field for it.

  Add `author: hoangbm` and `social-share: true` to any that are missing them.

- [ ] **Step 2: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 3: Commit**

  ```bash
  git add _posts/2021-*.md _posts/2022-*.md _posts/2023-*.md _posts/2025-*.md
  git commit -m "content: add tags, subtitle, author to 2021-2025 posts"
  ```

---

## Task 4: Publish LLM post and add metadata

**Files:**
- `_posts/2026-03-26-llm-architectures.md` (currently untracked)
- `images/llm/` (currently untracked)

- [ ] **Step 1: Verify untracked files exist**

  ```bash
  ls /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2026-03-26-llm-architectures.md
  ls /Users/harbui/axon/axon-ai/minhhoangbui.github.io/images/llm/
  ```
  Expected: both exist.

- [ ] **Step 2: Add front matter to LLM post**

  Current front matter:
  ```yaml
  ---
  layout: post
  title: "Demystifying LLM Architectures: Encoders, Decoders, and Inference"
  image: /images/llm/transformer.png
  ---
  ```

  Replace with:
  ```yaml
  ---
  layout: post
  title: "Demystifying LLM Architectures: Encoders, Decoders, and Inference"
  subtitle: Encoders, decoders, and how inference actually works
  author: hoangbm
  tags: [llm, deep-learning]
  social-share: true
  image: /images/llm/transformer.png
  ---
  ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors. The new post should appear in `_site/`.

- [ ] **Step 4: Verify post appears in built site**

  ```bash
  ls /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_site/2026-03-26-llm-architectures/
  ```
  Expected: `index.html` present.

- [ ] **Step 5: Commit**

  ```bash
  git add _posts/2026-03-26-llm-architectures.md images/llm/
  git commit -m "content: publish LLM architectures post with metadata"
  ```

---

## Verification Checklist

After all tasks:

- [ ] `bundle exec jekyll build` passes with no YAML errors
- [ ] `grep -L "tags:" _posts/*.md` returns only `2018-01-04-first-post.md` (the intentional no-tags post)
- [ ] `grep -L "subtitle:" _posts/*.md` returns nothing (all posts have subtitles)
- [ ] `grep -L "author:" _posts/*.md` returns nothing
- [ ] The tags page at `http://localhost:4000/tags` shows populated tag sections
- [ ] The homepage card grid shows tag pills on post cards
