# Blog Content & Writing — Design Spec

**Date:** 2026-05-31  
**Scope:** Two sequential sub-projects — (1) metadata for all 28 posts, (2) structural edits for 6 recent posts  
**Order:** Metadata first (unblocks features), structure second

---

## Sub-project 1: Metadata

### Goal

Add tags, subtitles, and missing front matter to all 28 posts. Publish the untracked LLM architectures post.

### Tag Taxonomy

9 tags. Posts get 1–2 tags maximum.

| Tag | Used for |
|---|---|
| `deep-learning` | CNN posts, distributed training, triplet loss, YOLACT, SOLO |
| `computer-vision` | Image similarity, object detection, pose extraction, YOLACT, SOLO |
| `nlp` | NER tagger, retrieval, attention, TTS |
| `llm` | Attention, TTS, LLM architectures post |
| `systems` | Recommendation system, K8s, SageMaker |
| `infra` | Hadoop, HDFS, K8s, distributed TensorFlow |
| `recsys` | Multi-armed bandits, Amazon Personalize, retrieval, recommendation system |
| `statistics` | Randomization, clustering, mixture models |
| `aws` | SageMaker, Amazon Personalize |

### Tag Assignments (all 28 posts)

| File | Tags |
|---|---|
| `2018-01-04-first-post.md` | _(none — meta post)_ |
| `2018-01-08-convnet-element.md` | `deep-learning` |
| `2018-01-12-convnet-architecture.md` | `deep-learning, computer-vision` |
| `2018-01-16-challenges.md` | `deep-learning` |
| `2018-01-18-distributed-tensorflow.md` | `deep-learning, infra` |
| `2018-02-25-improvement-distributed-training.md` | `deep-learning, infra` |
| `2018-03-20-image-similarity.md` | `computer-vision, deep-learning` |
| `2018-06-01-sagemaker-tut.md` | `aws, systems` |
| `2018-10-20-multi-armed-bandits.md` | `recsys, statistics` |
| `2018-10-28-randomization.md` | `statistics` |
| `2018-10-30-apache-hadoop-introduction.md` | `infra` |
| `2018-10-8-ner-tagger.md` | `nlp, deep-learning` |
| `2018-11-07-hdfs-1.md` | `infra` |
| `2018-11-10-hdfs-2.md` | `infra` |
| `2019-01-12-amazon-personalize.md` | `aws, recsys` |
| `2019-01-16-retrieval.md` | `nlp, recsys` |
| `2019-02-22-clustering.md` | `statistics, deep-learning` |
| `2019-03-04-mixture-models.md` | `statistics` |
| `2019-05-19-triplet.md` | `deep-learning, computer-vision` |
| `2019-09-06-pose-extraction.md` | `computer-vision, deep-learning` |
| `2021-02-06-yolact.md` | `computer-vision, deep-learning` |
| `2021-02-22-solo.md` | `computer-vision, deep-learning` |
| `2022-12-24-object-detection.md` | `computer-vision, deep-learning` |
| `2023-01-01-attention.md` | `nlp, deep-learning` |
| `2023-11-4-tts.md` | `nlp, deep-learning` |
| `2025-02-12-notes-recommendation-system.md` | `recsys, systems` |
| `2025-02-18-notes-k8s.md` | `infra, systems` |
| `2026-03-26-llm-architectures.md` | `llm, deep-learning` |

### Front Matter Fields

Add to every post that is missing them:

- `subtitle:` — one sentence describing the post (written from content, see below)
- `author: hoangbm`
- `social-share: true`

Do NOT add `image:` to posts that don't have one — adding placeholder images is out of scope.

### Subtitles (all posts)

| File | Subtitle |
|---|---|
| `2018-01-04-first-post.md` | Why I started writing about machine learning |
| `2018-01-08-convnet-element.md` | The building blocks of convolutional neural networks |
| `2018-01-12-convnet-architecture.md` | A tour of landmark CNN architectures from AlexNet to ResNet |
| `2018-01-16-challenges.md` | Common pitfalls when training deep neural networks |
| `2018-01-18-distributed-tensorflow.md` | Setting up distributed training with TensorFlow |
| `2018-02-25-improvement-distributed-training.md` | Techniques for faster and more stable distributed training |
| `2018-03-20-image-similarity.md` | Learning visual representations for similarity search |
| `2018-06-01-sagemaker-tut.md` | Training and deploying models with Amazon SageMaker |
| `2018-10-20-multi-armed-bandits.md` | Balancing exploration and exploitation in recommendation |
| `2018-10-28-randomization.md` | Why randomization matters for valid experiment design |
| `2018-10-30-apache-hadoop-introduction.md` | A practical introduction to the Hadoop ecosystem |
| `2018-10-8-ner-tagger.md` | Building a named entity recognition system from scratch |
| `2018-11-07-hdfs-1.md` | How the Hadoop Distributed File System stores data |
| `2018-11-10-hdfs-2.md` | HDFS architecture: NameNode, DataNode, and replication |
| `2019-01-12-amazon-personalize.md` | Using Amazon Personalize for production recommendation |
| `2019-01-16-retrieval.md` | Candidate retrieval in large-scale recommendation systems |
| `2019-02-22-clustering.md` | From K-Means to DBSCAN: clustering algorithms compared |
| `2019-03-04-mixture-models.md` | Gaussian mixture models and the EM algorithm |
| `2019-05-19-triplet.md` | Learning embeddings with triplet loss |
| `2019-09-06-pose-extraction.md` | Estimating human pose from images |
| `2021-02-06-yolact.md` | Real-time instance segmentation with YOLACT |
| `2021-02-22-solo.md` | Instance segmentation without bounding boxes |
| `2022-12-24-object-detection.md` | A personal guide to modern object detection |
| `2023-01-01-attention.md` | Understanding the attention mechanism in transformers |
| `2023-11-4-tts.md` | How modern text-to-speech systems work |
| `2025-02-12-notes-recommendation-system.md` | A practical three-stage recommendation system pipeline |
| `2025-02-18-notes-k8s.md` | Kubernetes concepts for ML engineers |
| `2026-03-26-llm-architectures.md` | Encoders, decoders, and how inference actually works |

### LLM Post Publishing

`_posts/2026-03-26-llm-architectures.md` and `images/llm/` are currently untracked. Add front matter additions (subtitle, author, tags) then commit both with `git add`.

---

## Sub-project 2: Post Structure (6 recent posts)

### Goal

Improve readability and section structure for the 6 posts from 2022 onwards. Each post gets what it specifically needs — not a template.

### Guiding Principles

- **Opening**: every post gets an explicit first paragraph answering "what will I learn here?" — if it already exists, improve it; if missing, add it
- **Sections**: `##` for main sections, `###` for subsections — no mixed heading levels at the top
- **Closing**: every post ends with a named section (title varies by post: "Putting it together", "In practice", "Final thoughts", "Where to go next") — no generic "Conclusion"
- **Paragraphs**: split any prose block longer than 5 lines
- **Transitions**: add a linking sentence between sections where the jump is abrupt
- **Examples**: add a short concrete example (pseudocode, analogy, or real-world scenario) where a concept is abstract and has none
- **No rewriting** of technical content — structure and readability only

### Per-Post Edits

#### `2022-12-24-object-detection.md`
Longest post. Needs: section heading cleanup (`##` consistency), split long paragraphs, add linking sentences between the major algorithm families (anchor-based vs anchor-free), closing section "In practice".

#### `2023-01-01-attention.md`
Dense mathematical content. Needs: stronger opening that contextualises why attention matters, short pseudocode example for the attention computation, closing section "Where to go next" pointing toward the transformer series.

#### `2023-11-4-tts.md`
Needs: opening paragraph (currently jumps straight to architecture), add one concrete listening analogy for the vocoder section, closing section "Final thoughts".

#### `2025-02-12-notes-recommendation-system.md`
"Notes" format — keep the loose structure. Needs: opening paragraph framing the three-stage pipeline, closing section "Putting it together" that ties the three stages back to each other.

#### `2025-02-18-notes-k8s.md`
"Notes" format — keep the loose structure. Needs: opening paragraph stating the audience ("for ML engineers, not ops"), closing section "In practice" with a one-paragraph summary of how these concepts connect.

#### `2026-03-26-llm-architectures.md`
Already has a good structure and opening. Needs: split one or two long paragraphs in the decoder section, add a brief closing section "Putting it together" that recaps encoder vs decoder vs encoder-decoder in two sentences.

### Out of Scope

- Posts from 2018–2021
- Adding new technical content or correcting factual claims
- Reformatting code blocks or diagrams
- Adding images

---

## Files Changed

**Sub-project 1:**
- All 28 `_posts/*.md` files — add `tags`, `subtitle`, `author`, `social-share`
- `_posts/2026-03-26-llm-architectures.md` — also newly tracked (was untracked)
- `images/llm/` — newly tracked

**Sub-project 2:**
- `_posts/2022-12-24-object-detection.md`
- `_posts/2023-01-01-attention.md`
- `_posts/2023-11-4-tts.md`
- `_posts/2025-02-12-notes-recommendation-system.md`
- `_posts/2025-02-18-notes-k8s.md`
- `_posts/2026-03-26-llm-architectures.md`
