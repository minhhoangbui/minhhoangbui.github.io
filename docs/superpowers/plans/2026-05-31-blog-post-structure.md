# Blog Post Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve readability and section structure for the 6 posts from 2022 onwards — stronger openings, varied closings, split long paragraphs, add concrete examples where concepts are abstract.

**Architecture:** One task per post. Each task is a focused prose edit: read the post, apply the specific changes listed, build to verify, commit. No new technical content is added. No template is imposed — each post gets what it specifically needs.

**Tech Stack:** Jekyll Markdown posts, kramdown

---

## Editing Principles (apply to every task)

- **Do not rewrite technical content** — only structure and readability
- **Opening**: if the post jumps straight into content without framing what the reader will learn, add 1–3 sentences before the first heading
- **Sections**: use `##` for top-level sections, `###` for subsections — fix any inversions
- **Long paragraphs**: split any prose block longer than 5 lines into two
- **Transitions**: if two `##` sections feel disconnected, add one linking sentence at the end of the first
- **Closing**: every post must end with a named `##` section — title varies by post (see each task)
- **Examples**: where a concept is explained abstractly with no illustration, add a short one (2–4 lines of pseudocode, or a one-sentence analogy)

---

## Task 1: `2022-12-24-object-detection.md`

**File:** `_posts/2022-12-24-object-detection.md`

This is the longest recent post. Focus: heading cleanup, paragraph splits, transitions between algorithm families, closing section.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2022-12-24-object-detection.md
  ```

- [ ] **Step 2: Apply edits**

  Make the following specific changes:

  1. **Opening**: Check the first paragraph before any `##`. If it doesn't explicitly tell the reader what they'll learn (e.g., which detectors are covered, what the structure is), add a 2-sentence framing paragraph at the very top.

  2. **Heading levels**: Scan all headings. Any `###` that appears without a parent `##` should be promoted to `##`. Fix the hierarchy so the outline is coherent.

  3. **Anchor-based → anchor-free transition**: Find where the post moves from anchor-based detectors (Faster R-CNN, YOLO variants) to anchor-free methods (FCOS, CenterNet, etc.). Add a transition sentence at the end of the anchor-based section, e.g.: *"Anchor-based methods dominated the field for years, but they come with a significant tuning burden. Anchor-free detectors emerged to address exactly this."*

  4. **Paragraph splits**: Find any prose block longer than 5 lines. Split it at a logical sentence boundary into two paragraphs.

  5. **Closing section**: Add at the very end of the file:
     ```markdown
     ## In practice

     Choosing a detector depends more on your constraints than on benchmark numbers. If you need real-time inference on edge hardware, a single-stage anchor-free model like YOLOX or FCOS is a strong starting point. If accuracy matters more than speed and you have GPU headroom, a two-stage detector like Faster R-CNN with a strong backbone will serve you well. Either way, the concepts here — anchor design, feature pyramids, NMS — will reappear in almost every modern detection system you encounter.
     ```

- [ ] **Step 3: Build**

  ```bash
  cd /Users/harbui/axon/axon-ai/minhhoangbui.github.io
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2022-12-24-object-detection.md
  git commit -m "content: improve structure and readability of object detection post"
  ```

---

## Task 2: `2023-01-01-attention.md`

**File:** `_posts/2023-01-01-attention.md`

Dense mathematical post. Focus: contextualising opening, pseudocode example for the attention computation, closing pointing forward.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2023-01-01-attention.md
  ```

- [ ] **Step 2: Apply edits**

  1. **Opening**: Before the first `##`, add or strengthen the introductory paragraph to answer: *why does attention matter?* If the current opening doesn't include a sentence like "Before attention, sequence models had to compress everything into a single vector" (or equivalent), add it. Keep to 3 sentences max.

  2. **Pseudocode example**: Find the section that explains the scaled dot-product attention formula. Immediately after the formula, add a short pseudocode block if one doesn't already exist:

     ```markdown
     ```python
     # Scaled dot-product attention (simplified)
     def attention(Q, K, V):
         scores = Q @ K.T / sqrt(d_k)   # similarity between query and each key
         weights = softmax(scores)        # normalise to a probability distribution
         return weights @ V              # weighted sum of values
     ```
     ```

  3. **Paragraph splits**: Find any prose block longer than 5 lines and split at a logical boundary.

  4. **Closing section**: Add at the end:
     ```markdown
     ## Where to go next

     The attention mechanism described here is the core of the transformer. Once you're comfortable with it, the next natural step is understanding how it scales: multi-head attention runs several attention operations in parallel, each learning to focus on different aspects of the input. From there, the architecture of BERT (encoder-only) and GPT (decoder-only) falls into place naturally — both are just transformers with different masking strategies.
     ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2023-01-01-attention.md
  git commit -m "content: add opening context, pseudocode, and closing to attention post"
  ```

---

## Task 3: `2023-11-4-tts.md`

**File:** `_posts/2023-11-4-tts.md`

Needs an opening paragraph and a concrete analogy for the vocoder section, plus a closing.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2023-11-4-tts.md
  ```

- [ ] **Step 2: Apply edits**

  1. **Opening**: If the post starts directly with a heading or jumps into architecture without framing, add before the first `##`:

     ```markdown
     Modern text-to-speech systems are a two-part pipeline: a model that converts text into an intermediate acoustic representation (a mel spectrogram), and a vocoder that converts that representation into raw audio. This post walks through how each part works and where the interesting engineering happens.
     ```

     If the post already has an opening paragraph, strengthen it to cover this same ground — don't add a duplicate.

  2. **Vocoder analogy**: Find the section covering the vocoder (WaveNet, WaveGlow, HiFi-GAN, or similar). If the explanation of how vocoders convert spectrograms to audio is abstract, add after the first sentence:

     *"Think of the spectrogram as a musical score — it describes what notes to play and when. The vocoder is the instrument that actually produces the sound."*

  3. **Paragraph splits**: Split any prose block longer than 5 lines.

  4. **Closing section**: Add at the end:
     ```markdown
     ## Final thoughts

     The gap between the best open-source TTS systems and human speech has closed dramatically in the last few years. What makes modern systems work is not any single breakthrough but the combination: better acoustic models (non-autoregressive, parallel), better vocoders (learned priors, adversarial training), and more data. If you're building on top of TTS, the practical implication is that pre-trained models like VITS or Bark are now good enough for most use cases — the interesting work is in fine-tuning for specific voices and domains.
     ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2023-11-4-tts.md
  git commit -m "content: add opening, vocoder analogy, and closing to TTS post"
  ```

---

## Task 4: `2025-02-12-notes-recommendation-system.md`

**File:** `_posts/2025-02-12-notes-recommendation-system.md`

Notes-style post — keep the loose format. Add an opening that frames the three stages and a closing that ties them together.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2025-02-12-notes-recommendation-system.md
  ```

- [ ] **Step 2: Apply edits**

  1. **Opening**: Before the first `##` (or before the numbered list if there's no heading), add:

     ```markdown
     A production recommendation system rarely lives in a single model. The standard architecture is a three-stage funnel: retrieve a broad set of candidates from millions of items, rank them with a more expensive model, then re-rank with business rules and diversity constraints. Each stage makes a different tradeoff between speed and precision. These notes cover the key ideas in each stage.
     ```

     If a similar framing already exists, skip this step.

  2. **Paragraph splits**: Split any prose block longer than 5 lines.

  3. **Closing section**: Add at the end:
     ```markdown
     ## Putting it together

     The three stages aren't independent — they compound. A retrieval stage that misses relevant candidates can't be rescued by ranking. A ranking model that ignores diversity will produce a list that a re-ranker can only partially fix. In practice, most of the engineering effort goes into retrieval (because it's the bottleneck on recall) and into the feedback loops that keep all three stages aligned as user behaviour shifts.
     ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2025-02-12-notes-recommendation-system.md
  git commit -m "content: add opening framing and closing to recsys post"
  ```

---

## Task 5: `2025-02-18-notes-k8s.md`

**File:** `_posts/2025-02-18-notes-k8s.md`

Notes-style post. Add an ML-engineer-focused opening and a practical closing.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2025-02-18-notes-k8s.md
  ```

- [ ] **Step 2: Apply edits**

  1. **Opening**: Before the first `##`, add:

     ```markdown
     These notes are written for ML engineers who use Kubernetes as a deployment target but don't want to become Kubernetes operators. The goal is to build a working mental model of the abstractions — Pods, Deployments, Services, Ingress — so that reading a Helm chart or debugging a failing rollout feels tractable rather than opaque.
     ```

     If a similar audience-statement already exists, skip or merge.

  2. **Paragraph splits**: Split any prose block longer than 5 lines.

  3. **Closing section**: Add at the end:
     ```markdown
     ## In practice

     Most ML engineers interact with Kubernetes through a layer of abstraction — Helm charts, Argo Workflows, Kubeflow Pipelines, or a platform team's internal tooling. Understanding the primitives here doesn't replace that, but it means you can reason about what's happening when a deployment is stuck, a pod is OOMKilled, or a service isn't routing traffic. That's usually enough to unblock yourself without filing a ticket.
     ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2025-02-18-notes-k8s.md
  git commit -m "content: add ML-engineer-focused opening and closing to K8s post"
  ```

---

## Task 6: `2026-03-26-llm-architectures.md`

**File:** `_posts/2026-03-26-llm-architectures.md`

Already has a good opening and structure. Focus: split long paragraphs in the decoder section, add a brief closing recap.

- [ ] **Step 1: Read the post**

  ```bash
  cat /Users/harbui/axon/axon-ai/minhhoangbui.github.io/_posts/2026-03-26-llm-architectures.md
  ```

- [ ] **Step 2: Apply edits**

  1. **Decoder section paragraph splits**: Find the section covering decoder-only models (GPT family). If any prose block runs longer than 5 lines, split it at a logical sentence boundary.

  2. **Inference section paragraph splits**: Find the section covering the prefill/decode split in inference. Apply the same paragraph-split rule.

  3. **Closing section**: Add at the end of the file:
     ```markdown
     ## Putting it together

     The architecture choice — encoder-only, decoder-only, or encoder-decoder — is really a choice about what the model needs to know at generation time. Encoders see everything; decoders see only the past; encoder-decoders separate understanding from generation. For most applications today, decoder-only models (the GPT family) have won because scale and RLHF have made them surprisingly good at tasks that look like they'd need bidirectional context. But understanding why each architecture exists makes it easier to reason about when the exceptions still apply.
     ```

- [ ] **Step 3: Build**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add _posts/2026-03-26-llm-architectures.md
  git commit -m "content: split long paragraphs and add closing to LLM architectures post"
  ```

---

## Verification Checklist

After all tasks:

- [ ] `bundle exec jekyll build` passes with no errors
- [ ] Each of the 6 posts ends with a `##` closing section
- [ ] No post has a prose block longer than 5 lines (rough visual check)
- [ ] The attention post has a pseudocode block for the attention computation
- [ ] The TTS post has the vocoder analogy
