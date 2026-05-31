# Blog Visual Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reskin the Jekyll blog to a dark-mode, soft-purple-accent aesthetic with a Featured + Grid homepage layout.

**Architecture:** CSS-override approach — update `_config.yml` color vars (which Jekyll injects into CSS via Liquid), promote the existing `prefers-color-scheme: dark` block to always-on, swap the blue accent to purple, and replace the flat post list in `index.html` with a featured-hero + 2-column card grid. No theme switch, no new dependencies.

**Tech Stack:** Jekyll (Beautiful Jekyll theme), Liquid templates, plain CSS

---

## File Map

| File | What changes |
|---|---|
| `_config.yml` | Update all color vars to dark values; remove `page-img` |
| `css/main.css` | Strip `prefers-color-scheme: dark` wrapper → always-on; blue→purple accent; add new `.featured-post`, `.post-grid`, `.post-card`, `.post-tag-pill` classes |
| `index.html` | Replace flat paginator loop with featured-hero (page 1) + post-card grid |
| `_layouts/post.html` | No structural changes needed — `.blog-tags` already uses the CSS class we're updating |

---

## Task 1: Update `_config.yml` colors

**Files:**
- Modify: `_config.yml:42-57`

- [ ] **Step 1: Edit color values and remove page-img**

  Replace the color block (lines 42–57) with:

  ```yaml
  navbar-col: "#161b22"
  navbar-text-col: "#c9d1d9"
  navbar-children-col: "#21262d"
  page-col: "#0d1117"
  link-col: "#d2a8ff"
  hover-col: "#d2a8ff"
  footer-col: "#161b22"
  footer-text-col: "#8b949e"
  footer-link-col: "#8b949e"
  ```

  Delete this line entirely (it enables the bg image):
  ```yaml
  page-img: "/images/bgimage.png"
  ```

- [ ] **Step 2: Verify Jekyll builds without error**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: `...done in X seconds.` with no errors.

- [ ] **Step 3: Commit**

  ```bash
  git add _config.yml
  git commit -m "config: switch to dark mode colors with purple accent"
  ```

---

## Task 2: Promote dark-mode CSS to always-on and swap blue → purple

**Files:**
- Modify: `css/main.css`

The existing file already has a `@media (prefers-color-scheme: dark) { ... }` block at the bottom (lines 1022–1131). This task removes the media-query wrapper so the styles apply unconditionally, swaps all blue values to purple, and adds a body text-color override (since body `color: #404040` is hardcoded in the base styles).

- [ ] **Step 1: Remove the media query wrapper**

  In `css/main.css`, delete line 1022 (`@media (prefers-color-scheme: dark) {`) and line 1131 (the closing `}`). Leave all the rules inside intact.

  Also delete the matching `@media (prefers-color-scheme: dark)` block for `.share-btn` at lines 840–846:
  ```css
  @media (prefers-color-scheme: dark) {
    .share-btn {
      background-color: #21262d;
      border-color: #30363d;
      color: #c9d1d9;
    }
  }
  ```
  Replace with the unwrapped version:
  ```css
  .share-btn {
    background-color: #21262d;
    border-color: #30363d;
    color: #c9d1d9;
  }
  ```

- [ ] **Step 2: Add body text-color override**

  Directly after the `body { background-color: #0d1117; ... }` rule (which is now unwrapped), add one line so the full rule reads:

  ```css
  body {
    background-color: #0d1117;
    background-image: none !important;
    color: #c9d1d9;
  }
  ```

- [ ] **Step 3: Swap blue accent → purple throughout the former dark-mode block**

  Do a find-and-replace **only inside the block you just unwrapped** (the rules from the old media query). Apply these substitutions in order:

  | Find | Replace | Used for |
  |---|---|---|
  | `#58a6ff` | `#d2a8ff` | links, navbar hover, blockquote border, code border, progress bar |
  | `#79c0ff` | `#e2c0ff` | link hover |
  | `#388bfd` | `#9a6fd8` | pager hover bg, back-to-top button, code badge bg, selection bg |

  After replacement the block should contain no `#58a6ff`, `#79c0ff`, or `#388bfd`.

  The `.blog-tags a` rule inside the block should end up as:
  ```css
  .blog-tags a {
    background-color: #21262d;
    color: #d2a8ff;
  }
  .blog-tags a:hover {
    background-color: #9a6fd8;
    color: #fff;
  }
  ```

  And the blockquote rule:
  ```css
  blockquote {
    background-color: #161b22;
    color: #8b949e;
    border-left-color: #d2a8ff;
  }
  ```

  And `.blog-post h3`:
  ```css
  .blog-post h3 { border-left-color: #d2a8ff; }
  ```

- [ ] **Step 4: Update the footer border rule**

  In the unwrapped block find:
  ```css
  footer {
    background-color: #161b22;
    border-top-color: #30363d;
  }
  ```
  And add a border-top style line (it's already there but verify it reads exactly):
  ```css
  footer {
    background-color: #161b22;
    border-top: 1px solid #30363d;
  }
  ```

- [ ] **Step 5: Add post-preview border-left accent on hover**

  Inside the unwrapped block, update the `.post-preview` rule to add a subtle purple left accent:
  ```css
  .post-preview {
    background-color: #161b22;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    border-left: 3px solid transparent;
    transition: box-shadow 0.2s ease, border-left-color 0.2s ease;
  }
  .post-preview:hover {
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6);
    border-left-color: #d2a8ff;
  }
  ```

- [ ] **Step 6: Serve and verify**

  ```bash
  bundle exec jekyll serve --livereload
  ```
  Open `http://localhost:4000`. Check:
  - Background is dark (`#0d1117`) ✓
  - Navbar is `#161b22` ✓
  - Links are purple, not blue ✓
  - Code blocks have dark bg with purple left border ✓
  - Blockquote has purple left border ✓

- [ ] **Step 7: Commit**

  ```bash
  git add css/main.css
  git commit -m "style: promote dark mode to always-on, swap accent blue→purple"
  ```

---

## Task 3: Homepage featured-hero + post-card grid

**Files:**
- Modify: `index.html`
- Modify: `css/main.css` (append new classes at bottom)

- [ ] **Step 1: Add new CSS classes to `css/main.css`**

  Append the following to the end of `css/main.css`:

  ```css
  /* --- Featured post hero --- */

  .featured-post {
    display: block;
    text-decoration: none;
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #d2a8ff;
    border-radius: 6px;
    padding: 28px 32px;
    margin-bottom: 32px;
    transition: border-left-color 0.2s ease, box-shadow 0.2s ease;
  }
  .featured-post:hover,
  .featured-post:focus {
    border-left-color: #e2c0ff;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    text-decoration: none;
  }
  .featured-post-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }
  .featured-label {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .featured-title {
    font-size: 22px;
    color: #e6edf3;
    font-weight: 700;
    margin: 0 0 10px;
    line-height: 1.3;
  }
  .featured-excerpt {
    font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #8b949e;
    line-height: 1.7;
    margin: 0 0 14px;
  }

  /* --- Post tag pill --- */

  .post-tag-pill {
    display: inline-block;
    background: #1f2937;
    color: #d2a8ff;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 2px 10px;
    border-radius: 3px;
    text-decoration: none;
  }
  .post-tag-pill:hover,
  .post-tag-pill:focus {
    background: #9a6fd8;
    color: #fff;
    text-decoration: none;
  }

  /* --- Post meta line --- */

  .post-meta-line {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #6e7681;
  }

  /* --- Post card grid --- */

  .post-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 40px;
  }
  @media only screen and (max-width: 600px) {
    .post-grid { grid-template-columns: 1fr; }
  }
  .post-card {
    display: block;
    text-decoration: none;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 20px;
    transition: border-color 0.2s ease, background 0.2s ease;
  }
  .post-card:hover,
  .post-card:focus {
    border-color: #d2a8ff;
    background: #1a1f29;
    text-decoration: none;
  }
  .card-title {
    font-size: 15px;
    color: #e6edf3;
    font-weight: 700;
    margin: 8px 0;
    line-height: 1.4;
  }
  ```

- [ ] **Step 2: Rewrite `index.html`**

  Replace the entire content of `index.html` with:

  ```liquid
  ---
  layout: page
  title: Bonjour
  subtitle: A record of how I become a ML Engineer
  use-site-title: true
  ---

  {% assign all_posts = paginator.posts %}

  {% if paginator.page == 1 %}
    {% assign featured = all_posts[0] %}
    {% assign grid_posts = all_posts | slice: 1, all_posts.size %}
  {% else %}
    {% assign featured = nil %}
    {% assign grid_posts = all_posts %}
  {% endif %}

  {% if featured %}
  <a href="{{ featured.url | prepend: site.baseurl }}" class="featured-post">
    <div class="featured-post-meta">
      {% if featured.tags.size > 0 %}
        <span class="post-tag-pill">{{ featured.tags[0] }}</span>
      {% endif %}
      <span class="featured-label">Latest post</span>
    </div>
    <h2 class="featured-title">{{ featured.title }}</h2>
    <p class="featured-excerpt">
      {{ featured.excerpt | strip_html | xml_escape | truncatewords: 45 }}
    </p>
    <div class="post-meta-line">
      {% assign words = featured.content | number_of_words %}
      {% assign read_time = words | divided_by: 200 | plus: 1 %}
      {{ featured.date | date: "%B %-d, %Y" }} &nbsp;&middot;&nbsp; {{ read_time }} min read
    </div>
  </a>
  {% endif %}

  <div class="post-grid">
    {% for post in grid_posts %}
    <a href="{{ post.url | prepend: site.baseurl }}" class="post-card">
      {% if post.tags.size > 0 %}
        <span class="post-tag-pill">{{ post.tags[0] }}</span>
      {% endif %}
      <h3 class="card-title">{{ post.title }}</h3>
      <div class="post-meta-line">
        {% assign words = post.content | number_of_words %}
        {% assign read_time = words | divided_by: 200 | plus: 1 %}
        {{ post.date | date: "%b %-d, %Y" }} &nbsp;&middot;&nbsp; {{ read_time }} min read
      </div>
    </a>
    {% endfor %}
  </div>

  {% if paginator.total_pages > 1 %}
  <ul class="pager main-pager">
    {% if paginator.previous_page %}
    <li class="previous">
      <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&larr; Newer Posts</a>
    </li>
    {% endif %}
    {% if paginator.next_page %}
    <li class="next">
      <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Older Posts &rarr;</a>
    </li>
    {% endif %}
  </ul>
  {% endif %}
  ```

- [ ] **Step 3: Serve and verify**

  ```bash
  bundle exec jekyll serve --livereload
  ```
  Open `http://localhost:4000`. Check:
  - Page 1: purple-bordered hero card for latest post, 2-column grid for remaining posts ✓
  - Featured post shows tag pill, excerpt, date + read time ✓
  - Grid cards show tag pill, title, date + read time (no excerpt) ✓
  - Clicking any card navigates to the post ✓
  - On mobile (resize to <600px): grid collapses to 1 column ✓
  - Navigate to page 2 (`http://localhost:4000/page2`): no hero, all posts in grid ✓
  - Pagination "Older Posts →" / "← Newer Posts" links work ✓

- [ ] **Step 4: Commit**

  ```bash
  git add index.html css/main.css
  git commit -m "feat: featured-hero + post-card grid homepage layout"
  ```

---

## Task 4: Final polish and cleanup

**Files:**
- Modify: `css/main.css` (minor fixes after visual review)

- [ ] **Step 1: Check post page rendering**

  With `bundle exec jekyll serve` still running, open any post (e.g. `http://localhost:4000` → click the featured post). Verify:
  - Post title is `#e6edf3` (bright white) ✓
  - Body text is `#c9d1d9` (light grey) ✓
  - Tags at bottom render as purple pills ✓
  - Code blocks have dark bg, purple left border ✓
  - Blockquote has purple left border ✓
  - Reading progress bar is purple ✓
  - Back-to-top button is purple ✓

- [ ] **Step 2: Check the tags page**

  Open `http://localhost:4000/tags`. Verify tag buttons render correctly (purple on dark bg).

- [ ] **Step 3: Remove now-redundant post-preview CSS from light-mode section**

  In `css/main.css`, the original `.post-preview` rules (lines ~354–504) set light backgrounds and box shadows that are now fully overridden. They're harmless but add noise. Optionally, remove or comment the hardcoded light values that are superseded:

  ```css
  /* these are overridden by the always-on dark block below */
  .post-preview {
    padding: 24px;
    border-bottom: none;
    margin-bottom: 20px;
    /* background: #ffffff; — removed, dark block sets #161b22 */
    border-radius: 8px;
    /* box-shadow: ...; — removed, dark block overrides */
    transition: box-shadow 0.2s ease, border-left-color 0.2s ease;
  }
  ```

  > **Note:** This step is optional — the site works correctly without it. Skip if you prefer not to touch the base rules.

- [ ] **Step 4: Add `.superpowers/` to `.gitignore`**

  The visual companion wrote session files to `.superpowers/`. Add this to `.gitignore` so they don't get committed:

  ```bash
  echo ".superpowers/" >> /Users/harbui/axon/axon-ai/minhhoangbui.github.io/.gitignore
  ```

  If there is no `.gitignore`, create it with that one line.

- [ ] **Step 5: Final commit**

  ```bash
  git add css/main.css .gitignore
  git commit -m "chore: polish dark mode styles, add .superpowers to gitignore"
  ```

---

## Verification Checklist (after all tasks)

Run `bundle exec jekyll serve` and check:

- [ ] Homepage (page 1): dark bg, purple hero card, 2-column grid
- [ ] Homepage (page 2+): dark bg, all posts in 2-column grid, no hero
- [ ] Any post page: dark bg, purple tags, dark code blocks, purple blockquote border
- [ ] Tags page: purple tag buttons on dark bg
- [ ] About Me page: dark bg, readable text
- [ ] No white flash / light-mode flickering
- [ ] Mobile (<600px): grid collapses to single column
- [ ] Disqus comments section still loads on post pages
