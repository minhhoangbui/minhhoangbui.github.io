# Blog Features & UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a floating TOC sidebar, tag-based related posts section, and verify the reading progress bar (already implemented) on the Jekyll blog.

**Architecture:** TOC is pure JavaScript injected at page load; related posts is pure Liquid rendered at Jekyll build time; progress bar is already implemented via existing scroll handler in `main.js`. No new Jekyll plugins. Fully GitHub Pages compatible.

**Tech Stack:** Jekyll 3.5.2, Liquid templates, jQuery (already loaded), vanilla JS `IntersectionObserver`

---

## File Map

| File | Change |
|---|---|
| `js/main.js` | Add `main.initTOC()` method + call from `main.init()` |
| `css/main.css` | Append `#toc` styles + `.related-posts` styles |
| `_layouts/post.html` | Insert related posts Liquid block between social-share and pager |

---

## Task 1: Table of Contents — JavaScript

**Files:**
- Modify: `js/main.js`

- [ ] **Step 1: Add `initTOC` method to the `main` object**

  In `js/main.js`, inside the `main = { ... }` object, add this method after `setImg` (before the closing `}`):

  ```javascript
  initTOC: function() {
    if ($('.blog-post').length === 0) return;
    var headings = $('.blog-post').find('h2, h3');
    if (headings.length < 2) return;

    var nav = $('<nav id="toc"><div class="toc-label">Contents</div><ul class="toc-list"></ul></nav>');
    var list = nav.find('.toc-list');

    headings.each(function() {
      var $h = $(this);
      var tag = this.tagName.toLowerCase();
      var text = $h.text();
      if (!$h.attr('id')) {
        var id = text.toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .trim();
        $h.attr('id', id);
      }
      var li = $('<li><a href="#' + $h.attr('id') + '">' + text + '</a></li>');
      if (tag === 'h3') li.addClass('toc-l2');
      list.append(li);
    });

    $('body').append(nav);

    if (!('IntersectionObserver' in window)) return;
    var tocLinks = nav.find('a');
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var id = '#' + entry.target.getAttribute('id');
          tocLinks.removeClass('toc-active');
          nav.find('a[href="' + id + '"]').addClass('toc-active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });

    headings.each(function() {
      if (this.id) observer.observe(this);
    });
  },
  ```

  > **Note on placement:** Add a comma after `setImg: function(...) { ... }` if needed. The `initTOC` method goes inside the `main` object, before the closing `};`.

- [ ] **Step 2: Call `initTOC` from `main.init()`**

  Inside `main.init()`, after the line `main.initImgs();` (line ~101), add:

  ```javascript
  main.initTOC();
  ```

- [ ] **Step 3: Build to verify no JS syntax errors**

  ```bash
  cd /Users/harbui/axon/axon-ai/minhhoangbui.github.io
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: `done in X seconds.` — no errors.

- [ ] **Step 4: Serve and verify TOC appears on a post**

  ```bash
  bundle exec jekyll serve --port 4000 &
  sleep 5 && open http://localhost:4000
  ```

  Click into any post that has `h2`/`h3` headings (e.g. the LLM architectures post). On a wide screen (>1300px) you should see a floating sidebar on the right with links. Scrolling should highlight the current section in purple.

  If the screen is narrow, resize to >1300px to see it.

- [ ] **Step 5: Commit**

  ```bash
  git add js/main.js
  git commit -m "feat: add floating TOC sidebar with IntersectionObserver active tracking"
  ```

---

## Task 2: Table of Contents — CSS

**Files:**
- Modify: `css/main.css` (append at end)

- [ ] **Step 1: Append TOC styles to `css/main.css`**

  Add the following to the very end of `css/main.css`:

  ```css
  /* --- Table of Contents sidebar --- */

  #toc {
    position: fixed;
    top: 140px;
    right: calc((100vw - 830px) / 2 - 220px);
    width: 200px;
    max-height: calc(100vh - 180px);
    overflow-y: auto;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 14px;
    z-index: 100;
  }
  @media (max-width: 1300px) {
    #toc { display: none; }
  }
  .toc-label {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #d2a8ff;
    margin-bottom: 8px;
  }
  .toc-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .toc-list li a {
    display: block;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #8b949e;
    text-decoration: none;
    padding: 3px 0;
    line-height: 1.4;
  }
  .toc-list li a:hover,
  .toc-list li a:focus { color: #d2a8ff; }
  .toc-list li a.toc-active { color: #d2a8ff; font-weight: 600; }
  .toc-list li.toc-l2 a { padding-left: 12px; font-size: 10px; }
  ```

- [ ] **Step 2: Build and verify styling**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

  Open a post at `http://localhost:4000` on a >1300px wide screen. The TOC sidebar should:
  - Have dark `#161b22` background with subtle border
  - Show "CONTENTS" label in purple
  - Links in `#8b949e`, active link in `#d2a8ff` bold
  - `h3` links indented and slightly smaller than `h2`

- [ ] **Step 3: Commit**

  ```bash
  git add css/main.css
  git commit -m "style: add TOC sidebar positioning and link styles"
  ```

---

## Task 3: Related Posts — Liquid + CSS

**Files:**
- Modify: `_layouts/post.html`
- Modify: `css/main.css` (append at end)

The current `_layouts/post.html` structure around the insertion point:

```html
{% if page.social-share %}
  {% include social-share.html %}
{% endif %}

<ul class="pager blog-pager">   ← insert BEFORE this line
```

- [ ] **Step 1: Insert the related posts Liquid block in `_layouts/post.html`**

  Find the line `<ul class="pager blog-pager">` in `_layouts/post.html` and insert the following block immediately before it:

  ```liquid
  {% if page.tags.size > 0 %}
    {% assign related = "" | split: "" %}
    {% assign related_urls = "" | split: "" %}
    {% for post in site.posts %}
      {% unless post.url == page.url %}
        {% unless related_urls contains post.url %}
          {% for tag in post.tags %}
            {% if page.tags contains tag %}
              {% assign related = related | push: post %}
              {% assign related_urls = related_urls | push: post.url %}
              {% break %}
            {% endif %}
          {% endfor %}
        {% endunless %}
      {% endunless %}
    {% endfor %}
    {% assign related = related | slice: 0, 3 %}
    {% if related.size > 0 %}
    <div class="related-posts">
      <h4 class="related-posts-title">Related Posts</h4>
      <div class="post-grid">
        {% for post in related %}
        <a href="{{ post.url | prepend: site.baseurl }}" class="post-card">
          {% if post.tags.size > 0 %}
            <span class="post-tag-pill">{{ post.tags[0] }}</span>
          {% endif %}
          <h3 class="card-title">{{ post.title }}</h3>
          <div class="post-meta-line">{{ post.date | date: "%b %-d, %Y" }}</div>
        </a>
        {% endfor %}
      </div>
    </div>
    {% endif %}
  {% endif %}
  ```

- [ ] **Step 2: Append related-posts CSS to `css/main.css`**

  Add to the end of `css/main.css`:

  ```css
  /* --- Related Posts --- */

  .related-posts {
    margin: 40px 0 30px;
  }
  .related-posts-title {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6e7681;
    margin-bottom: 16px;
  }
  ```

  > The `.post-grid`, `.post-card`, `.card-title`, `.post-meta-line`, and `.post-tag-pill` classes are already defined in `css/main.css` from the homepage redesign.

- [ ] **Step 3: Build and verify**

  ```bash
  bundle exec jekyll build 2>&1 | tail -5
  ```
  Expected: no errors.

  Open a post that has tags (check `_posts/2025-02-12-notes-recommendation-system.md` or `_posts/2025-02-18-notes-k8s.md`). Scroll below the social share buttons. You should see a "RELATED POSTS" label and up to 3 post cards below it.

  Open a post with no tags (e.g. `_posts/2018-01-04-first-post.md`). No related posts section should appear.

- [ ] **Step 4: Commit**

  ```bash
  git add _layouts/post.html css/main.css
  git commit -m "feat: add tag-based related posts section to post layout"
  ```

---

## Task 4: Verify Reading Progress Bar

**Files:** None — already implemented in `js/main.js` lines 19–25 and 91–93.

- [ ] **Step 1: Confirm progress bar works on a post page**

  With `bundle exec jekyll serve --port 4000` running, open any post. The thin purple bar at the very top of the page should:
  - Be invisible on the homepage (bar is hidden when `.blog-post` is absent)
  - Appear and fill left-to-right as you scroll down a post
  - Reset to 0% when you scroll back to the top

  If it does not appear, check that `#reading-progress-bar` exists in the page source and that `$('.blog-post').length > 0` evaluates correctly.

- [ ] **Step 2: No commit needed**

  The progress bar is already committed. This step is verification only.

---

## Verification Checklist (after all tasks)

Run `bundle exec jekyll serve --port 4000` and check:

- [ ] Post with 2+ headings on screen >1300px: TOC sidebar visible on right, links highlight on scroll
- [ ] Post on screen ≤1300px: no TOC (not just hidden — not in DOM either since JS skips at narrow widths)
- [ ] Post with no headings (or only 1): no TOC injected
- [ ] Post with tags: related posts section shows up to 3 matching posts
- [ ] Post without tags: no related posts section
- [ ] All post pages: purple progress bar fills on scroll
- [ ] Homepage: no progress bar, no TOC
- [ ] Jekyll build: no errors or warnings beyond pre-existing Ruby version noise
