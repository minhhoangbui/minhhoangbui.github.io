# Blog Features & UX — Design Spec

**Date:** 2026-05-30  
**Scope:** Table of Contents (floating sidebar), Related Posts (tag-based), Reading Progress Bar  
**Constraint:** 100% GitHub Pages compatible — no unsupported plugins, no server-side logic

---

## Goal

Add three reader-experience features to the blog without introducing any Jekyll plugins or server dependencies. All features are either pure JavaScript (client-side) or pure Liquid (build-time).

---

## Feature 1: Table of Contents — Floating Sidebar

### Behavior

- Appears on post pages only, on screens wider than 1300px
- Positioned fixed in the right margin beside the post content column — the content column itself does not change width or layout
- Hidden entirely on screens ≤ 1300px (no collapsed/hidden state, simply absent)
- Automatically built from the post's `h2` and `h3` headings at page load
- Active section highlight: the TOC link for the heading currently in the viewport is styled purple + bold, updated via `IntersectionObserver`

### Implementation

**`js/main.js`** — add TOC builder function:
- Runs only if `.blog-post` exists on the page
- Queries all `h2, h3` inside `.blog-post`
- If fewer than 2 headings found, does nothing (no TOC for very short posts)
- Each heading gets an auto-generated `id` if it doesn't already have one (slugified from text content)
- Builds `<nav id="toc">` with `<ul>` list: `h2` items at top level, `h3` items indented with `.toc-l2` class
- Appends `<nav id="toc">` to `document.body`
- `IntersectionObserver` watches all headings; when one enters the viewport, marks its corresponding TOC link as `.toc-active`

**`css/main.css`** — add TOC styles:
```
#toc {
  position: fixed;
  top: 140px;
  right: calc((100vw - 830px) / 2 - 220px);  /* sits in right margin */
  width: 200px;
  max-height: calc(100vh - 180px);
  overflow-y: auto;
}
@media (max-width: 1300px) { #toc { display: none; } }
```
- TOC background: `#161b22`, border `#30363d`, border-radius `6px`, padding `14px`
- "Contents" label: purple, monospace, uppercase, small
- Links: `#8b949e` default, `#d2a8ff` + font-weight 600 when `.toc-active`
- `.toc-l2` links: `padding-left: 12px`, slightly smaller font

**`_layouts/post.html`** — no structural changes needed; JS appends the nav to `document.body`.

---

## Feature 2: Related Posts — Tag-Based

### Behavior

- Rendered at Jekyll build time via Liquid — zero JavaScript, zero runtime cost
- Appears below social share buttons and above Disqus comments on post pages
- Shows up to 3 posts that share at least one tag with the current post, ordered by date descending
- If the current post has no tags, or no other post shares a tag: section is not rendered (no empty heading)
- Layout: 3 cards in a row (same `.post-card` + `.post-tag-pill` + `.card-title` + `.post-meta-line` classes from the homepage grid); collapses to 1 column on mobile

### Implementation

**`_layouts/post.html`** — add Liquid block between social-share and Disqus:

```liquid
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
      <div class="post-meta-line">
        {{ post.date | date: "%b %-d, %Y" }}
      </div>
    </a>
    {% endfor %}
  </div>
</div>
{% endif %}
```

**`css/main.css`** — add minimal styles:
```css
.related-posts { margin: 40px 0 30px; }
.related-posts-title {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #6e7681;
  margin-bottom: 16px;
}
```
The `.post-grid`, `.post-card`, `.card-title`, `.post-meta-line`, `.post-tag-pill` classes are already defined from the homepage redesign.

---

## Feature 3: Reading Progress Bar

### Behavior

- Thin 3px bar fixed at the very top of the page, above the navbar
- Fills left-to-right as the user scrolls down the page
- Shows on post pages only
- Color: `#d2a8ff` (already set in existing CSS)

### Implementation

The DOM element (`<div id="reading-progress-bar">`) already exists in `_layouts/base.html`.  
The CSS (`position: fixed; top: 0; height: 3px; background-color: #d2a8ff; width: 0%`) is already in `css/main.css`.

**`js/main.js`** — add ~15 lines:
```javascript
(function () {
  if (!document.querySelector('.blog-post')) return;
  var bar = document.getElementById('reading-progress-bar');
  if (!bar) return;
  bar.style.display = 'block';
  window.addEventListener('scroll', function () {
    var scrolled = window.scrollY;
    var total = document.body.scrollHeight - window.innerHeight;
    bar.style.width = total > 0 ? (scrolled / total * 100) + '%' : '0%';
  }, { passive: true });
}());
```

No CSS changes. No template changes. No new files.

---

## Files Changed

| File | Change |
|---|---|
| `js/main.js` | Add TOC builder + IntersectionObserver + progress bar scroll handler |
| `css/main.css` | Add `#toc` positioning and link styles + `.related-posts` / `.related-posts-title` |
| `_layouts/post.html` | Add related posts Liquid block |

---

## Out of Scope

- Search (deferred)
- TOC on mobile (hidden; adds complexity for limited benefit on small screens)
- Manually curated related posts (tag-based matching is sufficient)
- TOC toggle/collapse behavior (sidebar is always open when visible)
