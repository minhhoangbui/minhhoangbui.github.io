# Blog Visual Redesign — Design Spec

**Date:** 2026-05-30  
**Approach:** CSS Override + Minimal Template Edits (Approach A)  
**Scope:** Visual Design & UI only (content and features deferred)

---

## Goal

Refresh the blog's visual identity to a dark-mode, engineer-aesthetic design without switching the Jekyll theme or breaking existing functionality (Disqus, Google Analytics, permalinks, pagination, RSS).

---

## Color System

All color changes are applied via CSS overrides in `css/main.css` and updated values in `_config.yml`.

| Token | Current | New |
|---|---|---|
| Page background | `#FFFFFF` + bg image | `#0d1117` |
| Navbar / footer bg | `#F5F5F5` | `#161b22` |
| Navbar / footer border | none | `#30363d` |
| Body text | `#404040` | `#c9d1d9` |
| Heading text | inherited | `#e6edf3` |
| Links | `#008AFF` | `#d2a8ff` |
| Link hover | `#0085A1` | `#e2c0ff` |
| Muted / meta text | inherited | `#8b949e` |
| Subtle / timestamp text | inherited | `#6e7681` |
| Card / surface background | n/a | `#161b22` |
| Card border | n/a | `#21262d` (normal), `#30363d` (raised) |
| Code block background | light | `#161b22` |
| Code block border | none | `#30363d` |
| Blockquote border | teal (`#0085A1`) | `#d2a8ff` |
| Background image | `bgimage.png` | Removed |

`_config.yml` color keys to update: `navbar-col`, `navbar-text-col`, `navbar-children-col`, `page-col`, `link-col`, `hover-col`, `footer-col`, `footer-text-col`, `footer-link-col`. Remove `page-img`.

---

## Typography

No font changes. Existing stack is kept:

- **Body:** Lora (serif), 18px, line-height 1.8
- **Headings:** Georgia, 700 weight
- **Meta / tags / code labels:** `'Courier New', monospace` — adds terminal feel without a new font dependency

---

## Homepage Layout (`index.html`)

Replace the flat paginated list with a **Featured + Grid** layout:

### Featured Hero (first post on page 1 only)
- Full-width card: `background #161b22`, `border-left: 3px solid #d2a8ff`
- Contents: category tag (pill), "Latest post" label, title (22px), 2–3 line excerpt, date + read time
- Links to post on click

### Post Grid (remaining posts)
- 2-column CSS grid, `gap: 16px`
- Each card: `background #161b22`, `border: 1px solid #21262d`, hover highlights to `#d2a8ff` border
- Contents: category tag, title (15px), date + read time (no excerpt)
- On page 2+, no featured hero — all posts shown as grid cards

### Pagination
- Unchanged logic; restyled links with `border: 1px solid #30363d`, purple text

---

## Post Page (`_layouts/post.html` + CSS)

Structural changes: none. CSS-only reskin:

- Post title: `#e6edf3`, large
- Subtitle: `#8b949e`
- Meta line (date, read time): monospace, `#6e7681`
- Body text: `#c9d1d9`, Lora 18px, line-height 1.8
- Code blocks: `background #161b22`, `border: 1px solid #30363d` — rouge syntax highlighting works on dark without changes
- Blockquotes: left border `#d2a8ff`, background `#0d1117` (drop current light tint)
- Tags: pill style — `background #1f2937`, `color #d2a8ff`, monospace, uppercase

---

## Files Changed

| File | Change type |
|---|---|
| `_config.yml` | Update color values, remove `page-img` |
| `css/main.css` | CSS overrides for dark mode, card styles, post styles |
| `index.html` | New featured + grid layout (replaces flat list) |
| `_layouts/post.html` | Minor: ensure tag rendering uses new pill class |

No new dependencies. No theme switch. No changes to `_includes/`, `_layouts/base.html` structure, or Jekyll plugins.

---

## Out of Scope

- Dark/light toggle (deferred — adds JS complexity)
- Content improvements (separate initiative)
- New features: search, TOC, related posts (separate initiative)
- SEO / Open Graph meta tags (separate initiative)
