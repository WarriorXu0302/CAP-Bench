# CAP-Bench Project Site

Static project site served by **GitHub Pages** at:

    https://warriorxu0302.github.io/CAP-Bench/

The site is plain HTML/CSS/JS — there is no build step. Edit the
files in this directory and `git push` to publish.

## Layout

```
docs/
├── index.html              # Main overview page
├── leaderboard.html        # Full leaderboard with sortable columns
├── assets/                 # CSS + figures (figures mirror /assets/figures/)
├── data/leaderboard.json   # Single source of truth for leaderboard rows
└── js/leaderboard.js       # Renders the leaderboard table on both pages
```

## Updating the leaderboard

Edit `data/leaderboard.json` and push. Both the homepage preview and
the full leaderboard re-render automatically. Ranks are computed from
`partial_completion` so you don't need to update them manually.

To accept a community submission, paste their numbers into
`entries[]`, mark `category` as `"Open-source agent"` or
`"Commercial"`, and bump `last_updated`.

## Enabling GitHub Pages (one-time setup)

In the GitHub repo: **Settings → Pages → Source** → choose the
`main` branch and the `/docs` folder. The site goes live within a
minute at the URL above.
