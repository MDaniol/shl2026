# Ideas & lanes — who's working on what

One month, 12 of us. This board stops five people independently re-running the
same logreg. **Claim a lane before you sink a day into it**, and keep your row's
best number current so the team always knows the state of play.

## How to use it

1. **Claim:** add a row (or put your name on an empty one) — idea, your name,
   status `🔵 claimed`.
2. **Work:** as you get results, update **Best val macro-F1** and the **MLflow
   experiment** name (so anyone can open your runs / `leaderboard()`).
3. **Status:** `🔵 claimed` → `🟢 active` → `✅ promising` / `⚪ parked`.
4. Edit your own row freely; don't overwrite someone else's. This file is
   nbstripout-free plain markdown, so just commit it on `main` like a notebook.

Two people *may* share a lane if you coordinate (e.g. different FMs) — say so in
Notes. If your idea needs a new head or a new FM embedded, that's a lead task —
note it and ping the lead (see `STUDENTS.md`).

## The board

| Idea / lane | Owner | Status | Best val macro-F1 | MLflow experiment | Notes |
|-------------|-------|--------|-------------------|-------------------|-------|
| Baseline: logreg, all locations | _lead_ | 🟢 active | — | `baseline` | reference everyone compares against |
| Baseline: test-matching locations (no Hand) | — | ⚪ open | — | — | honest baseline vs. test conditions |
| MLP head sweep (depth/width/alpha) | — | ⚪ open | — | — | seed-sensitive — run a few seeds |
| Per-location vs. pooled locations | — | ⚪ open | — | — | does training on all locations' samples beat the best single location? (true feature-fusion needs an SDK change — ping the lead) |
| Feature transform before head (PCA / extra norm) | — | ⚪ open | — | — | — |
| FM comparison (when >1 FM is cached) | — | ⚪ open | — | — | needs lead to extract a 2nd FM |
| Confusing the motorised trio (Car/Bus/Train) | — | ⚪ open | — | — | targeted: where most points are lost |

*(Rows above are starter suggestions — replace/extend them. `⚪ open` = up for
grabs.)*

## Currently available to build on

- **Embedding caches:** _(lead keeps this current)_ — e.g. `synthetic` (always),
  then real FMs as they're extracted: `…`.
- **Heads:** see `list_heads()` — `linear`, `logreg`, `mlp`.

> New to this? Read `STUDENTS.md` (Part 2) first — it explains what each lane
> actually means and the rules that keep results comparable.
