# Ferox News — Ireland

A coverage-comparison tool in the spirit of Ground News, scoped to Irish
outlets: it groups the same story across multiple publishers and tags each
one with who owns it, so the comparison is the point — not hosting anyone's
articles.

## Why this is legally safe

Only three things are ever stored or shown: **headline, a short summary,
and a link back to the original article** — pulled from each outlet's own
public RSS feed, which they publish specifically to be syndicated. No full
article text is fetched, stored, or displayed. This is the same design
pattern Google News and Ground News use, and it's structurally different
from mirroring full articles: nothing here substitutes for visiting the
original, so it doesn't compete with the ad revenue that funds it.

Ownership notes are stated as sourced public record, not opinion — each
one links to where it came from (Wikipedia today; see
[Media Ownership Monitor Ireland](https://ireland.mom-gmr.org) for deeper
independent research on Irish media ownership). There are deliberately no
political-bias labels yet: Ground News can make those claims because they
cite three independent monitoring organizations (AllSides, Ad Fontes
Media, Media Bias/Fact Check); no equivalent Irish body was found during
research, so bias labeling is out of scope until there's something
credible to cite it to.

## How it works

1. `src/sources.ts` — the outlet registry: name, RSS feed URL, ownership.
2. `src/ownership.ts` — sourced ownership facts, one entry per group.
3. `src/fetchFeeds.ts` — pulls and normalizes each outlet's RSS/Atom feed.
4. `src/cluster.ts` — groups articles covering the same story by headline
   word-overlap, and keeps only stories covered by more than one outlet.
5. `src/index.ts` — serves the comparison page and refreshes every 10
   minutes.

## Setup

```bash
npm install
npm run dev
# then open http://localhost:8790
```

## Known limitation

Three registered sources — Irish Independent, Belfast Telegraph (both
Mediahuis), and The Irish Sun (News UK) — currently return 403 to this
server's requests specifically (curl succeeds; Node's fetch doesn't),
most likely fingerprint-based bot protection rather than anything about
the request itself. Rather than working around their anti-bot measures,
these stay registered in `sources.ts` but are silently skipped each
refresh — the app degrades gracefully and runs on the five that work
(Irish Times, Irish Examiner, TheJournal.ie, RTÉ News, The Irish News).

## Status

7 outlets registered, 5 currently reachable, word-overlap clustering,
ownership tags, no styling beyond basic readability. Next steps: a
working feed for Business Post (no public RSS path found yet), upgrade
clustering to embeddings via the FeroxAi provider router once story
volume grows, and a proper front end.
