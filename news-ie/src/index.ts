import express from "express";
import { sources } from "./sources.js";
import { fetchAllArticles } from "./fetchFeeds.js";
import { clusterArticles, type StoryCluster } from "./cluster.js";

const PORT = Number(process.env.PORT ?? 8790);
const REFRESH_MS = 10 * 60 * 1000;

const sourceById = new Map(sources.map((s) => [s.id, s]));

let cachedClusters: StoryCluster[] = [];
let lastRefreshed: Date | null = null;

async function refresh(): Promise<void> {
  const articles = await fetchAllArticles(sources);
  cachedClusters = clusterArticles(articles);
  lastRefreshed = new Date();
  console.log(
    `Refreshed: ${articles.length} articles fetched, ${cachedClusters.length} covered by more than one outlet`
  );
}

const app = express();

app.get("/api/stories", (_req, res) => {
  res.json({ lastRefreshed, stories: cachedClusters });
});

app.get("/", (_req, res) => {
  res.type("html").send(renderPage(cachedClusters, lastRefreshed));
});

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string)
  );
}

function renderPage(clusters: StoryCluster[], refreshedAt: Date | null): string {
  const stories = clusters
    .map((cluster) => {
      const items = cluster.articles
        .map((a) => {
          const owner = sourceById.get(a.sourceId)?.ownership;
          return `
            <li>
              <div class="outlet">
                <span class="outlet-name">${escapeHtml(a.sourceName)}</span>
                <span class="owner" title="${escapeHtml(owner?.note ?? "")}">${escapeHtml(owner?.name ?? "")}</span>
              </div>
              <a href="${escapeHtml(a.link)}" target="_blank" rel="noopener">${escapeHtml(a.title)}</a>
            </li>`;
        })
        .join("");
      return `<article class="story"><ul>${items}</ul></article>`;
    })
    .join("");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ferox News — Ireland</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 780px; margin: 0 auto; padding: 28px 20px 60px; color: #1a1a1a; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 28px; }
  .story { border-bottom: 1px solid #e2e2e2; padding: 16px 0; }
  .story ul { list-style: none; margin: 0; padding: 0; }
  .story li { margin-bottom: 8px; }
  .outlet { display: flex; gap: 8px; align-items: baseline; font-size: 12px; color: #666; margin-bottom: 2px; }
  .outlet-name { font-weight: 600; min-width: 130px; }
  .owner { font-size: 11px; color: #888; }
  a { color: #14314f; text-decoration: none; }
  a:hover { text-decoration: underline; }
  footer { margin-top: 40px; font-size: 12px; color: #888; }
</style>
</head>
<body>
  <h1>Ferox News — Ireland</h1>
  <p class="meta">${clusters.length} stories currently covered by more than one outlet &middot; last refreshed ${
    refreshedAt ? refreshedAt.toLocaleString("en-IE") : "never"
  }</p>
  ${stories || "<p>No multi-outlet stories yet — try again shortly.</p>"}
  <footer>Headlines and links only, sourced from each outlet's own public RSS feed. Ownership notes are public record — see each outlet's Wikipedia entry, linked in the source.</footer>
</body>
</html>`;
}

refresh()
  .catch((err) => console.error("Initial fetch failed, starting anyway:", err))
  .finally(() => {
    app.listen(PORT, () => console.log(`Ferox News IE listening on port ${PORT}`));
  });

setInterval(() => {
  refresh().catch((err) => console.error("Refresh failed:", err));
}, REFRESH_MS);
