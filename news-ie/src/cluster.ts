import type { Article } from "./fetchFeeds.js";

export interface StoryCluster {
  articles: Article[];
}

const STOPWORDS = new Set([
  "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
  "was", "were", "with", "at", "by", "from", "as", "it", "its", "this",
  "that", "after", "over", "amid", "into", "says", "said", "how", "why",
  "what", "will", "new", "now",
]);

function significantWords(title: string): Set<string> {
  return new Set(
    title
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !STOPWORDS.has(w))
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let intersection = 0;
  for (const word of a) if (b.has(word)) intersection++;
  return intersection / (a.size + b.size - intersection);
}

/**
 * Word-overlap clustering. A cheap heuristic to group same-story headlines
 * across outlets without calling a model for every batch. Good enough to
 * prove the concept; swap for embedding-based clustering (via the FeroxAi
 * provider router) once this needs to scale past a handful of sources.
 */
const SIMILARITY_THRESHOLD = 0.35;

export function clusterArticles(articles: Article[]): StoryCluster[] {
  const wordSets = articles.map((a) => significantWords(a.title));
  const assigned = new Array(articles.length).fill(false);
  const clusters: StoryCluster[] = [];

  for (let i = 0; i < articles.length; i++) {
    if (assigned[i]) continue;
    const cluster: Article[] = [articles[i]];
    assigned[i] = true;
    for (let j = i + 1; j < articles.length; j++) {
      if (assigned[j]) continue;
      if (jaccard(wordSets[i], wordSets[j]) >= SIMILARITY_THRESHOLD) {
        cluster.push(articles[j]);
        assigned[j] = true;
      }
    }
    clusters.push({ articles: cluster });
  }

  return clusters
    .filter((c) => c.articles.length > 1)
    .sort((a, b) => b.articles.length - a.articles.length);
}
