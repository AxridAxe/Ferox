import { XMLParser } from "fast-xml-parser";
import type { Source } from "./sources.js";

export interface Article {
  sourceId: string;
  sourceName: string;
  title: string;
  link: string;
  summary: string;
  publishedAt: string | null;
}

const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });

function asArray<T>(value: T | T[] | undefined): T[] {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function extractLink(raw: unknown): string {
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) {
    const alt = raw.find((l: any) => !l["@_rel"] || l["@_rel"] === "alternate");
    return alt?.["@_href"] ?? raw[0]?.["@_href"] ?? "";
  }
  if (raw && typeof raw === "object") {
    return (raw as any)["@_href"] ?? "";
  }
  return "";
}

function textOf(raw: unknown): string {
  if (typeof raw === "string") return raw;
  if (raw && typeof raw === "object" && "#text" in (raw as any)) {
    return String((raw as any)["#text"]);
  }
  return raw == null ? "" : String(raw);
}

function stripHtml(input: string): string {
  return input.replace(/<[^>]+>/g, "").trim();
}

export async function fetchSourceArticles(source: Source): Promise<Article[]> {
  const res = await fetch(source.feedUrl, {
    headers: {
      "User-Agent": "Mozilla/5.0 (compatible; FeroxNewsIE/0.1; +https://github.com/AxridAxe/Ferox)",
      Accept: "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
      "Accept-Language": "en-IE,en;q=0.9",
    },
  });
  if (!res.ok) {
    throw new Error(`${source.name} feed failed: ${res.status}`);
  }
  const xml = await res.text();
  const parsed = parser.parse(xml);

  const rssItems = asArray(parsed?.rss?.channel?.item);
  if (rssItems.length > 0) {
    return rssItems.map((item: any) => ({
      sourceId: source.id,
      sourceName: source.name,
      title: stripHtml(textOf(item.title)),
      link: extractLink(item.link),
      summary: stripHtml(textOf(item.description ?? item["content:encoded"])).slice(0, 400),
      publishedAt: item.pubDate ?? null,
    }));
  }

  const atomEntries = asArray(parsed?.feed?.entry);
  return atomEntries.map((entry: any) => ({
    sourceId: source.id,
    sourceName: source.name,
    title: stripHtml(textOf(entry.title)),
    link: extractLink(entry.link),
    summary: stripHtml(textOf(entry.summary ?? entry.content)).slice(0, 400),
    publishedAt: entry.updated ?? entry.published ?? null,
  }));
}

export async function fetchAllArticles(sources: Source[]): Promise<Article[]> {
  const results = await Promise.all(
    sources.map((s) =>
      fetchSourceArticles(s).catch((err) => {
        console.error(`Failed to fetch ${s.name}:`, err instanceof Error ? err.message : err);
        return [] as Article[];
      })
    )
  );
  return results.flat();
}
