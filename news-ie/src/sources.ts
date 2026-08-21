import { ownershipGroups } from "./ownership.js";

export interface Source {
  id: string;
  name: string;
  feedUrl: string;
  ownership: typeof ownershipGroups[keyof typeof ownershipGroups];
}

/** Each feedUrl was verified reachable and parseable before being added here. */
export const sources: Source[] = [
  {
    id: "irish-times",
    name: "The Irish Times",
    feedUrl: "https://www.irishtimes.com/cmlink/the-irish-times-news-1.1319192",
    ownership: ownershipGroups.irishTimesGroup,
  },
  {
    id: "irish-examiner",
    name: "Irish Examiner",
    feedUrl: "https://www.irishexaminer.com/feed/35-top_news.xml",
    ownership: ownershipGroups.irishTimesGroup,
  },
  {
    id: "irish-independent",
    name: "Irish Independent",
    feedUrl: "https://www.independent.ie/rss/",
    ownership: ownershipGroups.mediahuis,
  },
  {
    id: "the-journal",
    name: "TheJournal.ie",
    feedUrl: "https://www.thejournal.ie/feed/",
    ownership: ownershipGroups.journalMedia,
  },
  {
    id: "rte-news",
    name: "RTÉ News",
    feedUrl: "https://www.rte.ie/feeds/rss/?index=/news",
    ownership: ownershipGroups.rte,
  },
  {
    id: "irish-sun",
    name: "The Irish Sun",
    feedUrl: "https://www.thesun.ie/feed/",
    ownership: ownershipGroups.newsUk,
  },
  {
    id: "belfast-telegraph",
    name: "Belfast Telegraph",
    feedUrl: "https://www.belfasttelegraph.co.uk/rss/",
    ownership: ownershipGroups.mediahuis,
  },
  {
    id: "irish-news",
    name: "The Irish News",
    feedUrl: "https://www.irishnews.com/arc/outboundfeeds/rss/",
    ownership: ownershipGroups.fitzpatrickFamily,
  },
];
