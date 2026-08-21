/**
 * Ownership is stated as public-record fact, not opinion, and each entry
 * cites where it came from. No political-bias labels here on purpose —
 * that would need a citable third-party methodology (the way Ground News
 * relies on AllSides/Ad Fontes/MBFC), and no Irish equivalent was found
 * yet. See Media Ownership Monitor Ireland (https://ireland.mom-gmr.org)
 * for the closest existing independent research on this.
 */
export interface OwnershipGroup {
  id: string;
  name: string;
  type: "private-foreign" | "private-domestic" | "public-trust" | "state";
  note: string;
  source: string;
}

export const ownershipGroups: Record<string, OwnershipGroup> = {
  mediahuis: {
    id: "mediahuis",
    name: "Mediahuis Ireland",
    type: "private-foreign",
    note: "Belgian media group; bought Independent News & Media's Irish titles from Denis O'Brien and Dermot Desmond in 2019.",
    source: "https://en.wikipedia.org/wiki/Mediahuis_Ireland",
  },
  irishTimesGroup: {
    id: "irishTimesGroup",
    name: "The Irish Times Group",
    type: "public-trust",
    note: "Operates as The Irish Times Designated Activity Company, an editorially independent trust structure rather than a conventional shareholder-owned publisher. Added the Irish Examiner and The Echo in 2018.",
    source: "https://en.wikipedia.org/wiki/The_Irish_Times",
  },
  journalMedia: {
    id: "journalMedia",
    name: "Journal Media Ltd",
    type: "private-domestic",
    note: "Independent Irish-owned digital publisher.",
    source: "https://en.wikipedia.org/wiki/TheJournal.ie",
  },
  rte: {
    id: "rte",
    name: "RTÉ",
    type: "state",
    note: "Ireland's statutory public-service broadcaster, funded via TV licence fee and state grants.",
    source: "https://en.wikipedia.org/wiki/RT%C3%89",
  },
};
