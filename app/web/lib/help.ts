// The "?" popups: how each page works. Mechanisms only, never findings.
// One entry is a term and one or two sentences.

export type HelpEntry = { term: string; text: string };
export type HelpSection = { title: string; entries: HelpEntry[] };

export const HELP: Record<string, HelpSection[]> = {
  "/": [{
    title: "The demo",
    entries: [
      { term: "VUS", text: "The volume under the PR or ROC surface over every buffer from 0 to L around each labelled anomaly." },
      { term: "FastVUS", text: "The engine: the same volume for all buffers at once. Exact on the CPU; on the GPU every term is exact except the existence term." },
      { term: "AutoVUS", text: "The metric: for every anomaly, one buffer on each side, where the detector's gain over a random score peaks (the paper rule). The score is chance-adjusted." },
      { term: "Adjusted VUS", text: "VUS-PR with the fixed buffer L on every anomaly, chance-adjusted: the score minus what a random score gets under the same buffered labels, over the gap to 1. No per-anomaly selection." },
    ],
  }],
  "/series": [{
    title: "The series page",
    entries: [
      { term: "A lane", text: "One detector's score on this series, min-max scaled, time aligned with the series above. The yellow band is the labelled anomaly." },
      { term: "A shaded zone", text: "The buffer AutoVUS selected before (blue) and after (red) that anomaly. The faint band is the range up to L." },
      { term: "A label", text: "From the selected buffer: wide = a buffer on both sides, exact = no buffer and the detector beats random inside the anomaly, missed = no buffer and it does not." },
      { term: "Top, by", text: "The detectors shown: the best ones on this series under the chosen metric, plus the pinned ones." },
      { term: "The drawer", text: "For the clicked anomaly: the detector curve and the random curve over the buffer size, and the rule curve whose first argmax is the selected buffer." },
      { term: "L", text: "The largest buffer considered on each side: 20 or 128, the switch in the top bar." },
    ],
  }],
  "/cases": [{
    title: "Cases",
    entries: [
      { term: "Global vs local: a gap", text: "AutoVUS Global minus the mean (or the lowest) AutoVUS Local over the anomalies of one series and detector." },
      { term: "Global vs local: a row", text: "A dataset row is the mean gap over its series and the chosen detectors. A series row is the largest gap over the chosen detectors, with its detector; a click opens the series in Explore with that detector pinned." },
      { term: "Global vs local: the plot", text: "One point per series and detector, the local score against the global one. A point on the diagonal has equal scores; the colour grows with the gap, blue below the diagonal and orange above. A click opens the series with that detector pinned; hovering a row of either table marks its points." },
      { term: "Global vs local: one anomaly", text: "Series with one anomaly are left out of the plot and of both tables: their local and global scores are equal, so they add nothing." },
    ],
  }],
  "/rankings": [{
    title: "The rankings page",
    entries: [
      { term: "The pair", text: "Two metrics, each ranking every detector by its mean (or median) over the scope. The curves between them are the rank moves." },
      { term: "A box", text: "The values of one detector under one metric: the box holds the middle half, the line is the median, the dot the mean, and the whiskers reach the furthest values within 1.5 box lengths." },
      { term: "AutoVUS Local", text: "One value per series: the mean of the AutoVUS Local scores of its anomalies." },
      { term: "AutoVUS Local per anomaly", text: "One value per anomaly, so a series counts as many times as it has anomalies. The reference detectors have no local scores." },
      { term: "L", text: "VUS-PR, Adjusted VUS and AutoVUS with buffers up to L on each side: 20 or 128, the switch in the top bar." },
      { term: "Oracle", text: "On each series, the detector with the best AUC-PR, with all its values." },
      { term: "Stitched score", text: "On each series, the stitched score of the Explore page: each anomaly's part takes the score of its best-coverage winner, shifted so its median is 0, clipped below 0, then divided by its max." },
      { term: "Weighted tau", text: "Kendall's tau with hyperbolic weights (Vigna 2015): a disagreement near the top of the rankings counts more." },
    ],
  }],
  "/local": [{
    title: "Explore",
    entries: [
      { term: "Your data", text: "One CSV file with a header row. The first column is the series, the second the label (0 or 1), and every further column is the score of one detector, named by the header. At most 4 MB. It is scored live at the L of the top bar and is not kept." },
      { term: "AutoVUS Local", text: "AutoVUS of one anomaly: its own buffers, and a fair share (one over the number of anomalies) of the series' false alarms. Chance-adjusted, so a random score sits near 0." },
      { term: "Scores", text: "One lane per detector, time aligned with the series; the number above each anomaly is the detector's AutoVUS Local there." },
      { term: "Heat map", text: "The same numbers as cells, one column per anomaly in view. With more than 20 anomalies in view, each row becomes one colour band along time." },
      { term: "Columns", text: "AUC: AUC-PR. VUS: VUS-PR at L. AutoVUS Global: AutoVUS on the whole series. AutoVUS Local: the mean of the per-anomaly values. Both at L, the switch in the top bar. A click on a column orders the detectors by it." },
      { term: "Best coverage", text: "The detectors that are best on at least one anomaly; a tie goes to the detector that is best on the most anomalies." },
      { term: "Stitched score", text: "One score built from the best-coverage winners: the series is cut halfway between anomalies, and each part takes the score of the detector that wins its anomaly, shifted so its median is 0, clipped below 0, then divided by its max. All metrics are then recomputed on it." },
      { term: "Zoom", text: "Drag across the series or a lane to zoom in, Back to step out, double click for the whole series." },
      { term: "Why this buffer", text: "A click on a number opens, for that anomaly and detector, the score with the selected buffers and the ratio curve of each side; the first argmax of a curve is that side's buffer." },
    ],
  }],
  "/profiles": [{
    title: "The profiles page",
    entries: [
      { term: "Stitched vs best detector", text: "One gap per series: AutoVUS of the stitched score minus the highest AutoVUS Global among the chosen detectors. Largest first. Series with one anomaly are left out. The stitched score always uses all detectors." },
      { term: "A point", text: "One labelled anomaly, described by its local AutoVUS under every detector." },
      { term: "A link", text: "Each anomaly is linked to the 10 anomalies whose scores are closest to its own (Euclidean distance)." },
      { term: "A profile", text: "A Louvain community of the linked anomalies, lettered by size." },
      { term: "Stability", text: "How much the profiles depend on chance: Louvain is run with 10 seeds and the mean pairwise Adjusted Rand Index of the 10 groupings is shown as a percentage. 100% means every seed gives the same profiles." },
      { term: "The map", text: "The points sit where a force layout of the links puts them: linked anomalies pull together." },
      { term: "Best per anomaly", text: "For a set of detectors: on every anomaly the best local AutoVUS any detector of the set reaches, averaged over the anomalies of the dataset. Each row shows the set up to that detector; the last row is every detector together." },
      { term: "The three detectors", text: "The best detector alone, then twice the detector that raises the best-per-anomaly score the most. Seconds are the mean time a detector takes to score one series of the dataset." },
      { term: "Similar and faster", text: "Among the detectors that take less time, the one whose local AutoVUS over the anomalies of the dataset correlates most with this detector's (Pearson). The correlation says how similar it is." },
      { term: "A click on a detector", text: "Colours the map by that detector: an anomaly takes its local AutoVUS, a series the mean over its anomalies. A second click returns to the profiles." },
      { term: "Series", text: "The points become the series. A point is painted with the profiles of its anomalies, one slice per profile, and grows with the number of anomalies. A series is the share of its anomalies in each profile; each series is linked to the 10 series with the closest shares." },
      { term: "The bars of a profile", text: "For every detector, its mean local AutoVUS over the anomalies of the profile." },
    ],
  }],
  "/coverage": [{
    title: "The coverage page",
    entries: [
      { term: "Detected", text: "A detector detects an anomaly when its AutoVUS Local on it is at least the threshold on the horizontal axis." },
      { term: "The three bands", text: "At each threshold, every series is in one band: one detector detects all its anomalies; no single detector does, but every anomaly is detected by some detector; or some anomaly is detected by none." },
      { term: "The series", text: "Series with at least two anomalies. With one anomaly, one detector is always enough." },
      { term: "A point", text: "One series. Across: the highest threshold at which one detector still detects every anomaly. Up: the highest threshold at which the detectors together still do. On the diagonal the series never needs several detectors; the height above it is the range of thresholds where it does. Series at the same position are spread around it." },
      { term: "A click on a point", text: "Opens the series with two detectors pinned: the best one alone, and the one that is best on that detector's lowest anomaly." },
    ],
  }],
  "/coverage-old": [{
    title: "The coverage page",
    entries: [
      { term: "Coverage", text: "For a set of detectors, each anomaly takes the best local AutoVUS in the set; coverage is the mean over the anomalies. It takes the best detector of every anomaly as known." },
      { term: "Best coverage", text: "The set of detectors that are best on at least one anomaly. A tie goes to the detector that is best on the most anomalies." },
      { term: "Fewer detectors", text: "The best set of each size k: exact search up to k = 4, then one detector added at a time." },
      { term: "A dataset", text: "Means over its series with at least two anomalies." },
      { term: "Found", text: "A detector finds an anomaly when its local AutoVUS on it is at least the threshold of the slider." },
      { term: "Minimum set", text: "The fewest detectors such that every anomaly found by some detector is found by one of the set. Exact search; among the sets of that size, the one with the highest coverage. Over all series, also those with one anomaly." },
    ],
  }],
  "/buffers": [{
    title: "The buffers page",
    entries: [
      { term: "A detection type", text: "From the buffers AutoVUS selects on each side of an anomaly: early = only before, late = only after, wide = both, exact = none and the detector beats random inside, missed = none and it does not." },
      { term: "A cell", text: "The share of that dataset's (or series') anomalies whose type, for that detector, is the selected one." },
      { term: "L", text: "The largest buffer considered on each side." },
    ],
  }],
  "/suspicious": [{
    title: "The suspicious page",
    entries: [
      { term: "Suspicious", text: "An anomaly that most detectors rank among their worst in the series while they catch the others: a mean-rank test against chance (q below 0.05 after Benjamini-Hochberg) and a median local AutoVUS at least 0.2 below the other anomalies." },
      { term: "Worst for", text: "How many of the informative detectors (scores that vary over the anomalies) score it lowest." },
      { term: "Median detector", text: "The median of all detectors' scaled scores around the anomaly: where they fire compared with the label." },
    ],
  }],
  "/play": [{
    title: "Find the buffer",
    entries: [
      { term: "A round", text: "One anomaly and one detector. Drag the blue and red edges to give the detection the buffer you think it deserves on each side, then lock in. The number above an edge is the gain of that buffer." },
      { term: "AutoVUS", text: "Selects both buffers itself, per side, up to L (the switch in the top bar): the first peak of the ratio curve of each side." },
      { term: "Why this buffer", text: "The ratio curve of each side with AutoVUS's pick (filled dot) and yours (hollow dot), and the AutoVUS value of the whole series with each pair of buffers on this anomaly." },
      { term: "The result", text: "How often your verdict (before, after, on, both sides) was AutoVUS's, and how close your buffers came to the peak gain, averaged over the rounds and the two sides." },
    ],
  }, {
    title: "Draw your detection",
    entries: [
      { term: "The lane", text: "A window of a real series with its labels. Draw a detector score with the pointer; along the bottom erases. Every stroke is scored live." },
      { term: "Above each label", text: "AutoVUS's verdict for that label and its two buffers, before and after." },
      { term: "Played so far", text: "Every game and every scored drawing, under the name given on the Play page, latest first. No ranking: a line says what was played and what AutoVUS answered. A game left early counts the rounds played." },
    ],
  }],
  "/speed": [{
    title: "The speed page",
    entries: [
      { term: "A total", text: "The time to score every detector on every series of the benchmark with that metric and engine, on one machine." },
      { term: "A point", text: "One series: its length and the median time over the detectors." },
      { term: "Your data", text: "A label and a score you upload, timed live on this machine. Nothing leaves it." },
    ],
  }],
};
