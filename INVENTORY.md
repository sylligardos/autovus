# FF-VUS / VUS v2 — Directory & Experiment Inventory

_Last updated: 2026-06-23_

This inventory was reorganized around the **VUS v2 reframing** (ICDE, combined paper):
speed is the *enabler*, the contribution is **Adjusted VUS (aVUS)** via per-anomaly
adaptive buffers. Experiments now split into three groups: the **current** TSB-AD /
aVUS set, the **legacy** TSB-UAD set (pre-reframing, mostly needs re-running or
dropping), and **archived** dead ideas.

---

## Status table

Legend: **Done** = produced its intended result on the dataset we want for the paper.
"Want" judgements for legacy experiments are recommendations open to revision.

### Current paper (TSB-AD / Adjusted-VUS)

| Experiment | Done | Dataset | Purpose | Action needed |
|---|---|---|---|---|
| ranking_adapt | Yes | TSB-AD (772 series) | aVUS reranks detectors vs VUS; τ=0.74, 28% top-1 flips; Stock case study | None — full run complete (2026-06-04) |
| per_anomaly_buffer | Yes | TSB-AD (23 ds sweep) | Per-anomaly adaptive buffer selection (marginal-ratio argmax) + detection taxonomy | None — design final, validated (13/13 tests + sweep) |
| rule_ablation | Yes | TSB-AD subset | "argmax suffices" ablation (R1-R4 rules vs plain argmax) | None — summary.csv + plots |
| compare_threshold_cap | Yes | TSB-AD (21 ds) | Justifies pr_relevant_fuzzy capping; exact uncapped, 2-3x speedup at cap 1-2k | None — see SUMMARY.md |
| compare_threshold_methods | Yes | TSB-AD | Threshold-set strategy comparison (companion to _cap) | None |
| decumulated_vus | Yes | TSB-AD (NAB demo) | Figure: marginal (de-cumulated) vs cumulative skill ratio | None — figure asset |
| adaptive_buffer | Yes | TSB-AD | Precomputed adaptive-buffer pickles (K100, L128/L1024) | None — supporting data |
| eval_vus | Yes | TSB-AD | Equivalence: VUS==sklearn (7.9e-17); CPU/GPU gap (max 0.21) | None — correctness eval |
| large_buffer | Yes | TSB-AD (Daphnet) | Per-detector large-buffer visualizations | None — figure assets |
| showcase | Yes | TSB-AD (Daphnet) | Per-detector showcase plots | None — figure assets |

### Legacy (TSB-UAD, pre-reframing)

| Experiment | Done | Dataset | Purpose | Action needed |
|---|---|---|---|---|
| allmetrics_defparams_tsb | No | TSB-UAD | Main metric comparison table | **Re-run on TSB-AD** (WANT) |
| allmetrics_defparams_syn | Partial | synthetic | Same benchmark on synthetic | Recompute if final metric defs changed; else keep |
| allmetrics_defparams_syn_nomask | Partial | synthetic | global_mask=False ablation of the above | Keep as ablation; recompute with rest of syn |
| vus_buffer_comparison_tsb | No | TSB-UAD | Buffer-size sensitivity | Re-run on TSB-AD (WANT, background) |
| vus_step_comparison_tsb | No | TSB-UAD | Step-size sensitivity | Re-run on TSB-AD (WANT, background) |
| vus_ffvus_auc_0_tsb | No | TSB-UAD | buffer=0 degenerates to AUC | Re-run on TSB-AD (WANT, small) |
| vus_ffvus_auc_0_syn | No | synthetic | Same equivalence on synthetic | Empty dir — run once on synthetic |
| ablation_study | Partial | TSB-UAD/ECG | Optimization/speed ablation (timing) | Timing is machine-dependent; re-run if ECG in TSB-AD |
| existence_comparison | No | TSB-AD | Trivial vs optimized vs **GPU** existence (~1000x, zero diff) | **Re-run full benchmark incl. GPU existence** — current GPU run covers only 119/2040 series |
| existence_skip_analysis | Yes | small | How often existence skips compute | Re-run only if cited |
| softed_tsb | No | TSB-UAD/SVDB | SOFTED baseline metric | Re-run on TSB-AD only if SOFTED kept as baseline |
| cummulative_vus | Yes | TSB-UAD/YAHOO | Cumulative-VUS figures | Regenerate from TSB-AD series if used |
| compare_train_metrics | No | TSB-UAD/KDD21 | VUS-as-training-objective comparison | **Decide if still in scope** (likely cut under combined-paper framing) |
| compare_rank | No | TSB-UAD | VUS-vs-FF-VUS engine ranking agreement | **Likely superseded** by eval_vus / _cap torch-vs-numpy |
| compare_rank_ff | No | TSB-UAD | Same, FF-VUS / FF-VUS-GPU variants | **Likely superseded** (FF-VUS name dropped) |
| compare_rank_vus | No | TSB-UAD | Same, per-dataset VUS rankings | **Likely superseded** |
| figures | No | derived | Paper-ready PDF figures | Regenerate after TSB-AD re-runs |
| synthetic_data_generation | Yes | N/A | Generates syn_0..syn_9 | None |
| precomputed | Yes | N/A | rankings.npz / existence maps cache | None — supporting cache |

### Archived (`experiments/archive/`)

| Experiment | Reason |
|---|---|
| vus_inv_yahoo | Abandoned idea — inverted weight slope |
| vus_v2_yahoo | Abandoned idea — uniform weight slope |
| allmetrics_defparams_syn__ffvusgpuweird | One-off GPU debug run |
| preliminary | Superseded by allmetrics_defparams_tsb |
| to_delete_* (5 dated dirs) | Stale outputs from Sept-Oct 2025; safe to delete |

---

## Open decisions for the user

1. **compare_rank / compare_rank_ff / compare_rank_vus** — these compared the VUS and
   FF-VUS *engines* for ranking agreement. With the FF-VUS name dropped and CPU/GPU
   parity now shown by `eval_vus` and the torch-vs-numpy check in `compare_threshold_cap`,
   these look redundant. Archive them too, or keep one as an explicit "ranking unchanged
   by the fast engine" check?
2. **compare_train_metrics** — the "VUS as a differentiable training loss" angle. Keep it
   in the combined paper, or cut to keep the story focused on aVUS?
3. **Legacy sensitivity studies** (buffer / step / auc_0) — re-run on TSB-AD now, or only
   if a reviewer asks? They are cheap but background to the main aVUS story.

---

## Directory structure

```
ff-vus/
├── src/
│   ├── vus/                  # Core implementation: vus_numpy.py, vus_torch.py only
│   ├── legacy/               # Old metrics codebase, pre-dates this project — do not touch
│   ├── archive/              # Dead experiment code (vus_v2.py, compute_vus_inv.py, …)
│   ├── notebooks/            # Jupyter notebooks
│   ├── compute_ranking_adapt.py     # Stage 1: per-pair aVUS selection + scoring → CSV
│   ├── analyze_ranking_adapt.py     # Stage 2: τ/Spearman vs VUS, flip rate, tables/figures
│   ├── compare_threshold_cap.py     # Threshold-cap experiment (pr_relevant_fuzzy vs subsampled)
│   ├── evaluate_ffvus.py     # Equivalence eval (VUS vs sklearn, CPU vs GPU)
│   ├── compute_metric.py     # Bulk metric computation over a dataset
│   ├── vus_visualizer.py     # Paper figure generation (--plot <name>)
│   ├── precompute_existence.py / precompute_rankings.py
│   └── demo_vus.py           # Streamlit interactive demo
├── scripts/                  # CLEPS launch/check + per-experiment shell scripts
│   ├── launch.py / check.py  # Submit to CLEPS / poll status + rsync results back
│   └── create_shell_scripts.py, templates.py
├── experiments/              # Experiment outputs (see status table above)
│   └── archive/              # Abandoned ideas + dated stale dirs (was to_delete/)
├── data/                     # Empty — data lives in tsbadparser repo
├── references/ffvus-paper    # Gitignored symlink to the paper repo (PAPER_PLAN.md)
├── environment.yml / requirements.txt
└── INVENTORY.md / TODO.md
```

Data now comes from **tsbadparser** (`TSBADParser`), not the deleted `src/utils/`.
TSB-AD has 23 datasets; complete-case detector coverage gives ~772-870 series depending
on the experiment's drop policy.

---

## Per-experiment detail (current paper)

### ranking_adapt
**What:** The headline aVUS experiment. For each (dataset, series, detector), pick a
per-anomaly buffer b* (marginal-ratio argmax) then score with `score_adapt`; compare the
resulting detector ranking against baseline VUS. `vus_full` reproduces standard VUS-PR.
**Result:** mean Kendall-τ vs VUS = 0.737, top-1 detector flips in 28.1% of 772 series;
Stock saturation case study (flat VUS wall vs spread aVUS). Full run complete 2026-06-04.
**Scripts:** `compute_ranking_adapt.py` (Stage 1), `analyze_ranking_adapt.py` (Stage 2).

### per_anomaly_buffer
**What:** `compute_per_anomaly_adaptive_slicing` in vus_numpy.py — slices each anomaly,
runs cumulative VUS, picks b* = argmax of the de-cumulated (marginal) skill ratio.
Per-side selection (b*_left / b*_right) drives a 4-state detection taxonomy
(early/late/exact/missed). Validated by 13/13 unit tests + a 23-dataset / ~202k-anomaly
sweep (~56% b*=0, rest sensible).

### rule_ablation
**What:** Shows the guard rules (R1-R4) are not needed: plain argmax matches the full
rule set. Output `summary.csv` (per-anomaly b* under each rule subset) + plots.

### compare_threshold_cap
**What:** Justifies the pr_relevant_fuzzy threshold-capping trick. Uncapped VUS matches
sklearn to machine precision; pr_relevant_fuzzy dominates uniform subsampling on
worst-case error; cap ~1000-2000 gives near-exact VUS at 2-3x speedup. See
`experiments/compare_threshold_cap/SUMMARY.md`. 21 datasets / 37 series / 1059 pairs.

### compare_threshold_methods
**What:** Companion threshold-set strategy comparison (single `results.csv`).

### decumulated_vus
**What:** Figure asset showing the marginal (de-cumulated) ratio peaks at the true
detection extent while the cumulative ratio overshoots ~2x (NAB Facility series).

### adaptive_buffer
**What:** Precomputed adaptive-buffer pickles per dataset (K100, L128 and L1024).

### eval_vus
**What:** `evaluate_ffvus.py` output. VUS vs sklearn APS mean |err| ~8e-17 (exact);
CPU vs GPU mean ~2e-3, max ~0.21 (the existence-matrix divergence). Correctness eval.

### large_buffer / showcase
**What:** Per-detector PNG visualizations on Daphnet (large-buffer behavior / detector
showcase). Figure assets, no quantitative results.

---

## Per-experiment detail (legacy — TSB-UAD)

Descriptions retained from the pre-reframing inventory; see the status table for the
current re-run / drop decision. All "tsb" experiments ran on **TSB-UAD** via the deleted
`src/utils/dataloader.py`.

- **allmetrics_defparams_tsb** — full metric benchmark (FF-VUS, VUS, AUC, RANGE-AUC,
  AFFILIATION, RF) at default params (buffer=128, step=1, optimized existence,
  dynamic_plus, global_mask) on TSB-UAD (1887 series). The main comparison table; re-run
  on TSB-AD.
- **allmetrics_defparams_syn / _nomask** — same benchmark on synthetic (syn_0..9); the
  `_nomask` variant is the global_mask=False ablation.
- **vus_buffer_comparison_tsb** — buffer ∈ {0,2,…,512} sensitivity.
- **vus_step_comparison_tsb** — step ∈ {1,…,512} sensitivity at buffer=512.
- **vus_ffvus_auc_0_tsb / _syn** — buffer=0 makes FF-VUS degenerate to AUC.
- **compare_rank / compare_rank_ff / compare_rank_vus** — per-dataset ranking agreement
  between the VUS and FF-VUS engines across TSB-UAD sub-datasets.
- **compare_train_metrics** — AUC/VUS/FF-VUS as MLP training objectives on KDD21.
- **ablation_study** — baseline → +precomputed slopes → +optimized existence →
  +dynamic_plus, timing + value (machine-dependent timing).
- **softed_tsb** — SOFTED (R implementation) on SVDB.
- **existence_comparison** — trivial vs optimized vs GPU existence: ~1000x speedup, zero
  diff. `tsb_all_128.csv` is the full numpy run (2040 series, paper figure);
  `existence_comparison_results.csv` adds the GPU benchmark but only 119 series.
  **TODO: re-run `compare_existence_reward.py` over the full TSB-AD benchmark with GPU
  existence** so the GPU column matches the numpy coverage.
- **existence_skip_analysis** — existence skip-rate stats.
- **cummulative_vus** — cumulative-VUS figures on YAHOO.
- **figures** — PDF figures from `vus_visualizer.py`; regenerate after TSB-AD re-runs.
- **synthetic_data_generation** — generates syn_0..9.
- **precomputed** — rankings / existence-map caches.
