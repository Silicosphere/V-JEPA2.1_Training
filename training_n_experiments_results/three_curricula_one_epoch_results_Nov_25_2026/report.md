# V-JEPA 2.1 pretraining on Ego4D under three data-order curricula

**Question.** Does the order in which a self-supervised video model sees its training data change how fast it learns? Three V-JEPA 2.1 ViT-B/16 models were pretrained from scratch on the same 207,562 Ego4D clips. Each run made exactly one pass over the clips, and the runs differed only in the order of the clips:

| Curriculum | Order |
|---|---|
| `top` | most surprising clips first |
| `low` | least surprising clips first |
| `random` | a fixed random permutation (the control) |

"Surprise" is the prediction error a pretrained V-JEPA 2.1 model makes on the clip (Section 1.2).

**Short answer.** The held-out prediction loss of `random` is lowest for almost the entire pass. `top` is close to `random` until about 100k clips, then worsens. `low` is clearly worst until the end. Section 7 lists the reasons this is not yet a conclusion about which curriculum learns better features. The main ones:

- one seed per curriculum;
- a loss that is measured against each run's own moving target;
- hyperparameters tuned on the random order;
- two GPU types.

Figures: `figures/curricula_loss.png` and `figures/curricula_rankme.png`. Section 6 defines every quantity they show.

### Contents of this folder

The folder is self-contained. Appendix A explains how to rebuild the whole experiment from it, even if the `curriculum-prep` branch and the run folders on `/data` no longer exist.

| Path | What it is |
|---|---|
| `report.md` | this document |
| `figures/curricula_loss.png`, `figures/curricula_rankme.png` | the two figures (Section 6) |
| `code/curriculum-prep.patch` | **every code change of the experiment**: the 13 commits of branch `curriculum-prep`, in `git format-patch` form, on top of `main` commit `1b60c66`. Verified: `git am` on `1b60c66` reproduces the branch tip `d263b9c` exactly (identical tree `74810996…`). |
| `code/make_splits.py.from-Ego-Curation-3f7095b` | copy of the script that produced the original curriculum lists (Section 1.3) |
| `configs/base_vitb16-pretrain-256px-16f.yaml` | the experiment config (Section 4.3) |
| `configs/run_{top,low,random}.yaml` | the exact resolved config each run trained with |
| `data/removed_clips.csv.gz` | the 9,531 clips removed from the original lists, each with its reason (`black`, `dim`, `heldout`). The original lists minus these give the training lists exactly (verified). |
| `data/heldout_eval.csv.gz` | the 1,000 evaluation clips, in evaluation order |
| `data/heldout_videos.txt` | the 30 held-out video ids |
| `data/summary.json` | counts and arguments of the filtering / held-out split |
| `results/{top,low,random}/eval_r0.csv` | held-out evaluations of each run (the data behind the figures) |
| `results/{top,low,random}/log_r0.csv.gz` | per-step training log of each run: step, CSV rows of the batch, losses, λ, LR, timings |
| `results/sweep/bs<B>-lr<LR>/` | config and held-out evaluations of each of the 12 sweep runs (Section 4.2) |

Not in this folder, because of their size:

- the model weights (1.7 GB per run, at `/data/asem.a.abdelaziz/gp-2027a/runs/vitb16-ego4d-*/latest.pth.tar`);
- the video files;
- the full clip-scan table (`/data/asem.a.abdelaziz/gp-2027a/ego4d_lists_filtered/clip_scan.csv`).

The scan's conclusions are recorded in Section 3, and its decisions are fully captured by `data/removed_clips.csv.gz`.

---

## 1. Inputs

### 1.1 Videos

- **Storage:** six HDF5 files, `/data/asem.a.abdelaziz/gp-2027a/ego4d_clips/clips{1..6}.h5`, 367 GB in total (56.7–74.0 GB each).
- **Format:** each dataset in a file is one clip, stored as the raw bytes of an MP4 file. The dataset key is `<video_uid>/<video_uid>_seg<rank:03d>_<start_sec:07.2f>s`.
- **Count:** 217,093 clips cut from 1,203 Ego4D v2 videos (`ego4d_data/v2/full_scale`, folds 1–6).
- **Clips per video:** median 151, range 2–596.
- **Decoding:** every clip decoded to exactly 180 frames at 30 fps (6.0 s). Frame heights are 540–1,080 px and widths 540–1,440 px.

### 1.2 Surprise scores

- **Source:** Ego-Curation scoring run 1184 (`Ego-Curation/runs_results/run_1184/segments{1..6}/*_segments.csv`).
- **Columns:** `rank, video, start_sec, end_sec, surprise, copy_error`.
- **Scoring model:** the released V-JEPA 2.1 *giant* checkpoint (`vjepa2_1_vitg_384`).
- **How a window is scored:**
  1. Each 6 s window is sampled at 4 fps and split into a 4 s context and the following 2 s target.
  2. The predictor predicts the target's hierarchical, layer-normalised embeddings from the context alone.
  3. `surprise` is the error between those predicted embeddings and the embeddings the encoder produces from the full window.
- **Range over the 217,093 clips:** 0.204–0.482, median 0.346.

### 1.3 Curriculum lists

- **Files:** `/data/asem.a.abdelaziz/gp-2027a/ego4d_lists/ego4d_{top,low,random}.csv`. Each holds all 217,093 clips, one per line, as `<h5 path>::<key> 0`. This is the space-separated V-JEPA list format; the trailing `0` is an unused label.
- **Generated by:** Ego-Curation `scripts/train/make_splits.py` at commit `3f7095b`, run as:

  ```
  python scripts/train/make_splits.py runs_results/run_1184 \
      --clips-dir /data/asem.a.abdelaziz/gp-2027a/ego4d_clips --fraction 1 --global --seed 0
  ```

  Re-running this command on 2026-11-25 reproduced all three files byte for byte.
- **`top`:** sorted by surprise, descending, with a stable sort.
- **`low`:** sorted by surprise, ascending, with a stable sort.
- **`random`:** `DataFrame.sample(frac=1, random_state=0)`.

Because the sort is stable, tied scores keep their original relative order in *both* `top` and `low`. So `low` is not exactly `top` reversed: after filtering (Section 3), 19,708 of 207,562 positions differ from a reversal.

---

## 2. Code and environment

### 2.1 Code

- **Repository:** `V-JEPA2.1_Training`, branch `curriculum-prep`. It was branched from `main` at `1b60c66` and lives in the worktree `/data/asem.a.abdelaziz/gp-2027a/V-JEPA2.1_Training-curriculum-prep`.
- **Commits:** the three training runs used commit `70b272b`. The plots come from `scripts/analysis/plot_curricula.py` at `d263b9c`.
- **Push status:** the branch has not been pushed. **All of its changes are preserved in `code/curriculum-prep.patch`** (Appendix A).

Commits on the branch, in order (hashes as on the original branch):

| Commit | Summary |
|---|---|
| `08b09b5` | Fix `compute_mask_distance` at batch size 1 and vectorize it |
| `d067996` | Train curricula in strict CSV order with exact resume |
| `abe922d` | Add Ego4D curriculum configs and pretraining benchmark scripts |
| `80de021` | Add held-out evaluation and clip-based schedules to V-JEPA 2.1 training |
| `66b051b` | Add a clip scanner to validate all curriculum clips before training |
| `64503cc` | Let the clip scanner split the list over jobs and resume after a timeout |
| `29bb7a2` | Add a script that filters degenerate clips and carves out a whole-video held-out set |
| `9829918` | Train on the filtered Ego4D lists and evaluate on the held-out videos |
| `b4bb3ab` | Add an LR × batch-size sweep on the first 40k clips of the random list |
| `96c7513` | Checkpoint sweep runs every 8,000 clips so batch 32 divides an epoch |
| `70b272b` | Set the ViT-B curriculum config from the sweep and add the run script (**the code the three runs used**) |
| `df6f454`, `d263b9c` | Plotting script for the two figures |

Changes made to the upstream training code, and why:

| Change | Reason |
|---|---|
| **Strict order** (`data.shuffle: false`). A `ResumableSequentialSampler` yields rows 0, 1, 2, … of the CSV. On resume it restarts exactly at row `epochs_done × ipe × batch`. | Data order is the experimental variable; it must not be shuffled, replayed or skipped. |
| **No substitution of failed clips.** In strict mode a clip that fails to load is retried 3 times; then training stops with an error. Upstream would silently load a random other clip instead. | Substitution would change the order. Section 3 shows that every clip decodes. |
| **Per-clip random numbers** (`data.per_clip_rng: true`). Before a clip is loaded, Python, NumPy and PyTorch are seeded with `SHA-256(seed, "clip", clip path)`. | The temporal window, crop and flip of a clip are then identical in every curriculum and independent of loader worker count, so order is the only difference. |
| **Per-step masks.** The masks of a batch are drawn from a generator seeded with `SHA-256(seed, "mask", first row index of the batch, frames per clip, mask-generator index)`. | All three runs use identical masks at a given step; only the clips under them differ. Tests confirmed that pixels per clip and masks per step are identical between a list and its reversal, and independent of worker count. |
| **`compute_mask_distance` fixed and vectorised.** | The original crashed at batch size 1 (a `.squeeze()` removed the batch dimension). The vectorised version matches the original to within 2.4 × 10⁻⁷. |
| **Held-out evaluation** (`eval.*` config keys). | Defined in Section 6. |
| **Schedules expressed in clips seen** (`optimization.warmup_clips`, `optimization.ema_horizon_clips`, `model.lambda_start_clips` / `lambda_end_clips`), plus `optimization.epochs: auto` (enough epochs for one pass). | A schedule then means the same thing at every batch size, which the batch-size sweep required. |
| **Per-step log** `log_r0.csv`. Columns: step, `first_row`, `last_row` (the CSV rows in the batch), `loss_pred`, `loss_ctx`, λ, learning rate, timings. | Every logged loss is tied to the exact clips that produced it. |

### 2.2 Environment

- **Cluster:** SBME RCS Slurm cluster.
  - At most 2 jobs per user, queued and running combined.
  - 1 GbE between nodes, no InfiniBand.
  - The data is on an NFS-mounted `/data`.
- **Python:** 3.12.3 in the venv `/data/asem.a.abdelaziz/gp-2027a/myenv`.
- **Libraries:** torch 2.13.0+cu130 (CUDA 13.0, cuDNN 9.20), decord 0.6.0, h5py 3.16.0, numpy 2.5.2, timm 1.0.30.
- **Module:** FFmpeg/7.0.2-GCCcore-13.3.0.
- **Allocator:** runs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 3. Data processing

### 3.1 Scanning every clip

Before training, every one of the 217,093 clips was decoded once (`scripts/data/scan_clips.py`):

- 8 evenly spaced frames per clip;
- run as CPU-only jobs.

For each clip the scan recorded:

| Statistic | Definition |
|---|---|
| decode status | whether the clip decoded |
| frame count, fps, resolution | as reported by the decoder |
| `mean` | pixel mean over the 8 frames (0–255 scale) |
| `std` | pixel standard deviation over the 8 frames |
| `motion` | mean absolute difference between consecutive sampled frames |

**Results:**

- **Decoding:** all 217,093 clips decode, all with 180 frames at 30 fps. So the strict-order "stop on failure" rule never triggers.
- **Fully black clips: 7,170** (`std = 0`) from 53 videos. Every one carries one of five *tied* surprise values, 0.356337–0.356458: the scorer gives degenerate black input an almost constant score. Because of the tie they formed contiguous blocks:
  - `top`, rows 63,441–71,077 (29–33 % of the list): 94 % black;
  - `low`, rows 146,015–153,651 (67–71 %).

  At batch 16 that would have been about 477 consecutive steps of nearly all-black batches.
- **Near-black clips: 151** (`0 < std ≤ 5`). Sampled frames showed them to be essentially unusable. Their median position is 99.7 % of the way down `top`, so they sit at the start of `low`.
- **Dark but real clips: 1,768** (`5 < std ≤ 20`): night streets, dim rooms. Kept.
- **Static clips: 129** (`motion ≤ 0.5`, `std > 5`): a camera lying still, a static street. Real footage. Kept.

### 3.2 Filtering

All clips with `std ≤ 5` (7,321) were removed from all three lists, and the order of the remaining clips was preserved.

Dark and static clips were deliberately kept. They are concentrated at the low-surprise end, so removing them would have changed the curriculum itself rather than just cleaned the data.

### 3.3 Held-out set (whole videos)

The held-out clips come from videos that contribute *no* clips to training, so the evaluation measures generalisation to unseen videos rather than to unseen moments of seen videos. Built by `scripts/data/make_curriculum_split.py` (seed 0):

1. **Eligible videos:** those with no removed clip and with 30–120 clips.
2. **Sorting:** eligible videos were sorted by the median position of their clips in `top`.
3. **Strata:** that sorted list was split into 30 strata of equal video count.
4. **Selection:** one video was drawn at random from each stratum, giving 30 videos and 2,210 clips. All of them were removed from all three training lists.
5. **Evaluation set:** the 2,210 held-out clips were sorted by their `top` position, and 1,000 evenly spaced ones were taken (`numpy.linspace` indices).

The evaluation clips span the whole surprise ranking. Their positions in `top` have quantiles (0, 25, 50, 75, 100 %) of (0.00, 0.24, 0.51, 0.72, 1.00) of the list length.

Selecting whole videos costs little diversity because the clips of one video spread over most of the ranking: the median within-video standard deviation of rank fraction is 0.225, against 0.289 for a uniform spread.

### 3.4 Result

| | Clips |
|---|---|
| Original lists | 217,093 |
| Removed: black (`std = 0`) | 7,170 |
| Removed: near-black (`0 < std ≤ 5`) | 151 |
| Removed: held-out videos (30) | 2,210 |
| **Training lists (each)** | **207,562** |
| Evaluation set (from the held-out videos) | 1,000 |

Files (in `/data/asem.a.abdelaziz/gp-2027a/ego4d_lists_filtered/`):

| File | Contents |
|---|---|
| `ego4d_{top,low,random}.csv` | the three filtered training lists |
| `heldout_eval.csv` | the 1,000 evaluation clips |
| `heldout_videos.txt` | the 30 held-out video ids |
| `removed_clips.csv` | every removed clip, with its reason |
| `clip_scan.csv` | the full scan results |
| `summary.json` | counts and the selection arguments |

Checks on the result:

- The three lists hold the same set of clips.
- Each list is a subsequence of its original, i.e. the order is preserved.
- No held-out video appears in training.
- No training clip has `std ≤ 5`.
- In the filtered lists the longest run of consecutive clips from one video is 5 (`top`, `low`) or 3 (`random`). No batch of 16 comes from a single video.

---

## 4. Model and hyperparameters

### 4.1 Model size and single pass

- **Single pass.** Each model sees each clip exactly once, so "when" a clip is seen is well defined.
- **No multi-node training.** It was ruled out by the 1 GbE network and the 2-job limit.
- **Benchmarks** (RTX 5070 Ti, 16 GB):
  - ViT-L/16 trains at 7.3 clips/s at batch 8, about 8.3 h per pass, and runs out of memory at batch 16.
  - ViT-B/16 trains at 19.5 clips/s at batch 16 (7.6 GB), about 3 h per pass.
- **Choice:** ViT-B/16.

### 4.2 Batch size and learning-rate sweep

**Setup:**

- **Data:** the first 40,000 clips of the filtered `random` list.
- **Evaluation:** the same 1,000 held-out clips, every 10,000 clips.
- **Schedules:** the ones in Section 4.3. Because they are defined in clips, they mean the same at every batch size.
- **Learning-rate grid:** at each batch size, a grid scaled by √(batch / 16) around 1e-4 at batch 16, with one step of about 3× on either side. Upstream's 6e-4 was set for a global batch of 3,072.

**Results after 40,000 clips:**

| Batch | LR | Held-out prediction loss | RankMe |
|---|---|---|---|
| 1 | 7.5e-6 / 2.5e-5 / 7.5e-5 | 0.591 / 0.557 / 0.543 | 59 / 67 / 60 |
| 4 | 1.5e-5 / 5e-5 / 1.5e-4 | 0.536 / 0.546 / 0.586 | 59 / 61 / 47 |
| 16 | 3e-5 / 1e-4 / 3e-4 / 6e-4 | 0.539 / 0.498 / 0.492 / **0.466** | 51 / 54 / 54 / **28** |
| 32 | 2e-4 / 4e-4 | 0.498 / 0.517 | 54 / 50 |

The starting RankMe is 57.

**Decisions:**

- **Batch 16.**
  - At an equal number of clips seen, batch 16 reached lower held-out loss than batch 4, and batch 4 lower than batch 1.
  - Batch 1 was also about 5× slower per clip (3.8 against 19.5 clips/s on the same GPU), and its largest step-to-step loss jumps (99.9th percentile) were about 4× larger than at batch 16.
  - Batch 32 matched batch 16 at 1.8× the GPU time per step.
- **Learning rate 2e-4.**
  - 1e-4 and 3e-4 tie (0.498 vs 0.492, same RankMe).
  - 6e-4 has the lowest loss but RankMe 28: a partial collapse (Section 6.4). It was rejected despite its loss.
  - 2e-4 is the middle of the flat optimum, a factor of 3 below the collapsing value. It was not run directly; both neighbours tie.

### 4.3 Final configuration

The config is `configs/train_2_1/ego4d/vitb16-pretrain-256px-16f.yaml`. Each run's resolved copy is `config.yaml` in its run folder. It derives from the upstream `configs/train_2_1/vitb16/pretrain-256px-16f.yaml`.

**Data and input:**

| Setting | Value | Origin / reason |
|---|---|---|
| Modality | video only | Upstream also trains on images; a single GPU cannot host the image ranks. |
| Frames per clip | 16 at `fps: 4` | Upstream. The frame step is 30 // 4 = 7, so each sample covers 112 frames (3.73 s) of the 6 s clip, starting at a per-clip seeded random frame. |
| Spatial augmentation | random resized crop to 256 × 256 (scale 0.3–1.0, aspect 0.75–1.35), random horizontal flip, ImageNet mean/std normalisation | Upstream. |
| Tokens | tubelet 2 × patch 16 → 8 × 16 × 16 = 2,048 per clip | Upstream. |
| Batch | 16 | Sweep (4.2). |

**Model and loss:**

| Setting | Value | Origin / reason |
|---|---|---|
| Encoder | ViT-B/16 (`vit_base`) with RoPE; 4 hierarchical output levels after blocks 3, 6, 9, 12; activation checkpointing | Upstream. |
| Predictor | depth 12, width 384, 12 heads | Upstream. |
| Masks | generator A: 8 blocks of 15 % frame area; generator B: 2 blocks of 70 %. Aspect 0.75–1.5, every block spans all 8 time steps | Upstream. |
| Loss | L1 (`loss_exp 1`); context loss on (`predict_all`) with distance weighting | Upstream. |
| Context-loss weight λ | 0 → 0.5, linear between 10,000 and 20,000 clips (steps 625–1,250) | Upstream ramps at 5–10 % of the run; same fraction here. |

**Optimisation:**

| Setting | Value | Origin / reason |
|---|---|---|
| Learning rate | linear warmup from 3.33e-5 (lr/6) to 2e-4 over 8,000 clips (500 steps), then constant | Sweep. The warmup is about 4 % of the pass, as upstream; the start/peak ratio and the constant phase (`final_lr = lr`) are upstream defaults. |
| Weight decay | 0.04, constant | Upstream. |
| Target encoder | EMA with momentum 0.99925 per step, expressed as a horizon of 21,333 clips (m = 1 − 16/21,333) | The upstream value at batch 16. |
| Precision | bfloat16 autocast | Upstream. |
| Seed | 239 | Upstream. |
| Length | one pass: 12,972 steps (`epochs: auto`, `ipe: 625`) | Single-pass decision. |
| Checkpoints | every 10,000 clips; only the latest is kept | Saves disk while allowing exact resume. |
| Held-out evaluation | every 10,000 clips; batch 20 | Batch 20 divides the 1,000 clips exactly (partial batches are dropped). |

---

## 5. The three training runs

All three runs used commit `70b272b` and the script `scripts/train/curriculum.sbatch <curriculum>`.

| Run | GPU (node) | Wall time | Rows trained | Output |
|---|---|---|---|---|
| `top` | RTX 5090 (gpu-06) | 3 h 03 min | 0–207,551 | `runs/vitb16-ego4d-top/` |
| `random` | RTX 5070 Ti (gpu-01) | 6 h 06 min | 0–207,551 | `runs/vitb16-ego4d-random/` |
| `low` | RTX 5090 (gpu-06), after `top` | 3 h 09 min | 0–207,551 | `runs/vitb16-ego4d-low/` |

**Placement.** `top` and `random` ran in parallel; `low` ran after `top` on the same GPU. That keeps the main contrast, `top` against `low`, on identical hardware.

**Completeness.** Each run trained 12,972 contiguous steps, ran in a single job (no resume), and ended with "Sequential data exhausted". The paths above are under `/data/asem.a.abdelaziz/gp-2027a/`.

**Contents of each run folder:**

| File | Contents |
|---|---|
| `config.yaml` | the resolved config |
| `log_r0.csv` | per-step training log |
| `eval_r0.csv` | held-out evaluations |
| `train-<job>.log` | full training log |
| `latest.pth.tar` | final encoder, predictor, target encoder and optimiser state (1.7 GB) |

**Why `random` took twice as long.** It was loader-starved: the two parallel runs together pulled about 77 MB/s of clips from the NFS server, near what 1 GbE delivers, and the faster 5090 run took most of it. This changed wall time only, not what was trained: order, crops and masks are deterministic.

---

## 6. The metrics and how they are evaluated

### 6.1 Notation

For a clip, the **target encoder** (the EMA copy of the encoder, not trained by gradients) sees the *whole* clip. For each of the 2,048 tokens *t* it produces a feature vector **h**_t ∈ ℝ³⁰⁷², built as follows:

1. Take the output tokens of blocks 3, 6, 9 and 12, each 768-dimensional.
2. Layer-normalise each of the four 768-dimensional parts separately.
3. Concatenate them.

The **encoder** (the trained, "online" network) sees only the *context* tokens of the clip. The **predictor** then outputs a vector **ẑ**_t ∈ ℝ³⁰⁷² for every token: for the masked tokens these are predictions, and for the context tokens a reconstruction of their own target features.

Each clip is masked twice, independently, by the two mask generators *g* ∈ {A, B}. For generator *g*:

- **M**_g is the set of masked (predicted) tokens: the union of its blocks;
- **C**_g is the set of remaining context tokens.

### 6.2 Held-out prediction loss (left panel of `curricula_loss.png`)

For one clip and one generator:

  L_pred(g) = mean over t ∈ M_g and d = 1…3072 of | ẑ_t,d − h_t,d |

That is, the mean absolute error between the predicted and the target features of the masked tokens, averaged over all four hierarchy levels. The clip's loss is the average over the two generators, ½ (L_pred(A) + L_pred(B)). The plotted value is this loss averaged over the 1,000 evaluation clips (50 batches of 20, equal weight per clip).

**How it is evaluated:**

- **When:** before the first step (clips seen = 0); after every step that brings the clips seen to a multiple of 10,000 (10,000 / 16 = 625 steps, so exactly at 10k, 20k, …); and once at the end (207,552 clips). That is 22 points per run.
- **Mode:** no gradients, the same bfloat16 autocast as training, and the model state exactly as it is at that step.
- **Inputs:** each evaluation clip gets the same temporal window, crop and flip at every evaluation and in every run (per-clip seeding). Each evaluation batch gets the same masks (seeded by its row position in `heldout_eval.csv`). The *inputs are therefore fixed*; only the model changes.
- **Training augmentation:** the evaluation uses the training augmentation (random resized crop, flip) but frozen per clip. The views are representative of training views, not centre crops.
- **Not a fixed yardstick:** the target **h** is produced by *this run's own* EMA target encoder at that moment. Different runs, and the same run at different times, are scored against different targets (6.4).

Not plotted but logged in `eval_r0.csv`: the **held-out context loss**. It is the same L1 error on the context tokens C_g, each token weighted by 1/√(distance to the nearest masked token), the distance measured in token-grid units over (time, row, column). It only becomes part of the training objective once λ > 0, at 10,000 clips.

### 6.3 Training prediction loss (right panel of `curricula_loss.png`)

At every training step, `log_r0.csv` records L_pred for that step's batch of 16 clips. It uses the same formula as 6.2, computed in the forward pass *before* that step's parameter update, with that step's masks. (The optimised loss is L_pred + λ · L_ctx; only L_pred is plotted.)

The curve is a moving average over 312 consecutive steps (4,992 clips), plotted against the number of clips consumed by the end of the step. It starts at about 5,000 clips because the first 311 steps have no full window.

**How to read it.** The three runs are scored on *different clips* at each point: whatever their curriculum puts there. This curve shows how hard the current data is for the current model, not how good the model is.

- `top` sees progressively less surprising clips, so its training loss falls; the final steep drop is its least surprising, mostly dark or static, clips.
- `low` sees progressively more surprising clips, so its training loss rises after an early dip on its easy clips.
- Only `random` sees a stationary mix of data.

Compare runs with the held-out panel, not this one.

### 6.4 Held-out RankMe (`curricula_rankme.png`)

RankMe (Garrido et al., 2023) is the effective rank of a set of representations. It measures how many dimensions the features actually use. It is computed from the same forward passes as the held-out loss:

1. **Clip features:** for each evaluation clip *i*, take the target encoder's last-level features (the layer-normalised block-12 output, 768-dimensional) and average them over the clip's 2,048 tokens: **f**_i ∈ ℝ⁷⁶⁸.
2. **Matrix:** stack the 1,000 clips into F ∈ ℝ^(1000 × 768) (float32) and subtract the column means.
3. **Spectrum:** take the singular values σ₁ ≥ … ≥ σ₇₆₈ of F and normalise them to a distribution, p_k = σ_k / Σ_j σ_j.
4. **RankMe** = exp( − Σ_k p_k log p_k ), the exponential of the entropy of that distribution.

**Range and meaning:**

- RankMe lies between 1 (all clips map to one direction) and 768 (variance spread equally over all dimensions; the bound is min(1,000, 768)).
- A collapsing model, one whose features stop distinguishing clips, shows a falling RankMe.

**Differences from the original definition:**

- the features are centred before the decomposition, so RankMe measures the spread *between* clips rather than also counting the shared mean direction;
- it uses the target (EMA) encoder's mean-pooled last level;
- it is computed from only 1,000 clips.

### 6.5 Why the two metrics must be read together

The prediction loss compares the predictor against a target that the same run produces and keeps changing. If the target features become less diverse (lower RankMe), they are easier to predict and the loss falls, without the model having learnt anything useful.

The sweep showed this directly: batch 16 at learning rate 6e-4 had the *lowest* held-out loss (0.466) and a collapsed RankMe (28 vs about 54).

Conversely, a run whose features spread over more dimensions faces a harder target and can show a *rising* loss while improving.

So:

- a lower held-out loss is only evidence of better learning when RankMe is comparable or higher;
- a rising loss with a rising RankMe is ambiguous.

---

## 7. Results

| Clips seen | `top` loss / RankMe | `low` loss / RankMe | `random` loss / RankMe |
|---|---|---|---|
| 0 | 0.858 / 57 | 0.858 / 57 | 0.858 / 57 |
| 10k | 0.556 / 26 | 0.606 / 27 | 0.560 / 26 |
| 20k | 0.518 / 33 | 0.580 / 28 | 0.513 / 34 |
| 50k | 0.495 / 68 | 0.594 / 51 | 0.490 / 64 |
| 100k | 0.490 / 93 | 0.581 / 80 | 0.479 / 84 |
| 150k | 0.497 / 122 | 0.562 / 108 | 0.481 / 107 |
| 207,552 (end) | 0.540 / 157 | 0.553 / 140 | 0.487 / 137 |
| Lowest held-out loss | 0.490 at 80k | 0.550 at 200k | 0.478 at 90k |

### 7.1 Common shape

- **First 10k clips:** RankMe drops from 57 to about 26 in all runs. This is the learning-rate warmup (0–8k clips) before the context loss switches on.
- **After that:** RankMe rises almost linearly to the end. No run collapsed.

### 7.2 Per curriculum

- **`low`:**
  - Its first ~70k clips are the least surprising, mostly dark and static footage. Its held-out loss stalls near 0.59 there and its RankMe lags.
  - Both improve steadily once more surprising clips arrive, and are still improving at the end.
- **`top`:**
  - Its held-out loss follows `random` for 20k clips, then flattens near 0.490.
  - It rises from about 100k clips on, as its data becomes progressively less surprising. The final 7.5k clips (its least surprising) raise it from 0.520 to 0.540.
  - Its RankMe is the highest from about 40k clips on, which accounts for part of that rise (6.5).
- **`random`:** lowest loss from 20k clips on, nearly flat (0.478–0.487) over the second half.

### 7.3 Comparison at matched RankMe

The ranking holds when runs are compared at matched RankMe. At RankMe about 84:

| Run | Clips seen | Held-out loss |
|---|---|---|
| `random` | 100k | 0.479 |
| `top` | 80k | 0.490 |
| `low` | 110k | 0.576 |

---

## 8. Caveats for interpreting the results

1. **One run per curriculum.**
   - There is no estimate of seed-to-seed variation.
   - The `top`–`random` gap (about 0.01) is the same size as the gap between neighbouring learning rates in the sweep, so it may not be significant.
   - The gap to `low` (about 0.09 mid-run) is much larger.
2. **The held-out loss is not a fixed yardstick (6.2, 6.5).**
   - Each run is scored against its own moving target, so it confounds prediction quality with target diversity.
   - The decision was to use held-out loss together with RankMe and no other measure; there is no target-independent evaluation such as a frozen-feature probe.
   - Statements about *feature quality* need such an evaluation; these results describe *training dynamics*.
3. **Hyperparameters were tuned on the random order.**
   - The sweep trained on the first 40k clips of `random`.
   - Its winner was chosen with the same held-out set later used to compare the curricula.
   - Both choices can favour `random` slightly.
4. **Two GPU types.**
   - `top` and `low` ran on an RTX 5090, `random` on an RTX 5070 Ti.
   - Data, crops and masks were identical, but bfloat16 kernels round differently across GPU generations.
   - Training on GPUs is also not bitwise deterministic even on one card. The same command will reproduce the curves closely but not exactly.
5. **Curriculum and data quality are entangled.**
   - Low surprise coincides with dark and static footage: the 1,768 dark and 129 static clips kept in the data concentrate at the low end.
   - Part of what looks like a curriculum effect at the start of `low` and the end of `top` is an effect of uninformative video.
6. **What "surprise" means here.**
   - Surprise is the error of a *different, much larger* model (V-JEPA 2.1 giant) with a 4 s context / 2 s target split.
   - It is not the error of the ViT-B being trained, and its ranking may not match what is hard for that model.
7. **The held-out set is small and clustered.**
   - It holds 1,000 clips from 30 videos.
   - Clips of one video are correlated, so the effective sample size is closer to 30 videos than to 1,000 clips.
   - Small differences in held-out loss should be read with that in mind.
8. **Small compute regime.**
   - Each run is 12,972 steps of batch 16, about 207k clips, against upstream's hundreds of thousands of steps at batch 3,072.
   - There is a constant learning rate and no cool-down, and the models are far from converged.
   - The results describe early training and may not transfer to larger scales.
9. **The last 10 clips of each list were not trained on.**
   - 207,562 is not a multiple of 16, so the final partial batch was dropped.
   - These are different clips per curriculum: the 10 least surprising for `top`, the 10 most surprising for `low`, 10 arbitrary ones for `random`.
   - That is 0.005 % of the data.
10. **Ties make `low` differ from a reversed `top`.** 19,708 positions differ (1.3).
11. **The filtered lists differ from the curated originals.**
    - 7,321 near-black or black clips were removed, plus the 2,210 clips of the held-out videos.
    - The relative order of all remaining clips is unchanged.
12. **The training-loss panel does not compare models.** Each run is scored on different data at each step (6.3).

---

## Appendix A. Rebuilding the experiment from this folder

This works even if the `curriculum-prep` branch, its worktree and the run folders on `/data` are gone. It needs this repository's `main` history, which contains commit `1b60c66`, plus the video files and surprise scores of Section 1.

1. **Code.** Recreate the branch from the patch:

   ```bash
   git checkout -b curriculum-prep 1b60c66
   git am training_n_experiments_results/three_curricula_one_epoch_results_Nov_25_2026/code/curriculum-prep.patch
   ```

   This gives back the exact files of `d263b9c`; the commit hashes differ, but the tree is identical. To use exactly the code the runs used, check out the commit whose message is "Set the ViT-B curriculum config from the sweep and add the run script" (originally `70b272b`).

2. **Environment.** Section 2.2: Python 3.12, torch 2.13.0+cu130, decord 0.6.0, h5py 3.16, FFmpeg 7.0.2.

3. **Original lists.** In Ego-Curation at `3f7095b` (a copy of the script is in `code/`), run:

   ```bash
   python scripts/train/make_splits.py runs_results/run_1184 \
       --clips-dir <ego4d_clips> --fraction 1 --global --seed 0 --out-dir <lists>
   ```

   Verified to be byte-identical to the lists used.

4. **Training lists and evaluation set.**
   - **Exact copy:** remove every path in `data/removed_clips.csv.gz` from each original list, keeping the order. This reproduces the training lists exactly (verified). Unzip `data/heldout_eval.csv.gz` for the evaluation set.
   - **From scratch:** re-run the scan, then `scripts/data/make_curriculum_split.py --dim-std 5 --seed 0`, as described in Section 3.

   The h5 paths inside the lists are absolute. If the clips move, rewrite the prefix before `::`.

5. **Training.** From the repository root, edit the paths in `configs/train_2_1/ego4d/vitb16-pretrain-256px-16f.yaml` and `scripts/train/curriculum.sbatch` (data lists, held-out list, run folder, venv), then submit:

   ```bash
   sbatch --nodelist=<node> --gres=gpu:<gpu>:1 scripts/train/curriculum.sbatch top   # and low, random
   ```

   - **Resolved configs:** each run's exact settings are in `configs/run_*.yaml`.
   - **GPUs:** to remove the GPU-type caveat (Section 8, item 4), run all three on the same GPU type.
   - **Resuming:** a run stopped early resumes exactly from its last checkpoint when the same command is resubmitted.

6. **Sweep.** Run `scripts/sweep/sweep.sbatch <batch>:<lr> …`, with the pairs of Section 4.2.

7. **Figures.** Run:

   ```bash
   python scripts/analysis/plot_curricula.py --out loss.png --rankme-out rankme.png
   ```

   It reads `runs/vitb16-ego4d-*/{eval_r0,log_r0}.csv`. To re-plot from this folder instead, point `RUNS` in the script at `results/` and unzip the `log_r0.csv.gz` files.

Even with identical code and data, GPU kernels are not bitwise deterministic. Expect curves that match closely, not exactly.

## Appendix B. The order-control mechanisms, in code

These are the parts of the patch the experiment's validity rests on. They are reproduced here so they can be re-implemented even if the patch cannot be applied, for example to a much newer upstream.

**Seeds from strings** (`src/utils/seeding.py`). Python's `hash()` is salted per process, so SHA-256 is used to get the same seed in every run and every worker:

```python
def stable_seed(*parts):
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1

def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
```

**Per-clip randomness** (`VideoDataset.get_item_video`). This runs before the clip is decoded and transformed. `sample` is the clip's path string (`<h5>::<key>`), and `rng_seed` is the config seed (239):

```python
if self.rng_seed is not None:
    seed_all(stable_seed(self.rng_seed, "clip", sample))
```

**Strict order** (`ResumableSequentialSampler`):

- The sampler yields rows `start, start+1, …` of the CSV; with several ranks, rank *r* gets `start + r, start + r + world_size, …`.
- The tail that does not fill a batch is dropped, never padded by wrapping around.
- On resume, `train.py` calls `set_start_index(epochs_done × ipe × batch × world_size)`.
- If a row fails to load, the same row is retried up to 3 times (2 s apart). Then training stops with an error; the upstream code would have substituted a random row instead.

**Per-step masks** (`MaskCollator.__call__`). The seed comes from the row position of the batch, not from the clips in it. So all curricula draw identical masks at the same step:

```python
first_row = min(int(s[2]) for s in fpc_batch)          # dataset row index of the batch
generator = torch.Generator()
generator.manual_seed(stable_seed(self.seed, "mask", first_row, fpc, i))   # i = mask generator
masks_enc, masks_pred = mask_generator(batch_size, generator=generator)
```

**Schedules in clips** (`train.py`), with `global_batch = batch × world_size`:

| Schedule | Conversion |
|---|---|
| Warmup | `warmup_clips / (ipe × global_batch)` epochs |
| EMA momentum | `m = 1 − global_batch / ema_horizon_clips` |
| Context-loss ramp | from step `lambda_start_clips // global_batch` to step `lambda_end_clips // global_batch` |
| Number of epochs | `ceil(batches in one pass / ipe)` (`epochs: auto`) |

**Held-out evaluation** (`evaluate_heldout`):

- **Loss:** run the same `compute_losses` as training, under `no_grad` and autocast, on the fixed evaluation loader (`shuffle=False`, per-clip seeding, its own mask collator seeded with the config seed). Average the losses weighted by batch size.
- **RankMe:**

  ```python
  feats.append(h[0][:, :, -embed_dim:].float().mean(dim=1))  # last level (last 768 dims), mean over tokens
  f = torch.cat(feats); f = f - f.mean(dim=0, keepdim=True)
  s = torch.linalg.svdvals(f); p = s / s.sum()
  rankme = torch.exp(-(p * torch.log(p + 1e-12)).sum())
  ```
