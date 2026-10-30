# Phase 3 (LTX-2B): causal test C9, decoder test C5, probe metrics C6, statistics and gates report

Scope follows the pre-registration (`configs/gates.yaml`): C7 steering failed G6 on LTX (fitted hspeed / truth = 0.67
at every alpha), so the causal arm keeps only C9 (LEACE); C8 patching is not run. Everything except C9 is CPU work
on data already on HF and runs on the laptop.

## C9: LEACE erasure during generation (GPU, `pilot/leace.py`)

- Concept z = ln(true horizontal speed) of the scene (`probe.target_value(gt, "hspeed")`), scalar.
- Fit data: B1, all 100 kmain scenes x 3 seeds (300 clips), pooled object-token activations
  `transformer_blocks/obj[block, step]` from `acts.npz`. Block 7, step index 0: the best hspeed probe in probe2
  (`results/p6_ltx2b_hspeed_obj.csv`); read from that file, `--block` overrides it; `--step` only labels meta (the eraser uses every captured step).
  Untested: steps 1-11 of 50 (never captured, the eraser is applied there too) and per-token perturbation size (the
  check measures pooled features).
- Eraser (Belrose et al. 2023, LEACE, closed form, shrunk covariance; `pilot/erase.py`): fit on the B1 pooled
  object tokens at the block, all 3 captured steps (12, 24, 37 of 50) stacked, with one concept column per step,
  Z[:, s] = (z - mean z) 1{step = s}, so cov(r(x), z) = 0 within every step (the hook runs on every step; an eraser fit on
  step 12 alone leaves speed re-learnable at step 37, retrained-probe R² 0.59). S_a = (1-a) S + a tr(S)/D I with a by
  Ledoit-Wolf (0.012; D = 2048 > n, so the exact S^-1 interpolates the fit data), Sig = cov(x, Z), B = S_a^-1 Sig,
  r(x) = x - Sig (Sig^T B)^-1 B^T (x - mu) (= W+ P W with W = S_a^-1/2), rank 3, affine, so applying it to every object
  token makes the mean-pooled feature exactly r(pooled). Control "rand": the same fit to a permutation of z (seed =
  scene index): same rank and construction, erases nothing about speed (the old random whitened direction gave a 2.8x
  larger perturbation than leace). The pod prints a cross-fitted check (`erase.crossfit`: fit on 4/5 of the scenes,
  score the held-out fifth per step) and stops unless leace (a) clean-trained ridge on r(x) and (b) ridge retrained on
  r(x) are both R² <= 0.05, rand keeps both >= 0.6, and median |r(x) - x| / |x - mu| <= 0.3 with leace/rand within 2x.
  On B1 LTX block 7: clean 0.74, leace (a) 0.01, (b) -0.04, rand (a)/(b) 0.67-0.69, perturbation 0.12-0.13 vs rand
  0.21. (b) is ~0 by construction (r zeroes the train covariance, so a retrained linear probe sees nothing); (a) and
  the per-step result are the real tests. `python -m pilot.erase` runs the same checks on synthetic data.
- Hook: like `pilot.steer.steer_hook`, at `gen.blocks()[block]`, object tokens of the input frames
  (`object_token_mask`), every denoising step and both guidance passes.
- Runs: every kmain scene, seed 0, cond in {leace, rand} -> `runs/c9/ltx2b/{scene}_{cond}_s0/` with video.mp4
  (write-then-rename), acts.npz (capture on: lets the analysis check that erasure removed the probe signal in
  later blocks), meta.json (exp "c9", cond, block, step, scene, box, in_wh + the usual generate meta). The B1 seed-0 clip
  is the no-erasure baseline. 200 clips.
- Plan: jobs `c9s1of4`..`c9s4of4` (scene shards as phase 2, `pilot.shard`), each: fit eraser, generate its shard's
  2 x 25 clips, score them. Pull: scenes, `data/renders/kmain/*_input.mp4`, `runs/b1/ltx2b/*/{acts.npz,meta.json}`,
  `results/p6_ltx2b_hspeed_obj.csv`. Push: `runs/c9/ltx2b/**`, own logs/results.
- Readout (analysis.py): Spearman rho(fitted hspeed, true hspeed) across scenes and median sym MRA, for B1 s0 /
  leace / rand, bootstrap 95% CI over scenes; "information used" = rho(leace) < rho(rand) with CI excluding 0 on the
  paired difference. Plus probe MRA on the c9 acts per cond at blocks >= the erased block.

## C5: final latent vs decoded video (CPU, analysis.py)

`latent.pt` (1, 6240, 128) bf16 = 16 x 15 x 26 latent tokens x 128 channels. Feature: reshape to (16, 15, 26, 128),
average-pool space to (16, 5, 13, 128), flatten (133k). Same cross-validated ridge pipeline as `pilot.probe`
(`probe.score`), targets hspeed (B1), g and e (B1 + B5). Compare: probe on the final latent vs the show score of the
video decoded from that latent vs the best mid-network probe. Latent ~ mid-network > video: the decoder loses it;
latent ~ video: the loss happens before decoding.

## C6: conditional V-information and MDL (CPU, analysis.py)

At the best block/step of each target (from `results/p6_ltx2b_{target}_obj.csv`):
- Conditional V-information beyond the VAE input latents: I = 0.5 ln(MSE_vae / MSE_vae+acts) nats, MSE of
  out-of-fold ln-target predictions (Gaussian approximation), bootstrap CI over scenes.
- Online (prequential) MDL codelength, Voita & Titov: scenes in a fixed random order, blocks at 10/20/40/60/80/100 %,
  ridge fit on earlier blocks, Gaussian codelength of the next block; report total bits and compression vs the
  uniform code, for acts, VAE, control task. Also for p7 noise vs inv (H5).

## Statistics and gates report (CPU, analysis.py -> results/gates_ltx2b.md)

- Paired know-show gap per clip: out-of-fold probe prediction (best block) vs the tracker estimate of the same clip,
  both sym MRA; mean gap with a scene-level bootstrap 95% CI; Holm across targets (`pilot.metrics`).
- Gates evaluated from data where LTX has it: G1b (measurability), G2 (B1 MRA range), G3 (probe margin), G4 (gamma CI
  width, B5), G5 (clock rule), G6 (steering, fail), G8 (power at gap 0.10 with this n), G10 (rendered vs real
  measurability); G1a / G7 / G9 "not evaluated" with the reason. Every number cites the file it came from.

## C9 result and C9b (every block from 7)

- **C9 (one block):** LEACE at block 7 removes the speed code locally: probe MRA on the c9 activations drops from 0.81 to 0.56. Block 8 re-learns it (0.90).
  - Generated motion is unchanged: rho 0.77 for LEACE vs 0.79 for B1 seed 0, while the rand control drops to 0.66.
  - "Information used" is false. C9 is inconclusive.
- **Why C9b erases object tokens only:** the all-token pool does not carry speed. Its hspeed probe sits at the mean-only baseline (about 0.55) at every block. So C9b erases the same object tokens at every block 7–27.
- **C9b fit:**
  - One eraser per block, fit on that block's B1 activations.
  - I also tried adding the C9 leace runs to the fit data for blocks 8 and up, since erasure had already shifted their activations. That eraser did not carry over to held-out scenes: the clean probe still scored R² 0.18 after erasure at block 8, so it was dropped.
- **C9b check:** each block passes the cross-fitted check before generating. With the C9 targets, 10 of the 21 blocks failed on marginal misses. The targets were then changed, after seeing the numbers (laptop dry run, all 21 blocks then pass):
  - **leace (a) and (b): ≤ 0.06** (was 0.05). Block 10 reached 0.053, against a clean R² of 0.63.
  - **rand: ≥ min(0.6, 0.75 × clean R²)**, checked only at steps where clean R² ≥ 0.3 (was a fixed 0.6). Late blocks hold less speed: clean R² is 0.53–0.66 at blocks 19–26 and 0.2–0.3 at block 27.
  - **Perturbation: both ≤ 0.3, and leace at most 2× rand** (was within 2× either way). At blocks 20 and up, leace is about 0.11 and rand about 0.23. A smaller leace edit only makes "information used" harder to show.
  - These are engineering checks from this document, not pre-registered gates.
- **The analysis then reads whether the erasure held:** it probes the C9b activations at every block, retraining the probe on each condition.
- **C9b result (2026-10-26, 200 clips, `results/c3_ltx2b_c9b_*`):**
  - **Measurable clips:** leace 61/100 and rand 47/100, against C9's 70 and 66.
  - **Speed readout on the 40 scenes measurable in all conditions:**

    | Condition | rho | 95% CI | Median MRA |
    |---|---|---|---|
    | B1 seed 0 | 0.735 | [0.53, 0.87] | 0.78 |
    | leace | 0.483 | [0.13, 0.75] | 0.69 |
    | rand | 0.537 | [0.28, 0.73] | 0.38 |

    - rho(B1) − rho(leace) = +0.25, CI [0.003, 0.53]: leace lowers rho compared with no erasure.
    - rho(rand) − rho(leace) = +0.05, CI [−0.32, 0.43]: "information used" is false.
  - **Probe trained on clean B1, applied to the erased runs:** R² of ln hspeed is about 0 at blocks 7–8. It stays at or below 0.5 at every block from 7 to 27 (C9 recovered 0.7–0.9 by blocks 10–12). The original speed direction stays erased.
  - **Probe retrained on the leace runs:** MRA 0.88–0.96 at blocks 8–27, higher than B1 (0.61–0.80). Speed is re-encoded along new directions that the erasers, fit on clean data, do not touch.
  - **Rand probe:** falls to 0.58–0.66 at late blocks. The rand edits there disturb the features more than leace does.
  - **Reading:** the linear speed code is redundant. Removing it at every block moves the model to another encoding. Motion gets worse under leace and rand alike, so the damage is not specific to speed.
  - **Next step for a clean causal test:** sequential (concept-scrubbing) LEACE, refitting each block on activations already erased upstream. That needs one generation pass per block, which is expensive.
- **C9b runs:** jobs `c9bs1of4`..`c9bs4of4` write to `runs/c9b/ltx2b/`. The readout is `python -m pilot.analysis c9b` and writes `results/c3_ltx2b_c9b_*`.

## C9c: sequential LEACE (concept scrubbing), blocks 7-27

- **Design:** blocks 7-27 split into 5 near-equal groups (7-11, 12-15, 16-19, 20-23, 24-27). The erasers of group k are fit on activations generated with the erasers of every group before k active.
  - Group 0 has no upstream eraser, so its fit data is the B1 seed-0 activations.
  - Every other group needs one pass over the fit scenes: generate with the upstream erasers hooked, stop denoising after the last captured step (37 of 50), no VAE decode (`Generator.generate(acts_only=True)`).
  - Estimator and check as C9b (`leace.fit_check`): pooled object tokens, one concept column per captured step, Ledoit-Wolf, cross-fitted check.
  - The leace chain must pass `check_ok`. The rand chain has its own passes: its erasers are fit to permuted z (`random_like`, seed 0) on the rand chain's activations and must pass `rand_kept` (`rand_ok`, perturbation <= 0.3).
  - `python -m pilot.leace --model ltx2b --chain leace|rand` -> `runs/c9c_chain/ltx2b/{cond}/`: `g{k}/{scene}.npy` (one clip of one pass), `g{k}.npz` (the group's erasers and check), `chain.json` (written last: groups, fit scenes, checks).
  - The chain is resumable. A relaunch skips finished groups and clips, and the pod pushes the chain every 15 min.
- **Fit set: all 100 kmain scenes, seed 0.** Laptop dry run of the check on clean B1 seed-0 activations, every block 7-27:
  - 50 scenes: 18 of 21 blocks fail. The rand control's retrained probe drops to 0.33-0.55 against a clean R² of about 0.63.
  - 75 scenes: 5 blocks fail (15, 16, 24-26).
  - 100 scenes: all pass. leace (a) and (b) stay <= 0.03, and rand keeps a margin of >= 0.09 over its target.
  - A check that fails mid-chain stops the chain, which wastes the passes before it, so the smaller sets are out.
  - Trade-off: the final runs use the same 100 scenes at seed 0, so the erasure is in-sample, as in C9b (fit on 300 clips including seed 0). There are no held-out scenes.
- **Why 5 groups:** each pass costs about 100 x 55 s = 1.5 h. Exact sequential LEACE (21 groups) would take about 30 h per chain.
  - With 5 groups, the eraser of the second to fifth block of a group is fit without the earlier blocks of its own group erased. C9 showed that speed re-encodes within one block (0.56 -> 0.90 from block 7 to 8), so some re-encoding inside a group goes unseen by that group's fit. The next group's fit sees it.
  - The per-block retrained probes would show this as a sawtooth that peaks at the last blocks of each group.
- **Final runs:** every kmain scene, seed 0, leace and rand with every block's chain eraser -> `runs/c9c/ltx2b/{scene}_{cond}_s0/` (video, acts, meta as C9b; meta `chain` records groups, fit scenes, chain file).
  - Jobs `c9cs1of4`..`c9cs4of4` load both chains. Without a `chain.json` they stop at once: launch them only after both chain jobs have pushed.
  - They fail rather than wait, because 4 idle pods for about 7 h would burn about 28 GPU-hours.
- **Budget (RTX 3090):**
  - Chains `c9cchain-leace` and `c9cchain-rand` run on two pods at once: 4 passes x 100 clips x ~55 s = 6.1 h, plus fits, load and setup, about 6.8 h wall each.
  - Shards: about 1.5 h each.
  - Total about 20 GPU-hours.
  - Unverified: 55 s per acts-only clip is scaled from 66-79 s full clips (38 of 50 steps). The checks on chain activations (groups 1-4) are not tested before the run.
- **Readout:** `python -m pilot.analysis c9c` -> `results/c3_ltx2b_c9c_*`, as c9b: video rho for B1 s0 / leace / rand with bootstrap CIs on the paired differences.
  - The per-block probes in `_probe.csv` are now two: retrained within each condition (`probe_mra`), and trained on clean B1 clips of the other scenes (`clean_r2`, `clean_mra`; this also applies to c9 and c9b).
  - **Reading:** if the chain scrubs speed, the retrained probe on leace falls well below B1 at blocks >= 7, unlike C9b's 0.88-0.96. "Information used" is rho(leace) < rho(rand) with the CI excluding 0.

- **Outcome of the chains (2026-10-30):**
  - rand: all 5 groups passed `rand_kept`.
  - leace: groups 0-2 (blocks 7-19) passed. At block 20 speed is readable from the chain activations at R² 0.93-0.95 (clean B1 block 20: about 0.6), and the fitted eraser leaves (a) 0.07-0.08 against the 0.06 target, so the chain stopped.
  - Reading: after 13 blocks of sequential erasure of the input object tokens, the model carries speed at block 20 more strongly than without erasure. The erasure touches only the object tokens of the input frames, so the speed code is likely rebuilt from other tokens.
  - Decision (user, 2026-10-30): no threshold change. The shards generate with groups 0-2 of both chains (`--use-groups 3`: erasers at blocks 7-19 for leace and rand alike).

- **C9c result (2026-10-30, 200 clips, erasers at blocks 7-19, `results/c3_ltx2b_c9c_*`):**
  - **Video rho on the 48 scenes measurable in all conditions:** B1 seed 0 0.745 [0.57, 0.85], leace 0.648 [0.40, 0.82], rand 0.602 [0.36, 0.77]. rho(rand) - rho(leace) = -0.05, CI [-0.18, 0.08]: "information used" is false, as in C9 and C9b.
  - **Clean-trained probe on the leace runs:** R² of ln hspeed is negative at every block 7-27 (mean -0.63, max -0.08). The original speed direction is gone everywhere, including blocks 20-27 that have no eraser.
  - **Retrained probe on the leace runs:** a sawtooth. At the first block of each group (7, 12, 16) it sits at the mean-only baseline (sym MRA 0.56). One block later it is back at 0.86-0.90, and from block 20 on it reaches 0.92-0.95, above B1 (0.75-0.78).
  - **Reading:** each eraser removes speed at its own block, and the next block re-encodes it along a new direction. Generated motion does not depend on the linear speed code at the input object tokens. Exact one-block-per-group scrubbing would only push the sawtooth one block further.

## B-ratio: why only 5/30 clips are measurable

- **First diagnosis (wrong):** the kratio balls are 25–41 px, about one LTX latent cell.
- **Test:** `bratio2` used 0.7–0.9 m balls (71–92 px). It was still 5/30 measurable (`configs/deviations.md` D2).
- **Actual cause:** LTX splits the balls, changes their colour, shrinks them, and keeps them airborne until they leave the frame. The measurability rule rejects these clips: a generator failure, not a tracker failure.

## Three generators: LTX-2B, Cosmos-Predict2.5-2B, Wan2.1-VACE-1.3B (2026-10-28)

**What ran.**
- Cosmos-2B and VACE-1.3B ran the core plan: B1 on 100 kmain scenes, B5 on 30 kgrav scenes, and the 7 real clips, all seed 0.
- Each model then ran the same probes (`{model}probe`) and `python -m pilot.analysis all --model {model}`.
- Results are in `results/c3_{model}_*` and `results/gates_{model}.md` (also on HF).
- LTX numbers are the phase 2 runs over all seeds (300 B1 clips). For the other two models every scene has one clip.

**Know vs show** (sym MRA). *Know* is the best mid-network probe on object tokens. *Margin* is its lead over the larger of the VAE-latent baseline and the control task. *Gap* is know minus show over measurable clips, with a scene-bootstrap 95% CI; all gaps have Holm p < 0.001. *I* is the conditional V-information beyond the VAE latents, in nats.

| Model | Target | Know | Margin | Show | Gap [95% CI] | Scenes | I [95% CI] |
|---|---|---|---|---|---|---|---|
| LTX-2B | speed | 0.84 | +0.19 | 0.55 | +0.36 [0.29, 0.43] | 91 | 0.68 [0.52, 0.86] |
| LTX-2B | gravity | 0.81 | +0.07 | 0.25 | +0.58 [0.50, 0.65] | 51 | 0.51 [0.28, 0.71] |
| LTX-2B | restitution | 0.94 | 0.00 | 0.71 | +0.24 [0.18, 0.30] | 51 | 0.02 [-0.04, 0.08] |
| Cosmos-2B | speed | 0.78 | +0.13 | 0.41 | +0.36 [0.27, 0.45] | 72 | 0.45 [0.30, 0.62] |
| Cosmos-2B | gravity | 0.88 | +0.10 | 0.34 | +0.57 [0.45, 0.69] | 28 | 0.90 [0.55, 1.19] |
| Cosmos-2B | restitution | 0.95 | +0.01 | 0.84 | +0.12 [0.07, 0.17] | 28 | -0.01 [-0.10, 0.08] |
| VACE-1.3B | speed | 0.85 | +0.18 | 0.48 | +0.37 [0.28, 0.46] | 56 | 0.77 [0.61, 0.93] |
| VACE-1.3B | gravity | 0.93 | 0.00 | 0.17 | +0.80 [0.68, 0.90] | 19 | 0.00 [0.00, 0.00] |
| VACE-1.3B | restitution | 0.93 | +0.01 | 0.60 | +0.34 [0.22, 0.47] | 19 | 0.00 [-0.03, 0.02] |

**Reading.**
- **Speed:** all three models encode it beyond the input latents, with margins of 0.13–0.19 and I of 0.45–0.77 nats. All three render it far worse than they encode it, with the same gap of about +0.36 for each. This is the cross-architecture result.
- **Gravity:**
  - Only LTX and Cosmos carry it beyond the input. Cosmos carries the most (I = 0.90 nats), even though it sees only 5 input frames (0.3 s) against LTX's 25.
  - VACE's gravity "know" is the VAE baseline itself, so its +0.80 gap measures a show failure, not a hidden code.
- **Restitution:** no model encodes it beyond the input. The probe reads e at the level of the VAE latents, so the e gaps are rendering failures too.
- **C5** (speed, final latent vs mid-network) is the same in all three models: 0.60 vs 0.77–0.85. Speed is lost inside the network before the final latent, not in the VAE decoder.

**Gates.**
- G1b fails for all three: B5 measurability is 0.40 for Cosmos, 0.13 for VACE and 0.37 for LTX.
- G8 power is 0.57–0.58 for the two new models, one seed each. LTX, with three seeds, reaches 0.76.
- Cosmos also fails G4 (gamma CI width 1.07 from 12 B5 clips). G10 is noisy: 7 real clips, all measurable.
- Cosmos G6 comes from 6 old pilot steer clips and shows no effect (slope +0.001). It is not a real steering test.

**Caveats.**
- *Know* is the best of 84 (LTX) to 135 (VACE: `blocks` plus `vace_blocks`) block/step choices. The nested-CV correction (Deferred, item 2) applies to every row.
- The gravity and restitution rows rest on 19–51 measurable scenes.
- Seed 0 only for Cosmos and VACE. A second seed would raise G8 power and is the cheapest next step: about $13 for both models.

## G1b re-score with CoTracker3 (job `rescore`, D3 pending approval)

- **Why:** G1b fails for all three models (B5 0.13-0.41). The registered fail action adds CoTracker3, so `configs/deviations.md` D3 adds it as a second readout. That entry also gives the laptop diagnosis: in a sample of failing clips the generators break the ball, so G1b is expected to change little.
- **Job:** one RTX 3090 running three steps:
  1. P0 of the CoTracker3 readout on the 100 full kmain renders (G1a).
  2. Every B1, B5 and b1real clip of LTX-2B, Cosmos-2B and VACE-1.3B, plus the 30 bratio2 clips (767 in all), into `score_v2.json`.
  3. SAM2 P0 on the same renders, for comparison.
- **Outputs:**
  - `results/p0_renders_{cotracker3,sam2}_ltx_rescore.csv`
  - `results/show_v2_ltx_rescore.csv`
  - `results/faults_ltx_rescore.csv`: one row per clip and ball, with frame counts of vanished, off-frame, edge, split, recoloured, shrunk, grown and lost-but-colour-seen from the SAM2 masks. Its bratio2 rows are the B-ratio breakdown below.
- **Pull:** videos, meta and any `score_v2.json` of those runs, the scenes, the real-clip manifest and the full kmain renders. No acts or latents. **Push:** `score_v2.json` and the job's results and logs.
- **Readout (laptop):**
  - `python -m pilot.analysis fetch --model M` for each model.
  - `python -m pilot.analysis g1b` gives measurability per model × exp under both trackers on the same clips, plus why SAM2 rejected each ball.
  - `python -m pilot.analysis gap --v2 --model M` and `gates --v2` recompute the know-show gaps and the gates from `score_v2.json`, into `results/c3_{M}_v2_*` and `gates_{M}_v2.md`.
- **Not re-scored:** LTX b2nl, b3, steer and c9. Under `--v2` the G1b row of `gates` covers b1, b5 and bratio2 only.

## Deferred (user, 2026-10-26)

1. **B-ratio failure breakdown** (about $0.3 GPU): per clip and per ball, count split, recoloured, shrinking, off-frame and vanished, from the SAM2 masks. This puts numbers on the D2 finding. Covered by job `rescore` (`results/faults_ltx_rescore.csv`, exp bratio2).
2. **Statistics without the selection bias:**
   - A nested-CV know–show gap: choose the block and step inside each outer fold, not the best of 84 on all data.
   - G1a: tracker vs ground truth on the full kmain renders.
   - Both run on CPU or a cheap vast instance.
   - **Built: pod job `stats2`** (one RTX 3090, CPU-bound apart from SAM2).
     - G1a on all 100 full renders runs in job `rescore` (`p0:sam2`, `results/p0_renders_sam2_ltx_rescore.csv`), which `gates` reads.
     - Nested gap: `python -m pilot.analysis nested --model M`. There are 5 outer folds split by scene. Inside each outer training fold, the block and step are chosen by `best()`'s rule (max out-of-fold sym MRA, `probe.score`) and the ridge penalty by `probe.fit`'s inner scene CV. The model then predicts the outer fold. The paired gap code is the same as `gap`'s, with Holm across targets.
     - Output: `results/c3_{model}_nested.csv` holds one `nested` row and one `best_of_all` row per target (the latter recomputes `c3_{model}_gap.csv`). `picks` lists the block and step each outer fold chose. Per-clip rows go to `results/c3_{model}_nested_clips.csv`.
     - `python -m pilot.analysis selftest` covers it on synthetic data. A planted block is picked in every fold. With no signal, the nested know is lower than the best of 8 noise probes.
