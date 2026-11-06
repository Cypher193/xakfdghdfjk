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

## Round 3: the rest of the B runs on the three small models (2026-11-01)

- **Scope (user, 2026-10-31 and 11-01):** B2-nl seeds 1-2, B2-num, B3 on Cosmos, B-L3, B5-frames on Cosmos and LTX, B-ratio. VACE's B5 extension, B5-text and Cosmos's G6-dependent work (C7, C8) are out. Budget about $125-135 of $164.80, verified and datacenter hosts allowed.

| Experiment | Input | Models | Scenes × seeds | New clips |
|---|---|---|---|---|
| B2-nl | plain-word text, no video | Cosmos, VACE | 100 × seeds 1-2 | 400 |
| B2-num | numeric text, no video | all 3 | 60 × 3 | 540 |
| B3 | input window + plain-word text | Cosmos | 100 × 3 | 300 |
| B-L3 (`bl3`, `bl3nl`) | caption + input window; plain-word text only | Cosmos, LTX | 30 × 2 × 3 | 360 |
| B5-frames | input window at 0.5, 1, 2 g | Cosmos, LTX | 90 × 5 | 660 |
| B-ratio (`bratio2`) | input window of two throws | all 3 | 40 × 3 | 330 |

- **New scenes (`pilot/scenes.py`; old scenes unchanged, `tests/test_round3.py`):**
  - `kgrav` grows from 30 to 90 scenes (20 new bases × 3 gravity values, own random stream).
  - `kratio2` grows from 10 to 40 scenes (same stream, the first 10 are the old ones).
  - `kl3` holds 15 collisions and 15 ramps. See `configs/deviations.md` D4.
- **Numeric text (`describe_num`):** the plain-word description with numbers in place of the words. Size, speed and positions at t = 0 and 1 s for constant velocity; size, height, vertical speed, drift, gravity and restitution for bounces.
- **Readouts (`track.readout`):**
  - `collide`: speed of the struck ball over the last 40% of the scored track, stored as `v`, so it also fills `hspeed`.
  - `ramp`: horizontal acceleration `a_slope` from a quadratic fit up to the fastest point.
  - H7 compares each level's own number: L1 `v`, L2 `g`, L3 `v` and `a_slope`.
- **Jobs (`run_pilot.ROUND3`):**
  - `render3sKof3`: the new input windows. Pod 1 also renders 6 full kl3 clips and runs SAM2 on them against ground truth (`results/p0_kl3_ltx_render3s1of3.csv`).
  - `kl3diag`: 15 full ramp renders scored the same way (`results/kl3diag_ltx_kl3diag.csv`). All 15 are measurable and the readout is about 2% low, after two tracker fixes for the tilted box: the scale in `px_per_m` and the size of the SAM2 prompt.
  - `{model}gKofN`: Cosmos 20 pods, VACE 11, LTX 2. Each pod runs one model, a slice of every experiment, then SAM2 scoring. They start after `render3`, because a pod pulls its inputs once.
  - `{model}rescore3`: CoTracker3 on every round 3 clip and the steer clips, one pod per model, about 3550 clips at 15 s. The VACE pod first runs P0 on the 21 full kl3 renders, because the kl3 readouts were only validated with SAM2.
  - `rederr`: rescores, with both trackers, the 153 clips in `configs/rederr.txt`. Their score files held `ValueError: Initial guess is outside of provided bounds` from `track.fit_bounce`, which crashed when a track started below the floor. Mostly text-only Cosmos and VACE clips, so their measurable share was understated. `pilot.score` now treats such a file as not scored.
- **Hosts:** measured s/clip, 3090 vs 4090: Cosmos 619 vs 316, VACE 856 vs 389, LTX 69. At $0.25 and $0.43/h the 4090 costs the same or less per clip, so Cosmos and VACE fill the 4090s first. LTX and renders use 3090s.
- **Next:** `rescore3` (CoTracker3 on the new clips), the probes on the larger B1 + B5 sets, `stats` with `--v2`, then C2-g and C2. C2 needs new velocity-scaled renders and is priced after generation.

## After round 3: nested gap with CoTracker3, C2 and C2-g (2026-11-03, user approved, credit $44.52)

- **`stats2v2`** (one RTX 3090, CPU-bound): `pilot.analysis nested --v2` for the three models on the round 3 B1 + B5 sets. Output `results/c3_{model}_v2_nested*.csv`.
- **C2 and C2-g** (`pilot/likelihood.py`, deviation D5):
  - Scene sets `kvelc` (Cosmos window) and `kvell` (LTX window): each kmain scene with its velocity scaled by 0.5, 0.8, 1, 1.25 and 2 after the input window.
  - C2-g reuses the 70 `kcont` renders with 16 noise levels instead of the pilot's 8.
  - Jobs `c2cKof4` (Cosmos, RTX 4090): render 25 bases x 5 copies, C2-g on 2-3 kcont bases, then C2. Jobs `c2lKof2` (LTX, RTX 3090): render 50 bases x 5 copies, then C2.
  - Rows are saved after every base (`results/c2{,g}_{set}_{model}_ltx_{job}.csv`), so a relaunched pod resumes. Summaries per base are in `*_summary.csv`: `k_implied`, `sym_mra` and `pref_acc`.
  - Estimates: Cosmos about 140 s per copy at 32 forward passes, LTX about 25 s. These are guesses from the generation timings, to be replaced by the measured `wall_s`.
  - Cost estimate: about $13 for the Cosmos pods, $1.5 for LTX and $1 for `stats2v2`.

### Results (2026-11-04, all pods verified on HF; `python scripts/c2_report.py`)

- **Cost:** $16.62 in all, credit $44.52 to $27.90. Measured: Cosmos about 4.2 s per forward pass (about 670 s per base at 5 copies x 32 passes), LTX about 128 s per base.
- **`stats2v2`:** the nested-CV gap matches the best-of-all gap in every cell (LTX hspeed .227 vs .223, g .460 vs .484, e .189 vs .193; Cosmos .414/.424, .601/.601, .218/.220; VACE .459/.473, .746/.751, .321/.329; Holm p about 0 for all). Choosing the block and step on all data does not inflate the gap.
- **C2 (100 bases per model, 4 altered copies each).** `pref_acc` is the share of altered copies whose loss is above the true copy's (chance 0.5).

| | Cosmos (kvelc) | LTX (kvell) |
|---|---|---|
| pref_acc [95% CI over bases] | 0.688 [0.642, 0.733] | 0.733 [0.677, 0.785] |
| k = 0.5 / 0.8 (slower) | 0.40 / 0.40 | 0.54 / 0.53 |
| k = 1.25 / 2 (faster) | 0.95 / 1.00 | 0.88 / 0.98 |
| constant velocity (cv) / bounce (ca) | 0.87 / 0.505 | 0.975 / 0.49 |
| in-frame copies only | 0.629 (337 of 400) | 0.712 (371 of 400) |
| loss(0.8) < loss(1.25) | 97% | 94% |
| loss(0.5) < loss(2) | 100% | 99% |
| lowest-loss k: 0.5 / 0.8 / 1 / 1.25 | 54 / 17 / 29 / 0 | 38 / 14 / 47 / 1 |
| k_implied median | 0.570 | 0.800 |

- **C2-g (Cosmos, 10 kcont bases, 16 noise levels x 4 draws):** pref_acc 0.583 [0.467, 0.700]. Win share by gravity multiplier: 0.25 1.0, 0.5 0.9, 0.75 0.7, 1.5 0.6, 2 0.2, 4 0.1. The lowest loss is at 4 g in 6 bases, 2 g in 3 and 1.5 g in 1, never at the true g. k_implied median 2.645 (about 26 m/s²), symmetric MRA 0.188. The g probe on the same model reads 0.90 MRA.
- **Reading:**
  - Both models' likelihoods reject faster copies almost always, and are at or below chance on slower ones. The lowest loss sits at a slower speed than the truth in most bases (Cosmos 71, LTX 52).
  - Above chance comes only from constant-velocity scenes. On bounces both models are at chance.
  - The Cosmos likelihood favours stronger gravity, while its activations hold g well (probe g obj +.317 over control). This is the proposal's "know but do not show" pattern measured on likelihood.
  - A common cause is plausible but untested: a prior for less motion. Slower copies move less, and with stronger gravity the ball lands and settles sooner. A motion-matched comparison (copies with equal pixel displacement) would test it.
- **Correction:** LTX in-frame-only pref_acc is 0.712 with `scripts/c2_report.py`; earlier notes said 0.72 from an ad hoc count.

### Follow-ups (2026-11-04, user approved A, B, C; deviation D6)

- **A. LTX C2-g** on `kcontl`: the kcont scenes with the switch at LTX's window, 16 noise levels.
- **B. C2-turn** on `kturnc` (Cosmos) and `kturnl` (LTX): the same speed, reversed or turned 30 degrees up, so the amount of motion matches the true copy. 250 copies per model.
- **C. VACE C2** on `kvelv` with `pilot.likelihood.VaceLoss`.
- **Jobs (`run_pilot.C3_PODS`):** `c3vKof4` (VACE, RTX 4090), `c3cKof3` (Cosmos, RTX 4090), `c3lKof2` (LTX, RTX 3090: kcontl and kturnl). Each renders its own bases first. Results `results/{c2,c2g}_{set}_{model}_ltx_{job}.csv`.
- **Cost estimate:** VACE about $9, Cosmos about $5, LTX about $1.5, from about 125 s per VACE copy (3.9 s per pass on a 4090 in generation), 134 s per Cosmos copy and 25 s per LTX copy.

### Follow-up results (2026-11-05, all 9 pods verified on HF; `python scripts/c2_report.py`)

- **Cost:** $27.93, the whole remaining credit (balance now -$0.03). About $10-11 of it was waste: pod `c3v3of4` (Japan, machine 103782) verified its data at 13:43 UTC, then `vastai destroy` from the pod failed with `401: Unable to authorize instance API key`, which the CLI reports with exit code 0. The pod sat idle until the credit ran out. Fixed in `finalize.stop_pod` (reads the output and retries 6 times over 12 min); the host is in `BAD_MACHINES`. A pod still listed after its `final_check` must be destroyed from the laptop.
- **A. LTX C2-g (`kcontl`, 10 bases, 16 x 4 passes):** pref_acc 0.583 [0.40, 0.77]; win share by gravity multiplier 0.25 0.8, 0.5 0.7, 0.75 0.7, 1.5 0.4, 2 0.5, 4 0.4. Lowest loss at 1.5 g in 4 bases, 2 g in 2, the true g in 2. k_implied median 1.82, symmetric MRA 0.344. Like Cosmos (2.65), LTX favours stronger gravity, less strongly.
- **B. C2-turn (100 bases per model, same speed after the turn):**

| | Cosmos (kturnc) | LTX (kturnl) |
|---|---|---|
| pref_acc [95% CI] | 0.860 [0.79, 0.93] | 0.735 [0.645, 0.815] |
| cv reversed / cv turned 30 degrees up | 1.00 / 1.00 | 1.00 / 0.98 |
| bounce reversed | 0.72 | 0.48 |
| true copy has the lowest loss | 86 | 73 |

- **C. VACE C2 (`kvelv`, 100 bases):** pref_acc 0.517 [0.487, 0.547]; slower copies 0.07 (k = 0.5) and 0.12 (0.8), faster 0.89 (1.25) and 0.99 (2); cv 0.545, bounce 0.49; in-frame only 0.480. Lowest loss at k = 0.5 in 80 bases, 0.8 in 15, the true speed in 4. k_implied median 0.500, symmetric MRA 0.333. VACE has the strongest slow-motion preference of the three.
- **Reading:**
  - A plain preference for less motion does not explain C2 by itself. At matched speed, both likelihoods reject wrong directions on constant-velocity scenes every time. Cosmos also rejects reversed bounces (0.72); LTX does not (0.48).
  - The speed bias is real in all three models: true speed has the lowest loss in 29 (Cosmos), 47 (LTX) and 4 (VACE) of 100 bases. VACE's likelihood prefers half speed almost always.
  - Both gravity tests favour stronger than true gravity (Cosmos 2.65 g, LTX 1.82 g).
  - So the likelihood holds direction and continuity, but not the size of speed or gravity; the probes read g well from the same models. Speed and gravity are known but not shown by the likelihood.

## Scale study (Cosmos-14B, H6) and V-JEPA 2 (2026-11-05, user approved; credit $149.96; deviation D7)

- **V-JEPA 2** (`pilot/vjepa.py`, job `vjepa2`, one RTX 4090, about $3-5): ViT-L and ViT-g on existing renders. C1 acts on the B1 / B5 input windows and the usual probes; C2 / C2-g surprise on kvelv, kturnl and kcontl (`results/c2{,g}_{set}_{model}_ltx_vjepa2.csv`, same layout as the generators' C2); C3 gamma per layer (`results/c3_{model}_gamma_ltx_vjepa2.csv`). No pixels, so no show score and no gap.
- **Cosmos-14B** (jobs `cosmos14bgKof6`, 48 GB cards, about 1,100 s per clip on an L40S): B1 100 kmain x 2 seeds and B5 90 kgrav x 2 seeds, scored with SAM2 and CoTracker3, then `cosmos14bprobe` (probes + nested gap with `--v2`). About $92. H6: gap(14B) - gap(2B) on the same scenes, scene-bootstrap CI inside +-0.05.
- **B2-nl on 14B** (jobs `cosmos14btKof3`, seed 0, 100 clips, about $22-35): first gated to run after `cosmos14bg*`; the user then approved running it in parallel (2026-11-05), since the projected total (about $130 of $149.96) keeps the $10 reserve. Shards t1 and t2 on RTX PRO 6000 cards, t3 when a card frees up; stopped if a clip costs more than $0.35.
- **Deferred (user, 2026-11-05): B2-nl seed 1 on Cosmos-14B,** about $22, when there is more budget.

### V-JEPA 2 results (2026-11-05, jobs `vjepa2` and `vjepa2s`, both verified on HF, about $0.90)

- **C1, know** (best block, object tokens, margin over the patch-embedding baseline and the control task):

| | speed | gravity | restitution |
|---|---|---|---|
| ViT-L | 0.738 (+0.097, layer 23) | 0.828 (+0.137, layer 18) | 0.960 (none) |
| ViT-g | 0.803 (+0.164, layer 37) | 0.783 (+0.091, layer 33) | 0.963 (none) |

  A predictor that never makes pixels encodes speed and gravity about as well as the generators (speed margins 0.13-0.19, gravity LTX +0.07 and Cosmos +0.10). Its predicted future tokens, pooled over the frame, carry neither (below the control task). Pooled over all tokens, the encoder carries little: speed margin below 0.
- **C3, gamma per layer** (out-of-fold probe on kgrav, no tracker involved): ViT-L 0.79 at layer 18, falling to 0.61 at the last layer; ViT-g rising to 0.81 at the last layer (object tokens). H3 asks for a drop of at least 0.2 by the last block: ViT-L drops 0.18, ViT-g not at all.
- **C2 / C2-g, surprise** (pref_acc, chance 0.5; frame stride 1 is the render rate, 16 fps for kvelv and 24 fps for the others):

| set | ViT-L stride 1 / 2 / 3 / 4 | ViT-g stride 1 / 2 / 3 / 4 |
|---|---|---|
| kvelv (speed x 0.5-2) | 0.515 / 0.515 / 0.432 / 0.393 | 0.527 / 0.500 / 0.458 / 0.438 |
| kturnl (same speed, turned) | 0.465 / 0.490 / 0.565 / 0.505 | 0.465 / 0.550 / 0.570 / 0.620 [0.53, 0.71] |
| kcontl (gravity, 10 bases) | 0.300 / 0.450 / 0.400 / 0.450 | 0.567 / 0.500 / 0.450 / 0.450 |

  - Surprise does not tell the true speed or gravity from the altered copies at any stride. On kturnl it flags the 30-degree turn on constant-velocity scenes (0.66-0.82: an object rising off the floor) but not a reversal (cv 0.28-0.54, bounce 0.42-0.62). Only ViT-g at stride 4 clears chance overall.
  - At the render rate the reversed copy had the lowest surprise in 54-55 of 100 bases; with sparser frames this moves back to chance, so it was not a physics preference.
- **Reading:** V-JEPA 2 shows the know-without-use pattern more strongly than the generators. Its encoder reads speed and gravity, but its own prediction error barely depends on them, while the generators' likelihoods at least reject wrong directions. Caveats: frames are squashed to a square, 10 gravity bases only, and the context is 1 s (V-JEPA 2's own intuitive-physics evaluation uses longer clips).
- Order: `cosmos14bg1of6` first as the smoke test (one clip with activations, VRAM and time per clip), with `vjepa2` in parallel; then the other five 14B pods.

## Cosmos 3 Nano (H8) on the RunPod team account (2026-11-06, user approved)

- **Why RunPod:** the vast credit is nearly spent; the user added a key for a new, well-funded RunPod team account.
- **Jobs** (`run_pilot`): `cosmos3_nanosmoke` (2 B1 clips, both trackers, must end itself), then `cosmos3_nanogKof8` (B1 100 kmain x 2 seeds, B5 90 kgrav x 2 seeds), `cosmos3_nanotKof2` (B2-nl seed 0), `cosmos3_nanoprobe` (probes + nested gap). Measured on the timing clips: 248 s per clip on an RTX A6000, 31.3 GB peak, activations captured (`layers/obj`, 36 x 3 x 4096). About 480 clips, about $15-25 on A40s at $0.35/h.
- **Pods** (`scripts/runpod_launch.py`): the image is unchanged; a start command sets up sshd, clones the repo and runs `scripts/ltx_pod.sh`. HF_TOKEN is the account's RunPod secret. The launcher only terminates pods named `kvs-*` that it created, and terminates a pod at once if it costs more than `--max-price`.
- **Ending pods:** `finalize.runpod_stop` deletes the pod with RunPod's pod-scoped `RUNPOD_API_KEY` once HF has every file (else stops it), checking the HTTP status, 6 tries. Backstop: `scripts/rpmon.py` on the cheapest vast box (`vast_launch.py rpmon`, about $0.01/h) terminates kvs-* RunPod pods whose final check is verified or whose hard cap (budget + 3 h, in `data/logs/runpod_caps.json`) passed, and destroys verified vast workers still listed. It holds a separate RunPod key that the user revokes after the run.

### H8 result (2026-11-06)

All ten generation shards and `cosmos3_nanoprobe` finished on RunPod, verified on HF, and the pods ended themselves (GraphQL `podTerminate`). H8 says Cosmos 3 Nano's gap is at most half the mean gap of the other three generators. Below, the gap is the nested-CV know-show gap on the CoTracker3 readout (`results/c3_{model}_v2_nested.csv`, `nested` rows; scene-bootstrap 95% CI).

| Target | Nano gap [95% CI] | Cosmos-2B / LTX-2B / VACE-1.3B | Threshold (half the mean) | H8 | Nano detail |
|---|---|---|---|---|---|
| hspeed | 0.298 [0.213, 0.379] | 0.414 / 0.227 / 0.459 | 0.183 | FAIL | know 0.688, show 0.445, 57 scenes |
| g | 0.442 [0.331, 0.550] | 0.601 / 0.460 / 0.746 | 0.301 | FAIL | know 0.843, show 0.385, 30 scenes |
| e | 0.426 [0.312, 0.538] | 0.218 / 0.189 / 0.321 | 0.121 | FAIL | know 0.962, show 0.483, 30 scenes |

**H8 fails on every target, and the CI's lower bound is above the threshold each time.** Nano's gap on speed and gravity is smaller than Cosmos-2B's and VACE's, but not by half. Part of the smaller speed gap is a weaker probe: Nano knows speed less well (know 0.688 vs 0.75-0.82), while it shows speed and gravity somewhat better (show 0.445 and 0.385, the best of the four). On restitution its gap is the largest of the four, because its bounces are poor (show 0.483). Fewer scenes are measurable for Nano (57 for speed, 30 for g and e), so its CIs are wider.

## G4 re-fit and C2 on Nano and VACE (2026-11-06, deviations D8, D9)

All shards below finished on RunPod, verified on HF, and the pods ended themselves.

### G4 with 8 B5 seeds

B5 seeds 5-7 (jobs `ltx2bhKof2`, `cosmos2bhKof6`) bring LTX-2B and Cosmos-2B to 8 seeds on the 90 kgrav scenes, which is G4's own fail action. The gamma fit is `pilot.analysis gates` on all 720 B5 clips per model; the CI is a scene bootstrap.

| Model | Readout | Before (seeds 0-4) | After (seeds 0-7) | G4 (width <= 0.5) |
|---|---|---|---|---|
| LTX-2B | SAM2 | gamma 0.11 [-0.20, 0.42], width 0.62, 201 clips | gamma 0.15 [-0.12, 0.40], width 0.53, 295 clips | FAIL |
| LTX-2B | CoTracker3 | gamma 0.04 [-1.09, 0.63], width 1.73, 14 clips | gamma 0.15 [-0.38, 0.50], width 0.88, 102 clips | FAIL |
| Cosmos-2B | SAM2 | gamma 0.80 [0.33, 1.25], width 0.92, 179 clips | gamma 0.92 [0.50, 1.29], width 0.79, 241 clips | FAIL |
| Cosmos-2B | CoTracker3 | gamma 1.22 [0.09, 2.34], width 2.25, 30 clips | gamma 0.89 [0.59, 1.21], width 0.62, 267 clips | FAIL |

**G4 still fails on both models.** The CIs narrow by 15-70%, but the scene bootstrap is limited by the 90 scenes, not the seeds. The estimates are stable, though. LTX's gamma is near 0.15 with a CI that excludes 1: its fall barely tracks the scene's gravity. Cosmos-2B's gamma is near 0.9 with a CI that includes 1. The other fail action, a 0.25-4 g gravity range, was not run.

### C2 on Nano and VACE (`python scripts/c2_report.py`)

pref_acc is the share of altered copies whose denoising loss is above the true clip's (0.5 = chance); bootstrap CI over bases.

| Test | Model | pref_acc [95% CI] | Bases | Preferred copy (argmin) | Other models |
|---|---|---|---|---|---|
| C2 velocity (kveln) | Nano | 0.593 [0.555, 0.632] | 100 | k 0.5: 59, 0.8: 28, 1.0: 12 | LTX 0.733, Cosmos-2B 0.688, VACE 0.517 |
| C2 velocity (kvelc) | Cosmos-14B | 0.668 [0.587, 0.745] | 52 | k 0.5: 21, 0.8: 6, 1.0: 18, 1.25: 7 | Cosmos-2B on the same 52 bases 0.702; 14B - 2B = -0.034 [-0.101, 0.034] |
| C2-turn (kturnv) | VACE | 0.825 [0.745, 0.895] | 100 | no turn: 82, 180 deg: 17 | Cosmos-2B 0.860, LTX 0.735, V-JEPA 0.47-0.62 |
| C2-g (kcontv) | VACE | 0.533 [0.383, 0.683] | 10 | k >= 2: 7 of 10 (k_implied median 2.40) | Cosmos-2B 0.583, LTX 0.583 |

- **Nano** is above chance on velocity but below both 2B generators. As with Cosmos-2B (k_implied 0.570), its likelihood prefers slow motion: k_implied median 0.584, and slower copies beat the true clip on 75-78% of bases, while faster copies lose 90-100% of the time. It scores 0.68 on constant-velocity bases and 0.505 on constant-acceleration ones.
- **Cosmos-14B** (shards 1-6 of 12 of kvelc, jobs `c3bKof6`) scores the same as Cosmos-2B on the same bases: 7x the parameters buys no better velocity likelihood. Its slow bias is weaker (k_implied median 0.736 vs 0.570; 18 of 52 bases prefer the true clip), and it splits the families the same way (constant velocity 0.846, constant acceleration 0.49). Every shard was OOM-killed once at the A40 pod's 50 GB RAM and relaunched with the prompts encoded first and the text encoder freed (7aa6239); rows already written were kept.
- **VACE** detects turns as well as Cosmos-2B (constant velocity 0.99, constant acceleration 0.66). On gravity it is at chance and prefers stronger-than-true gravity, as Cosmos-2B does (k_implied 2.65); 10 bases make this loose.
- **Across generators,** the likelihood pulls velocity slow (preferred k about 0.5-0.8) and gravity strong (preferred k about 2-2.6), and detects turns well.

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
