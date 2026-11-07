# Deviations from the pre-registered gates

Each entry: what changed, when, whether the data had been seen, and why. Thresholds in `gates.yaml` are not edited.

## D1 — G1a real-footage check scored on speeds only (2026-10-17, after seeing data)

- **Rule as registered:** symmetric MRA ≥ 0.80 against ground truth on the main renders and on QuantiPhy val.
- **Result:**
  - Renders: 0.947, passed.
  - QuantiPhy val (2D clips, 7 questions): 0.696, failed.
    - The 4 speed questions score 0.94–1.00 (mean ≈ 0.98).
    - The 3 "acceleration at time t" questions score 0.00–0.63.
- **Change:** the QuantiPhy part of G1a is scored on the speed questions only, and passes at ≈ 0.98.
- **Why:**
  - The local-acceleration questions ask for very small accelerations (0.34 m/s², 11 cm/s²) at a single instant. At that scale, 1–2 px of tracking noise dominates.
  - The pilot never reads accelerations that way. Gravity comes from a fit of the bouncing-ball law over the whole track, which recovers g at 0.99 on the renders.
  - The QuantiPhy acceleration questions therefore test a quantity the pilot does not use.
- **Approved by:** the user, after seeing the data, on 2026-10-17.
- **Also changed before scoring:**
  - The QuantiPhy clips first chosen (video type S3) move toward and away from the camera, and their labels are 3D speeds. They were replaced with 2D-motion clips (simulation_0163, internet_0027).
  - The tracker's metre scale now comes from min(mask width, height) instead of mask area.
  - Both changes fix measurement errors and do not alter any threshold.

## D2 — B-ratio rerun with bigger balls (2026-10-26, after seeing data)

- **Result that prompted it:** only 5/30 LTX B-ratio clips were measurable, which fails G1b on its own.
- **Diagnosis:** the kratio balls are 0.25–0.40 m at the 9 m ratio camera, i.e. 25–41 px wide. One LTX latent cell is 32 px, so LTX smears or drops the balls.
- **Change:** a new scene set `kratio2` with 0.7–0.9 m balls (71–92 px) and throws capped at 8.5 m/s, so the whole ball stays in frame. It is run as experiment `bratio2`. The old `kratio`/`bratio` data are kept, and G1b uses `bratio2` once it exists.
- **What is not changed:** the camera, the throw law, the tracker and every threshold.
- **Side effect:** the speed cap narrows the throw-speed ratio to about 1.33–1.43 (was 1.3–1.5).
- **Approved by:** the user, on 2026-10-26.
- **Outcome:** still 5/30 measurable. The diagnosis was wrong: the bigger balls are clearly visible, but LTX breaks them.
  - Balls split in two, change colour (red to white), shrink as if moving away from the camera, and fly off the top of the frame because they stay airborne far too long.
  - These failures trip the area rule (0.5–2× the frame-0 area) of the measurability gate. They are generator failures, not tracker failures.
  - Among the 5 measurable clips, median sym MRA is t_ratio 0.69 and h_ratio 0.81 (was 0.56 and 0.75).

## D3 — G1b fail action: CoTracker3 readout as a second tracker (2026-10-30, after seeing data)

- **Result that prompted it:** G1b (measurability ≥ 0.70 per model) fails for all three generators. B5 (kgrav): LTX-2B 0.41 (150 clips, 5 seeds), Cosmos-2B 0.40, VACE-1.3B 0.13 (30 clips each). B1: LTX 0.72, Cosmos 0.72, VACE 0.56. bratio2: LTX 0.17.
- **Registered fail action:** "add CoTracker3 or stability filter; drop hard scenes; rerun P0 and P12".
- **What the data had shown before this entry:**
  - score.json does not record which part of the rule failed.
  - Contact sheets of 12 failing B5 seed-0 clips (4 per model) show 11 generator failures. The ball grows to 2-3x as it comes toward the camera (Cosmos 3/4, VACE 1/4). It fades out or vanishes (LTX 2/4, VACE 1/4). It shrinks to a dot (VACE 1/4). It flies off the top of the frame (VACE 1/4). The twelfth (LTX `kgrav_06_g1.0`) keeps a constant-size ball that rises to the top edge, so it could be a tracker or an off-frame failure.
  - Failures are not concentrated in a few scenes. 4 of 30 kgrav scenes are never measurable, and the per-scene measurability of the three models correlates at only 0.07-0.48. Dropping hard scenes would therefore not reach 0.70, and choosing them now would be choosing on the outcome. That part of the fail action is not taken.
- **Change (opt-in, nothing registered is replaced):**
  - `pilot.score --tracker cotracker3` re-reads every clip with CoTracker3 (offline, torch.hub `facebookresearch/co-tracker`). It tracks a grid of up to 64 points seeded inside the frame-0 SAM2 mask (`track.cotracker_tracks`).
  - Centroid: the mean of the visible points.
  - Area for the measurability rule: the frame-0 mask area × (convex hull of the visible points / its frame-0 hull).
  - A point counts as visible only if CoTracker marks it visible and its pixel is within 30 CIELAB units of its own frame-0 colour. This guard is a precaution, so that points left on the background where a ball faded out cannot count as the ball. On the 4 clips tried on the laptop it changed nothing, because CoTracker had already marked those points invisible.
  - The metre scale still comes from the SAM2 masks of the input frames.
  - Output goes to `score_v2.json` beside `score.json`, which is never rewritten. `pilot.analysis g1b` and `--v2` read it.
  - The same pass records per-object SAM2 mask faults (`track.mask_faults`: vanished, off-frame, edge, split, recoloured, shrunk, grown). This gives the rule-level breakdown that score.json lacks, and the D2 B-ratio counts.
- **What is not changed:** the measurability rule and its numbers (≥ 80% of scored frames present, area 0.5-2x frame 0), every threshold in `gates.yaml`, the scenes and the clips.
- **P0 re-check (G1a):** the new readout must reach sym MRA ≥ 0.80 on the 100 full kmain renders (`results/p0_renders_cotracker3_ltx_rescore.csv`). SAM2 runs on the same renders for comparison (`p0_renders_sam2_ltx_rescore.csv`). P12 is not rerun here.
- **Laptop check of the new readout (CPU, seeded from the ground-truth disk on frame 0):**
  - On a kmain render (48 frames) the median centroid error is 0.42 px and the area stays at 0.86-1.17x.
  - 4 failing B5 clips (LTX `kgrav_08_g2.0`, VACE `kgrav_02_g2.0`, Cosmos `kgrav_03_g0.5`, LTX `kgrav_06_g1.0`) all stay not measurable. The share of scored frames in the area band is 0.32, 0.09, 0.06 and 0.45.
- **Expected outcome, stated before the run:** G1b should change little. Most failures in the sample are what the generator drew, and an honest second tracker has to reject those too. If CoTracker3 agrees with SAM2, G1b's failure is a show result (the generators break the object) rather than a measurement failure.
- **Approved by:** the user, on 2026-10-30, after the rescore run: report SAM2 (primary) and CoTracker3 side by side, with the fault breakdown.
- **Outcome (job `rescore`, 2026-10-30):** P0 passes with both trackers on the 100 full renders (sym MRA: SAM2 0.96, CoTracker3 0.94). CoTracker3 does not rescue G1b. Measurability SAM2 -> CoTracker3, same clips: B1 Cosmos 0.70 -> 0.71, LTX 0.72 -> 0.56, VACE 0.55 -> 0.46; B5 Cosmos 0.40 -> 0.47, LTX 0.41 -> 0.09, VACE 0.13 -> 0.17. No model reaches 0.70 on B5 with either. CoTracker3 rejects 96 LTX clips SAM2 accepts (it accepts 2 SAM2 rejects); why is not yet checked. The dominant fault in unmeasurable clips is a ball that grows (Cosmos), vanishes (LTX) or a mix (VACE). G1b stays FAIL.

## D4 — B-L3 and B2-num definitions (2026-11-01, before any B-L3 or B2-num clip)

- **Proposal v5:** B-L3 uses "scenes that combine two laws", and an earlier draft named collision, slope with friction and projectile. B2-num uses "a matching description in numbers".
- **Change:**
  - `kl3` holds 15 collisions (momentum + restitution) and 15 ramps (gravity + sliding friction), rendered in Blender rather than Kubric like every other scene set. Projectiles are left out: a thrown ball that bounces is the same physics as the L2 `ca` family.
  - In a collision the contact falls at 1.3-2 s, after every input window, so the input shows only the approach. The scored number is the struck ball's speed.
  - On a ramp the box reaches the foot at t >= 1.6 s, also after every input window. The scored number is the horizontal acceleration on the slope.
  - B2-num gives the size, speed or height, gravity and restitution directly, plus positions at t = 0 and 1 s for constant velocity. It does not list positions per frame, because frame rates differ per model.
- **What is not changed:** gates, thresholds, the metric, the measurability rule and the tracker.
- **New P0:** the `collide` and `ramp` readouts are new, so `results/p0_kl3_ltx_render3s1of3.csv` checks them against ground truth on 6 full renders before generation starts (G1a threshold, sym MRA >= 0.80).
- **Approved by:** the user approved the experiment list on 2026-11-01. The scene families are my choice within it.

## D5 — C2 and C2-g setup (2026-11-03, before any C2 or C2-g run)

- **Proposal v5:** C2 is the denoising loss of the real continuation against copies with the velocity scaled by 0.5, 0.8, 1.25 and 2 (480 clips, 120 scenes), on Cosmos-2B and LTX. C2-g is the denoising loss of Kubric continuations at 7 gravity values.
- **Change:**
  - The C2 scenes are the 100 kmain scenes, not 120. Each one gets copies with the velocity (both components for a bouncing ball) multiplied by k from the end of the model's input window on, at 0.3125 s for Cosmos (`kvelc`) and 1.04 s for LTX (`kvell`). The true k = 1 copy is rendered with the others in the same pod, so render version and noise cannot separate it from them. That makes 500 clips per model.
  - Fast copies may leave the frame. At k = 2, 32 of 50 cv and 23 of 50 bounce scenes leave it at some point within the Cosmos clip. The share of frames in frame is stored per copy (`gt.in_frame`), and the analysis reports the preference both on all copies and on the copies that stay in frame.
  - C2-g uses the 70 existing Blender `kcont` renders, not Kubric, like every other scene set.
  - Loss: 8 noise levels evenly spread over the model's sampling schedule x 4 noise draws for C2, and 16 x 4 for C2-g ("more noise levels"). The draws are the same for every variant of a base. The input-window latents stay clean, as each model's own conditioning does.
  - Per base: the implied multiplier (minimum of a parabola in ln k), its symmetric MRA against 1, and the share of altered copies whose loss is higher than the true copy's (`pref_acc`, chance 0.5).
- **What is not changed:** gates, thresholds and the scenes of every other experiment. Every earlier scene file is byte-identical (`tests/test_round3.py`, checked by regenerating them).
- **Approved by:** the user approved running C2 and C2-g on 2026-11-03. The scene choice and loss settings are mine.

## D6 — C2 follow-ups: VACE C2, C2-turn, LTX C2-g (2026-11-04, after seeing the C2 results)

- **Why:** C2 on Cosmos and LTX rejects faster copies and accepts slower ones, and the Cosmos C2-g minimum sits at 2-4 g. One explanation is a preference for less motion. These runs test that and add the third model.
- **Change:**
  - VACE C2 (`kvelv`): the kmain scenes with the velocity scaled from the end of VACE's window (17 frames, 1.06 s). VACE's own conditioning noises every latent frame and gives the window through the VACE context (window frames, then grey, mask 0 on the window), so the loss does too; it still counts only the continuation frames.
  - C2-turn (`kturnc` Cosmos, `kturnl` LTX): at the end of the window the velocity turns by 180 degrees (reversed) or 30 degrees upward, at the same speed. Floor objects then glide in a straight line and balls stay ballistic. The object starts so that it is at the frame centre at the turn, so the reversed copy has as much room as the true one; this moves the start away from the kmain scenes, so the true copy (0 degrees) is rendered again. A 30-degree turn throws a falling ball sideways out of frame (31% of frames in frame for Cosmos), so bounces have only the 0 and 180 degree copies. Reading: if the likelihood prefers the true copy over a reversed copy with the same speed, it is not only a preference for less motion.
  - LTX C2-g (`kcontl`): the 70 kcont scenes with the gravity switch moved to the end of LTX's window (1.04 s), 16 noise levels x 4 draws as for Cosmos.
  - The implied multiplier and its MRA are not defined for the turn sets; only `pref_acc` is reported.
- **What is not changed:** every earlier scene file is byte-identical (regenerated and compared), and the loss settings of D5.
- **Approved by:** the user approved runs A (LTX C2-g), B (motion-matched C2) and C (VACE C2) on 2026-11-04. The 30-degree-for-cv-only rule and the centring are mine.

## D7 — Scale study and V-JEPA 2 design (2026-11-05, before any 14B or V-JEPA 2 run)

- **Proposal v5:** Cosmos-14B runs B1 and B2-nl on 60 scenes x 3 samples (360 clips) for H6. V-JEPA 2 gets C1, C2 and C3.
- **Change:**
  - Cosmos-14B runs B1 on all 100 kmain scenes x 2 seeds and B5 on the 90 kgrav scenes x 2 seeds. H6 is an equivalence test (gap within 0.05 MRA of Cosmos-2B's), and the 2B gap's scene-bootstrap interval is already +-0.07-0.09 at 100 scenes, so 60 scenes could not show it; scenes matter more than seeds for that interval. B5 gives the gravity gap at scale.
  - B2-nl on 14B runs with seed 0 only (100 clips), and only if the credit allows after B1 and B5. **Seed 1 of B2-nl on 14B is deferred until there is more budget (user, 2026-11-05).**
  - V-JEPA 2 (ViT-L 256 and ViT-g 384) reads frames squashed to its square crop, not centre-cropped, because objects cross the full 832 px width. Its encoder sees only the context frames (the library's forward pass encodes the whole clip first). C2 uses surprise (L1 between predicted and encoded future tokens) on the rendered kvelv, kturnl and kcontl sets; C1 pools the context tokens per layer as the generators' acts, with the patch embeddings as the input-level baseline; C3 is the per-layer probe gamma on kgrav.
- **What is not changed:** scenes, probes, gates and thresholds.
- **Approved by:** the user approved V-JEPA 2, 14B B1 + B5 and one seed of 14B B2-nl on 2026-11-05. The scene and seed split is mine.

## D8: C2 on Cosmos-14B and Cosmos 3 Nano (2026-11-06)

- **What changes:** the proposal runs C2 on the three main generators only. The user added C2 for Cosmos-14B (50 bases) and Cosmos 3 Nano, run on the RunPod team account.
- **How:**
  - Cosmos-14B uses the Cosmos-2B loss code and the same kvelc renders, since its window and clip match. It runs shards 1-6 of 12 of the 100 kvelc bases, 52 bases in all.
  - Nano gets its own scene set, kveln: the same kmain scenes with the speed switch at the end of its 5-frame window at 24 fps (0.208 s). All 100 bases run. The loss (`pilot.likelihood.NanoLoss`) keeps the window's two latent frames clean and noises the rest, as the pipeline's denoising loop does. It uses the text-conditional pass only, the same 8 noise levels x 4 draws, and target v = eps - x0. A one-time sign check is printed on the pod.
- **What is not changed:** the copies (k in 0.5, 0.8, 1, 1.25, 2), the passes per copy and the summary.
- **Approved by:** the user, 2026-11-06.

## D9: G4 seeds, 14B B2-nl seed 1, VACE C2-g and C2-turn (2026-11-06)

- **What changes (user, 2026-11-06, RunPod):**
  - **G4:** B5 seeds 5-7 on LTX and Cosmos-2B, so B5 has 8 seeds (the gate's own fail rule). The gamma CI is a scene bootstrap, so more seeds may narrow it less than more scenes would.
  - **14B B2-nl seed 1:** deferred on 2026-11-05, now run.
  - **VACE C2-g and C2-turn:** the kcont and kturn scenes with the switch at VACE's window end (kcontv, kturnv; 1.0625 s), with `VaceLoss`.
- **How:** jobs `ltx2bhKof2`, `cosmos2bhKof6`, `cosmos14buKof6` and `c3wKof6`. They pull only their own shard's scenes, inputs and clips from HF (`run_pilot.NARROW_PULL`), after ten parallel setups that pulled about 2,700 files each hit HF's rate limit.
- **What is not changed:** the scenes of every existing set (regenerated byte-identical), the scoring and the gates.

## D10: Nano C2-turn and C2-g on vast (2026-11-06)

- **What changes (user, 2026-11-06: spend the vast credit down to about $8 on the smaller experiments):** C2-turn and C2-g on Cosmos 3 Nano, on its own window (5 frames at 24 fps, 0.208 s): new scene sets kturnn (250) and kcontn (70), made the same way as kturnv / kcontv. Every existing set regenerates byte-identical.
- **How:** jobs `c3mKof3` (render on the pod, then `pilot.likelihood` with `NanoLoss`), on vast 48 GB cards.
- **Then (user, 2026-11-07, vast):** Nano B2-nl seed 1 (jobs `cosmos3_nanouKof2`), so H2 at Nano has two text-only seeds like the other models.
- **What is not changed:** the copies (turn 0 / 30 / 180 degrees; gravity x0.25-4), the passes per copy, the summary (`scripts/c2_report.py`).

## D11: 14B C2-g and C2-turn, Nano B5 seeds 2-4 (2026-11-07)

- **What changes (user, 2026-11-07, RunPod):** Cosmos-14B C2-g on kcont (10 bases, jobs `c3gKof5`) and C2-turn on kturnc (100 bases, jobs `c3tKof8`), on the renders Cosmos-2B used (same window and clip, so no render step); Cosmos 3 Nano B5 seeds 2-4 on its 90 kgrav scenes (jobs `cosmos3_nanovKof5`).
- **Why:** complete the 14B column of the C2 tables next to the H6 scale result, and tighten Nano's gravity slope.
- **What is not changed:** the copies, noise levels and draws per copy, the scoring and the summaries.

## D12: replication on fresh scenes (2026-11-07)

- **What changes (user, 2026-11-07, RunPod):** the main (kmain, 100) and gravity (kgrav, 90) sets drawn again with seed 7 (`python -m pilot.scenes --fresh 7`) as kmainf / kgravf: same families, parameter ranges and counts, new draws. Experiments b1f / b5f (configs/pilot.yaml).
- **How:** jobs `freshKof10`: each shard renders its input windows, then LTX-2B (seeds 0-1) and Cosmos-2B (seed 0) generate B1 and B5, and SAM2 and CoTracker3 score. `freshprobe` (CPU pod): the same probes, then the nested-CV gap (`pilot.analysis nested --v2 --fresh`), written to `results/p6_{model}f_*` and `results/c3_{model}f_v2_*`.
- **Why:** every earlier result is on one scene set the pipeline was built on; this checks that the know-show gap replicates on scenes it never saw. VACE is not rerun (budget and deadline).
- **What is not changed:** the generators, windows, captions, trackers, probes, nested CV and the gap definition.
