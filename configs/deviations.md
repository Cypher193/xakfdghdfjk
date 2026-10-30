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
