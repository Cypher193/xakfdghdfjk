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

## B-ratio: why only 5/30 clips are measurable (CPU, read-only diagnosis first)
