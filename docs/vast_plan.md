# Full run on vast.ai: plan, parallelism and live pricing

How the ~790 A100-hour main study runs on vast.ai instances, as many in parallel as the market offers, at the lowest
total cost (compute + reserved disk + bandwidth). Prices on vast.ai change by the hour, so this document gives the
*method* and a live tool; the numbers below are a dated snapshot.

    python scripts/vast_quote.py                    # datacenter hosts only, on-demand: the default policy
    python scripts/vast_quote.py --hosts verified   # + verified hosts (individuals' machines): cheaper, bigger pool
    python scripts/vast_quote.py --bid              # interruptible prices (about half), instance can be pre-empted
    python scripts/vast_quote.py --pod-h 8 --top 5  # smaller instances = more of them in parallel

Run it right before launching. The account key stays on the laptop (`~/.vast_key`, read as `VAST_API_KEY`); the
CLI is `%APPDATA%\Python\Python313\Scripts\vastai.exe` (not on PATH). **The account balance is $0.00 (2026-10-24);
vast.ai is prepaid, so top up before anything is launched.**

## 1. Host policy: no personal GPUs

vast.ai has three kinds of host. *Unverified* hosts are excluded by the CLI's default query. *Verified* hosts
(`verified=true`, `hosting_type` 0) passed vast's benchmark but are mostly individuals' machines in homes and small
shops. *Datacenter* hosts (`datacenter=true`, `hosting_type` 1) are commercial facilities.

- Default: `datacenter=true`. Pool today: ~20 single-GPU offers that pass our filters.
- Fallback when the datacenter pool is too thin to parallelise (it is, today): `--hosts verified`, with
  `reliability > 0.98`, `duration > 2` days and `inet_down >= 400` Mb/s. That is still not the hobbyist tier, and
  everything is resumable, so a host that drops costs only the clips since the last 15 min push.
- Always: `cuda_vers >= 12.8` (the image is torch 2.8 cu128, needed for the Blackwell cards), `num_gpus=1`.

## 2. Work inventory

A100-hours from the pilot's measured medians at 480p (Cosmos-2B 243 s/clip, VACE-1.3B 284 s, LTX 27 s,
Cosmos-14B 834 s, Wan2.2-5B 43 s on A100 SXM); Nano re-estimated from the screen (248 s/clip on an A6000, ~124 s
A100-equivalent, 300 clips, plus the same again for its question passes). Other blocks keep the v5 estimates.

| Wave | Block | Clips | A100-h | Min VRAM | Min RAM | Disk (cache + outputs + 40) | Needs |
|---|---|---|---|---|---|---|---|
| 1 | Cosmos-Predict2.5-2B B-runs | 2,160 | 146 | 24 GB | 48 GB | 156 GB | renders on HF |
| 1 | Wan2.1-VACE-1.3B B-runs | 1,680 | 133 | 24 GB | 48 GB | 89 GB | renders |
| 1 | LTX-Video-2B B-runs | 2,160 | 16 | 24 GB | 48 GB | 96 GB | renders |
| 1 | Cosmos-Predict2.5-14B | 360 | 83 | 45 GB (CPU offload) or 80 GB | 64 GB | 106 GB | renders |
| 1 | Cosmos 3 Nano + questions | 300 | ~20 | 45 GB | 64 GB | 81 GB | renders |
| 1 | Wan2.2-TI2V-5B C1-inv / C1-noise | 200 | 3 | 24 GB | 48 GB | 76 GB | real clips |
| 2 | Tracking not done on the gen pods | — | ~20 | 16 GB | 32 GB | 44 GB | wave 1 videos |
| 2 | Probing C1–C6, V-JEPA 2, baselines | — | 80 | 24 GB | 64 GB | 75 GB | wave 1 `acts.npz` |
| 2 | B-ratio, C2-g likelihood, MDL, V-information | 360 | 39 | 24 GB | 48 GB | 121 GB | wave 1 |
| 3 | Causal C7–C9: steer, patch, LEACE (Cosmos-2B, LTX) | 2,020 | 152 | 24 GB | 48 GB | 162 GB | wave 2 probe directions |
| | **Total** | **~16,100** | **~692** | | | | |

(692 rather than 790: Nano measured at a fifth of the v5 guess. Keep the 10% rerun buffer: ~760.)

Model cache sizes are the HF repo sizes after each config's `ignore` patterns (Cosmos-2B 76 GB, VACE-1.3B 19,
LTX 26, Cosmos-14B 58, Nano 35, Wan2.2-5B 34). Outputs are small: a clip is ~3.4 MB of video plus ~0.6 MB of
captured activations, so the whole run adds ~65 GB to the HF dataset (private repos: 100 GB free, then $25/TB/month).

## 3. GPU classes and speed factors

Wall-time multiplier against the A100 SXM, the card the pilot was timed on. Measured: A100 1.00, L40S 1.27
(Cosmos-14B with offload). Everything else in `GPUS` in `scripts/vast_quote.py` is an estimate from published
bf16 throughput and memory bandwidth, and must be replaced before trusting a quote that depends on it:

| Class | VRAM | Factor | Status | Fit |
|---|---|---|---|---|
| A100 SXM4 80 / 40 | 80 / 40 | 1.00 | measured | everything / 2B-class only |
| L40S, RTX 6000Ada, 5880Ada | 45–48 | 1.27–1.35 | L40S measured | everything with offload |
| RTX A6000, A40 | 48 | 2.0–2.1 | estimate | everything with offload, slow |
| RTX PRO 6000 (WS, Max-Q, S), RTX 6000D | 84–96 | 0.7–1.0 | estimate | everything, no offload |
| RTX PRO 5000 | 48 | 1.2 | estimate | everything with offload |
| RTX 5090 | 32 | 1.0 | estimate | 2B-class + scorer |
| RTX 4090, RTX PRO 4000, 3090 | 24 | 1.3 / 2.0 / 2.0 | estimate | 2B-class, scorer on a separate pod |
| H100 / H200 / B200 | 80–180 | 0.3–0.55 | estimate | everything; rarely cheapest per A100-h |

Not used: Volta and Turing cards (V100, Quadro RTX 8000, Titan RTX: no bf16), CMP mining cards.

**Step 0 of the run is a calibration pass**: on each class the quote proposes, one instance for ~30 min runs
`python -m pilot.generate --model cosmos2b --exp b1 --limit 3 --no-capture` (and `vace13`, `ltx2b`; `cosmos14b`
where VRAM allows), the medians go into `GPUS`, and the quote is re-run. Cost: well under $1 per class.

## 4. Cost model (what the quote adds up)

For a block on a GPU class, at the offer that minimises the total:

- compute = (A100-h × factor + pods × (0.3 h setup + model download at the host's line speed)) × $/h
- storage = hours × reserved disk × storage_cost / 730 (disk is reserved at creation and billed whether used or
  not: size it per block, not 300 GB for everything; on a $0.36/h card a 300 GB disk at $0.6/GB/month is 20% of
  the bill)
- bandwidth = pods × model cache × inet_down_cost + outputs × inet_up_cost (a few dollars; a 39 $/TB host with a
  76 GB model download costs $3 per pod)
- pods = min(ceil(GPU-hours / `--pod-h`), offers available now)

Setup overhead is why instances should carry ≥ 8 h of work: at 10 h per pod the overhead is ~5%, at 5 h ~10%.

### Snapshot, 2026-10-24 16:50 (will be stale: re-run the quote)

| Policy | Sum of per-block best | Longest wave-1 instance | Best classes |
|---|---|---|---|
| Datacenter, on-demand | **$654** | 168 h (one A100 PCIE 40 GB offer; 4 RTX PRO 5000 at 44 h) | A100 PCIE $0.73, RTX PRO 5000 $0.81–0.92 |
| Verified, on-demand | **$379** | 27 h | A100 SXM4 40 GB $0.41 (9 offers), RTX 6000Ada $0.60, A6000 $0.40 |
| Verified, interruptible | **$249** | 28 h | A6000 $0.15, RTX PRO 5000 $0.16, A100 SXM4 $0.35 |

Add ~10% for reruns. Compare with RunPod: $1,098 all-A100, $699 all-L40S, $521 all-A6000 (8 Oct 2026 prices).

## 5. Parallel plan

Everything is sharded by scene: a shard is a (block, k of N) pair, and clips are skipped when their `meta.json`
exists, so a shard can be restarted or re-split at any time. Shards push to the shared HF dataset every 15 min
under disjoint paths (scene ids differ; tables and logs carry a `_full_{block}_{k}` tag like the screen tiers).

| Wave | What runs | Instances (at 10 h each) | Wall |
|---|---|---|---|
| 0 | calibration, one per candidate class | 3–5 × 0.5 h | 1 h |
| 1 | the six generation blocks, each split into N = ceil(GPU-h / 10) shards | ~40 shard-instances; bounded by offers (today 20–40) | 10–30 h |
| 2 | probing (sharded by model × target), tracking leftovers, extra metrics | ~15 | 10 h |
| 3 | causal C7–C9 (sharded by model × scene) | ~16 | 10–15 h |

Critical path ≈ 2 days wall with 20–40 instances alive at once, against 33 days on one A100. Waves 2 and 3 pull
their inputs (videos, `acts.npz`, probe directions) from HF, so no instance has to outlive its own shard.

On 24 GB cards the SAM2 scorer does not share the GPU with the generator (as on the L40S with the 14B models):
`run_pilot` must start `pilot.score --loop` only when VRAM ≥ 32 GB, and the tracking block in wave 2 covers the
rest. On 32–48 GB cards it runs alongside as in the pilot and the wave-2 tracking block shrinks.

Interruptible instances: generation is resumable and `guard.py` pushes on any crash, but a pre-empted instance is
*stopped*, not destroyed, and keeps billing its disk until it is restarted or destroyed. Use `--bid` only with a
laptop-side watcher that restarts or replaces stopped shards; on-demand is the default.

## 6. Code still needed (in order)

1. `pilot.generate --shard k/N`: `scenes = scenes[k::N]` after `--limit`. One line.
2. `scripts/run_pilot.py --plan full --block NAME --shard k/N`: a `FULL` dict of blocks → `gen(...)` steps, the
   shard passed through, tag `_full_{block}_{k}` for log, budget and table (reuse `tag()` / `score_args()`), scorer
   gated on VRAM. `sync.py` reads a `FULL_SHARD` env var the way it reads `SCREEN_TIER`.
3. `scripts/full_pod.sh BLOCK K N BUDGET`: `screen_pod.sh` with the new arguments (guard → setup → guardrail check →
   run_pilot → finalize, which already destroys a vast.ai instance with its own `CONTAINER_API_KEY` once HF is
   verified, commit 1567006).
4. `scripts/vast_launch.py BLOCK N` (laptop): for each shard, pick the cheapest fitting offer with the quote's
   filters, `vastai create instance OFFER --template_hash 61949e9297f2fb86b541659282d8ad3c --disk <per block>
   --onstart-cmd "cd /workspace && git clone https://github.com/Cypher193/xakfdghdfjk.git cvpr_vail && cd cvpr_vail
   && setsid nohup bash scripts/full_pod.sh BLOCK K N BUDGET > /workspace/full.out 2>&1 < /dev/null &"`, poll
   `show instance` until `running` (destroy and re-pick on `exited` / `offline`), append instance ids to
   `data/logs/vast_instances.json`. `HF_TOKEN` comes from the account-level environment variable, never from the
   template or the command line. Back off on HTTP 429 (vast sends no `Retry-After`).
5. Laptop-side `vast_status.py`: `show instances` + each shard's `budget_*.json` from HF; relaunches a shard whose
   instance died before `finalize` verified it (the guard's push means only the last ≤ 15 min of clips are lost).

Concurrent pushes from 20–40 instances to one HF repo worked for two screen pods; `upload_large_folder` retries
commit conflicts, but staggering `--every` by a random 0–300 s per pod keeps 429s away.

## 7. Launch checklist

1. Top up the vast.ai balance; confirm `vastai show user` (key via `VAST_API_KEY`), `HF_TOKEN` set as an account
   environment variable, template 778394 current (`vastai show templates`).
2. `python scripts/vast_quote.py` (and `--hosts verified`): pick policy and classes; note the per-block disk sizes.
3. Wave 0 calibration; update `GPUS`; re-quote.
4. Launch wave 1 shards; watch `vast_status`; every instance destroys itself when its shard is verified on HF.
5. Pull, merge the shard tables (`results/show_full_*.csv`), check gates; launch wave 2, then 3.
6. After the run: revoke and replace the HF token and the vast.ai key; `vastai show instances` must be empty.
