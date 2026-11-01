"""Job kl3diag: why the ramp readout misses (docs/phase3.md "Round 3"). One row per full ramp render, plus a contact sheet
with the SAM2 mask outlined -> results/kl3diag_ltx_kl3diag.csv, results/kl3diag_ltx_kl3diag.png."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import metrics, track  # noqa: E402

rows, sheet = [], []
for spec_file in sorted(Path("data/scenes/kl3").glob("kl3_ra_*.json")):
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    video = Path("data/renders/kl3full") / f"{spec['id']}.mp4"
    frames, fps = track.clip_frames(video, spec)
    r = track.track_clip(video, spec, cond_frames=5)
    masks = track.sam2_masks(frames, track.gt_boxes(spec))
    area = masks[0].sum((1, 2)).astype(float)
    in_band = ((area >= 0.5 * area[0]) & (area <= 2 * area[0])).mean()
    u0 = np.nonzero(masks[0, 0])[1]
    rows.append(dict(scene=spec["id"], measurable=r["measurable"], in_band=in_band, area_mean=area.mean() / area[0],
                     mask_u_min=u0.min() if len(u0) else -1, a_gt=spec["gt"]["a_slope"], a_tr=r.get("a_slope"),
                     a_mra=float(metrics.sym_mra(r["a_slope"], spec["gt"]["a_slope"])) if r.get("a_slope") else None,
                     t_foot=spec["gt"]["t_foot"], theta=spec["gt"]["theta_deg"], color=spec["objects"][0]["color"]))
    print(rows[-1], flush=True)
    panels = []
    for i in (0, 24, 72):
        f = frames[i].copy()
        contours, _ = cv2.findContours(masks[0, i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(f, contours, -1, (0, 255, 0), 2)
        panels.append(cv2.resize(f, (284, 160)))
    sheet.append(np.hstack(panels))
df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/kl3diag_ltx_kl3diag.csv", index=False)
cv2.imwrite("results/kl3diag_ltx_kl3diag.png", cv2.cvtColor(np.vstack(sheet), cv2.COLOR_RGB2BGR))
print(f"measurable {df.measurable.mean():.2f}, mean sym MRA {df.a_mra.mean():.3f}")
