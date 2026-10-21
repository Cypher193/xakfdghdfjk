"""Download real clips and write data/real/manifest.json (inputs for b1real) and the QuantiPhy tracker checks.

    python -m pilot.download

Boxes were drawn by hand on the first input frame (pixels of the source video).
Physics-IQ: centre view, take 1, 30 fps; only scenarios where the object stays in view after the input window.
QuantiPhy val: only clips whose kinematic questions are about a visible rigid object with a size prior.
"""
import json
from pathlib import Path
from urllib.request import urlretrieve

PIQ_URL = "https://storage.googleapis.com/physics-iq-benchmark/full-videos/take-1/30FPS/{}.mp4"
PIQ = {  # name: (file stem, box on the frame at 2.6875 s, caption without quantities)
    "ball-and-block-fall": ("0002_full-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall",
                            [880, 170, 1230, 555], "A tennis ball held by a clamp above two pillows. Static camera."),
    "ball-ramp": ("0020_full-videos_30FPS_perspective-center_take-1_trimmed-ball-ramp",
                  [115, 700, 385, 945], "A tennis ball rolls across a table towards a cardboard ramp. Static camera."),
    "balls-collide": ("0032_full-videos_30FPS_perspective-center_take-1_trimmed-balls-collide",
                      [295, 775, 555, 1010], "A blue ball and a yellow ball roll towards each other on a table. Static camera."),
    "mirror-ball-fall": ("0116_full-videos_30FPS_perspective-center_take-1_trimmed-mirror-ball-fall",
                         [1710, 1150, 2010, 1450], "An orange ball hangs from a magnet next to a mirror. Static camera."),
}
PIQ_INPUT_END_S = 3.0          # Physics-IQ conditioning ends at 3 s; the continuation is the ground truth

# Tracker check (G1a) uses only 2D-motion clips (video_type "?2??"): the "?3??" clips move toward or away from
# the camera and their labels are 3D speeds, which a 2D tracker cannot recover. Boxes are on the prompt frame.
QVAL_CHECK = {  # clip: objects {name: box, prompt time s, size m}, questions (object, quantity, time, unit to SI, answer)
    "simulation_0163": ({"ball": dict(box=[98, 272, 136, 310], prompt_t=0.25, size_m=0.21)},
                        [("ball", "speed", 0.5, 1.0, 3.148), ("ball", "speed", 1.0, 1.0, 2.974),
                         ("ball", "accel", 0.5, 1.0, 0.359), ("ball", "accel", 1.0, 1.0, 0.339)]),
    "internet_0027": ({"white ball": dict(box=[474, 426, 526, 478], prompt_t=1.0, size_m=0.0572),
                       "orange ball": dict(box=[181, 184, 233, 236], prompt_t=1.0, size_m=0.0572),
                       "yellow ball": dict(box=[199, 362, 251, 414], prompt_t=1.0, size_m=0.0572)},
                      [("white ball", "avg_speed", (1.0, 2.0), 0.01, 26.82), ("orange ball", "speed", 1.0, 0.01, 24.99),
                       ("yellow ball", "accel", 2.0, 0.01, 11.0)]),
}
# Real inputs for b1real (rendered-vs-real check): boxes on frame 0 of the source clip.
QVAL = {"captured_0021x": [474, 267, 492, 285], "captured_0027s": [405, 208, 453, 262],
        "captured_0034as": [148, 300, 199, 354]}
QVAL_CAPTIONS = {
    "captured_0021x": "A white ping pong ball on a wooden table. Static camera.",
    "captured_0027s": "A red ball hangs on a string. Static camera.",
    "captured_0034as": "Two tennis balls, a football and a pencil case on a tilting table. Static camera.",
}


def main(root="data"):
    real = Path(root) / "real"
    (Path(root) / "physics_iq").mkdir(parents=True, exist_ok=True)
    real.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (stem, box, cap) in PIQ.items():
        f = Path(root) / "physics_iq" / f"{stem}.mp4"
        if not f.exists():
            urlretrieve(PIQ_URL.format(stem), f)
        manifest.append(dict(id=f"piq_{name}", source="physics-iq", video=f.as_posix(), wh=[3840, 2160],
                             input_end_s=PIQ_INPUT_END_S, box=box, caption=cap))
    qdir = Path(root) / "quantiphy_val"
    if not (qdir / "validation_dataset.csv").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("PaulineLi/QuantiPhy-validation", repo_type="dataset", local_dir=qdir)
    for clip, box in QVAL.items():
        manifest.append(dict(id=f"qval_{clip}", source="quantiphy-val", video=(qdir / "validation_videos" / f"{clip}.mp4").as_posix(),
                             wh=[854, 480], start_s=0.0, box=box, caption=QVAL_CAPTIONS[clip]))
    (real / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (real / "qval_questions.json").write_text(json.dumps(
        {c: dict(objects=o, questions=q) for c, (o, q) in QVAL_CHECK.items()}, indent=1))
    print(f"{len(manifest)} real clips -> {real / 'manifest.json'}")


if __name__ == "__main__":
    main()
