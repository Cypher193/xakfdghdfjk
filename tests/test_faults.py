"""CPU check of track.mask_faults on a synthetic red ball whose mask goes through every fault once.
    python -m tests.test_faults"""
import cv2
import numpy as np

from pilot import track

H, W, R = 120, 160, 10


def disk(img, u, v, r, colour):
    cv2.circle(img, (u, v), r, colour, -1)


frames, masks = [], []
# frame: (ball drawn in the video, mask drawn as SAM2's output)
plan = [((80, 60, R, "red"), [(80, 60, R)]),            # 0 reference
        ((80, 60, R, "red"), [(80, 60, R)]),            # 1 fine
        ((80, 60, R // 2, "red"), [(80, 60, R // 2)]),  # 2 shrunk (area 1/4)
        ((80, 60, 2 * R, "red"), [(80, 60, 2 * R)]),    # 3 grown (area 4x)
        ("parts", [(50, 60, 8), (110, 60, 8)]),        # 4 split in two red halves of 0.64x each
        ((80, 60, R, "white"), [(80, 60, R)]),          # 5 recoloured
        ((5, 60, R, "red"), [(5, 60, R)]),              # 6 touches the left border
        (None, []),                                     # 7 empty after the border: offframe
        ((80, 60, R, "red"), [(80, 60, R)]),            # 8 back
        ((80, 60, R, "red"), []),                       # 9 empty, ball still drawn: vanished + lost_seen
        (None, [])]                                     # 10 empty, nothing there: vanished
for ball, parts in plan:
    f = np.full((H, W, 3), 128, np.uint8)
    for b in parts if ball == "parts" else [ball] if ball else []:
        disk(f, *b[:3], (250, 250, 250) if b[3:] == ("white",) else (220, 30, 30))
    m = np.zeros((H, W), np.uint8)
    for p in parts:
        disk(m, *p, 1)
    frames.append(f)
    masks.append(m.astype(bool))
frames, masks = np.stack(frames), np.stack(masks)

f = track.mask_faults(frames, masks, i0=1)
want = dict(vanished=2, offframe=1, edge=1, split=1, recoloured=1, shrunk=1, grown=1, lost_seen=1, n=10)
assert {k: f[k] for k in want} == want, f
_, area = track.centroids(masks)
ok_share = (f["n"] - f["vanished"] - f["offframe"] - f["shrunk"] - f["grown"]) / f["n"]
assert ok_share == 0.5 and track.measurable(area, 1) is False        # same frames as the measurability rule
print("mask_faults checks passed")
