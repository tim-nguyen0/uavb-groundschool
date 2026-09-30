from pathlib import Path

import cv2
import numpy as np

import detector
import utils

# anchored to this file, not the shell's cwd, so the script runs from any directory
ROOT = Path(__file__).resolve().parent.parent

out_path = utils.make_unique_dir(ROOT / "out", "blob", identifier=utils.id_type.DATETIME)

#the architecture is essentially doing different ways to detect blobs, and then voting on what counts as a blob.
#the bandpass from week 1 (also from CS180 project 1), and saturation modicifcaitons.
channels = (("gray", lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, "trust"),
            ("saturation", detector.saturation, 255, "bypass"),
            ("bgdistance", detector.background_distance, 255, "trust"),
            ("sat_or_bgdist", lambda bgr: np.maximum(detector.saturation(bgr),
                                                     detector.background_distance(bgr)), 255, "trust"),
            ("bp_gray_L2", lambda bgr: detector.band_pass_channel(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), level=2), None, "vote"),
            ("bp_gray_L3", lambda bgr: detector.band_pass_channel(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), level=3), None, "vote"),
            ("bp_bgdist_L2", lambda bgr: detector.band_pass_channel(
                detector.background_distance(bgr), level=2), 255, "vote"),
            ("bp_bgdist_L3", lambda bgr: detector.band_pass_channel(
                detector.background_distance(bgr), level=3), 255, "vote"),
            ("mc_bgdist_L1", lambda bgr: detector.matched_circle_channel(
                detector.background_distance(bgr), level=1), 255, "vote"),
            ("mc_bgdist_L2", lambda bgr: detector.matched_circle_channel(
                detector.background_distance(bgr), level=2), 255, "vote"),
            ("mc_bgdist_L3", lambda bgr: detector.matched_circle_channel(
                detector.background_distance(bgr), level=3), 255, "vote"),
            ("bp_clahe_L2", lambda bgr: detector.band_pass_channel(detector.enhance_contrast(
                np.maximum(detector.saturation(bgr), detector.background_distance(bgr))),
                level=2), 255, "vote"),
            ("bp_clahe_L3", lambda bgr: detector.band_pass_channel(detector.enhance_contrast(
                np.maximum(detector.saturation(bgr), detector.background_distance(bgr))),
                level=3), 255, "vote"))

# a band pass is linear and band_pass_channel renormalizes
probe = cv2.cvtColor(cv2.imread(str(ROOT / "images" / "polka_dots_1.png"), cv2.IMREAD_COLOR),
                     cv2.COLOR_BGR2GRAY)
plain = detector.band_pass_channel(probe, level=2)
gained = detector.band_pass_channel(np.clip(probe * 0.5 + 40, 0, 255).astype(np.uint8), level=2)
print(f"linear contrast before band_pass: max channel difference = "
      f"{int(np.abs(plain.astype(int) - gained.astype(int)).max())}/255\n")

print(f"{'level':>7s}{'band_pass r':>14s}{'matched r':>12s}")
for level in range(5):
    print(f"{level:7d}{detector.band_pass_radius(level):13.1f}px{4 * 2 ** level:10d}px")
print()

probe2 = cv2.imread(str(ROOT / "images" / "polka_dots_2.jpg"), cv2.IMREAD_COLOR)
counts = [len(detector.detect_blobs(detector.background_distance(probe2), min_area=600,
                                    min_circularity=0.6, blob_color=255)) for _ in range(6)]
print(f"background_distance determinism over 6 runs (polka_dots_2): {counts}")

# which colour sits closest to the background?
lab2 = cv2.cvtColor(cv2.medianBlur(probe2, 5), cv2.COLOR_BGR2LAB).astype(np.float32)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, lbl, cen = cv2.kmeans(lab2.reshape(-1, 3), 6, None, crit, 5, cv2.KMEANS_PP_CENTERS)
share = np.bincount(lbl.ravel()) / lbl.size
bg = cen[share.argmax()]
print(f"  {'cluster BGR':22s}{'share':>7s}{'Lab dist from background':>26s}")
for i in np.argsort([np.linalg.norm(c - bg) for c in cen]):
    bgr = cv2.cvtColor(cen[i].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).ravel()
    print(f"  {str(tuple(int(v) for v in bgr)):22s}{share[i]:6.1%}"
          f"{np.linalg.norm(cen[i] - bg):20.1f}")

# can saturation rescue red where bgdistance struggles? compare in HSV per cluster
hsv2 = cv2.cvtColor(cv2.medianBlur(probe2, 5), cv2.COLOR_BGR2HSV).reshape(-1, 3)
labels2 = lbl.ravel()
print(f"  {'cluster BGR':22s}{'hue':>6s}{'sat':>6s}{'val':>6s}")
for i in np.argsort([np.linalg.norm(c - bg) for c in cen]):
    bgr = cv2.cvtColor(cen[i].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).ravel()
    px = hsv2[labels2 == i]
    print(f"  {str(tuple(int(v) for v in bgr)):22s}"
          f"{px[:, 0].mean():6.0f}{px[:, 1].mean():6.0f}{px[:, 2].mean():6.0f}")
print()

# min_area per image so a dot
min_areas = {"polka_dots_1.png": 150, "polka_dots_2.jpg": 600, "polka_dots_3.jpg": 150}

rows = []

for name in ("polka_dots_1.png", "polka_dots_2.jpg", "polka_dots_3.jpg"):
    image = cv2.imread(str(ROOT / "images" / name), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {ROOT / 'images' / name}")
    stem = name.rsplit(".", 1)[0]
    per_channel = []

    for label, to_channel, blob_color, role in channels:
        channel = to_channel(image)
        keypoints = detector.detect_blobs(channel, min_area=min_areas[name],
                                          min_circularity=0.6, blob_color=blob_color)
        per_channel.append((role, keypoints))

        cv2.imwrite(str(out_path / f"{stem}_{label}.png"), detector.draw_blobs(image, keypoints))
        cv2.imwrite(str(out_path / f"{stem}_{label}_channel.png"), channel)

        rows.append((name, label, len(keypoints)))
        print(f"{name:20s} {label:14s} -> {len(keypoints):3d} blobs")

    # union across channels, richest first, then drop whatever landed on the background
    bypassed = [kps for role, kps in per_channel if role == "bypass"]
    trusted = [kps for role, kps in per_channel if role == "trust"]
    voting = [kps for role, kps in per_channel if role == "vote"]

    confirmed = detector.consensus_keypoints(voting, min_votes=2)
    print(f"{name:20s} {'bandpass':14s} -> {sum(len(k) for k in voting):3d} raw, "
          f"{len(confirmed):3d} confirmed by >=2 channels")

    checked = detector.merge_keypoints(sorted(trusted, key=len, reverse=True) + [confirmed])

    foreground = np.maximum(detector.saturation(image), detector.background_distance(image))
    before_reject = len(checked)
    checked = detector.reject_background(checked, foreground)
    rejected = before_reject - len(checked)

    # bypassed channels skip the background check entirely
    merged = detector.merge_keypoints(sorted(bypassed, key=len, reverse=True) + [checked])
    cv2.imwrite(str(out_path / f"{stem}_merged.png"), detector.draw_blobs(image, merged))
    rows.append((name, "MERGED", len(merged)))
    print(f"{name:20s} {'MERGED':14s} -> {len(merged):3d} blobs "
          f"({rejected} dropped as background, saturation exempt)\n")

with (out_path / "summary.txt").open("w") as f:
    f.write(f"{'image':22s}{'channel':16s}blobs\n")
    for name, label, count in rows:
        f.write(f"{name:22s}{label:16s}{count}\n")

print(f"wrote images + summary.txt to {out_path.resolve()}")
