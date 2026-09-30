import cv2
import numpy as np

import pyramid


def detect_blobs(gray: np.ndarray, min_area: float=50, max_area: float=1e5,
                 min_circularity: float=0.7, min_convexity: float=0.8,
                 min_inertia: float=0.1, blob_color: int=None,
                 min_threshold: float=0, max_threshold: float=255,
                 threshold_step: float=10, min_repeatability: int=2) -> list:
    """Runs cv2.SimpleBlobDetector on a grayscale image; kp.pt is (x, y) and kp.size a diameter.

    cv2's default 50-220 threshold window silently clips blobs outside it (25 of 112 dots on
    polka_dots_1), so this opens it to the full range.
    """
    if gray.ndim != 2:
        raise ValueError("detect_blobs expects a grayscale image, got shape %r" % (gray.shape,))

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    params.filterByCircularity = True
    params.minCircularity = min_circularity

    params.filterByConvexity = True
    params.minConvexity = min_convexity

    params.filterByInertia = True
    params.minInertiaRatio = min_inertia

    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.thresholdStep = threshold_step
    params.minRepeatability = min_repeatability

    # off by default: cv2 defaults this on at blobColor=0, which silently drops every light blob
    params.filterByColor = blob_color is not None
    if blob_color is not None:
        params.blobColor = blob_color

    return cv2.SimpleBlobDetector_create(params).detect(gray)


def draw_blobs(image: np.ndarray, keypoints: list, color: tuple=(255, 255, 255),
               radius_scale: float=1.15, thickness: int=2) -> np.ndarray:
    """Draws each keypoint as a circle, haloed in black so it reads against any dot colour."""
    out = image.copy()

    for keypoint in keypoints:
        centre = (int(round(keypoint.pt[0])), int(round(keypoint.pt[1])))
        radius = max(int(round(keypoint.size / 2 * radius_scale)), 1)
        cv2.circle(out, centre, radius, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.circle(out, centre, radius, color, thickness, cv2.LINE_AA)

    return out


def saturation(bgr: np.ndarray, blur: int=5) -> np.ndarray:
    """HSV saturation as a single channel: how *colourful* a pixel is, ignoring how bright."""
    return cv2.cvtColor(cv2.medianBlur(bgr, blur), cv2.COLOR_BGR2HSV)[:, :, 1]


def background_distance(bgr: np.ndarray, clusters: int=6, blur: int=5) -> np.ndarray:
    """Distance in CIELAB from the image's dominant colour, normalized to 0-255."""
    lab = cv2.cvtColor(cv2.medianBlur(bgr, blur), cv2.COLOR_BGR2LAB).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(lab.reshape(-1, 3), clusters, None, criteria, 5,
                                    cv2.KMEANS_PP_CENTERS)
    background = centers[np.bincount(labels.ravel()).argmax()]

    distance = np.linalg.norm(lab - background, axis=2)
    return cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def band_pass_radius(level: int, low: int=2, high: int=5) -> float:
    """Blob radius in original pixels that band_pass(low, high) at pyramid (level) is tuned to."""
    accumulated = (1.0 - 4.0 ** -level) / 3.0          # blur inherited from the downsampling
    sigma_sq = 4.0 ** level * (accumulated + 0.5 * (low + high))
    return np.sqrt(2.0 * sigma_sq)


def band_pass_channel(channel: np.ndarray, low: int=2, high: int=5, level: int=0) -> np.ndarray:
    """band_pass is a DoG, so it is a LoG blob detector already.

    (level) runs the same small band on a downsampled image, doubling the radius it selects per
    level. 128 means zero response; the sign is kept so a light blob stays bright and a dark one
    stays dark.
    """
    small = channel.astype(np.float32) / 255.0
    for _ in range(level):
        small = pyramid.downsample_half(small)

    response = pyramid.band_pass(small, low, high)
    scale = max(np.abs(response).max(), 1e-6)
    rendered = np.clip(128 + 127 * response / scale, 0, 255).astype(np.uint8)

    return cv2.resize(rendered, (channel.shape[1], channel.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def merge_keypoints(keypoint_lists: list, overlap: float=0.5) -> list:
    """Merging keypoints
    """
    kept = []
    for keypoints in keypoint_lists:
        for candidate in keypoints:
            cx, cy = candidate.pt
            duplicate = False
            for other in kept:
                ox, oy = other.pt
                if np.hypot(cx - ox, cy - oy) < overlap * (candidate.size + other.size) / 2:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
    return kept


def reject_background(keypoints: list, reference: np.ndarray, min_value: float=40,
                      sample_frac: float=0.5) -> list:
    """Drops keypoints sitting on background, gaps between dots are blobby as a dot."""
    height, width = reference.shape
    kept = []

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = max(int(keypoint.size * sample_frac / 2), 1)

        patch = reference[max(y - radius, 0):min(y + radius + 1, height),
                          max(x - radius, 0):min(x + radius + 1, width)]
        if patch.size and patch.mean() >= min_value:
            kept.append(keypoint)

    return kept


def circle_kernel(radius: float, surround: float=1.6) -> np.ndarray:
    """Zero-mean disk matched filter: +1 on the disk, negative on the annulus around it."""
    outer = int(np.ceil(radius * surround))
    yy, xx = np.mgrid[-outer:outer + 1, -outer:outer + 1]
    distance = np.hypot(yy, xx)

    kernel = np.zeros(distance.shape, np.float32)
    kernel[distance <= radius] = 1.0
    kernel[(distance > radius) & (distance <= radius * surround)] = -1.0

    inside, ring = (kernel > 0).sum(), (kernel < 0).sum()
    kernel[kernel < 0] *= inside / max(ring, 1)   # zero mean, so flat regions score 0
    return kernel / max(inside, 1)                # unit peak, so a perfect disk scores 1


def matched_circle_channel(channel: np.ndarray, radius: float=4, level: int=0) -> np.ndarray:
    """Correlation with a disk kernel, rendered 0-255 with 128 meaning no match.

    A radius-40 kernel is 113x113 taps, so instead (level) runs a small kernel on a downsampled
    image: the radius it matches is radius * 2**level.
    """
    small = channel.astype(np.float32) / 255.0
    for _ in range(level):
        small = pyramid.downsample_half(small)

    response = cv2.filter2D(small, -1, circle_kernel(radius), borderType=cv2.BORDER_REPLICATE)
    scale = max(np.abs(response).max(), 1e-6)
    rendered = np.clip(128 + 127 * response / scale, 0, 255).astype(np.uint8)

    return cv2.resize(rendered, (channel.shape[1], channel.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def enhance_contrast(channel: np.ndarray, clip: float=2.0, grid: int=8) -> np.ndarray:
    """CLAHE, i.e. histogram equalisation per tile, which a band pass cannot cancel the way it
    cancels a linear gain."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(channel)


def consensus_keypoints(keypoint_lists: list, min_votes: int=2, overlap: float=0.5) -> list:
    """Keeps only blobs that at least (min_votes) of the given channels independently found."""
    entries = [(keypoint, source)
               for source, keypoints in enumerate(keypoint_lists)
               for keypoint in keypoints]
    entries.sort(key=lambda entry: -entry[0].size)

    taken = [False] * len(entries)
    kept = []

    for i, (keypoint, source) in enumerate(entries):
        if taken[i]:
            continue
        taken[i] = True
        voters = {source}

        for j in range(i + 1, len(entries)):
            if taken[j]:
                continue
            other, other_source = entries[j]
            if np.hypot(keypoint.pt[0] - other.pt[0], keypoint.pt[1] - other.pt[1]) \
                    < overlap * (keypoint.size + other.size) / 2:
                taken[j] = True
                voters.add(other_source)

        if len(voters) >= min_votes:
            kept.append(keypoint)

    return kept
