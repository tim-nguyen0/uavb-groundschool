import numpy as np
import cv2
import pyramid as pyramid
import utils as utils
import align as align

#This stitcher works on the assumption of the given minecraft video (linear motion). 

def video_to_frame_array(input: cv2.VideoCapture, grayscale: bool=False, step: int=1) -> np.ndarray:
    """Decodes the video, keeping every (step)-th frame."""
    kept = []

    i = 0
    while True:
        ok, frame = input.read()
        if not ok:
            break
        if i % step == 0:
            if grayscale:
                kept.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                kept.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1

    if not kept:
        raise ValueError("No frames could be read from the video")

    return np.stack(kept)


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    """Typical grayscaling (using standard values)"""
    return frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def calculate_relative_offsets_ncc(frames: np.ndarray) -> np.ndarray:
    """Calculates offsets of frames in image (by pixels) relative to previous sample frame

    Args:
        frames (np.ndarray): Sampled grayscale frames

    Returns:
        offsets (np.array): offset array
    """

    offsets = np.zeros((frames.shape[0], 2))

    for i in range(1,frames.shape[0]):
        offsets[i]=np.array(align.calculate_offset_pyramid(frames[i-1], frames[i], initial_search_frac=0.45, step_max_offset=5, crop_frac=0.1))
    return offsets

def stitch_frames_from_offset(frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Stitches frames together based on offsets"""
    N, h, w = frames.shape[:3]

    c   = np.cumsum(offsets, axis=0)         
    low  = c.min(axis=0)
    pos = np.round(c - low).astype(int)        

    H_c = int(pos[:, 0].max()) + h
    W_c = int(pos[:, 1].max()) + w
    canvas = np.zeros((H_c, W_c) + frames.shape[3:], frames.dtype)

    for t in range(N):
        y, x = pos[t]
        canvas[y:y+h, x:x+w] = frames[t]

    return canvas
