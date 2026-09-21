import cv2
import skimage.io as skio
import stitch
import utils

video_path = "../stitch_in.mp4"
sample_rate = 1 # Hz
out_dir = "out"

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise SystemExit("Could not open video")

out_path = utils.make_unique_dir(out_dir, "stitch", identifier=utils.id_type.DATETIME)

fps = video.get(cv2.CAP_PROP_FPS)
diff = max(int(fps / sample_rate), 1)
frames = stitch.video_to_frame_array(video, step=diff)
video.release()

offsets  = stitch.calculate_relative_offsets_ncc(stitch.to_grayscale(frames))
stitched = stitch.stitch_frames_from_offset(frames, offsets)

skio.imsave(out_path / "stitch.png", stitched)
