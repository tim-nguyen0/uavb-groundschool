import cv2
import skimage.io as skio
import stitch
import utils

video_path = "../stitch_in.mp4"
out_dir = "out"



for i in range (5):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise SystemExit("Could not open video")
    sample_rate = 1/(2**i)
    out_path = utils.make_unique_dir(out_dir, "stitch", identifier=utils.id_type.DATETIME)

    fps = video.get(cv2.CAP_PROP_FPS)
    diff = max(int(fps / sample_rate), 1)
    frames = stitch.video_to_frame_array(video, step=diff)
    video.release()

    offsets  = stitch.calculate_relative_offsets_ncc(stitch.to_grayscale(frames))
    stitched = stitch.stitch_frames_from_offset(frames, offsets)


    imname_str = "stitch"+ str(sample_rate) + "hz.png"

    skio.imsave(out_path / imname_str , stitched)
