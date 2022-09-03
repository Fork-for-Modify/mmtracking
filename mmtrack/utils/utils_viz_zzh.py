from tqdm import tqdm
import cv2
import os
import os.path as osp
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)


def frames2video(frame_dir: str,
                 video_file: str,
                 fps: float = 30,
                 fourcc: str = 'XVID',
                 filename_tmpl: str = '*.jpg',
                 frame_index: list = None,
                 show_progress: bool = True) -> None:
    """Read the frame images from a directory and join them as a video.

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template, e.g. '{:06d}.jpg', the frame index serves as the variable here (index mode). 
            Default is '*.jpg', i.e. use all .jpg frames in the frame_dir (non index mode)
        frame_index (list[int]): frame index selected. Valid in index mode.
        show_progress (bool): Whether to show a progress bar.
    """
    assert osp.isdir(frame_dir), f'{frame_dir} is not a directory'
    # whether use frame index to select frames
    index_mode = True if (
        '{:' in filename_tmpl and 'd}' in filename_tmpl) else False
    if index_mode and frame_index is None:
        raise ValueError(
            '`frame_idx` should be assigned for index-mode `filename_tmpl`')

    # get selected frame paths
    if index_mode:
        frame_paths = [osp.join(frame_dir, filename_tmpl.format(idx))
                       for idx in frame_index]
    else:
        frame_names = sorted(os.listdir(frame_dir))
        frame_names = list(
            filter(lambda x: x.endswith(filename_tmpl.rsplit('.', 1)[-1]), frame_names))
        frame_paths = [osp.join(frame_dir, frame_name)
                       for frame_name in frame_names]

    img = cv2.imread(frame_paths[0])
    assert img is not None, f'{frame_paths[0]} is not a image path'
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    def write_frame(frame_path):
        img = cv2.imread(frame_path)
        assert img is not None, f'{frame_path} is not a image path'
        vwriter.write(img)

    if show_progress:
        # track_progress(write_frame, range(start, end))
        for frame_path in tqdm(frame_paths):
            write_frame(frame_path)
    else:
        for frame_path in frame_paths:
            write_frame(frame_path)
    vwriter.release()


if __name__ == '__main__':

    # ---------- test frames2video start ------------------------
    frame_dir = './output/tmp/test/res_imgs/val/MVI_20011'
    video_file = './_out_video.mp4'
    fourcc = 'mp4v'
    fps = 5
    filename_tmpl = '{:05d}.jpg'
    frame_index = list(range(1, 600, 7))
    show_progress = True

    frames2video(frame_dir, video_file, fourcc=fourcc, fps=fps,
                 filename_tmpl=filename_tmpl, frame_index=frame_index, show_progress=show_progress)
    # ---------- test frames2video end ------------------------
