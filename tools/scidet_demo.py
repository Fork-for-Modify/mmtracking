# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv

from mmtrack.apis import inference_scidet, init_scidet_model


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or images')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--score-thr', type=float, default=0.8, help='bbox score threshold')
    parser.add_argument(
        '--Cr', type=int, default=10, help='compressive ratio for SCI encoding')
    parser.add_argument(
        '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()

    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        imgs = [osp.join(args.input, img) for img in imgs]  # full path
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_scidet_model(args.config, args.checkpoint, device=args.device)

    num_meas = len(imgs)//args.Cr  # discard rest images (<Cr)
    prog_bar = mmcv.ProgressBar(num_meas)
    # test and show/save the images
    # for i, img in enumerate(imgs):
    for i in range(num_meas):
        results = inference_scidet(model, imgs, meas_id=i, ref_img_sampler=dict(
            num_ref_imgs=args.Cr, method='right'))

        # rearrange results
        results = [{'det_bboxes': results['det_bboxes'][k]}
                   for k in range(args.Cr)]

        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = [osp.join(out_path, imgs[i*args.Cr+m].rsplit(
                    os.sep, 1)[-1]) for m in range(args.Cr)]
        else:
            out_file = None

        for k in range(args.Cr):
            model.show_result(
                imgs[i*args.Cr+k],
                results[k],
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file[k],
                thickness=args.thickness)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(
            f'\nmaking the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    main()
