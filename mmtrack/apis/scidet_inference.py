# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import tempfile

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose

from mmtrack.models import build_model


def init_scidet_model(config,
                      checkpoint=None,
                      device='cuda:0',
                      cfg_options=None,
                      verbose_init_params=False):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. Default as None.
        cfg_options (dict, optional): Options to override some settings in
            the used config. Default to None.
        verbose_init_params (bool, optional): Whether to print the information
            of initialized parameters to the console. Default to False.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'detector' in config.model:
        config.model.detector.pretrained = None
    model = build_model(config.model)

    if not verbose_init_params:
        # Creating a temporary file to record the information of initialized
        # parameters. If not, the information of initialized parameters will be
        # printed to the console because of the call of
        # `mmcv.runner.BaseModule.init_weights`.
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        file_handler = logging.FileHandler(tmp_file.name, mode='w')
        model.logger.addHandler(file_handler)
        # We need call `init_weights()` to load pretained weights in MOT
        # task.
        model.init_weights()
        file_handler.close()
        model.logger.removeHandler(file_handler)
        tmp_file.close()
        os.remove(tmp_file.name)
    else:
        # We need call `init_weights()` to load pretained weights in MOT task.
        model.init_weights()

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    if not hasattr(model, 'CLASSES'):
        if hasattr(model, 'detector') and hasattr(model.detector, 'CLASSES'):
            model.CLASSES = model.detector.CLASSES
        else:
            print("Warning: The model doesn't have classes")
            model.CLASSES = None
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_scidet(model,
                     imgs,
                     meas_id,
                     ref_img_sampler=dict(num_ref_imgs=10, stride=1, method='right')):
    """Inference image with the video object detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (list[str]): image paths
        meas_id (int): measurement id.
        ref_img_sampler (dict): The configuration for sampling reference
            images. Only used under video detector of fgfa style. Defaults to
            dict(num_left_ref_imgs=10, stride=1, method='right').

    Returns:
        dict[str : ndarray]: The detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    Cr = ref_img_sampler['num_ref_imgs']  # compressive ratio
    frame_id = meas_id * Cr  # id of the key frame in the compressive meas

    if cfg.data.test.pipeline[0].type == 'LoadImageFromFile':
        data = dict(
            img=image.astype(np.float32).copy(),
            img_info=dict(frame_id=frame_id))

        # remove the "LoadImageFromFile" in pipeline
        test_pipeline = Compose(cfg.data.test.pipeline[1:])

    elif cfg.data.test.pipeline[0].type == 'LoadMultiImagesFromFile':
        data = []
        method = ref_img_sampler.get('method')
        stride = ref_img_sampler.get('stride', 1)
        num_ref_imgs = ref_img_sampler.get('num_ref_imgs')
        if method == 'right':
            sampling_ticks = list(range(0, stride*num_ref_imgs, stride))
            # shift left tick to frame_id to get valid ref_frame_ids
            ref_frame_ids = [k+frame_id for k in sampling_ticks]
            for k in ref_frame_ids:
                assert isinstance(
                    imgs[k], str), f'image path should be str, but get {imgs[k]}'
                image = mmcv.imread(imgs[k])
                one_ref_img = dict(
                    img=image.astype(np.float32).copy(),
                    img_info=dict(frame_id=k),
                    img_shape=image.shape)
                data.append(one_ref_img)
        else:
            raise NotImplementedError

         # remove the "LoadMultiImagesFromFile" in pipeline
        test_pipeline = Compose(cfg.data.test.pipeline[1:])

    else:
        print('Not supported loading data pipeline type: '
              f'{cfg.data.test.pipeline[0].type}')
        raise NotImplementedError

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        data['frames']['img_metas'] = data['frames']['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result
