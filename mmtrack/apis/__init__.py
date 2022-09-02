# Copyright (c) OpenMMLab. All rights reserved.
from .train import init_random_seed, train_model
from .test import multi_gpu_test, single_gpu_test
from .inference import init_model, inference_mot, inference_sot, inference_vid
from .scidet_train import scidet_init_random_seed, scidet_train_model
from .scidet_test import scidet_multi_gpu_test, scidet_single_gpu_test
from .scidet_inference import inference_scidet,  init_scidet_model


__all__ = [
    'init_random_seed', 'train_model',
    'multi_gpu_test', 'single_gpu_test',
    'init_model', 'inference_mot', 'inference_sot', 'inference_vid',
    'scidet_init_random_seed', 'scidet_train_model',
    'scidet_multi_gpu_test', 'scidet_single_gpu_test',
    'inference_scidet', 'init_scidet_model'
]
