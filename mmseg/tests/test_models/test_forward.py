# Copyright (c) OpenMMLab. All rights reserved.
"""pytest tests/test_forward.py."""
import copy
from os.path import dirname, exists, join
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmengine.utils import is_list_of, is_tuple_of
from torch import Tensor

from mmseg.structures import SegDataSample

init_default_scope('mmseg')


def _demo_mm_inputs(batch_size=2, image_shapes=(3, 32, 32), num_classes=5):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        batch_size (int): batch size. Default to 2.
        image_shapes (List[tuple], Optional): image shape.
            Default to (3, 128, 128)
        num_classes (int): number of different labels a
            box might have. Default to 10.
    """
    if isinstance(image_shapes, list):
        assert len(image_shapes) == batch_size
    else:
        image_shapes = [image_shapes] * batch_size

    inputs = []
    data_samples = []
    for idx in range(batch_size):
        image_shape = image_shapes[idx]
        c, h, w = image_shape
        image = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)

        mm_input = torch.from_numpy(image)

        img_meta = {
            'img_id': idx,
            'img_shape': image_shape[1:],
            'ori_shape': image_shape[1:],
            'pad_shape': image_shape[1:],
            'filename': '<demo>.png',
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': None,
        }

        data_sample = SegDataSample()
        data_sample.set_metainfo(img_meta)

        gt_semantic_seg = np.random.randint(
            0, num_classes, (1, h, w), dtype=np.uint8)
        gt_semantic_seg = torch.LongTensor(gt_semantic_seg)
        gt_sem_seg_data = dict(data=gt_semantic_seg)
        data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        inputs.append(mm_input)
        data_samples.append(data_sample)
    return dict(inputs=inputs, data_samples=data_samples)


def _get_config_directory():
    """Find the predefined segmentor config directory."""
    try:
        # Assume we are running in the source mmsegmentation repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmseg
        repo_dpath = dirname(dirname(dirname(mmseg.__file__)))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmengine import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_segmentor_cfg(fname):
    """Grab configs necessary to create a segmentor.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def test_pspnet_forward():
    _test_encoder_decoder_forward(
        'pspnet/pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py')


def test_fcn_forward():
    _test_encoder_decoder_forward(
        'fcn/fcn_r18-d8_4xb2-80k_cityscapes-512x1024.py')


def test_deeplabv3_forward():
    _test_encoder_decoder_forward(
        'deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-512x1024.py')


def test_deeplabv3plus_forward():
    _test_encoder_decoder_forward(
        'deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py')


def test_gcnet_forward():
    _test_encoder_decoder_forward(
        'gcnet/gcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')


def test_ccnet_forward():
    if not torch.cuda.is_available():
        pytest.skip('CCNet requires CUDA')
    _test_encoder_decoder_forward(
        'ccnet/ccnet_r50-d8_4xb2-40k_cityscapes-512x1024.py')


def test_upernet_forward():
    _test_encoder_decoder_forward(
        'upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py')


def test_hrnet_forward():
    _test_encoder_decoder_forward(
        'hrnet/fcn_hr18s_4xb2-40k_cityscapes-512x1024.py')


def test_ocrnet_forward():
    _test_encoder_decoder_forward(
        'ocrnet/ocrnet_hr18s_4xb2-40k_cityscapes-512x1024.py')


def test_sem_fpn_forward():
    _test_encoder_decoder_forward(
        'sem_fpn/fpn_r50_4xb2-80k_cityscapes-512x1024.py')


def test_mobilenet_v2_forward():
    _test_encoder_decoder_forward(
        'mobilenet_v2/mobilenet-v2-d8_pspnet_4xb2-80k_cityscapes-512x1024.py')


def get_world_size(process_group):

    return 1


def _check_input_dim(self, inputs):
    pass


@patch('torch.nn.modules.batchnorm._BatchNorm._check_input_dim',
       _check_input_dim)
@patch('torch.distributed.get_world_size', get_world_size)
def _test_encoder_decoder_forward(cfg_file):
    model = _get_segmentor_cfg(cfg_file)
    model['pretrained'] = None
    model['test_cfg']['mode'] = 'whole'

    from mmseg.models import build_segmentor
    segmentor = build_segmentor(model)
    segmentor.init_weights()

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    packed_inputs = _demo_mm_inputs(
        batch_size=2, image_shapes=(3, 4, 4), num_classes=num_classes)
    # convert to cuda Tensor if applicable
    if torch.cuda.is_available():
        segmentor = segmentor.cuda()
    else:
        segmentor = revert_sync_batchnorm(segmentor)

    # Test forward train
    data = segmentor.data_preprocessor(packed_inputs, True)
    losses = segmentor.forward(**data, mode='loss')
    assert isinstance(losses, dict)

    packed_inputs = _demo_mm_inputs(
        batch_size=1, image_shapes=(3, 32, 32), num_classes=num_classes)
    data = segmentor.data_preprocessor(packed_inputs, False)
    with torch.no_grad():
        segmentor.eval()
        # Test forward predict
        batch_results = segmentor.forward(**data, mode='predict')
        assert len(batch_results) == 1
        assert is_list_of(batch_results, SegDataSample)
        assert batch_results[0].pred_sem_seg.shape == (32, 32)
        assert batch_results[0].seg_logits.data.shape == (num_classes, 32, 32)
        assert batch_results[0].gt_sem_seg.shape == (32, 32)

        # Test forward tensor
        batch_results = segmentor.forward(**data, mode='tensor')
        assert isinstance(batch_results, Tensor) or is_tuple_of(
            batch_results, Tensor)

        # Test forward predict without ground truth
        data.pop('data_samples')
        batch_results = segmentor.forward(**data, mode='predict')
        assert len(batch_results) == 1
        assert is_list_of(batch_results, SegDataSample)
        assert batch_results[0].pred_sem_seg.shape == (32, 32)

        # Test forward tensor without ground truth
        batch_results = segmentor.forward(**data, mode='tensor')
        assert isinstance(batch_results, Tensor) or is_tuple_of(
            batch_results, Tensor)
