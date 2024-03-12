# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.decode_heads import LRASPPHead


def test_lraspp_head():
    with pytest.raises(ValueError):
        # check invalid input_transform
        LRASPPHead(
            in_channels=(4, 4, 123),
            in_index=(0, 1, 2),
            channels=32,
            input_transform='resize_concat',
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

    with pytest.raises(AssertionError):
        # check invalid branch_channels
        LRASPPHead(
            in_channels=(4, 4, 123),
            in_index=(0, 1, 2),
            channels=32,
            branch_channels=64,
            input_transform='multiple_select',
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

    # test with default settings
    lraspp_head = LRASPPHead(
        in_channels=(4, 4, 123),
        in_index=(0, 1, 2),
        channels=32,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    inputs = [
        torch.randn(2, 4, 45, 45),
        torch.randn(2, 4, 28, 28),
        torch.randn(2, 123, 14, 14)
    ]
    with pytest.raises(RuntimeError):
        # check invalid inputs
        output = lraspp_head(inputs)

    inputs = [
        torch.randn(2, 4, 111, 111),
        torch.randn(2, 4, 77, 77),
        torch.randn(2, 123, 55, 55)
    ]
    output = lraspp_head(inputs)
    assert output.shape == (2, 19, 111, 111)
