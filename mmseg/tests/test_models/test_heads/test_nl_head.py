# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import NLHead
from .utils import to_cuda


def test_nl_head():
    head = NLHead(in_channels=8, channels=4, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'nl_block')
    inputs = [torch.randn(1, 8, 23, 23)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)
