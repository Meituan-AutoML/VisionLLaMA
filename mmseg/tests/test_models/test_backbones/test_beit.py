# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones.beit import BEiT
from .utils import check_norm_state


def test_beit_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = BEiT()
        model.init_weights(pretrained=0)

    with pytest.raises(TypeError):
        # img_size must be int or tuple
        model = BEiT(img_size=512.0)

    with pytest.raises(TypeError):
        # out_indices must be int ,list or tuple
        model = BEiT(out_indices=1.)

    with pytest.raises(AssertionError):
        # The length of img_size tuple must be lower than 3.
        BEiT(img_size=(224, 224, 224))

    with pytest.raises(TypeError):
        # Pretrained must be None or Str.
        BEiT(pretrained=123)

    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = BEiT(img_size=(224, ))
    model.init_weights()
    model(imgs)

    # Test img_size isinstance tuple
    imgs = torch.randn(1, 3, 224, 224)
    model = BEiT(img_size=(224, 224))
    model(imgs)

    # Test norm_eval = True
    model = BEiT(norm_eval=True)
    model.train()

    # Test BEiT backbone with input size of 224 and patch size of 16
    model = BEiT()
    model.init_weights()
    model.train()

    # Test  qv_bias
    model = BEiT(qv_bias=False)
    model.train()

    # Test out_indices = list
    model = BEiT(out_indices=[2, 4, 8, 12])
    model.train()

    assert check_norm_state(model.modules(), True)

    # Test image size = (224, 224)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test BEiT backbone with input size of 256 and patch size of 16
    model = BEiT(img_size=(256, 256))
    model.init_weights()
    model.train()
    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 16, 16)

    # Test BEiT backbone with input size of 32 and patch size of 16
    model = BEiT(img_size=(32, 32))
    model.init_weights()
    model.train()
    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 2, 2)

    # Test unbalanced size input image
    model = BEiT(img_size=(112, 224))
    model.init_weights()
    model.train()
    imgs = torch.randn(1, 3, 112, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 7, 14)

    # Test irregular input image
    model = BEiT(img_size=(234, 345))
    model.init_weights()
    model.train()
    imgs = torch.randn(1, 3, 234, 345)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 21)

    # Test init_values=0
    model = BEiT(init_values=0)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test final norm
    model = BEiT(final_norm=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)

    # Test patch norm
    model = BEiT(patch_norm=True)
    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[-1].shape == (1, 768, 14, 14)


def test_beit_init():
    path = 'PATH_THAT_DO_NOT_EXIST'
    # Test all combinations of pretrained and init_cfg
    # pretrained=None, init_cfg=None
    model = BEiT(pretrained=None, init_cfg=None)
    assert model.init_cfg is None
    model.init_weights()

    # pretrained=None
    # init_cfg loads pretrain from an non-existent file
    model = BEiT(
        pretrained=None, init_cfg=dict(type='Pretrained', checkpoint=path))
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # test resize_rel_pos_embed
    value = torch.randn(732, 16)
    ckpt = {
        'state_dict': {
            'layers.0.attn.relative_position_index': 0,
            'layers.0.attn.relative_position_bias_table': value
        }
    }
    model = BEiT(img_size=(512, 512))
    # If scipy is installed, this AttributeError would not be raised.
    from mmengine.utils import is_installed
    if not is_installed('scipy'):
        with pytest.raises(AttributeError):
            model.resize_rel_pos_embed(ckpt)

    # pretrained=None
    # init_cfg=123, whose type is unsupported
    model = BEiT(pretrained=None, init_cfg=123)
    with pytest.raises(TypeError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg=None
    model = BEiT(pretrained=path, init_cfg=None)
    assert model.init_cfg == dict(type='Pretrained', checkpoint=path)
    # Test loading a checkpoint from an non-existent file
    with pytest.raises(OSError):
        model.init_weights()

    # pretrained loads pretrain from an non-existent file
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = BEiT(
            pretrained=path, init_cfg=dict(type='Pretrained', checkpoint=path))
    with pytest.raises(AssertionError):
        model = BEiT(pretrained=path, init_cfg=123)

    # pretrain=123, whose type is unsupported
    # init_cfg=None
    with pytest.raises(TypeError):
        model = BEiT(pretrained=123, init_cfg=None)

    # pretrain=123, whose type is unsupported
    # init_cfg loads pretrain from an non-existent file
    with pytest.raises(AssertionError):
        model = BEiT(
            pretrained=123, init_cfg=dict(type='Pretrained', checkpoint=path))

    # pretrain=123, whose type is unsupported
    # init_cfg=123, whose type is unsupported
    with pytest.raises(AssertionError):
        model = BEiT(pretrained=123, init_cfg=123)
