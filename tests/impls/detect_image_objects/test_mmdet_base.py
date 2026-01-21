import gc
import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Type
import unittest.mock as mock

from smqtk_image_io import AxisAlignedBoundingBox
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from mmdet.datasets.pipelines.auto_augment import AutoAugment   
from mmdet.datasets import build_dataloader
from torch.utils.data import Dataset


# noinspection PyProtectedMember
from smqtk_detection.impls.detect_image_objects.mmdet_base import (
    _trivial_batch_collator,
    _to_tensor,
    _aug_one_image,
    MMDetectionBase,
)


LOG = logging.getLogger(__name__)

PARAM_np_dtype = pytest.mark.parametrize(
    "np_dtype",
    [np.uint8, np.float32, np.float64],
    ids=lambda v: f"dtype={v.__name__}"
)

TEST_CONFIG_PATH = Path("../../data/mmdet_config/retinanet/retinanet_r50_fpn_mstrain_640-800_3x_coco.py")


def test_trivial_batch_collector() -> None:
    """
    Should literally do nothing and return what was input.
    """
    new_type = type("rando_new_type")
    new_object = new_type()
    val = _trivial_batch_collator(new_object)
    assert val is new_object


@PARAM_np_dtype
def test_to_tensor_nochan(np_dtype: np.dtype) -> None:
    """
    Test that a tensor is appropriately output for the given ndarray with NO
    channel dimension.
    """
    n = np.empty((15, 16), dtype=np_dtype)
    t = _to_tensor(n)
    assert t.shape == (15, 16)
    assert t.dtype == torch.float32


@PARAM_np_dtype
def test_to_tensor_1chan(np_dtype: np.dtype) -> None:
    """
    Test that a tensor is appropriately output for the given ndarray with ONE
    channel dimension.
    """
    n = np.empty((17, 18, 1), dtype=np_dtype)
    t = _to_tensor(n)
    assert t.shape == (1, 17, 18)
    assert t.dtype == torch.float32


@PARAM_np_dtype
def test_to_tensor_3chan(np_dtype: np.dtype) -> None:
    """
    Test that a tensor is appropriately output for the given ndarray with THREE
    channel dimension.
    """
    n = np.empty((19, 20, 3), dtype=np_dtype)
    t = _to_tensor(n)
    assert t.shape == (3, 19, 20)
    assert t.dtype == torch.float32


def test_aug_one_image() -> None:
    """
    Test single-image augmentation pass with mock augmentation.
    """
    m_aug = mock.MagicMock(spec=AutoAugment)
    test_image = np.zeros((128, 224, 3), dtype=np.uint8)
    test_boxes = np.zeros((1,4), dtype=np.uint8)

    ret = _aug_one_image(test_image, m_aug, test_boxes)

    # The mock augmentation does nothing to the `AugInput` given to it, so
    # can treat it as a "NoOp" augmentation where the `aug_input.image` is the
    # same image as was input to the parent function.
    assert isinstance(ret, dict)
    assert len(ret) == 3
    assert 'image' in ret
    assert isinstance(ret['image'], torch.Tensor)
    assert ret['image'].shape == (3, 128, 224)
    assert 'height' in ret
    assert ret['height'] == 128
    assert 'width' in ret
    assert ret['width'] == 224

class TestMMDetBase:

    class StubPlugin(MMDetectionBase):
        """ Stub implementation of abstract base class used to test base class
        provided functionality. """
        def _get_augmentation(self) -> AutoAugment: ...

    @classmethod
    def teardown_class(cls) -> None:
        del cls.StubPlugin
        # Clean-up locally defined pluggable implementation.
        gc.collect()

    @mock.patch.object(MMDetectionBase, "_lazy_load_model")
    def test_init_lazy_load(self, _: Any) -> None:
        """
        Test that with lazy load on, construction does NOT attempt to load the
        model.
        """
        
        inst = self.StubPlugin(TEST_CONFIG_PATH.as_posix(), model_lazy_load=True)

        inst._lazy_load_model.assert_not_called()  # type: ignore

    @mock.patch.object(MMDetectionBase, "_lazy_load_model")
    def test_init_eager_load(self, _: mock.Mock) -> None:
        """
        Test that with lazy load off, the model attempts to initialize
        immediately.
        """
        inst = self.StubPlugin(TEST_CONFIG_PATH.as_posix(), model_lazy_load=False)

        inst._lazy_load_model.assert_called_once()  # type: ignore

    @pytest.mark.parametrize("initially_lazy", [False, True], ids=lambda v: f"initially_lazy={v}")
    def test_lazy_load_model_idempotent(self, initially_lazy: bool) -> None:
        """
        Test that the lazy loading function returns the same object on
        successive calls.
        """
        inst = self.StubPlugin(TEST_CONFIG_PATH.as_posix(), model_lazy_load=initially_lazy)
        inst_model1 = inst._lazy_load_model()
        inst_model2 = inst._lazy_load_model()
        assert inst_model1 is not None
        assert inst_model1 is inst_model2

