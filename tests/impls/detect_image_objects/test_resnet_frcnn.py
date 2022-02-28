import unittest.mock as mock

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper

from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN


@pytest.mark.skipif(not ResNetFRCNN.is_usable(),
                    reason="ResNetFRCNN is not usable")
class TestResNetFRCNN:

    def test_configuration(self) -> None:
        """ Test configuration stability """
        inst = ResNetFRCNN(
            box_thresh=0.1,
            num_dets=50,
            img_batch_size=2,
            use_cuda=True,
        )
        for i in configuration_test_helper(inst):
            assert i.box_thresh == 0.1
            assert i.num_dets == 50
            assert i.img_batch_size == 2
            assert i.use_cuda is True

    def test_get_model_idempotent(self) -> None:
        """
        Test that the same model instance is returned on a subsequent call.
        """
        inst = ResNetFRCNN()
        model_1 = inst.get_model()
        assert model_1 is not None
        model_2 = inst.get_model()
        assert model_1 is model_2

    @mock.patch("torch.cuda.is_available")
    @mock.patch("torchvision.models.detection.fasterrcnn_resnet50_fpn")
    def test_get_model_cuda_not_available(
        self,
        m_model_constructor: mock.MagicMock,
        m_torch_cuda: mock.MagicMock
    ) -> None:
        """
        Test that when requesting cuda when it's not available results in a
        RuntimeError.
        """
        m_torch_cuda.return_value = False

        inst = ResNetFRCNN(use_cuda=True)
        with pytest.raises(RuntimeError, match=r"not available"):
            inst.get_model()

    def test_smoketest(self) -> None:
        """Run on a dummy image for basic sanity."""
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        inst = ResNetFRCNN()
        list(inst.detect_objects([dummy_image]))
