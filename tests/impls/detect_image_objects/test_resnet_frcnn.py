from typing import Dict
from typing import Iterable
from typing import List
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
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case (no outputs even expected on
        such a random image).
        """
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        inst = ResNetFRCNN(img_batch_size=2)
        list(inst.detect_objects([dummy_image] * 3))

    def test_detect_batching_1(self) -> None:
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        inst = ResNetFRCNN(img_batch_size=1)

        # Mock model output to get something back since we are not testing
        # model performance here. Simulating the model outputting one detection
        # from (1,1) to (3,3) with confidence .77 in all classes
        inst.model = mock.MagicMock(spec="torch.nn.Module")
        inst.model.side_effect = mock_model_effect

        results = list(inst.detect_objects([dummy_image] * 7))

        # Should have been 7 total calls
        assert inst.model.call_count == 7

        # Should have a result for each input
        assert len(results) == 7
        # For each result, we should see the mocked returns
        for r in results:
            r_dets = list(r)
            assert len(r_dets) == 1
            assert np.allclose(r_dets[0][0].min_vertex, [1, 1])
            assert np.allclose(r_dets[0][0].max_vertex, [3, 3])
            assert np.allclose([v for k, v in r_dets[0][1].items()],
                               0.77)

    def test_detect_batching_3(self) -> None:
        """
        Test that the algorithm runs with batching with no specific errors.
        This will intentionally use a number of inputs that is not evenly
        divisible by the batch size in order to touch on more code than without
        batching.
        """
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        inst = ResNetFRCNN(img_batch_size=3)
        inst.model = mock.MagicMock(spec="torch.nn.Module")

        # Mock model output to get something back since we are not testing
        # model performance here.
        inst.model = mock.MagicMock(spec="torch.nn.Module")
        inst.model.side_effect = mock_model_effect

        # Pretend to call with 7 images, which should be 3 batches run, one
        # with only 2 inputs.
        results = list(inst.detect_objects([dummy_image] * 7))

        # Should have been 3 total calls (2 full batches, 1 partial)
        assert inst.model.call_count == 3

        # Should have a result for each input
        assert len(results) == 7
        # For each result, we should see the mocked returns
        for r in results:
            r_dets = list(r)
            assert len(r_dets) == 1
            assert np.allclose(r_dets[0][0].min_vertex, [1, 1])
            assert np.allclose(r_dets[0][0].max_vertex, [3, 3])
            assert np.allclose([v for k, v in r_dets[0][1].items()],
                               0.77)


def mock_model_effect(tensors: Iterable) -> List[Dict]:
    """
    Simulating the model outputting one detection
    from (1,1) to (3,3) with confidence .77 in all classes.
    This needs a separate function to return new instances of tensors
    due to some implementation in-place conversions into different types.
    """
    import torch  # type: ignore
    return [
        {
            # one box.
            "boxes": torch.tensor([[1, 1, 3, 3]]),
            # one vector of scores for the 90 non-background classes.
            "scores": torch.tensor([[0.77 for _ in range(90)]]),
        }
        for _ in tensors
    ]
