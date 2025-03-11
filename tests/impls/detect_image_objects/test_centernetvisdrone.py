from typing import Dict
import unittest.mock as mock

import numpy as np
import pathlib
import pytest
import requests
from smqtk_core.configuration import configuration_test_helper

from smqtk_detection.impls.detect_image_objects.centernet import CenterNetVisdrone

try:
    from smqtk_detection.impls.detect_image_objects.centernet import _gather
except ImportError:
    pass

try:
    import torch  # type: ignore
except ImportError:
    # CenterNetVisdrone will be unusable, so no use of this module will ensure
    # in this exception case.
    pass


MODEL_URL_RESNET18 = "https://data.kitware.com/api/v1/item/623de4744acac99f42f05fb1/download"
MODEL_URL_RESNET50 = "https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download"
MODEL_URL_RES2NET50 = "https://data.kitware.com/api/v1/item/623e18464acac99f42f40a4e/download"


def can_reach_internet() -> bool:
    """ Quick test if a known URL is reachable. """
    try:
        requests.get("https://kitware.com")
    except requests.ConnectionError:  # pragma: no cover
        return False
    return True


@pytest.fixture(scope="function")
def centernet_resnet18_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Download the resnet18 model for this test session.
    """
    f = tmp_path / "centernet_resnet18.pth"
    r = requests.get(MODEL_URL_RESNET18)
    f.write_bytes(r.content)
    return f


@pytest.fixture(scope="function")
def centernet_resnet50_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Download the resnet50 model for this test session.
    """
    f = tmp_path / "centernet_resnet50.pth"
    r = requests.get(MODEL_URL_RESNET50)
    f.write_bytes(r.content)
    return f


@pytest.fixture(scope="function")
def centernet_res2net50_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Download the res2net50 model for this test session.
    """
    f = tmp_path / "centernet_res2net50.pth"
    r = requests.get(MODEL_URL_RES2NET50)
    f.write_bytes(r.content)
    return f


@pytest.mark.skipif(not CenterNetVisdrone.is_usable(),
                    reason="CenterNetVisdrone is not usable.")
class TestCenterNetVisdrone:

    def test_configuration(self) -> None:
        """ Test configuration stability """
        inst = CenterNetVisdrone(
            "resnet18", "mock path",
            max_dets=33,
            k=100,
            scales=[0.3, 0.6],
            flip=True,
            nms=False,
            use_cuda=False,
            batch_size=3,
            num_workers=2,
            device="cpu"
        )
        for i in configuration_test_helper(inst):
            assert i.arch == "resnet18"
            assert i.model_file == "mock path"
            assert i.max_dets == 33
            assert i.k == 100
            assert i.scales == [0.3, 0.6]
            assert i.flip is True
            assert i.nms is False
            assert i.use_cuda is False
            assert i.batch_size == 3
            assert i.num_workers == 2
            assert str(i.device) == "cpu"

    def test_invalid_arch(self) -> None:
        """ Test that an exception is raised when the arch provided is not
        one that is supported. """
        with pytest.raises(ValueError, match=r"Invalid architecture provided"):
            CenterNetVisdrone("dummy architecture", "mock path")

    @pytest.mark.skipif(not can_reach_internet(),
                        reason="No internet access, test models will not be "
                               "accessible.")
    def test_smoketest_3channel_resnet18(self, centernet_resnet18_file: pathlib.Path) -> None:
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case (no outputs even expected on
        such a random image).
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        inst = CenterNetVisdrone(
            "resnet18", str(centernet_resnet18_file),
            batch_size=2
        )
        list(inst.detect_objects([dummy_image]))

    @pytest.mark.skipif(not can_reach_internet(),
                        reason="No internet access, test models will not be "
                               "accessible.")
    def test_smoketest_3channel_resnet50(self, centernet_resnet50_file: pathlib.Path) -> None:
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case (no outputs even expected on
        such a random image).
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        inst = CenterNetVisdrone(
            "resnet50", str(centernet_resnet50_file),
            batch_size=2
        )
        list(inst.detect_objects([dummy_image]))

    @pytest.mark.skipif(not can_reach_internet(),
                        reason="No internet access, test models will not be "
                               "accessible.")
    def test_smoketest_3channel_res2net50(self, centernet_res2net50_file: pathlib.Path) -> None:
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case (no outputs even expected on
        such a random image).
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        inst = CenterNetVisdrone(
            "res2net50", str(centernet_res2net50_file),
            batch_size=2
        )
        list(inst.detect_objects([dummy_image]))

    def test_flip_detection(self) -> None:
        """
        Test that the model is invoked twice as much when flip is enabled.
        Without flip, the model should be invoked once per image, but with flip
        on it should be invoked 2x as much.
        """
        # Using the same shape as known input network req. for simplicity.
        dummy_image = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
        inst = CenterNetVisdrone(
            "resnet18", "mock path",
            flip=False, batch_size=1
        )
        test_input = [dummy_image] * 7

        # Mock model output since we are not testing model performance here.
        # Simulating the model outputting one detection.

        inst.flip = False
        with mock.patch.object(inst, "model", side_effect=mock_model_forward) as m_model:
            list(inst.detect_objects(test_input))
            assert m_model.call_count == 7

        inst.flip = True
        with mock.patch.object(inst, "model", side_effect=mock_model_forward) as m_model:
            list(inst.detect_objects(test_input))
            assert m_model.call_count == 14


def mock_model_forward(img_tensors: "torch.Tensor") -> Dict:  # noqa: F821
    """
    Mock model forward results.
    Trying to simulate the model outputting one detection from (1,1) to (7,7)
    with confidence 0.99 in the first class (exact conf value changes because
    of the sigmoid).
    """
    import torch  # type: ignore
    batch_size = len(img_tensors)
    test_hm = torch.zeros([batch_size, 10, 480, 640])
    # output is half-scale of input
    test_hm[:, 0, 2, 2] = 0.99
    test_wh = torch.zeros([batch_size, 2, 480, 640])
    test_wh[:, :, 2, 2] = 3.
    test_reg = torch.zeros([batch_size, 2, 480, 640])
    return {
        "hm": test_hm,
        "wh": test_wh,
        "reg": test_reg,
    }


@pytest.mark.skipif(not CenterNetVisdrone.is_usable(),
                    reason="CenterNetVisdrone is not usable.")
def test_gather_feat_mps() -> None:
    """
    Check that alternative implementation for torch.gather on MPS
    matches torch.gather.
    """
    torch.manual_seed(42)

    batch_size = 2
    num_features = 76800
    selected_features = 256
    dim = 2

    feat = torch.randn(batch_size, num_features, dim, device="cpu")
    ind = torch.randint(0, num_features, (batch_size, selected_features, dim), device="cpu")

    feat_cpu = _gather(feat, ind, "cpu")
    feat_mps = _gather(feat, ind, "mps")

    assert torch.allclose(feat_cpu, feat_mps, atol=1e-6)
