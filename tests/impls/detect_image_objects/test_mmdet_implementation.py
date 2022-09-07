import numpy as np
from pathlib import Path

from smqtk_detection.impls.detect_image_objects.mmdet_base import MMDetectionBase 

TEST_CONFIG_PATH = Path("../../data/mmdet_config/retinanet/retinanet_r50_fpn_mstrain_640-800_3x_coco.py")


class TestMMDetectionBaseReal:

    def test_smoke_random(self) -> None:
        """
        Smoke-test running a model on a random RGB image.
        """
        cfg_fpath = TEST_CONFIG_PATH
        inst = MMDetectionBase(
            cfg_fpath.as_posix(),
            batch_size=1, model_lazy_load=False, num_workers=0
        )

        random_image = (np.random.rand(244, 244, 3) * 255).astype(np.uint8)
        results = list(list(inst.detect_objects([random_image]))[0])
