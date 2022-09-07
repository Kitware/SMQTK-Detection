import logging
from typing import Any, Dict, Hashable, Iterable, Iterator, List
from typing import Optional, Tuple, TypeVar, Union
from typing_extensions import Protocol, runtime_checkable
import warnings

import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.pipelines.auto_augment import AutoAugment   

import mmcv

import torch.nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from smqtk_detection import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox


LOG = logging.getLogger(__name__)
T_co = TypeVar("T_co", covariant=True)


class MMDetectionBase(DetectImageObjects):
    """
    Plugin base wrapping the loading and application of mmdetection models
    on images to yield object detections and classifications.

    It is expected that classes will be derived from this base that concretely
    defines the data augmentation to appropriate transform input imagery for
    the configured network.

    This plugin expects input image matrices to be in the dimension format
    ``[H x W]`` or ``[H x W x C]``. It is *not* the case that all input imagery
    must have matching shape values.

    This plugin attempts to be intelligent in how it handles different kinds of
    iterable inputs. When given a ``Dataset`` or countable sequence (has
    ``__len__`` and ``__getitem__``), any valid value may be provided to
    ``num_workers`` as ``DataLoader`` might accept.
    However, when the input to ``detect_objects`` is an uncountable iterable,
    like a generic generator or stream source, the ``num_workers`` value should
    usually be either 0 or 1.
    This is due to the input iterable being copied for each worker, which may
    not result in desired behavior.
    For example:
      * when the input iterable involves non-trivial operations per yield,
        these operations are duplicated for each copy of the iterable as
        traversed on each worker, probably resulting in excessive use of
        resources. E.g. if the iterable is loading images from disk, each
        worker is loading every image as it traverses their copy of the
        iterable, even though each worker may only operate on a minority of
        elements traversed.
      * when the input iterable yields real-time data or is otherwise **not**
        idempotent, like an iterable that yields images from a webcam stream,
        each traversal of a copy of that iterable will produce different values
        for equivalent "indices" since what is returned is conditional on when
        ``next()`` is requested. Since iterators are copied to N separate
        workers, each making independent next requests, the e.g. 64th
        ``next()`` for each worker might yield a different image matrix.

    :param config_path: Filesystem path to the mmdet model configuration for
        use in the model initialization
    :param load_device: The device to load the model onto.
    :param batch_size: Optionally provide prediction batch size override. If
        set, we will override the configuration's ``SOLVER.IMS_PER_BATCH``
        parameter to this integer value. Otherwise, we will use the batch size
        value set to that parameter.
    :param weights_uri: Optional reference to the model weights file to use
        instead of that referenced in the detectron configuration file.
        If not provided, we will
    :param model_lazy_load: If the model should be lazy-loaded on the first
        inference request (``True`` value), or if we should load the model up-
        front (``False`` value).
    :param num_workers: The number of workers to use for data loading. When set
        to ``None`` (the default) we will pull from the detectron config,
        otherwise we will obey this value. See torch ``DataLoader`` for
        ``num_workers`` value meanings.

    """

    def __init__(
        self,
        config_path: str,
        load_device: Union[int, str] = "cuda:0",
        batch_size: Optional[int] = None,
        weights_uri: Optional[str] = None,
        model_lazy_load: bool = True,
        num_workers: Optional[int] = None,
    ):
        self._mmdet_config_path = config_path

        self._load_device_prim = load_device  # int/str reference only
        self._batch_size = batch_size
        self._weights_uri = weights_uri
        self._model_lazy_load = model_lazy_load
        self._num_workers = num_workers

        self._model_device = torch.device(load_device)
        self._model: Optional[torch.nn.Module] = None
        self._classes = None

        if not model_lazy_load:
            self._lazy_load_model()

    def _lazy_load_model(self) -> torch.nn.Module:
        """
        Actually initialize the model and set the weights, storing on the
        requested device. If the model is already initialized, we simply return
        it. This method is idempotent and should always return the same model
        instance once loaded.

        If this fails to initialize the model, then nothing is set to the class
        and ``None`` is returned (reflective of the set model state).
        """
        if self._model is None:
            model = init_detector(self._mmdet_config_path, self._weights_uri, device=self._load_device_prim)
            model.to(self._model_device).eval()
            self._model = model
        self._classes = range(self._model.bbox_head.num_classes)

        return self._model

    def detect_objects(
        self,
        img_iter: Iterable[np.array]
    )-> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        model = self._lazy_load_model()
        batch = []
        with torch.no_grad():
            for batch_input in img_iter:
                batch.append(batch_input)
                if len(batch) < self._batch_size:
                    continue

                batch_output = self._forward(model, batch)
                # For each output, yield an iteration converting outputs into
                # the interface-defined data-types
                for output_dict in batch_output:
                    yield self._format_detections(output_dict)
                batch = []

    def _forward(self, model: torch.nn.Module, batch_inputs: List[Dict[str, Any]]) -> List[Any]:
        """
        Method encapsulating running a forward pass on a model given some batch
        inputs.

        This is a separate method to allow for potential subclasses to override
        this.

        :param model: Torch module as loaded by mmdet to perform forward
            passes with.
        :param batch_inputs: mmdet formatted batch inputs. It can be
            expected that this will follow the format described by [1] 
            which is a list[str/ndarray] or tuple[str/ndarray]

        Returns:
            Sequence of outputs for each batch input. Each item in this output
            is expected to be interpreted by ``_iterate_output``.

        [1]: https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.apis.inference_detector
        """
        return inference_detector(model, batch_inputs)

    def _format_detections(
            self,
            preds
    ):
        # Empty dict to fill
        zero_dict: Dict[Hashable, float] = {lbl: 0. for lbl in self._classes}

        # Loop over each prediction and format result
        formatted_dets = []
        for pred in preds:
            a_bboxes = []
            score_dicts = []
            for i, bbox in enumerate(pred):
                a_bboxes.append(AxisAlignedBoundingBox(
                    [bbox[0], bbox[1]], [bbox[2], bbox[3]]))
                class_dict = zero_dict.copy()
                class_dict[self._classes[i]] = bbox[4]
                score_dicts.append(class_dict)
                break

            formatted_dets.append(list(zip(a_bboxes, score_dicts)))
        return formatted_dets

    def _iterate_output(
        self, single_output: List[Any]
    ) -> Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]:
        """
        Given the model's output for a single image's input, yield out a number
        of ``(AxisAlignedBoundingBox, dict)`` pairs representing detections.

        :param single_output: mmdet formatted results output. 

        """
        bboxes = np.vstack(single_output)
        scores = bboxes[:, -1]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(single_output)
        ]
        labels = np.concatenate(labels)
        cpu_instances = single_output['instances'].to('cpu')
        for box, cls_idx, score in zip(
            bboxes,
            labels,
            scores,
        ):
            yield (
                AxisAlignedBoundingBox(box[:2], box[2:-1]),
                {cls_idx: float(score)}
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "mmdet_config": self._mmdet_config_path,
            "load_device": self._load_device_prim,
            "batch_size": self._batch_size,
            "weights_uri": self._weights_uri,
            "model_lazy_load": self._model_lazy_load,
            "num_workers": self._num_workers,
        }


def _trivial_batch_collator(batch: Any) -> Any:
    """
    A batch collator that does nothing.
    """
    return batch


def _to_tensor(np_image: np.ndarray) -> torch.Tensor:
    """
    Common transform to go from ``[H x W [x C]]`` numpy image matrix to a
    ``[[C x] x H x W]`` torch float32 tensor image. Image pixel scale is left
    alone.
    """
    aug_image = np_image.astype(np.float32)
    if aug_image.ndim == 3:
        aug_image = aug_image.transpose([2, 0, 1])
    return torch.as_tensor(aug_image)


def _aug_one_image(image: np.ndarray, aug: AutoAugment, gt_bboxes=[]) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Common augmentation operation for detectron2 inference passes, performed by
    datasets defined below.

    Args:
        image: Image matrix to be augmented
        aug: Augmentation to be performed on the input image.

    Returns:
        mmdet input with the augmented image tensor and
        original image height and width attributes.
    """
    # sorta replicating detectron2.engine.defaults.DefaultPredictor use of
    # input formatting, which passes along original image height and width
    height, width = image.shape[:2]

    # apply aug. `aug_input` will now contain the changed image matrix after
    # `aug` call.
    aug_input = {'image':image, 'gt_bboxes':gt_bboxes}
    aug(aug_input)

    # convert from numpy-common format to torch.Tensor-common format.
    aug_image = _to_tensor(aug_input['image'])

    return {
        "image": aug_image,
        "height": height,
        "width": width,
    }
