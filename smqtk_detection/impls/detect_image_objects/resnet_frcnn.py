import importlib.util
import logging
from typing import Tuple, Iterable, Dict, Hashable, List, Union
from types import MethodType

import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

try:
    import torch  # type: ignore
    import torchvision.models as models  # type: ignore
    from torchvision import transforms  # type: ignore
    from torchvision.ops import boxes as box_ops  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torchvision.models.detection.roi_heads import RoIHeads  # type: ignore
except ModuleNotFoundError:
    pass

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects


LOG = logging.getLogger(__name__)


class ResNetFRCNN(DetectImageObjects):
    """
    ``DetectImageObjects`` implementation using ``torchvision``'s Faster R-CNN
    with a ResNet-50-FPN backbone, pretrained on COCO train2017.

    :param box_thresh: Confidence threshold for detections.
    :param num_dets: Maximum number of detections per image.
    :param img_batch_size: Batch size in images for inferences.
    :param use_cuda: Attempt to use a cuda device for inferences. If no
        device is found, CPU is used.
    :param cuda_device: When using CUDA use the device by the given ID. By
        default, this refers to GPU ID 0. This parameter is not used if
        `use_cuda` is false.
    """

    def __init__(
        self,
        box_thresh: float = 0.05,
        num_dets: int = 100,
        img_batch_size: int = 1,
        use_cuda: bool = False,
        cuda_device: Union[int, str] = "cuda:0",
    ):
        self.box_thresh = box_thresh
        self.num_dets = num_dets
        self.img_batch_size = img_batch_size
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        # Set to None for lazy loading later.
        self.model: torch.nn.Module = None  # type: ignore
        self.model_device: torch.device = None  # type: ignore

        # The model already has normalization and resizing baked into the
        # layers.
        self.model_loader = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_model(self) -> "torch.nn.Module":
        """
        Lazy load the torch model in an idempotent manner.

        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self.model
        if model is None:
            model = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                progress=False,
                box_detections_per_img=self.num_dets,
                box_score_thresh=self.box_thresh
            )
            model = model.eval()
            model_device = torch.device('cpu')
            if self.use_cuda:
                if torch.cuda.is_available():
                    model_device = torch.device(device=self.cuda_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError(
                        "Use of CUDA requested, but not available."
                    )
            model.roi_heads.postprocess_detections = (
                MethodType(_postprocess_detections, model.roi_heads)
            )
            # store the loaded model for later return.
            self.model = model
            self.model_device = model_device
        return model

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:

        model = self.get_model()

        # batch model passes
        all_img_dets = []  # type: List[Dict]
        batch = []
        batch_idx = 0
        for img in img_iter:
            batch.append(img)

            if len(batch) is self.img_batch_size:
                batch_tensors = [self.model_loader(batch_img).to(device=self.model_device) for batch_img in batch]

                with torch.no_grad():
                    img_dets = model(batch_tensors)

                for det in img_dets:
                    det['boxes'] = det['boxes'].cpu().numpy()
                    det['scores'] = det['scores'].cpu().numpy()
                    all_img_dets.append(det)

                batch = []

                batch_idx += 1
                LOG.info(f"{batch_idx} batches computed")

        # compute leftover batch
        if len(batch) > 0:
            batch_tensors = [self.model_loader(batch_img).to(device=self.model_device) for batch_img in batch]

            with torch.no_grad():
                img_dets = model(batch_tensors)

            for det in img_dets:
                det['boxes'] = det['boxes'].cpu().numpy()
                det['scores'] = det['scores'].cpu().numpy()
                all_img_dets.append(det)

            batch_idx += 1
            LOG.info(f"{batch_idx} batches computed")

        formatted_dets = []  # AxisAlignedBoundingBox detections to return
        for img_dets in all_img_dets:
            bboxes = img_dets['boxes']
            scores = img_dets['scores']

            a_bboxes = [AxisAlignedBoundingBox(
                [box[0], box[1]], [box[2], box[3]]
            ) for box in bboxes]

            score_dicts = []

            for img_scores in scores:
                score_dict = {}  # type: Dict[Hashable, float]
                # Scores returned start at COCO i.d. 1
                for i, n in enumerate(img_scores, start=1):
                    score_dict[COCO_INSTANCE_CATEGORY_NAMES[i]] = n
                # Don't bother publishing the clobbered "N/A" category.
                del score_dict[COCO_INSTANCE_CATEGORY_NAMES_NA]
                score_dicts.append(score_dict)

            formatted_dets.append(list(zip(a_bboxes, score_dicts)))

        return formatted_dets

    def get_config(self) -> dict:
        return {
            "box_thresh": self.box_thresh,
            "num_dets": self.num_dets,
            "img_batch_size": self.img_batch_size,
            "use_cuda": self.use_cuda,
            "cuda_device": self.cuda_device,
        }

    @classmethod
    def is_usable(cls) -> bool:
        # check for optional dependencies
        torch_spec = importlib.util.find_spec('torch')
        torchvision_spec = importlib.util.find_spec('torchvision')
        if torch_spec is not None and torchvision_spec is not None:
            return True
        else:
            return False


try:
    def _postprocess_detections(
        self: RoIHeads,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Modified bounding box postprocessing function that returns class
        probabilites instead of just a confidence score. Taken from
        https://github.com/XAITK/xaitk-saliency/blob/master/examples/DRISE.ipynb
        """

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            scores_orig = scores.clone()
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            inds = inds[keep]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            inds = inds[keep]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Find corresponding row of matrix
            inds = inds // (num_classes - 1)

            all_boxes.append(boxes)
            all_scores.append(scores_orig[inds, :])
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    # Labels for this pretrained model are detailed here
    # https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    COCO_INSTANCE_CATEGORY_NAMES = (
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )
    COCO_INSTANCE_CATEGORY_NAMES_NA = "N/A"
except NameError:
    pass
