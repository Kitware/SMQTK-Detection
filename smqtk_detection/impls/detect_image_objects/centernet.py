"""
MIT License

Copyright (c) 2021 GNAYUOHZ
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Iterable, Tuple, Dict, Hashable, List, Optional
import numpy as np
import math
import logging
import warnings

from smqtk_image_io import AxisAlignedBoundingBox

try:
    import torch  # type: ignore
    from torch.cuda.amp import autocast  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.utils.model_zoo as model_zoo  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    import cv2  # type: ignore
    import numba  # type: ignore
except ModuleNotFoundError:
    pass

from smqtk_detection import DetectImageObjects

from importlib.util import find_spec
deps = ['torch', 'cv2', 'numba']
specs = [find_spec(dep) for dep in deps]
usable = all([spec is not None for spec in specs])

logger = logging.getLogger(__name__)


class CenterNetVisdrone(DetectImageObjects):
    """
    Implementation of CenterNet, pretrained on the visdrone2019 dataset. This
    particular implementation is taken from
    https://github.com/GNAYUOHZ/centernet-visdrone
    """

    def __init__(
        self,
        arch: str,
        model_file: str,
        max_dets: int = 100,
        k: int = 500,
        scales: List[float] = None,
        flip: bool = False,
        nms: bool = True,
        use_cuda: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        """
        :param arch: Backbone architecture to use. One of
            'resnet18',
            'resnet34',
            'resnet50',
            'resnet101',
            'resnet152',
            'res2net50',
            'res2net101'.
        :param model_file: .pth file to initialize model with. See linked
            source page above for pretrained model files.
        :param max_dets: Maximum number of detections returned.
        :param K: Number of proposals returned by the CenterNet model before
            filtering. This should be greater than ``max_dets``.
        :param scales: Scales for image pre-processing. Detections from each
            scale are combined for a given image. If this is ``None``, then no
            scaling is done (scale of 1 is used).
        :param flip: Combine detections in horizontally flipped image. This is
            done for each scaled version the images provided.
        :param nms: Use soft-nms to filter repeat detections. This defaults to
            true if more than one scale is used.
        :param use_cuda: Use a CUDA device to compute detections. This defaults
            to false if no such device is available.
        :param batch_size: Number of images to feed to the torch model at once.
        :param num_workers: Number of subprocesses to use for data loading.

        """
        self.model_urls = {
            "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
            "res2net50": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
            "res2net101": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth",
        }
        self.resnet_spec = {
            "resnet18": (_ResNet, _BasicBlock_ResNet, [2, 2, 2, 2]),
            "resnet34": (_ResNet, _BasicBlock_ResNet, [3, 4, 6, 3]),
            "resnet50": (_ResNet, _Bottleneck_ResNet, [3, 4, 6, 3]),
            "resnet101": (_ResNet, _Bottleneck_ResNet, [3, 4, 23, 3]),
            "resnet152": (_ResNet, _Bottleneck_ResNet, [3, 8, 36, 3]),
            "res2net50": (_Res2Net, _Bottleneck_Res2Net, [3, 4, 6, 3]),
            "res2net101": (_Res2Net, _Bottleneck_Res2Net, [3, 4, 23, 3]),
        }
        if arch not in self.model_urls:
            raise ValueError(f"Invalid architecture provided. Must be one of: "
                             f"{list(self.model_urls)}")

        self.arch = arch
        self.model_file = model_file
        self.max_dets = max_dets
        self.k = k
        if scales is None:
            self.scales = [1.0]
        else:
            self.scales = scales
        self.flip = flip
        self.use_cuda = use_cuda
        self.nms = nms
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CenterNet model input size
        self.input_h = 960
        self.input_w = 1280

        self.out_h = self.input_h // 2
        self.out_w = self.input_w // 2

        self.num_classes = 10
        assert self.num_classes == len(CLASS_NAMES), \
            "Mismatch in num_classes record and actual list of class names."

        # visdrone2019 mean and standard deviation, used for normalization
        self.mean = np.asarray([[[0.372949, 0.37837514, 0.36463863]]], dtype=np.float32)
        self.std = np.asarray([[[0.19171683, 0.18299586, 0.19437608]]], dtype=np.float32)

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                logger.info("CUDA device not available, using CPU.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.model: Optional[_CenterNet] = None

    def _get_model(self) -> "_CenterNet":
        """
        Lazy initialize the model. Idempotent.
        """
        if self.model is None:
            heads = {'hm': 10, 'wh': 2, 'reg': 2}   # type: Dict[Hashable, int]
            model = _CenterNet(self.resnet_spec[self.arch], heads)
            model.init_weights(self.model_urls[self.arch], pretrained=True)
            model = model.to(self.device)
            self.model = model = _load_model(model, self.model_file)
            model.eval()
        return self.model

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:

        img_set = _ImageDataset(img_iter, self.flip, self.scales, self._preprocess_img)
        batch_loader = DataLoader(img_set, batch_size=self.batch_size, num_workers=self.num_workers)
        model = self._get_model()

        # need to match flipped batched outputs if using flip and batch size is odd
        align_needed = self.flip and self.batch_size % 2 != 0

        if align_needed:
            score_carry = torch.empty((0, self.num_classes, self.out_h, self.out_w)).to(self.device)
            size_carry = torch.empty((0, 2, self.out_h, self.out_w)).to(self.device)
            offset_carry = torch.empty((0, 2, self.out_h, self.out_w)).to(self.device)

        dets_mat = np.empty((0, self.k, 6))
        for batch_i, (img_tensor, centers, trans_scales) in enumerate(batch_loader):

            logger.info(f"[{batch_i+1}/{len(batch_loader)}]")

            img_tensor = img_tensor.to(self.device)
            centers = centers.cpu().numpy()
            trans_scales = trans_scales.cpu().numpy()

            with torch.no_grad():
                output = model(img_tensor)

            score_maps = output['hm'].sigmoid_()  # detection scores
            size_maps = output['wh']  # bbox sizes
            offset_maps = output['reg']  # bbox offsets

            if align_needed:
                score_maps = torch.vstack((score_carry, score_maps))
                size_maps = torch.vstack((size_carry, size_maps))
                offset_maps = torch.vstack((offset_carry, offset_maps))

                # odd number of score_maps
                if len(score_maps) % 2 == 1:
                    score_carry = score_maps[[-1]]
                    score_maps = score_maps[:-1]

                    size_carry = size_maps[[-1]]
                    size_maps = size_maps[:-1]

                    offset_carry = offset_maps[[-1]]
                    offset_maps = offset_maps[:-1]
                else:
                    score_carry = torch.empty((0, self.num_classes, self.out_h, self.out_w)).to(self.device)
                    size_carry = torch.empty((0, 2, self.out_h, self.out_w)).to(self.device)
                    offset_carry = torch.empty((0, 2, self.out_h, self.out_w)).to(self.device)

            # combine outputs from flipped images
            if self.flip:
                score_maps = (score_maps[0:None:2] + torch.flip(score_maps[1:None:2], [3])) / 2
                size_maps = (size_maps[0:None:2] + torch.flip(size_maps[1:None:2], [3])) / 2
                offset_maps = offset_maps[0:None:2]
                centers = centers[0:None:2]
                trans_scales = trans_scales[0:None:2]

            # score_maps might be empty if batch size is one and flip is used
            if len(score_maps) > 0:
                # decode maps into detections
                dets_mat_batch = _ctdet_decode(score_maps, size_maps, offset_maps, self.k)
                dets_mat_batch = dets_mat_batch.cpu().numpy()

                dets_mat_batch = np.asarray([
                    self._postprocess_dets(dets, center, trans_scale)
                    for dets, center, trans_scale in zip(dets_mat_batch, centers, trans_scales)])

                dets_mat = np.vstack((dets_mat, dets_mat_batch))

        img_dets_list = self._combine_dets(dets_mat)

        formatted_dets = [self._dets_mat_to_list(img_dets_mat) for img_dets_mat in img_dets_list]

        return formatted_dets

    def _preprocess_img(
        self,
        img: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Scales, transforms, and normalizes image for passing to model.
        """

        height, width = img.shape[0:2]

        new_height = int(height * scale)
        new_width = int(width * scale)
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)

        trans_scale = max(height, width) * 1.0

        trans_input = _get_affine_transform(center=center,
                                            scale=trans_scale,
                                            rot=0,
                                            output_size=[self.input_w, self.input_h])

        resized_image = cv2.resize(img, (new_width, new_height))

        trans_img = cv2.warpAffine(
            src=resized_image,
            M=trans_input,
            dsize=(self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR
        )  # type: np.ndarray

        # normalize
        trans_img = (trans_img.astype(np.float32) / 255.)
        trans_img = (trans_img - self.mean) / self.std

        return trans_img, center, trans_scale

    def _postprocess_dets(
        self,
        img_dets: np.ndarray,
        center: List[float],
        trans_scale: float
    ) -> np.ndarray:
        """
        Transforms detections back to original image reference frame.
        """

        trans_dets = []
        trans = _get_affine_transform(center, trans_scale, 0, [self.out_w, self.out_h], inv=1)
        for det in img_dets:
            det[0:2] = _affine_transform(det[0:2], trans)
            det[2:4] = _affine_transform(det[2:4], trans)
            trans_dets.append(det)

        return np.asarray(trans_dets)

    def _combine_dets(
        self,
        dets_mat: np.ndarray
    ) -> List[np.ndarray]:
        """
        Combines scales from different image scales and returns top N
        detections for each image.
        """

        # organize by scale
        # dim0: img, dim1: scale, dim2: detection
        dets_mat = dets_mat.reshape(-1, len(self.scales), self.k, 6)  # organize by scale

        # scale dets
        dets_mat[:, :, :, :4] /= np.asarray(self.scales)[:, None, None]

        img_dets = []

        # combine detections from different image scales
        for img_scales_det_mat in dets_mat:
            img_filtered_dets = np.empty((0, 6))
            for c in range(self.num_classes):
                class_dets = img_scales_det_mat[img_scales_det_mat[:, :, -1] == c]  # all detections of given class
                if len(self.scales) > 1 or self.nms:
                    class_dets = _soft_nms(class_dets, Nt=0.5, method=2, threshold=0.3)  # filter repeat detections
                img_filtered_dets = np.vstack((img_filtered_dets, class_dets))

            img_filtered_dets = img_filtered_dets[img_filtered_dets[:, 4].argsort()]  # sort by score

            img_top_dets = img_filtered_dets[-self.max_dets:][::-1]  # grab top detections

            img_dets.append(img_top_dets)

        return img_dets

    def _dets_mat_to_list(
        self,
        dets_mat: np.ndarray
    ) -> List[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]:
        """
        Converts detection matrix to the format required by the
        ``DetectImageObjects`` interface.
        """
        # empty dict to fill
        zero_dict: Dict[Hashable, float] = {lbl: 0. for lbl in CLASS_NAMES}
        # batch round and cast index column.
        class_indices = dets_mat[:, 5].round().astype(int)
        # Create and collect tuples.
        dets_list = []
        for det, cls_idx in zip(dets_mat, class_indices):
            bbox = AxisAlignedBoundingBox(det[0:2], det[2:4])
            class_dict = zero_dict.copy()
            class_dict[CLASS_NAMES[cls_idx]] = det[4]
            dets_list.append((bbox, class_dict))
        return dets_list

    def get_config(self) -> dict:
        return {
            "arch": self.arch,
            "model_file": self.model_file,
            "max_dets": self.max_dets,
            "k": self.k,
            "scales": self.scales,
            "flip": self.flip,
            "nms": self.nms,
            "use_cuda": self.use_cuda,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    @classmethod
    def is_usable(cls) -> bool:
        if not usable:
            warnings.warn(
                f"CenterNetVisdrone is not usable. Dep status: "
                f"{ {k: (v is not None) for k, v in zip(deps, specs)} }",
                RuntimeWarning,
            )
        return usable


if usable:
    CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                   'tricycle', 'awning-tricycle', 'bus', 'motor']

    class _ImageDataset(Dataset):
        """
        Pytorch dataset that loads flipped and scaled version of images
        """

        def __init__(self, img_iter, flip, scales, transform):  # type: ignore
            self.imgs = list(img_iter)
            self.transform = transform

            mult = len(scales) * (int(flip) + 1)
            num_imgs = len(self.imgs) * mult

            idx_scales = np.repeat(scales, int(flip)+1)

            idx_map = {}
            for idx in range(num_imgs):
                idx_map[idx] = {}

                idx_map[idx]['img_idx'] = math.floor(idx / mult)

                idx_map[idx]['flip'] = (idx % 2 == 1 and flip)

                idx_map[idx]['scale'] = idx_scales[idx % mult]

            self.idx_map = idx_map

        def __len__(self):  # type: ignore
            return len(self.idx_map)

        def __getitem__(self, idx):  # type: ignore
            idx_dict = self.idx_map[idx]

            img = self.imgs[idx_dict['img_idx']]

            img, center, trans_scale = self.transform(img, idx_dict['scale'])

            if idx_dict['flip']:
                img = img[:, ::-1, :]

            img = np.moveaxis(img, -1, 0)
            img = torch.from_numpy(img.copy())

            return img, center, trans_scale

    """
    Everything defined below was taken directly from the original author's
    implementation (https://github.com/GNAYUOHZ/centernet-visdrone).
    """
# ==================== Functions used by CenterNetVisdrone ====================
    def _load_model(model, model_path, optimizer=None, lr=None, lr_step=None):  # type: ignore
        """
        Initialized CenterNet model using provided .pth file.
        """
        start_epoch = 0
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        logger.info(f'loaded {model_path}, epoch {checkpoint["epoch"]}')
        state_dict_ = checkpoint["state_dict"]
        state_dict = {}  # type: Dict[str, torch.Tensor]

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith("module") and not k.startswith("module_list"):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        # check loaded parameters and created model parameters
        msg = (
            "If you see this, your model does not fully load the "
            + "pre-trained weight. Please make sure "
            + "you have correctly specified --arch xxx "
            + "or set the correct --num_classes for your own dataset."
        )
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(
                        f"Skip loading parameter {k}, required shape {model_state_dict[k].shape},\
                        loaded shape {state_dict[k].shape}. {msg}"
                    )
                    state_dict[k] = model_state_dict[k]
            else:
                logger.info(f"Drop parameter {k}. {msg}")
        for k in model_state_dict:
            if k not in state_dict:
                logger.info(f"No param {k}. {msg}")
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        # resume optimizer parameters
        if optimizer is not None:  # not used in smqtk use-case  # pragma: no cover
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = checkpoint["epoch"]
                start_lr = lr
                if lr_step is not None:
                    for step in lr_step:
                        if start_epoch >= step:
                            start_lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = start_lr
                logger.info("Resumed optimizer with start lr", start_lr)
            else:
                logger.info("No optimizer parameters in checkpoint.")
        if optimizer is not None:  # not used in smqtk use-case  # pragma: no cover
            return model, optimizer, start_epoch
        else:
            return model

    def _get_dir(src_point, rot_rad):  # type: ignore
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result

    def _get_3rd_point(a, b):  # type: ignore
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _get_affine_transform(  # type: ignore
        center,
        scale,
        rot,
        output_size,
        shift=np.array([0, 0], dtype=np.float32),
        inv=0
    ):

        scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = _get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def _gather_feat(feat, ind):  # type: ignore

        dim = feat.size(2)  # c

        # ind 2,256 -> 2,256,1 -> 2,256,2
        # Number of sequences in ind fmap
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        # feat 2,76800,2-> 2,256,2
        feat = feat.gather(1, ind)

        return feat

    # Get the value of the corresponding center point calculated in the ground truth
    def _transpose_and_gather_feat(feat, ind):  # type: ignore

        # Some tensors do not occupy a whole block of memory, but are composed of different data blocks,
        # And tensor's view() operation depends on the whole block of memory，
        # At this time, you only need to execute the contiguous() function to turn
        # the tensor into a continuous distribution in the memory.
        feat = feat.permute(0, 2, 3, 1).contiguous()  # batch,c,h,w -> batch,h,w,c
        # Merge wh into one dimension
        feat = feat.view(feat.size(0), -1, feat.size(3))  # batch, w*h,c
        # ind represents the subscript of the target point set in ground truth
        feat = _gather_feat(feat, ind)
        return feat

    # non max suppression
    def _nms(heat, kernel=3):  # type: ignore
        hmax = torch.nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(scores, K=40):  # type: ignore
        batch, cat, height, width = scores.size()
        # The maximum value of statistics for each class channel
        # topk_scores and topk_inds are the top K largest scores and ids in each heatmap (each category) of each batch.
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        # torch.Size([1, 10, 500]) torch.Size([1, 10, 500])

        # Find the abscissa
        topk_ys = torch.true_divide(topk_inds, width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # Take the top K maximum scores and ids of all heatmaps in each batch,
        # without considering the impact of categories
        # topk_score：batch * K
        # topk_ind：batch * K
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        # Find the largest value among all categories
        topk_clses = torch.true_divide(topk_ind, K).int()
        topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    # Convert heatmap to bbox
    def _ctdet_decode(heat, wh, reg, K=100):  # type: ignore
        batch = heat.size(0)

        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)

        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def _affine_transform(pt, t):  # type: ignore
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    @numba.jit(nopython=True, nogil=True)
    def _soft_nms(  # type: ignore
        boxes,
        sigma=0.5,
        Nt=0.3,
        threshold=0.001,
        method=0
    ):  # pragma: no cover
        N = boxes.shape[0]
        pos = 0
        maxscore = 0
        maxpos = 0

        for i in range(N):
            maxscore = boxes[i, 4]
            maxpos = i

            tx1 = boxes[i, 0]
            ty1 = boxes[i, 1]
            tx2 = boxes[i, 2]
            ty2 = boxes[i, 3]
            ts = boxes[i, 4]

            pos = i + 1
            # get max box
            while pos < N:
                if maxscore < boxes[pos, 4]:
                    maxscore = boxes[pos, 4]
                    maxpos = pos
                pos = pos + 1

            # add max box as a detection
            boxes[i, 0] = boxes[maxpos, 0]
            boxes[i, 1] = boxes[maxpos, 1]
            boxes[i, 2] = boxes[maxpos, 2]
            boxes[i, 3] = boxes[maxpos, 3]
            boxes[i, 4] = boxes[maxpos, 4]

            # swap ith box with position of max box
            boxes[maxpos, 0] = tx1
            boxes[maxpos, 1] = ty1
            boxes[maxpos, 2] = tx2
            boxes[maxpos, 3] = ty2
            boxes[maxpos, 4] = ts

            tx1 = boxes[i, 0]
            ty1 = boxes[i, 1]
            tx2 = boxes[i, 2]
            ty2 = boxes[i, 3]
            ts = boxes[i, 4]

            pos = i + 1
            # NMS iterations, note that N changes if detection boxes fall below threshold
            while pos < N:
                x1 = boxes[pos, 0]
                y1 = boxes[pos, 1]
                x2 = boxes[pos, 2]
                y2 = boxes[pos, 3]
                # s = boxes[pos, 4]

                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                iw = (min(tx2, x2) - max(tx1, x1) + 1)
                if iw > 0:
                    ih = (min(ty2, y2) - max(ty1, y1) + 1)
                    if ih > 0:
                        ua = float((tx2 - tx1 + 1) *
                                   (ty2 - ty1 + 1) + area - iw * ih)
                        ov = iw * ih / ua  # iou between max box and detection box

                        if method == 1:  # linear
                            if ov > Nt:
                                weight = 1 - ov
                            else:
                                weight = 1
                        elif method == 2:  # gaussian
                            weight = np.exp(-(ov * ov)/sigma)
                        else:  # original NMS
                            if ov > Nt:
                                weight = 0
                            else:
                                weight = 1

                        boxes[pos, 4] = weight*boxes[pos, 4]

                        # if box score falls below threshold, discard the box by swapping with last box
                        # update N
                        if boxes[pos, 4] < threshold:
                            boxes[pos, 0] = boxes[N-1, 0]
                            boxes[pos, 1] = boxes[N-1, 1]
                            boxes[pos, 2] = boxes[N-1, 2]
                            boxes[pos, 3] = boxes[N-1, 3]
                            boxes[pos, 4] = boxes[N-1, 4]
                            N = N - 1
                            pos = pos - 1

                pos = pos + 1
        # keep = [i for i in range(N)]
        return boxes[:N, :]

# =================== Pytorch modules used by architectures ===================
    def _make_deconv_layer(inplanes, outplanes):  # type: ignore
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=outplanes,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(outplanes))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    class _cat_conv(nn.Module):
        def __init__(self, inc, outc, catc):  # type: ignore
            super(_cat_conv, self).__init__()
            self.conv3 = nn.Conv2d(
                catc, outc, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.deconv = _make_deconv_layer(inc, inc)
            self.bn = nn.BatchNorm2d(outc)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, feat1, feat2):  # type: ignore
            feat2 = self.deconv(feat2)
            cat_feat = torch.cat([feat1, feat2], dim=1)
            out = self.relu(self.bn(self.conv3(cat_feat)))
            return out

    class _fpn_deconv(nn.Module):
        def __init__(self, expansion):  # type: ignore
            super(_fpn_deconv, self).__init__()

            # Top layer
            self.toplayer = nn.Conv2d(512 * expansion, 256, 1, 1, 0)
            # Fpn
            self.elementwise1 = _elementwise(512 * expansion // 2, 256)
            self.elementwise2 = _elementwise(512 * expansion // 4, 256)
            self.elementwise3 = _elementwise(512 * expansion // 8, 256)
            self.reduce_chan = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)

            self.cat_conv1_4 = _cat_conv(64, 64, 128)

        def forward(self, c1_1, c2, c3, c4, c5):  # type: ignore

            # Top-down
            p5_1 = self.toplayer(c5)  # 256 16 16
            p4_1 = self.elementwise1(p5_1, c4)  # 256 32 32
            p3_1 = self.elementwise2(p4_1, c3)  # 256 64 64
            p2_1 = self.elementwise3(p3_1, c2)  # 256 128 128
            last_feat = self.reduce_chan(p2_1)

            # down ratio 2
            last_feat = self.cat_conv1_4(c1_1, last_feat)  # [1, 64, 256, 256]
            return last_feat

    class _elementwise(nn.Module):
        def __init__(self, inc, outc):  # type: ignore
            super(_elementwise, self).__init__()
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, bias=True)
            self.deconv = _make_deconv_layer(outc, outc)
            self.debn = nn.BatchNorm2d(outc)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, p, c):  # type: ignore
            out = self.relu(self.debn(self.deconv(p)) + self.conv1(c))
            return out

# ============================== CenterNet model ==============================

    class _CenterNet(nn.Module):
        def __init__(self, spec, heads):  # type: ignore
            super(_CenterNet, self).__init__()

            arch = spec[0]
            block = spec[1]
            layers = spec[2]
            backbone = arch(block, layers)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

            self.heads = heads
            self.neck = _fpn_deconv(block.expansion)

            # Head
            for head in sorted(self.heads):
                num_output = self.heads[head]
                fc = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, num_output, kernel_size=1, padding=0),
                )
                self.__setattr__(head, fc)

        @autocast()
        def forward(self, x):  # type: ignore
            c1 = self.conv1(x)  # 3
            c1 = self.bn1(c1)
            c1 = self.relu(c1)
            c1_1 = c1  # 64
            c1 = self.maxpool(c1)  # 64

            c2 = self.layer1(c1)  # 256
            c3 = self.layer2(c2)  # 512
            c4 = self.layer3(c3)  # 1024
            c5 = self.layer4(c4)  # 2048

            last_feat = self.neck(c1_1, c2, c3, c4, c5)

            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(last_feat)
            return ret

        def init_weights(self, url, pretrained=True):  # type: ignore
            if pretrained:
                for m in self.modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        nn.init.normal_(m.weight, std=0.001)
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

                for head in self.heads:
                    final_layer = self.__getattr__(head)
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            if m.weight.shape[0] == self.heads[head]:
                                if "hm" in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)

                pretrained_state_dict = model_zoo.load_url(url, map_location='cpu')
                logger.info("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

# ============================ ResNet architectures ===========================

    class _ResNet(nn.Module):
        def __init__(self, block, layers, **kwargs):  # type: ignore
            super(_ResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Bottom-up layers
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        def _make_layer(self, block, planes, blocks, stride=1):  # type: ignore
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

    class _Bottleneck_ResNet(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):  # type: ignore
            super(_Bottleneck_ResNet, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(
                planes, planes * self.expansion, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):  # type: ignore
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class _BasicBlock_ResNet(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):  # type: ignore
            super(_BasicBlock_ResNet, self).__init__()
            self.conv1 = nn.Conv2d(
                inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):  # type: ignore
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class _Res2Net(nn.Module):  # type: ignore
        def __init__(self, block, layers, **kwargs):  # type: ignore
            super(_Res2Net, self).__init__()
            self.inplanes = 64
            self.baseWidth = 26
            self.scale = 4
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        def _make_layer(self, block, planes, blocks, stride=1):  # type: ignore
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    ),
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    stype="stage",
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                )
            )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(
                    block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale)
                )

            return nn.Sequential(*layers)

    class _Bottleneck_Res2Net(nn.Module):
        expansion = 4

        def __init__(  # type: ignore
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            baseWidth=26,
            scale=4,
            stype="normal",
        ):
            """Constructor
            Args:
                inplanes: input channel dimensionality
                planes: output channel dimensionality
                stride: conv stride. Replaces pooling layer.
                downsample: None when stride = 1
                baseWidth: basic width of conv3x3
                scale: number of scale.
                type: 'normal': normal set. 'stage': first block of a new stage.
            """
            super(_Bottleneck_Res2Net, self).__init__()

            width = int(math.floor(planes * (baseWidth / 64.0)))
            self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width * scale)

            if scale == 1:
                self.nums = 1
            else:
                self.nums = scale - 1
            if stype == "stage":
                self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            convs = []
            bns = []
            for i in range(self.nums):
                convs.append(
                    nn.Conv2d(
                        width, width, kernel_size=3, stride=stride, padding=1, bias=False
                    )
                )
                bns.append(nn.BatchNorm2d(width))
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

            self.conv3 = nn.Conv2d(
                width * scale, planes * self.expansion, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stype = stype
            self.scale = scale
            self.width = width

        def forward(self, x):  # type: ignore
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            spx = torch.split(out, self.width, 1)
            for i in range(self.nums):
                if i == 0 or self.stype == "stage":
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
            if self.scale != 1 and self.stype == "normal":
                out = torch.cat((out, spx[self.nums]), 1)
            elif self.scale != 1 and self.stype == "stage":
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out
