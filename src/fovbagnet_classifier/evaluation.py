"""Evaluation utilities for classification and saliency/explanation tasks.

This module provides functions and classes to compute standard confusion-matrix metrics,
as well as dataset-wide pixel-level ranking for saliency/explanation evaluation.

Key design choices:
- Pixel-level rankings are computed by flattening all ROI pixels across the dataset
  into a single global ranking per score function. This corresponds to evaluating
  the scoring function as a dataset-wide foreground/background discriminator.

- This differs from per-image evaluation, where ROC/PR curves are computed
  separately per image and then averaged. The global formulation is commonly
  used in saliency/explanation benchmarks when the goal is to assess score
  calibration across the dataset rather than within-image ranking quality.

- All images are assumed to share the same spatial resolution, and score maps
  are comparable across samples because they come from the same model and
  preprocessing pipeline.
"""

import csv
import os
import xml.etree.ElementTree as ET
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from fovbagnet_classifier.data import get_dataset
from fovbagnet_classifier.utils import ZipDataset


def accuracy_from_confusion_matrix(m):
    """Compute classification accuracy from a confusion matrix.

    Args:
        m: Square confusion matrix of shape ``(C, C)``.

    Returns:
        Overall accuracy.
    """
    return m.trace() / m.sum()


def ious_from_confusion_matrix(m):
    """Compute per-class Intersection over Union (IoU).

    Args:
        m: Square confusion matrix of shape ``(C, C)``.

    Returns:
        Array containing one IoU value per class.
    """
    intersections = m.diagonal()
    unions = m.sum(axis=0) + m.sum(axis=1) - intersections
    return intersections / unions


def accuracies_from_confusion_matrices(ms):
    """Compute accuracies for multiple confusion matrices.

    Args:
        ms: Array of confusion matrices with shape
            ``(N, C, C)``.

    Returns:
        Accuracy for each confusion matrix.
    """
    return ms.trace(axis1=1, axis2=2) / ms.reshape(ms.shape[0], -1).sum(axis=1)


def ious_from_confusion_matrices(ms):
    """Compute IoUs from binary confusion matrices.

    Args:
        ms: Binary confusion matrices of shape ``(N, 2, 2)``.

    Returns:
        IoU value for each confusion matrix.
    """
    intersections = ms[:, 0, 0]
    unions = ms[:, 0, 0] + ms[:, 0, 1] + ms[:, 1, 0]
    return intersections / unions


def precisions_from_confusion_matrices(ms):
    """Compute precisions from binary confusion matrices."""
    true_positives, false_positives = ms[:, 0, 0], ms[:, 1, 0]
    return true_positives / (true_positives + false_positives)


def recalls_from_confusion_matrices(ms):
    """Compute recalls from binary confusion matrices."""
    true_positives, false_negatives = ms[:, 0, 0], ms[:, 0, 1]
    return true_positives / (true_positives + false_negatives)


class EvaluationConfMat:
    """Accumulate confusion-matrix statistics for classification and localization.

    This class collects dataset-wide evaluation statistics for both image-level
    classification and pixel-level localization. It maintains hard confusion
    matrices based on discrete predictions as well as soft confusion matrices
    based on predicted probabilities.

    Classification metrics are accumulated at the image level using spatially
    aggregated logits. Localization metrics are accumulated independently for
    each class using binary foreground/background predictions within annotated
    regions of interest (ROIs).

    The accumulated confusion matrices can be used to compute standard metrics
    such as accuracy, Intersection over Union (IoU), precision, and recall for
    both hard and soft prediction variants.

    Attributes:
        overall_hard_confusion_matrix:
            Classification confusion matrix based on argmax predictions.

        overall_soft_confusion_matrix:
            Classification confusion matrix accumulated using predicted class
            probabilities.

        roi_hard_binary_confusion_matrices:
            Per-class binary confusion matrices based on hard pixel-wise
            predictions.

        roi_soft_binary_confusion_matrices:
            Per-class binary confusion matrices based on class probabilities.

        num_classes:
            Number of target classes.
    """

    def __init__(self, num_classes):
        """Initialize evaluation statistics storage.

        Args:
            num_classes: Number of target classes.
        """
        # confusion matrices for the accumulated classification task label_pred vs label
        self.overall_hard_confusion_matrix = np.zeros((num_classes, num_classes))
        self.overall_soft_confusion_matrix = np.zeros((num_classes, num_classes))

        # confusion matrices for the intermediate explanation task: activation localized inside the roi?
        self.roi_hard_binary_confusion_matrices = np.zeros((num_classes, 2, 2))
        self.roi_soft_binary_confusion_matrices = np.zeros((num_classes, 2, 2))

        self.num_classes = num_classes

    def update(self, logits, label, roi):
        """Update evaluation statistics for a single image.

        Args:
            logits:
                Dense class logits of shape
                ``(1, H, W, num_classes)``.

            label:
                Ground-truth image class.

            roi:
                Binary region-of-interest mask identifying pixels belonging
                to the target object.

        Notes:
            The implementation assumes a batch size of one. Classification
            metrics are computed from spatially aggregated logits, while ROI
            metrics are computed from pixel-level predictions. ROI metrics
            evaluate whether high-scoring pixels are concentrated inside
            the ground-truth object region, serving as a proxy for explanation
            quality.
        """
        logits = logits.cpu()
        roi = roi.cpu()

        n, h, w, c = logits.shape

        assert n == 1, f"batch_size is required to be 1 but is {n}."
        assert (h, w) == roi.shape[2:], (
            f"logits.shape[1:3]={logits.shape[1:3]} must match roi.shape[2:4]={roi.shape[2:4]}."
        )
        assert c == self.num_classes, f"logits.shape[1]={c} does not match num_classes={self.num_classes}"

        roi = roi[0]

        overall_logits = logits.reshape(-1, self.num_classes).sum(dim=0)
        overall_probs = torch.softmax(overall_logits, dim=-1)
        label_pred = overall_logits.argmax().item()

        probs = torch.softmax(logits, dim=-1)
        probs = probs[0, :, :, label]

        not_probs = 1 - probs
        not_roi = 1 - roi

        preds = (logits.argmax(dim=-1) == label).to(int)
        not_preds = 1 - preds

        hard_true_positives = (preds * roi).sum()
        hard_false_negatives = (not_preds * roi).sum()
        hard_false_positives = (preds * not_roi).sum()
        hard_true_negatives = (not_preds * not_roi).sum()

        soft_true_positives = (probs * roi).sum()
        soft_false_negatives = (not_probs * roi).sum()
        soft_false_positives = (probs * not_roi).sum()
        soft_true_negatives = (not_probs * not_roi).sum()

        self.overall_hard_confusion_matrix[label, label_pred] += 1
        self.overall_soft_confusion_matrix[label] += overall_probs.cpu().numpy()
        self.roi_hard_binary_confusion_matrices[label] += np.array(
            [[hard_true_positives, hard_false_negatives], [hard_false_positives, hard_true_negatives]]
        )
        self.roi_soft_binary_confusion_matrices[label] += np.array(
            [[soft_true_positives, soft_false_negatives], [soft_false_positives, soft_true_negatives]]
        )

    def save(self, path):
        """Save accumulated evaluation statistics to disk."""
        np.savez(
            file=path,
            overall_hard_confusion_matrix=self.overall_hard_confusion_matrix,
            overall_soft_confusion_matrix=self.overall_soft_confusion_matrix,
            roi_hard_binary_confusion_matrices=self.roi_hard_binary_confusion_matrices,
            roi_soft_binary_confusion_matrices=self.roi_soft_binary_confusion_matrices,
        )

    @classmethod
    def load(cls, path):
        """Load evaluation statistics from a saved archive."""
        with np.load(path) as data:
            overall_hard_confusion_matrix = data["overall_hard_confusion_matrix"]
            overall_soft_confusion_matrix = data["overall_soft_confusion_matrix"]
            roi_hard_binary_confusion_matrices = data["roi_hard_binary_confusion_matrices"]
            roi_soft_binary_confusion_matrices = data["roi_soft_binary_confusion_matrices"]

        self = cls(num_classes=roi_hard_binary_confusion_matrices.shape[0])

        self.overall_hard_confusion_matrix = overall_hard_confusion_matrix
        self.overall_soft_confusion_matrix = overall_soft_confusion_matrix
        self.roi_hard_binary_confusion_matrices = roi_hard_binary_confusion_matrices
        self.roi_soft_binary_confusion_matrices = roi_soft_binary_confusion_matrices

        return self

    def get_overall_hard_accuracy(self):  # noqa: D102
        return accuracy_from_confusion_matrix(self.overall_hard_confusion_matrix)

    def get_overall_hard_ious(self):  # noqa: D102
        return ious_from_confusion_matrix(self.overall_hard_confusion_matrix)

    def get_overall_soft_accuracy(self):  # noqa: D102
        return accuracy_from_confusion_matrix(self.overall_soft_confusion_matrix)

    def get_overall_soft_ious(self):  # noqa: D102
        return ious_from_confusion_matrix(self.overall_soft_confusion_matrix)

    def get_roi_hard_accuracies(self):  # noqa: D102
        return accuracies_from_confusion_matrices(self.roi_hard_binary_confusion_matrices)

    def get_roi_hard_ious(self):  # noqa: D102
        return ious_from_confusion_matrices(self.roi_hard_binary_confusion_matrices)

    def get_roi_hard_precisions(self):  # noqa: D102
        return precisions_from_confusion_matrices(self.roi_hard_binary_confusion_matrices)

    def get_roi_hard_recalls(self):  # noqa: D102
        return recalls_from_confusion_matrices(self.roi_hard_binary_confusion_matrices)

    def get_roi_soft_accuracies(self):  # noqa: D102
        return accuracies_from_confusion_matrices(self.roi_soft_binary_confusion_matrices)

    def get_roi_soft_ious(self):  # noqa: D102
        return ious_from_confusion_matrices(self.roi_soft_binary_confusion_matrices)

    def get_roi_soft_precisions(self):  # noqa: D102
        return precisions_from_confusion_matrices(self.roi_soft_binary_confusion_matrices)

    def get_roi_soft_recalls(self):  # noqa: D102
        return recalls_from_confusion_matrices(self.roi_soft_binary_confusion_matrices)


def compute_curves(ranking, step=1):
    """Compute ROC and precision-recall curves from a ranked tensor of scores.

    Note:
        `ranking` is expected to contain scores for all pixels from all evaluation
        images concatenated together (optionally grouped by `step`). As a result,
        the computed ROC-AUC and PR-AUC measure dataset-level foreground/background
        separability rather than the mean per-image localization quality.

        If you want image-wise metrics, you should compute curves per image and
        then average the AUCs.

    Args:
        ranking:
            Binary ranking vector sorted by descending score, where
            positive entries correspond to true positives.

        step:
            Optional grouping factor used when multiple consecutive
            ranking elements should be aggregated.

    Returns:
        Tuple containing:

        - False positive rates
        - True positive rates
        - Area under the ROC curve
        - Recall values
        - Precision values
        - Area under the precision-recall curve
    """
    device = ranking.device
    if step > 1:
        ranking = ranking.view(-1, step).sum(dim=1)
    (n,) = ranking.shape
    true_positives = ranking.cumsum(dim=0)
    num_positives = true_positives[-1]
    num_negatives = (n * step) - num_positives
    pred_positives = step * torch.arange(start=1, end=n + 1, device=device)
    false_positives = pred_positives - true_positives

    # rates for ROC-Curve
    false_positive_rates = torch.cat([torch.zeros(1, device=device), false_positives / num_negatives], dim=0)
    true_positive_rates = torch.cat([torch.zeros(1, device=device), true_positives / num_positives], dim=0)
    area_under_roc = (false_positive_rates.diff() * (true_positive_rates[1:] + true_positive_rates[:-1]) / 2).sum()

    # recalls and precisions for Precision-Recall-Curve
    recalls = torch.cat([torch.zeros(1, device=device), true_positives / num_positives], dim=-1)
    precisions = torch.cat([torch.zeros(1, device=device), true_positives / pred_positives], dim=-1)
    precisions[0] = precisions[-1]
    area_under_pr = (recalls.diff() * (precisions[1:] + precisions[:-1]) / 2).sum()

    return false_positive_rates, true_positive_rates, area_under_roc, recalls, precisions, area_under_pr


def get_logits(logits, label):
    """Return logits corresponding to the ground-truth class."""
    return logits[0, :, :, label[0]]


def get_probabilities(logits, label):
    """Return probabilities corresponding to the ground-truth class."""
    return torch.softmax(logits, dim=-1)[0, :, :, label[0]]


def get_true_vs_strongest_competitor_per_pixel(logits, label):
    """Compute pixel-wise margins against the strongest competing class.

    Returns:
        Difference between the ground-truth class logit and the largest
        non-ground-truth class logit at each pixel.
    """
    label = label[0]
    logits_label = logits[0, :, :, label]
    logits_other = torch.cat((logits[0, :, :, :label], logits[0, :, :, label + 1 :]), dim=-1)
    logits_strongest_other = logits_other.max(dim=-1)[0]
    return logits_label - logits_strongest_other


def get_true_vs_strongest_competitor_overall(logits, label):
    """Compute margins against the strongest image-level competitor.

    The strongest competing class is determined using spatially aggregated
    logits. Pixel-wise margins are then computed relative to that class.
    """
    label = label[0]
    logits_overall = logits.sum(dim=(1, 2))
    strongest_other = torch.cat((logits_overall[:, :label], logits_overall[:, label + 1 :]), dim=-1).argmax(dim=-1)[0]
    if strongest_other >= label:
        strongest_other += 1
    return logits[0, :, :, label] - logits[0, :, :, strongest_other]


def get_num_images_per_class(dataloader, evaluation_resolution=None):
    """Count evaluable images per class.

    Args:
        dataloader:
            Evaluation dataloader providing image-label-mask tuples.

        evaluation_resolution:
            Optional central crop size applied before evaluating masks.

    Returns:
        Tuple containing:

        - Number of evaluable images for each class.
        - Dataset indices whose masks contain no target pixels.
    """
    num_images_per_class = torch.zeros(len(dataloader.dataset.classes), dtype=torch.int64)
    idxs_with_no_actual_class = []
    for idx, ((image, actual_class), mask) in enumerate(dataloader):
        if evaluation_resolution is not None:
            shift = (mask.shape[2] - evaluation_resolution) // 2
            mask = mask[:, :, shift : evaluation_resolution + shift, shift : evaluation_resolution + shift]
        if mask.sum() == 0:
            idxs_with_no_actual_class.append(idx)
        else:
            num_images_per_class[actual_class] += 1

    return [i.item() for i in num_images_per_class], idxs_with_no_actual_class


class SeparatedArray:
    """Store variable-length class-specific tensors efficiently.

    Data for all classes is stored in a single contiguous tensor while
    class-specific views provide convenient access to individual subsets.

    This structure avoids repeated memory allocations while maintaining
    separate storage regions for each class.

    Attributes:
        concatenation:
            Contiguous tensor containing all data.

        views:
            Per-class tensor views into ``concatenation``.
    """

    def __init__(self, lengths: list[int], shape: tuple[int] | None = None, dtype: torch.dtype = torch.float32):
        """Initialize a class-partitioned contiguous tensor.

        Allocates a single contiguous tensor and creates per-class views into
        non-overlapping regions of that tensor. The size of each region is
        determined by ``lengths``.

        Args:
            lengths: Number of elements to allocate for each class-specific partition.
            shape: Optional shape of each stored element. If ``None``, elements are treated as scalars.
            dtype: Data type of the allocated tensor.
        """
        shape = [] if shape is None else shape
        separations = np.cumsum([0] + lengths)
        length_total = separations[-1]
        self.concatenation = torch.empty([length_total] + shape, dtype=dtype)
        self.views = [
            self.concatenation[start_idx:end_idx] for start_idx, end_idx in zip(separations[:-1], separations[1:])
        ]

    def clone(self):
        """Create a deep copy preserving all class-specific views."""
        lengths = [view.shape[0] for view in self.views]
        separations = np.cumsum([0] + lengths)
        clone_of_self = self.__class__.__new__(self.__class__)
        clone_of_self.concatenation = self.concatenation.clone()
        clone_of_self.views = [
            clone_of_self.concatenation[start_idx:end_idx]
            for start_idx, end_idx in zip(separations[:-1], separations[1:])
        ]
        return clone_of_self


class Ranking:
    """Store and analyze score-based rankings.

    Rankings are used to evaluate how well different scoring functions
    separate foreground pixels from background pixels. Rankings may be
    computed globally across all classes or separately for each class.

    Note:
        Rankings are computed by flattening all pixels from all images of a class
        (or overall) into a single vector before sorting. Consequently, ROC and
        Precision-Recall curves computed on this ranking measure dataset-level
        foreground/background separability, not per-image performance.

    Attributes:
        num_classes:
            Number of classes.

        ranked_labels_overall:
            Global ranking across all classes.

        ranked_labels_per_class:
            Dictionary containing class-specific rankings.

        key:
            Name of the score function used to generate the ranking.
    """

    def __init__(
        self,
        num_classes: int,
        ranked_labels_overall: torch.Tensor,
        ranked_labels_per_class: torch.Tensor,
        key: str = None,
    ):
        """Initialize a ranking evaluation object.

        Args:
            num_classes:
                Number of classes represented in the ranking.

            ranked_labels_overall:
                One-dimensional tensor containing foreground/background labels
                sorted according to the global ranking induced by a score
                function across all classes.

            ranked_labels_per_class:
                Mapping from class-specific ranking identifiers to tensors
                containing foreground/background labels sorted by score within
                each class.

            key:
                Optional name of the score function used to generate the
                rankings. This value is stored for identification when saving
                or comparing rankings.
        """
        self.num_classes = num_classes
        self.ranked_labels_overall = ranked_labels_overall
        self.ranked_labels_per_class = ranked_labels_per_class
        self.key = key

    @classmethod
    def from_scores(cls, labels, scores, key=None):
        """Create rankings from binary labels and continuous scores."""
        labels, scores = labels.clone(), scores.clone()

        def permute_elements(x, args):
            return x.view(-1)[args].view(x.shape)

        for label, score in zip(labels.views, scores.views):
            args = score.view(-1).argsort(descending=True)
            label[:], score[:] = permute_elements(label, args), permute_elements(score, args)

        args_overall = scores.concatenation.view(-1).argsort(descending=True)
        ranked_labels_overall = labels.concatenation.view(-1)[args_overall]
        ranked_labels_per_class = {
            f"ranking_{class_id}": label.view(-1) for class_id, label in enumerate(labels.views) if label.numel() > 0
        }

        return cls(
            num_classes=len(labels.views),
            ranked_labels_overall=ranked_labels_overall,
            ranked_labels_per_class=ranked_labels_per_class,
            key=key,
        )

    @classmethod
    def load(cls, path: str):
        """Load a ranking saved by the save method of this class.

        Args:
            path:
                Path to the ranking that should be loaded.
        """
        with np.load(path) as data:
            num_classes = int(data["num_classes"])
            return cls(
                num_classes=num_classes,
                ranked_labels_overall=torch.from_numpy(data["ranking_overall"]),
                ranked_labels_per_class={
                    f"ranking_{class_id}": torch.from_numpy(data[f"ranking_{class_id}"])
                    for class_id in range(num_classes)
                    if f"ranking_{class_id}" in data
                },
                key=str(data["key"]) if "key" in data else None,
            )

    def save(self, path: str):
        """Save a ranking on the file system.

        Args:
            path:
                Path where the ranking should be saved.
        """
        np.savez(
            file=path,
            key=self.key,
            num_classes=self.num_classes,
            ranking_overall=self.ranked_labels_overall,
            **self.ranked_labels_per_class,
        )

    def get_curves_overall(self, step=1):
        """Compute ROC and PR curves for the global ranking."""
        return compute_curves(ranking=self.ranked_labels_overall, step=step)

    def get_curves(self, class_id, step=1):
        """Compute ROC and PR curves for a specific class."""
        if f"ranking_{class_id}" in self.ranked_labels_per_class:
            return compute_curves(ranking=self.ranked_labels_per_class[f"ranking_{class_id}"], step=step)
        else:
            return None


class EvaluationRankScores:
    """Accumulate pixel-level scores for ranking evaluation.

    This class stores ROI labels and score maps produced by multiple
    scoring functions. After all samples have been processed, rankings
    can be generated and evaluated using ROC and precision-recall
    analysis.

    Attributes:
        score_functions:
            Mapping from score names to score computation functions.

        rois:
            Ground-truth ROI masks.

        scores:
            Stored score maps for each scoring function.
    """

    def __init__(self, score_functions, num_classes, num_images_per_class, resolution):
        """Initialize storage for pixel-level ranking evaluation.

        Preallocates storage for ROI masks and score maps produced by multiple
        scoring functions. Data are organized per class to enable efficient
        construction of class-specific and global rankings after all evaluation
        samples have been processed.

        Args:
            score_functions:
                Mapping from score-function names to callables. Each callable
                must accept ``(logits, label)`` and return a score map used for
                ranking evaluation.

            num_classes:
                Number of target classes.

            num_images_per_class:
                Number of evaluable images for each class. These values are used
                to determine the amount of storage allocated per class.

            resolution:
                Spatial shape of stored ROI masks and score maps, typically ``(H, W)``.
        """
        self.rois = SeparatedArray(lengths=num_images_per_class, shape=resolution, dtype=torch.int8)
        self.scores = {
            key: SeparatedArray(lengths=num_images_per_class, shape=resolution, dtype=torch.float16)
            for key in score_functions.keys()
        }
        self.current_idxs = torch.zeros(num_classes, dtype=torch.int64)

        self.score_functions = score_functions
        self.num_classes = num_classes
        self.num_images_per_class = num_images_per_class

    def update(self, logits, label, roi):
        """Store scores and ROI labels for a single image."""
        shape = self.rois.concatenation.shape[1:]
        current_index = self.current_idxs[label]
        self.rois.views[label][current_index] = roi.to(torch.int8)
        for key, score_function in self.score_functions.items():
            self.scores[key].views[label][current_index] = score_function(logits, label).to(torch.float16).view(shape)

        self.current_idxs[label] += 1

    def get_ranking(self, key):
        """Construct a ranking for a given score function."""
        assert list(self.current_idxs) == self.num_images_per_class
        return Ranking.from_scores(labels=self.rois, scores=self.scores[key], key=key)


class BoundingBoxVal(Dataset):
    """ImageNet validation bounding-box annotations.

    This dataset provides binary ROI masks derived from ImageNet
    localization bounding-box annotations and aligned with the image
    preprocessing pipeline used during validation.

    Each sample returns a binary mask indicating the spatial extent of
    objects belonging to the image's ground-truth class.
    """

    def __init__(
        self,
        path_to_annotations: str,
        label_codes: list[str],
        imgs: list[tuple[str, int]],
        image_resolution: int,
        validation_crop_ratio: float,
    ):
        """Initializes the ImageNet validation bounding-box dataset.

        Loads ImageNet CLS-LOC validation annotations and prepares a dataset
        that converts bounding-box annotations into binary ROI masks aligned
        with the image preprocessing pipeline.

        The preprocessing pipeline resizes images to a fixed resolution,
        applies center cropping, and converts bounding boxes into spatial
        masks.

        Args:
            path_to_annotations: Root directory containing ImageNet CLS-LOC
                validation annotations.
            label_codes: Mapping from dataset label indices to ImageNet class
                codes used in XML annotation files.
            imgs: List of (image_path, label) tuples for the validation set.
            image_resolution: Target spatial resolution (H, W) of processed masks.
            validation_crop_ratio: Factor used to compute resize resolution prior
                to center cropping.
        """
        validation_resize_resolution = round(image_resolution / validation_crop_ratio)

        self.transform = v2.Compose(
            [
                v2.Resize(validation_resize_resolution, interpolation=v2.InterpolationMode.NEAREST),
                v2.CenterCrop(image_resolution),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.index_mapping = [int(x[0].split("/")[-1][15:23]) - 1 for x in imgs]
        self.path_to_annotations = osp.join(
            path_to_annotations,
            "val",
        )
        filenames = os.listdir(self.path_to_annotations)
        assert len(filenames) == 50000

        self.labels = np.array(list((zip(*imgs)))[1])
        self.label_codes = label_codes

    def __len__(self):  # noqa: D105
        return 50000

    def __getitem__(self, index: int) -> torch.tensor:
        """Retrieves a binary ROI mask for an ImageNet validation image.

        Loads the corresponding XML annotation file, extracts all bounding
        boxes matching the image's ground-truth class, and rasterizes them
        into a binary mask. The mask is then transformed using the same
        preprocessing pipeline applied to validation images.

        Args:
            index: Index of the validation sample.

        Returns:
            A binary ROI mask of shape (1, H, W) where foreground pixels
            correspond to bounding boxes of the ground-truth class.
        """
        label = self.labels[index]
        label_code = self.label_codes[label]
        index = self.index_mapping[index]
        xml_filename = f"ILSVRC2012_val_{index + 1:08}.xml"
        path_to_xml_file = osp.join(self.path_to_annotations, xml_filename)

        xml_tree = ET.parse(path_to_xml_file)
        xml_root = xml_tree.getroot()

        size = xml_root.find("size")
        height = int(size.find("height").text)
        width = int(size.find("width").text)
        bounding_boxes = []
        bndboxs = []
        for obj in xml_root.findall("object"):
            name = obj.find("name").text
            if name == label_code:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                bounding_boxes.append((xmin, ymin, xmax, ymax))
                bndboxs.append(bndbox)

        roi = torch.zeros(1, height, width)
        for xmin, ymin, xmax, ymax in bounding_boxes:
            roi[:, ymin:ymax, xmin:xmax] = 1

        roi = self.transform(roi)

        return roi


def get_dataset_val_with_bbox(cfg):
    """Create evaluation dataset using ImageNet bounding-box annotations."""
    dataset_val_images = get_dataset(cfg, mode="val")
    label_codes = sorted(os.listdir(osp.join(cfg.data.path_to_image_root, "val")))
    path_to_bbox_root = osp.normpath(cfg.path_to_annotations_root)
    dataset_val_bbox = BoundingBoxVal(
        path_to_bbox_root,
        label_codes,
        dataset_val_images.imgs,
        image_resolution=cfg.data.image_resolution,
        validation_crop_ratio=cfg.data.augmentation.validation_crop_ratio,
    )
    dataset = ZipDataset(dataset_val_images, dataset_val_bbox)
    dataset.classes = dataset_val_images.classes
    return dataset


class ImageNetSVal(ImageFolder):
    """ImageNet-S validation dataset wrapper.

    This dataset aligns ImageNet-S segmentation masks with ImageNet
    validation images and produces binary masks corresponding to the
    target object class.

    For images containing multiple object categories, the most prominent
    class in the segmentation mask is used when the annotated primary
    class is absent.

    Returns:
        ((image, class_label), roi_mask)
    """

    def __init__(self, path_to_imagenet_s: str, imagenet_val: Dataset):
        """Initializes the ImageNet-S validation dataset wrapper.

        Aligns ImageNet-S segmentation masks with ImageNet validation images
        and builds an index mapping between ImageNet validation samples and
        corresponding segmentation masks.

        The dataset also constructs a transformation pipeline for processing
        segmentation masks and maps ImageNet-S class labels to ImageNet class
        indices.

        Args:
            path_to_imagenet_s: Path to the ImageNet-S dataset directory.
            imagenet_val: An ImageNet validation dataset instance (typically
                torchvision.datasets.ImageFolder) used as the base image source.
        """
        super().__init__(osp.join(path_to_imagenet_s, "validation-segmentation"))

        with open(osp.join(path_to_imagenet_s, "ImageNetS_im919_mapping_validation.txt"), newline="") as csvfile:
            ids = [(left[:33], right[:33]) for left, right in csv.reader(csvfile, delimiter=" ")]

        ids_sorted = sorted(ids, key=lambda image_id_mask_id: image_id_mask_id[0])
        image_ids, mask_ids = zip(*ids_sorted)
        image_ids, mask_ids = list(image_ids), list(mask_ids)

        imagenet_val_idxs = []
        image_ids_iter = iter(image_ids + [None])
        image_id = next(image_ids_iter)

        def id_from_path(path_to_image):
            path_to_class_dir, image_name = osp.split(path_to_image)
            _, class_id = osp.split(path_to_class_dir)
            image_id = image_name[:23]
            return f"{class_id}/{image_id}"

        for j, (path_to_image, _) in enumerate(imagenet_val.imgs):
            if image_id == id_from_path(path_to_image):
                imagenet_val_idxs.append(j)
                image_id = next(image_ids_iter)
        assert image_id is None

        idx_to_imagenet_val_idx, _ = zip(*sorted(zip(imagenet_val_idxs, mask_ids), key=lambda x: x[1]))
        self.idx_to_imagenet_val_idx = np.array(idx_to_imagenet_val_idx)
        self.imagenet_val = imagenet_val
        self.transform = v2.Compose(
            [
                v2.Resize(
                    [int(0.9375 * imagenet_val.transform.transforms[0].size[0])],  # 240 seems to be a perfect match.
                    interpolation=v2.InterpolationMode.NEAREST,
                ),
                v2.CenterCrop(imagenet_val.transform.transforms[1].size),
                v2.ToImage(),
                v2.ToDtype(torch.int64),
            ]
        )
        self.wnids, self.classes = self.classes, self.imagenet_val.classes
        self.key_mapping = torch.tensor([self.imagenet_val.wnids.index(key) for key in self.wnids])

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, int], torch.Tensor]:
        """Retrieves an ImageNet-S sample and corresponding segmentation mask.

        For a given index, returns the ImageNet validation image paired with a
        binary region-of-interest mask derived from ImageNet-S segmentation
        annotations. When multiple object categories are present, the most
        prominent class in the segmentation mask is used unless the annotated
        primary class is present.

        Args:
            index: Index of the sample in the ImageNet-S dataset.

        Returns:
            tuple:
                ((image, class_label), roi_mask)

                image:
                    The ImageNet validation image.
                class_label:
                    The resolved ImageNet-object class label for the image.
                roi_mask:
                    Binary mask of shape (1, H, W) indicating pixels belonging
                    to the selected class.
        """
        image, _ = self.imagenet_val[self.idx_to_imagenet_val_idx[index]]
        mask_rgb, primary_class = super().__getitem__(index)
        mask_rgb = self.transform(mask_rgb)
        mask = mask_rgb[0] + 256 * mask_rgb[1] - 1
        ignored_mask = mask == 999
        mask[ignored_mask] = -1
        foreground_mask = 0 <= mask
        mask[foreground_mask] = self.key_mapping[mask[foreground_mask]]
        primary_class = self.key_mapping[primary_class]

        # estimate the actual class
        keys_in_mask = torch.unique(mask)
        objects_in_mask = keys_in_mask[keys_in_mask >= 0]
        if objects_in_mask.numel() > 0:
            most_prominent_class = objects_in_mask[(mask[:, :, None] == objects_in_mask).sum(dim=(0, 1)).argmax()]
            actual_class = primary_class if primary_class in objects_in_mask else most_prominent_class
        else:
            actual_class = primary_class

        return (image, actual_class), (mask == actual_class).to(torch.float32)[None]


def get_dataset_val_with_mask(cfg):
    """Create evaluation dataset using ImageNet-S segmentation masks."""
    imagenet_val = get_dataset(cfg, mode="val")
    path_to_imagenet_s = osp.normpath(cfg.path_to_annotations_root)
    return ImageNetSVal(
        path_to_imagenet_s=path_to_imagenet_s,
        imagenet_val=imagenet_val,
    )


def get_dataloader_evaluation(cfg):
    """Create the evaluation dataloader.

    Depending on the configuration, evaluation is performed using either
    bounding-box annotations or segmentation masks.
    """
    return DataLoader(
        dataset=(get_dataset_val_with_mask if cfg.evaluate_on_masks else get_dataset_val_with_bbox)(cfg),
        shuffle=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=2,
        persistent_workers=cfg.data.persistent_workers,
        drop_last=False,
    )
