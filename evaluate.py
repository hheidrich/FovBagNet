"""Evaluate trained BagNet-based classifiers on localization benchmarks.

This script loads a model checkpoint and performs post-training evaluation
on the validation dataset. In addition to standard classification accuracy,
it computes localization-aware metrics based on annotated regions of
interest (ROIs).

Supported models:
- MultiscaleDilationModel
- HighResNet50
- Highresnet26 Ensemble

For each evaluated checkpoint, the script computes:

- Confusion-matrix-based localization statistics
- Class ranking metrics using multiple score functions
- Top-k classification accuracy

Results are stored as NumPy archives in an ``evaluation`` directory
alongside the checkpoint outputs.

The evaluation operates on a fixed spatial resolution and processes
one image at a time to preserve pixel-level score maps.
"""

import os
from os import path as osp

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from fovbagnet_classifier.config import ConfigMode, get_config
from fovbagnet_classifier.evaluation import (
    EvaluationConfMat,
    EvaluationRankScores,
    get_dataloader_evaluation,
    get_logits,
    get_num_images_per_class,
    get_probabilities,
    get_true_vs_strongest_competitor_overall,
    get_true_vs_strongest_competitor_per_pixel,
)
from fovbagnet_classifier.metrics import AccuracyTopK
from fovbagnet_classifier.models import (
    MultiscaleDilationModel,
    get_highresnet26_ensemble,
    get_highresnet50,
)

PATH_TO_DEFAULT_CFG = "configs/default.yaml"


def main():
    """Run evaluation for a trained checkpoint.

    The evaluation workflow consists of:

    1. Loading the experiment configuration and checkpoint.
    2. Reconstructing the trained model architecture.
    3. Running inference on the evaluation dataset.
    4. Computing localization and ranking-based evaluation metrics.
    5. Computing top-k classification accuracy.
    6. Saving all evaluation results to disk.
    7. Print a selection of resulting metrics to std out.

    Evaluation is performed using ROI annotations (bounding boxes or
    segmentation masks depending on configuration) and produces output
    files that can be used for quantitative analysis and visualization.
    """
    cfg = get_config(PATH_TO_DEFAULT_CFG, config_mode=ConfigMode.EVALUATE_CHECKPOINT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    supported_models = {
        "multiscale_dilation_model": MultiscaleDilationModel,
        "highresnet50": get_highresnet50,
        "highresnet26_ensemble": get_highresnet26_ensemble,
    }
    assert cfg.model.name in supported_models.keys(), (
        f"model name is {cfg.model.name} but this evaluation script does only support {supported_models.keys()}."
    )
    kwargs = dict(cfg.model.kwargs)
    kwargs["stride"] = 1
    model = supported_models[cfg.model.name](**kwargs).to(device)

    state_dict = torch.load(cfg.ckpt, weights_only=False)
    state_dict = {
        key.removeprefix("model.").replace("level_to_model", "models"): val
        for key, val in state_dict["state_dict"].items()
    }

    model.load_state_dict(state_dict=state_dict)
    model = model.eval()
    if cfg.model.compile:
        model = torch.compile(model, dynamic=False)

    checkpoint_dir, checkpoint_name = osp.split(osp.normpath(cfg.ckpt))
    epoch = int(checkpoint_name.removeprefix("epoch=").removesuffix(".ckpt"))

    normalize_images = v2.Normalize(mean=cfg.data.normalization.mean, std=cfg.data.normalization.std)

    cfg["data"]["batch_size"] = 1

    path_to_evaluation = osp.join(
        osp.dirname(checkpoint_dir), "evaluation", "mask" if cfg.evaluate_on_masks else "bbox"
    )
    dataloader_evaluation = get_dataloader_evaluation(cfg)

    evaluation_conf_mat = EvaluationConfMat(num_classes=cfg.num_classes)

    evaluation_resolution = 192  # This is hard-coded for all our experiments with an image resolution of 224
    num_images_per_class, _ = get_num_images_per_class(
        dataloader_evaluation, evaluation_resolution=evaluation_resolution
    )
    evaluation_rank_scores = EvaluationRankScores(
        score_functions={
            "logits": get_logits,
            "probabilities": get_probabilities,
            "true_vs_strongest_competitor_per_pixel": get_true_vs_strongest_competitor_per_pixel,
            "true_vs_strongest_competitor_overall": get_true_vs_strongest_competitor_overall,
        },
        num_classes=cfg.num_classes,
        num_images_per_class=num_images_per_class,
        resolution=[evaluation_resolution, evaluation_resolution],
    )
    accuracy_top_k = AccuracyTopK(max_top_k=5, num_classes=cfg.num_classes)
    progress_bar = tqdm(enumerate(dataloader_evaluation), total=len(dataloader_evaluation))
    progress_bar.set_description("Processing validation images")
    with torch.no_grad():
        for index, ((image, label), roi) in progress_bar:
            progress_bar.set_postfix_str(f"Class {label.item()}/{cfg.num_classes - 1}")
            progress_bar.refresh()

            shift_roi = (roi.shape[2] - evaluation_resolution) // 2
            roi = roi[
                :, :, shift_roi : evaluation_resolution + shift_roi, shift_roi : evaluation_resolution + shift_roi
            ]

            if roi.sum() > 0:  # ignore images where the labeled class is absent
                image_normalized = normalize_images(image.to(device))

                logits = model(image_normalized, avg_pool=False).cpu()

                shift_logits = (logits.shape[1] - evaluation_resolution) // 2
                logits = logits[
                    :,
                    shift_logits : evaluation_resolution + shift_logits,
                    shift_logits : evaluation_resolution + shift_logits,
                ]

                evaluation_conf_mat.update(logits=logits, roi=roi, label=label)
                evaluation_rank_scores.update(logits=logits, roi=roi, label=label)
                accuracy_top_k.update(None, groundtruth=torch.tensor([label]), prediction=logits.sum(dim=[1, 2]))

    os.makedirs(name=path_to_evaluation, exist_ok=True)
    evaluation_conf_mat.save(path=osp.join(path_to_evaluation, f"epoch_{epoch}_confmat.npz"))

    score_function_key_to_area_under_roc = {}
    score_function_key_to_area_under_pr = {}
    for key in evaluation_rank_scores.score_functions.keys():
        ranking = evaluation_rank_scores.get_ranking(key=key)
        ranking.save(path=osp.join(path_to_evaluation, f"epoch_{epoch}_rankings_{key}.npz"))
        _, _, area_under_roc, _, _, area_under_pr = ranking.get_curves_overall(step=192**2)
        score_function_key_to_area_under_roc[key] = area_under_roc.item()
        score_function_key_to_area_under_pr[key] = area_under_pr.item()
    accuracy_top_k.save_cumulative_counts(
        path=osp.join(path_to_evaluation, f"epoch_{epoch}_accuracy_top_{accuracy_top_k.max_top_k}.npz")
    )
    print_results(
        evaluation_conf_mat,
        accuracy_top_k,
        evaluation_rank_scores,
        score_function_key_to_area_under_roc,
        score_function_key_to_area_under_pr,
    )


def valid_mean(xs: torch.Tensor):
    return xs[np.logical_not(np.isnan(xs))].mean()


def print_results(
    evaluation_conf_mat,
    accuracy_top_k,
    evaluation_rank_scores,
    score_function_key_to_area_under_roc,
    score_function_key_to_area_under_pr,
):
    for key, val in {
        "Accuracy Top-1": accuracy_top_k.get_top_k_accuracy(1).item(),
        "Accuracy Top-5": accuracy_top_k.get_top_k_accuracy(5).item(),
        "ROI Accuracy": valid_mean(evaluation_conf_mat.get_roi_hard_accuracies()).item(),
        "ROI Precision": valid_mean(evaluation_conf_mat.get_roi_hard_precisions()).item(),
        "ROI Recall": valid_mean(evaluation_conf_mat.get_roi_hard_recalls()).item(),
    }.items():
        print(f"{key}: {val}")

    for key in evaluation_rank_scores.score_functions.keys():
        print(f"{key} ROC-AUC: {score_function_key_to_area_under_roc[key]}")
        print(f"{key} PR-AUC: {score_function_key_to_area_under_pr[key]}")


if __name__ == "__main__":
    main()
