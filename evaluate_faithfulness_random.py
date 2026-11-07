from os import path as osp

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from fovbagnet_classifier.config import ConfigMode, get_config
from fovbagnet_classifier.evaluation import (
    get_dataloader_evaluation,
    get_probabilities,
)
from fovbagnet_classifier.models import (
    MultiscaleDilationModel,
    get_highresnet26_ensemble,
    get_highresnet50,
)

PATH_TO_DEFAULT_CFG = "configs/default.yaml"


def main():
    """Evaluate a model's area under perturbation curve on randonm saliency maps."""
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

    normalize_images = v2.Normalize(mean=cfg.data.normalization.mean, std=cfg.data.normalization.std)

    cfg["data"]["batch_size"] = 1

    dataloader_evaluation = get_dataloader_evaluation(cfg)

    evaluation_resolution = 192  # This is hard-coded for all our experiments with an image resolution of 224

    progress_bar = tqdm(enumerate(dataloader_evaluation), total=len(dataloader_evaluation))
    progress_bar.set_description("Processing validation images")
    with torch.no_grad():
        # precompute empty image
        empty_image = torch.zeros(
            1, 3, evaluation_resolution, evaluation_resolution, device=device, dtype=torch.float32
        )
        empty_pred = model(empty_image, avg_pool=True)
        empty_probs = torch.softmax(empty_pred, dim=1)
        area_linear_sum, area_log2_sum = 0.0, 0.0
        skipped_images = 0
        for index, ((image, label), roi) in progress_bar:
            progress_bar.set_postfix_str(f"Class {label.item()}/{cfg.num_classes - 1}")
            progress_bar.refresh()
            normalized_image = normalize_images(image.to(device))
            logits = model(normalized_image, avg_pool=False).cpu()
            predicted_class = logits.sum(dim=(1, 2)).argmax(dim=1)[0].item()
            shift_logits = (logits.shape[1] - evaluation_resolution) // 2
            cut_logits = logits[
                :,
                shift_logits : evaluation_resolution + shift_logits,
                shift_logits : evaluation_resolution + shift_logits,
            ]

            scores = get_probabilities(cut_logits, label=torch.tensor([predicted_class])).view(
                evaluation_resolution, evaluation_resolution
            )
            scores = torch.randn(evaluation_resolution, evaluation_resolution)
            steps = 16
            # q = torch.linspace(0,1,steps+1)[1:-1]
            a = 2 ** (torch.linspace(1, steps, steps) / 2)
            q = 1 - a / a[-1]
            thresholds = scores.reshape(-1).quantile(q=q, dim=0)

            shift_image = (image.shape[-1] - evaluation_resolution) // 2
            cut_normalized_image = normalized_image[
                :,
                :,
                shift_image : evaluation_resolution + shift_image,
                shift_image : evaluation_resolution + shift_image,
            ]

            initial_pred = model(cut_normalized_image, avg_pool=True)
            initial_probs = torch.softmax(initial_pred, dim=1)
            predicted_class = initial_pred.argmax(dim=1)[0].item()
            initial_prob = initial_probs[0, predicted_class].item()

            curve_x = [0.0]
            curve_y = [0.0]
            perturbation_mask_old_sum = scores.numel()
            for limit in thresholds[:-1]:
                perturbation_mask = scores <= limit
                perturbation_mask_sum = perturbation_mask.sum()
                if perturbation_mask_sum < perturbation_mask_old_sum:
                    perturbed_normalized_image = cut_normalized_image * perturbation_mask.to(device)
                    perturbed_pred = model(perturbed_normalized_image, avg_pool=True)
                    perturbed_probs = torch.softmax(perturbed_pred, dim=1)
                    perturbed_prob = (perturbed_probs)[0, predicted_class].item()
                    curve_x.append(1 - perturbation_mask_sum / perturbation_mask.numel())
                    curve_y.append(initial_prob - perturbed_prob)
                    perturbation_mask_old_sum = perturbation_mask_sum

            # add empty image
            curve_x.append(1.0)
            curve_y.append(initial_prob - empty_probs[0, predicted_class].item())
            curve_x = np.array(curve_x, dtype=np.float32)
            curve_y = np.array(curve_y, dtype=np.float32)
            area_linear = np.trapezoid(x=curve_x, y=curve_y)
            area_log2 = np.trapezoid(x=np.log2(curve_x[1:]), y=curve_y[1:])
            if np.isnan(area_linear) or np.isnan(area_log2):
                skipped_images += 1
                print(f"\n[Warning] NaN detected at index {index}. Skipping this image for AOPC.")
                continue

            area_linear_sum += area_linear
            area_log2_sum += area_log2
    n = len(dataloader_evaluation) - skipped_images
    area_linear_mean = area_linear_sum / n
    area_log2_mean = area_log2_sum / n
    print("AOPC_linear:", area_linear_mean, "AOPC_log2:", area_log2_mean)


if __name__ == "__main__":
    main()
