
import argparse, os, json
import cv2
import numpy as np
from lomave.preprocess import preprocess
from lomave.filters import build_orientation_bank, apply_filter_bank
from lomave.adaptive_enhance import adaptive_fusion
from lomave.bio_model import binarize, skeletonize_veins, leaf_mask_from_image
from lomave.feature_extraction import extract_features
from lomave.baselines import sobel_edges, prewitt_edges, canny_edges, log_edges
from lomave.utils.metrics import binarize as binarize_float, classification_metrics, ssim_score

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def process_image(path, args, kernels):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    gray = preprocess(img, denoise_method=args.denoise, ksize=args.ksize, sigma=args.sigma, eq_method=args.equalize)
    stack = apply_filter_bank(gray, kernels)
    enhanced, weight_map = adaptive_fusion(stack, gray, var_windows=(7,15,31), gain=(1.0,1.5,2.0))
    binary = binarize(enhanced, "otsu")
    skel = skeletonize_veins(binary, min_size=args.min_size)
    mask = leaf_mask_from_image(gray)

    feats = extract_features(skel, mask)

    # Baselines & metrics (optional, needs GT)
    results = {"features": feats}
    if args.ground_truth_dir:
        base = os.path.basename(path)
        gt_path = os.path.join(args.ground_truth_dir, base)
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt_bin = (gt > 0).astype(np.uint8)
                pred_bin = (binary > 0).astype(np.uint8)
                results["metrics"] = classification_metrics(pred_bin, gt_bin)
                results["ssim"] = ssim_score(enhanced, gt)
        else:
            results["metrics"] = None

    # Baseline edge maps for qualitative comparison
    sob = sobel_edges(gray)
    pre = prewitt_edges(gray)
    can = canny_edges(gray)
    log = log_edges(gray)

    return {
        "gray": gray,
        "enhanced": (enhanced*255).astype(np.uint8),
        "binary": binary,
        "skeleton": (skel*255).astype(np.uint8),
        "weight": (np.clip(weight_map,0,1)*255).astype(np.uint8),
        "sobel": sob,
        "prewitt": pre,
        "canny": can,
        "log": log,
        "report": results
    }

def save_outputs(out_dir, name, outputs):
    cv2.imwrite(os.path.join(out_dir, f"{name}_gray.png"), outputs["gray"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_enhanced.png"), outputs["enhanced"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_binary.png"), outputs["binary"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_skeleton.png"), outputs["skeleton"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_weights.png"), outputs["weight"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_sobel.png"), outputs["sobel"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_prewitt.png"), outputs["prewitt"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_canny.png"), outputs["canny"])
    cv2.imwrite(os.path.join(out_dir, f"{name}_log.png"), outputs["log"])

def main():
    parser = argparse.ArgumentParser(description="LOMAVE: Leaf Oriented Multi-scale Adaptive Vein Enhancement")
    parser.add_argument("--input_dir", type=str, default="data/input")
    parser.add_argument("--output_dir", type=str, default="outputs/images")
    parser.add_argument("--metrics_out", type=str, default="outputs/metrics/results.json")
    parser.add_argument("--ground_truth_dir", type=str, default=None, help="Optional dir with GT edge maps.")
    parser.add_argument("--n_orient", type=int, default=12)
    parser.add_argument("--ksize", type=int, default=21)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--denoise", type=str, default="gaussian", choices=["gaussian","median"])
    parser.add_argument("--equalize", type=str, default="clahe", choices=["clahe","hist"])
    parser.add_argument("--min_size", type=int, default=30)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.metrics_out))

    kernels, thetas = build_orientation_bank(n_orient=args.n_orient, ksize=args.ksize, sigma=3.0, lambd=8.0, gamma=0.5)

    all_reports = {}
    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith((".png",".jpg",".jpeg",".bmp","tif","tiff")):
            fpath = os.path.join(args.input_dir, fname)
            try:
                outs = process_image(fpath, args, kernels)
            except Exception as e:
                print(f"[WARN] {fname}: {e}")
                continue
            stem = os.path.splitext(fname)[0]
            save_outputs(args.output_dir, stem, outs)
            all_reports[fname] = outs["report"]

    with open(args.metrics_out, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Saved metrics to {args.metrics_out}")

if __name__ == "__main__":
    main()
