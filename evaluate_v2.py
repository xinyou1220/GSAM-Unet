import argparse, csv, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects
from tqdm import tqdm

from unet_v2        import UNetV2, soft_skeletonize
from data_loader_v2 import get_loader, VESSEL_TO_ID


FIXED_VIS_SAMPLES = {
    "LAD": {"Hard": [74, 128], "Simple": [158, 184]},
    "LCx": {"Hard": [75, 252], "Simple": [198,  97]},
    "RCA": {"Hard": [25, 114], "Simple": [ 99, 167]},
}

def dice_score(pred, gt, smooth=1e-6):
    p, g = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    return (2.0 * (p & g).sum() + smooth) / (p.sum() + g.sum() + smooth)


@torch.no_grad()
def cldice_score(prob, gt, n_iter=15, smooth=1e-6):
    p  = prob.unsqueeze(0).unsqueeze(0)
    g  = gt.unsqueeze(0).unsqueeze(0)
    sp = soft_skeletonize(p, n_iter)
    sg = soft_skeletonize(g, n_iter)
    tprec = ((sp * g).sum() + smooth) / (sp.sum() + smooth)
    tsens = ((sg * p).sum() + smooth) / (sg.sum() + smooth)
    return (2.0 * tprec * tsens / (tprec + tsens + 1e-7)).item()


@torch.no_grad()
def predict_with_tta(model, imgs, vids, device, use_amp=True):
    flips = [
        (lambda x: x,                           lambda x: x),
        (lambda x: torch.flip(x, dims=[-1]),    lambda x: torch.flip(x, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]),    lambda x: torch.flip(x, dims=[-2])),
        (lambda x: torch.flip(x, dims=[-1,-2]), lambda x: torch.flip(x, dims=[-1,-2])),
    ]
    prob_sum = None
    for fwd, inv in flips:
        x = fwd(imgs)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_amp)):
            out = model(x, vids)
            seg = out[0] if isinstance(out, tuple) else out
        prob      = torch.sigmoid(seg[:, 1].float())
        prob_back = inv(prob.unsqueeze(1)).squeeze(1)
        prob_sum  = prob_back if prob_sum is None else prob_sum + prob_back
    return prob_sum / len(flips)


def reconnect_by_dist(prob, high_thr=0.45, low_thr=0.25, max_iter=6):
    if hasattr(prob, "cpu"): prob = prob.cpu().numpy()
    seed = prob > high_thr
    cand = prob > low_thr
    if not seed.any(): return cand
    cur = seed.copy()
    for _ in range(max_iter):
        nxt = binary_dilation(cur, iterations=1) & cand
        if nxt.sum() == cur.sum(): break
        cur = nxt
    return cur


def postprocess(prob_np, high_thr=0.50, low_thr=0.25, min_size=50,
                max_reconnect_iter=6):
    if hasattr(prob_np, "cpu"): prob_np = prob_np.cpu().numpy()
    if low_thr is not None and low_thr < high_thr:
        mask = reconnect_by_dist(prob_np, high_thr=high_thr, low_thr=low_thr,
                                 max_iter=max_reconnect_iter)
    else:
        mask = prob_np > high_thr
    mask = mask.astype(bool)
    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)
    return mask

def plot_training_curves(csv_path, out_dir):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows: return

    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves (V2)", fontsize=14, fontweight="bold")

    def _plot(ax, key_tr, key_va, title, ylim=None):
        if key_tr not in rows[0] or key_va not in rows[0]: return
        ax.plot(epochs, [float(r[key_tr]) for r in rows], label="Train", lw=1.5)
        ax.plot(epochs, [float(r[key_va]) for r in rows], label="Val",   lw=1.5)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
        if ylim: ax.set_ylim(ylim)

    _plot(axes[0], "tr_total",      "va_total",      "Total Loss")
    _plot(axes[1], "tr_dice_coeff", "va_dice_coeff", "Dice Coefficient", (0, 1))

    ax = axes[2]
    ax.plot(epochs, [float(r["lr"]) for r in rows], lw=1.5, color="C4")
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(alpha=0.3)

    if "tr_reid_loss" in rows[0]:
        ax2 = ax.twinx()
        ax2.plot(epochs, [float(r.get("tr_reid_loss", 0)) for r in rows],
                 lw=1.0, color="C1", alpha=0.7, label="ReID Loss")
        ax2.set_ylabel("ReID Loss", color="C1")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Training curves → {out_dir / 'training_curves.png'}")


@torch.no_grad()
def find_best_threshold(model, loader, device,
                        high_thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
                        low_thr_ratios=(None, 0.4, 0.5, 0.6, 0.7),
                        min_size=50, use_tta=True, use_amp=True, set_name="Val"):
    model.eval()
    all_probs, all_masks = [], []

    for imgs, masks, vids, _ in tqdm(loader, desc=f"  Sweep[{set_name}]",
                                     leave=False, dynamic_ncols=True, unit="b"):
        imgs = imgs.to(device, non_blocking=True)
        vids = vids.to(device, non_blocking=True)
        if use_tta:
            prob = predict_with_tta(model, imgs, vids, device, use_amp)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs, vids)
                seg = out[0] if isinstance(out, tuple) else out
            prob = torch.sigmoid(seg[:, 1].float())
        for i in range(imgs.size(0)):
            all_probs.append(prob[i].cpu().numpy())
            all_masks.append(masks[i].numpy().astype(bool))

    n = len(all_probs)
    print(f"\n[Threshold Sweep on {set_name}] ({n} samples)")
    print(f"  {'high':>6} {'low':>6}  {'Dice':>7}  {'Prec':>7}  {'Rec':>7}")
    best_h, best_l, best_d = 0.5, None, -1.0

    for ht in high_thresholds:
        for lr in low_thr_ratios:
            lt = ht * lr if lr is not None else None
            inter_sum = pred_sum = gt_sum = 0.0
            for prob_np, mask_np in zip(all_probs, all_masks):
                pred       = postprocess(prob_np, high_thr=ht, low_thr=lt,
                                         min_size=min_size)
                inter_sum += (pred & mask_np).sum()
                pred_sum  += pred.sum()
                gt_sum    += mask_np.sum()
            d    = (2.0 * inter_sum + 1.0) / (pred_sum + gt_sum + 1.0)
            prec = inter_sum / max(pred_sum, 1.0)
            rec  = inter_sum / max(gt_sum, 1.0)
            flag = " ★" if d > best_d else ""
            lt_s = f"{lt:.2f}" if lt is not None else " None"
            print(f"  {ht:6.2f} {lt_s:>6}  {d:7.4f}  {prec:7.4f}  {rec:7.4f}{flag}")
            if d > best_d:
                best_h, best_l, best_d = ht, lt, d

    lt_disp = f"{best_l:.2f}" if best_l is not None else "None"
    print(f"  → Best ({set_name}): high={best_h:.2f}  low={lt_disp}  Dice={best_d:.4f}\n")
    return best_h, best_l, best_d

@torch.no_grad()
def evaluate_test(model, loader, device, out_dir, n_vis=16, cldice_iter=15,
                  threshold=0.5, low_thr=None, min_size=50, collect_reid=False):
    model.eval()
    compare_dir = out_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)

    overlay     = LinearSegmentedColormap.from_list("ov", [(0,0,0,0), (1,0.2,0.2,0.7)])
    all_metrics = []
    vis_count   = 0
    cache       = {}
    reid_embeds, reid_labels = [], []

    pbar = tqdm(loader, desc="  Test", leave=False, dynamic_ncols=True, unit="b")
    for imgs, masks, vids, paths in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        vids  = vids.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if collect_reid and hasattr(model, 'use_reid') and model.use_reid:
                seg_logits, reid_dict = model(imgs, vids, return_reid=True)
            else:
                out = model(imgs, vids)
                seg_logits = out[0] if isinstance(out, tuple) else out
                reid_dict = None

        prob = torch.sigmoid(seg_logits[:, 1].float())

        if reid_dict is not None and reid_dict.get("vessel_embed") is not None:
            reid_embeds.append(reid_dict["vessel_embed"].cpu())
            reid_labels.append(vids.cpu())

        for i in range(imgs.size(0)):
            prob_np  = prob[i].cpu().numpy()
            mask_np  = masks[i].cpu().numpy().astype(bool)
            img_np   = imgs[i, 0].cpu().numpy()
            pred_np  = postprocess(prob_np, high_thr=threshold, low_thr=low_thr,
                                   min_size=min_size)

            d_sc  = dice_score(pred_np, mask_np)
            cl_sc = cldice_score(prob[i].cpu(), masks[i].float().cpu(),
                                 n_iter=cldice_iter)

            fname = Path(paths[i]).stem
            all_metrics.append({"filename": fname, "dice": d_sc,
                                "cldice": cl_sc, "vessel_id": vids[i].item()})
            cache[fname] = (img_np, mask_np.astype(float),
                           pred_np.astype(float), prob_np)

            if vis_count < n_vis:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img_np, cmap="gray"); axes[0].set_title("Input")
                axes[1].imshow(img_np, cmap="gray")
                axes[1].imshow(mask_np.astype(float), cmap=overlay, vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(img_np, cmap="gray")
                axes[2].imshow(pred_np.astype(float), cmap=overlay, vmin=0, vmax=1)
                axes[2].set_title(f"Pred  D={d_sc:.3f}  clD={cl_sc:.3f}")
                axes[3].imshow(prob_np, cmap="hot", vmin=0, vmax=1)
                axes[3].set_title("Probability Map")
                for ax in axes: ax.axis("off")
                plt.tight_layout()
                fig.savefig(compare_dir / f"{fname}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                vis_count += 1

        pbar.set_postfix(n=len(all_metrics),
                         dice=f"{np.mean([m['dice'] for m in all_metrics]):.4f}")
    pbar.close()

    if reid_embeds:
        _plot_reid_tsne(reid_embeds, reid_labels, out_dir)

    return all_metrics, cache

def _plot_reid_tsne(embeds_list, labels_list, out_dir):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN] sklearn not found, skipping t-SNE."); return

    all_emb = torch.cat(embeds_list, dim=0).numpy()
    all_lab = torch.cat(labels_list, dim=0).numpy()
    if len(all_emb) < 10:
        print("[WARN] Too few samples for t-SNE."); return

    print(f"[Re-ID] t-SNE on {len(all_emb)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(all_emb) - 1))
    proj = tsne.fit_transform(all_emb)

    id_to_name = {v: k for k, v in VESSEL_TO_ID.items()}
    colors = plt.cm.Set1(np.linspace(0, 1, len(VESSEL_TO_ID)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for vid in sorted(np.unique(all_lab)):
        m = all_lab == vid
        ax.scatter(proj[m, 0], proj[m, 1], c=[colors[vid]],
                   label=id_to_name.get(vid, f"V{vid}"), alpha=0.6, s=20)
    ax.legend(fontsize=12)
    ax.set_title("Re-ID Embedding Space (t-SNE)", fontsize=14)
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "reid_tsne.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Re-ID t-SNE → {out_dir / 'reid_tsne.png'}")

def plot_fixed_samples(model, dataset, fixed_map, device, out_dir,
                       threshold=0.5, low_thr=None, min_size=50,
                       cldice_iter=15, use_amp=True):
    def _stem_to_int(stem):
        try:    return int(stem)
        except ValueError: return None

    lookup = {}
    for ds_idx, (img_p, _, vid) in enumerate(dataset.samples):
        si = _stem_to_int(img_p.stem)
        if si is not None:
            lookup[(vid, si)] = ds_idx

    rows = []
    for vessel_name, sub in fixed_map.items():
        if vessel_name not in VESSEL_TO_ID:
            print(f"[WARN] fixed samples 含不認識血管 {vessel_name}，略過")
            continue
        vid = VESSEL_TO_ID[vessel_name]
        for difficulty, ids in sub.items():
            samples = []
            for fid in ids:
                ds_idx = lookup.get((vid, int(fid)))
                if ds_idx is None:
                    print(f"[WARN] 找不到 {vessel_name}/{fid}.png 於 test set，該格留白")
                samples.append((fid, ds_idx))
            rows.append((f"{vessel_name}-{difficulty}", samples))

    if not rows:
        print("[WARN] 無任何固定樣本可繪，跳過 plot_fixed_samples")
        return

    panels_per_sample = 4
    n_samples = max(len(r[1]) for r in rows)
    n_rows, n_cols = len(rows), panels_per_sample * n_samples

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1: axes = axes[None, :]
    overlay = LinearSegmentedColormap.from_list("ov", [(0,0,0,0), (1,0.2,0.2,0.7)])

    for ri, (label, samples) in enumerate(rows):
        for si, (fid, ds_idx) in enumerate(samples):
            base = si * panels_per_sample
            if ds_idx is None:
                for j in range(panels_per_sample):
                    axes[ri, base + j].axis("off")
                axes[ri, base].set_title(f"[missing {fid}]", fontsize=10, color="red")
                continue

            img_t, mask_t, vid_int, _ = dataset[ds_idx]
            img_b = img_t.unsqueeze(0).to(device)
            vid_b = torch.tensor([vid_int], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda",
                                    enabled=(device.type == "cuda" and use_amp)):
                out = model(img_b, vid_b)
                seg = out[0] if isinstance(out, tuple) else out
            prob    = torch.sigmoid(seg[0, 1].float())
            prob_np = prob.cpu().numpy()
            pred_np = postprocess(prob_np, high_thr=threshold,
                                  low_thr=low_thr, min_size=min_size).astype(np.int64)

            img_np  = img_t[0].numpy()
            mask_np = mask_t.numpy()
            d_sc    = dice_score(pred_np, mask_np)
            cl_sc   = cldice_score(prob.cpu(), mask_t.float(), n_iter=cldice_iter)

            axes[ri, base + 0].imshow(img_np, cmap="gray")
            axes[ri, base + 0].set_title(f"#{fid}  Image", fontsize=10)
            axes[ri, base + 1].imshow(img_np, cmap="gray")
            axes[ri, base + 1].imshow(mask_np, cmap=overlay, vmin=0, vmax=1)
            axes[ri, base + 1].set_title("GT", fontsize=10)
            axes[ri, base + 2].imshow(img_np, cmap="gray")
            axes[ri, base + 2].imshow(pred_np, cmap=overlay, vmin=0, vmax=1)
            axes[ri, base + 2].set_title(f"Pred  D={d_sc:.3f}\nclD={cl_sc:.3f}",
                                          fontsize=10)
            axes[ri, base + 3].imshow(prob_np, cmap="hot", vmin=0, vmax=1)
            axes[ri, base + 3].set_title("Prob", fontsize=10)
            for j in range(panels_per_sample):
                axes[ri, base + j].axis("off")

            if si == 0:
                axes[ri, 0].text(-0.15, 0.5, label,
                                 transform=axes[ri, 0].transAxes,
                                 rotation=90, va="center", ha="right",
                                 fontsize=14, fontweight="bold")

    plt.suptitle(f"Fixed Test Samples  (threshold={threshold:.2f})",
                 fontsize=18, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(out_dir / "fixed_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Fixed samples → {out_dir / 'fixed_samples.png'}")

def plot_summary_grid(metrics, cache, out_dir, top_k=4):
    sorted_m = sorted(metrics, key=lambda x: x["dice"])
    n = len(sorted_m)
    if n < top_k * 3: return
    picks = {
        "Worst":  sorted_m[:top_k],
        "Median": sorted_m[n//2 - top_k//2 : n//2 + (top_k+1)//2][:top_k],
        "Best":   sorted_m[-top_k:],
    }
    ov = LinearSegmentedColormap.from_list("ov", [(0,0,0,0), (1,0.2,0.2,0.7)])
    fig, axes = plt.subplots(3, top_k * 3, figsize=(5 * top_k * 3, 15))
    for ri, (label, items) in enumerate(picks.items()):
        for ci, m in enumerate(items):
            fname = m["filename"]
            if fname not in cache: continue
            img_np, mask_np, pred_np, _ = cache[fname]
            bc = ci * 3
            axes[ri, bc].imshow(img_np, cmap="gray")
            if ci == 0: axes[ri, bc].set_ylabel(label, fontsize=14, fontweight="bold")
            axes[ri, bc].set_title(fname, fontsize=8)
            axes[ri, bc + 1].imshow(img_np, cmap="gray")
            axes[ri, bc + 1].imshow(mask_np, cmap=ov, vmin=0, vmax=1)
            axes[ri, bc + 1].set_title("GT", fontsize=9)
            axes[ri, bc + 2].imshow(img_np, cmap="gray")
            axes[ri, bc + 2].imshow(pred_np, cmap=ov, vmin=0, vmax=1)
            axes[ri, bc + 2].set_title(f"Dice={m['dice']:.3f}", fontsize=9)
            for j in range(3): axes[ri, bc + j].axis("off")
    plt.suptitle("Test Summary: Worst / Median / Best", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Summary grid → {out_dir / 'summary_grid.png'}")

def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = Path(args.log_csv)
    if csv_p.exists():
        print("\n[1/6] Training curves...")
        plot_training_curves(str(csv_p), out_dir)
    else:
        print(f"\n[1/6] {csv_p} not found, skipping.")

    print("\n[2/6] Loading model...")
    config_path = Path(args.ckpt).parent / "model_config.pth"
    if config_path.exists():
        config = torch.load(config_path, map_location="cpu")
        print(f"Config loaded from {config_path}")
    else:
        config = {}
        print("[WARN] model_config.pth not found, using defaults.")

    model = UNetV2(
        checkpoint=None, freeze=True,
        n_classes=2, n_vessels=len(VESSEL_TO_ID),
        use_semantic_prompt=config.get("use_semantic_prompt", True),
        use_sparse_gat=config.get("use_sparse_gat", True),
        use_reid=config.get("use_reid", True),
        gnn_iters=config.get("gnn_iters", args.gnn_iters),
        gat_layers=config.get("gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        k_neighbors=config.get("k_neighbors", 16),
        max_nodes=config.get("max_nodes", 4096),
        node_threshold=config.get("node_threshold", 0.3),
        n_prompt_tokens=config.get("n_prompt_tokens", 8),
        reid_embed_dim=config.get("reid_embed_dim", 128),
    ).to(device)

    st = torch.load(args.ckpt, map_location=device, weights_only=True)
    m, u = model.load_state_dict(st, strict=False)
    if m: print(f"[INFO] Missing keys: {len(m)}")
    if u: print(f"[INFO] Unexpected keys: {len(u)}")
    model.eval()
    print(f"Checkpoint : {args.ckpt}")

    vessels     = args.vessels.split(",")
    pp_min_size = args.pp_min_size
    collect_reid = config.get("use_reid", True)

    best_thr_val, best_low_val, best_dice_val = 0.5, None, None
    if args.skip_thr_sweep:
        print(f"\n[3/6] Threshold sweep skipped, using fixed = {best_thr_val}")
    else:
        print("\n[3/6] Threshold sweep on val set...")
        val_loader = get_loader(args.data, "val", vessels=vessels,
                                img_size=(args.img_size, args.img_size),
                                batch_size=args.batch, num_workers=args.workers,
                                cache_ram=not args.no_cache)
        if len(val_loader.dataset) == 0:
            print("[WARN] No val data. Falling back to threshold = 0.5")
        else:
            best_thr_val, best_low_val, best_dice_val = find_best_threshold(
                model, val_loader, device, use_tta=False,
                min_size=pp_min_size, set_name="Val")
        del val_loader

    print("\n[4/6] Test inference...")
    test_loader = get_loader(args.data, "test", vessels=vessels,
                             img_size=(args.img_size, args.img_size),
                             batch_size=args.batch, num_workers=args.workers,
                             cache_ram=not args.no_cache)
    print(f"Test samples : {len(test_loader.dataset)}")
    lt_disp = f"{best_low_val:.2f}" if best_low_val is not None else "None"
    print(f"Postprocess  : high={best_thr_val:.2f}  low={lt_disp}  "
          f"min_size={pp_min_size}")
    if len(test_loader.dataset) == 0:
        print("[ERROR] No test data."); return

    metrics, cache = evaluate_test(
        model, test_loader, device, out_dir,
        n_vis=args.n_vis, cldice_iter=args.cldice_iter,
        threshold=best_thr_val, low_thr=best_low_val, min_size=pp_min_size,
        collect_reid=collect_reid,
    )

    csv_out = out_dir / "test_metrics.csv"
    with open(csv_out, "w") as f:
        f.write("filename,dice,cldice,vessel_id\n")
        for m in metrics:
            f.write(f"{m['filename']},{m['dice']:.6f},{m['cldice']:.6f},"
                    f"{m['vessel_id']}\n")

    dices   = [m["dice"]   for m in metrics]
    cldices = [m["cldice"] for m in metrics]
    print(f"\n{'='*50}")
    print(f" Test Results  ({len(metrics)} samples, high={best_thr_val:.2f}, "
          f"low={lt_disp})")
    print(f"{'='*50}")
    print(f"  Dice   : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"  clDice : {np.mean(cldices):.4f} +/- {np.std(cldices):.4f}")

    id_to_name = {v: k for k, v in VESSEL_TO_ID.items()}
    for vid in sorted(set(m["vessel_id"] for m in metrics)):
        sub = [m for m in metrics if m["vessel_id"] == vid]
        name = id_to_name.get(vid, f"V{vid}")
        print(f"    {name:>3}: n={len(sub):3d}  "
              f"Dice={np.mean([m['dice'] for m in sub]):.4f}  "
              f"clDice={np.mean([m['cldice'] for m in sub]):.4f}")

    if not args.skip_thr_sweep and len(test_loader.dataset) > 0:
        best_thr_test, best_low_test, best_dice_test = find_best_threshold(
            model, test_loader, device, use_tta=False,
            min_size=pp_min_size, set_name="Test")
        lt_test = f"{best_low_test:.2f}" if best_low_test is not None else "None"
        print(f"{'='*50}")
        print(" Threshold Sanity Check")
        print(f"{'='*50}")
        if best_dice_val is not None:
            print(f"  Val   best : high={best_thr_val:.2f}  low={lt_disp}"
                  f"   Dice={best_dice_val:.4f}")
        print(f"  Test  best : high={best_thr_test:.2f}  low={lt_test}"
              f"   Dice={best_dice_test:.4f}")
        gap = abs(best_thr_test - best_thr_val)
        if gap >= 0.10:
            print(f"  [!] high threshold 差距 {gap:.2f} 偏大，val/test 分布有差異。")
        else:
            print(f"  [OK] val/test high threshold 接近 (差 {gap:.2f})。")

    plot_summary_grid(metrics, cache, out_dir, args.top_k)

    # ── [6] Fixed-sample visualization ──
    print("\n[6/6] Fixed-sample visualization...")
    plot_fixed_samples(model, test_loader.dataset, FIXED_VIS_SAMPLES,
                       device, out_dir,
                       threshold=best_thr_val, low_thr=best_low_val,
                       min_size=pp_min_size, cldice_iter=args.cldice_iter)

    print("\nDone.")


def _parse():
    p = argparse.ArgumentParser(description="SAM3 UNet V2 Evaluate")
    p.add_argument("--data",           type=str, required=True)
    p.add_argument("--ckpt",           type=str, default="./checkpoints/best_model.pth")
    p.add_argument("--log_csv",        type=str, default="./checkpoints/train_log.csv")
    p.add_argument("--vessels",        type=str, default="LAD,LCx,RCA")
    p.add_argument("--img_size",       type=int, default=512)
    p.add_argument("--batch",          type=int, default=4)
    p.add_argument("--workers",        type=int, default=0)
    p.add_argument("--no_cache",       action="store_true")
    p.add_argument("--out_dir",        type=str, default="./results")
    p.add_argument("--n_vis",          type=int, default=16)
    p.add_argument("--top_k",          type=int, default=4)
    p.add_argument("--cldice_iter",    type=int, default=10)
    p.add_argument("--gnn_iters",      type=int, default=3)
    p.add_argument("--skip_thr_sweep", action="store_true")
    p.add_argument("--pp_min_size",    type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
