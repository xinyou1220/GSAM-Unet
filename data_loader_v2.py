import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
from collections import defaultdict, Counter


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VESSEL_TO_ID = {"LAD": 0, "LCx": 1, "RCA": 2}

def elastic_transform(img: np.ndarray, mask: np.ndarray,
                      alpha: float = 80.0, sigma: float = 10.0) -> tuple:
    h, w = img.shape[:2]
    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    img_out  = map_coordinates(img,  coords, order=1, mode="reflect").astype(img.dtype)
    mask_out = map_coordinates(mask, coords, order=0, mode="reflect").astype(mask.dtype)
    return img_out, mask_out


def vessel_cutout(img: np.ndarray, mask: np.ndarray,
                  n_holes: int = 3, hole_range: tuple = (15, 40)) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img
    img_out = img.copy()
    for _ in range(n_holes):
        idx = np.random.randint(len(ys))
        cy, cx = ys[idx], xs[idx]
        hh = np.random.randint(hole_range[0], hole_range[1])
        hw = np.random.randint(hole_range[0], hole_range[1])
        y1, y2 = max(0, cy - hh // 2), min(img.shape[0], cy + hh // 2)
        x1, x2 = max(0, cx - hw // 2), min(img.shape[1], cx + hw // 2)
        img_out[y1:y2, x1:x2] = np.random.uniform(
            0.0, 0.3, (y2 - y1, x2 - x1)).astype(img_out.dtype)
    return img_out

def _cubic_bezier(p0, p1, p2, p3, n_points=200):

    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    pts = ((1-t)**3 * p0 + 3*(1-t)**2*t * p1
           + 3*(1-t)*t**2 * p2 + t**3 * p3)
    return pts.astype(np.float32)


def _quadratic_bezier(p0, p1, p2, n_points=100):

    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    pts = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
    return pts.astype(np.float32)


def _random_edge_point(h, w):

    edge = np.random.randint(4)
    if edge == 0:    # top
        return np.array([0, np.random.randint(w)], dtype=np.float32)
    elif edge == 1:  # bottom
        return np.array([h-1, np.random.randint(w)], dtype=np.float32)
    elif edge == 2:  # left
        return np.array([np.random.randint(h), 0], dtype=np.float32)
    else:            # right
        return np.array([np.random.randint(h), w-1], dtype=np.float32)


def _draw_smooth_curve(img, points, thickness, intensity, blur_sigma=0.8):

    canvas = np.zeros_like(img)
    pts_int = points.astype(np.int32)
    for i in range(len(pts_int) - 1):
        p1 = (int(pts_int[i, 1]), int(pts_int[i, 0]))    # (x, y)
        p2 = (int(pts_int[i+1, 1]), int(pts_int[i+1, 0]))
        cv2.line(canvas, p1, p2, 1.0, thickness, lineType=cv2.LINE_AA)
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), blur_sigma)
    # 混合：在線條區域，用 intensity 值覆蓋
    mask_line = canvas > 0.01
    blend = canvas / canvas.max().clip(min=1e-6)
    img_out = img.copy()
    img_out[mask_line] = img[mask_line] * (1.0 - blend[mask_line]) + intensity * blend[mask_line]
    return img_out


def draw_catheter(img, h, w):

    p0 = _random_edge_point(h, w)

    p3 = np.array([
        np.random.uniform(h * 0.2, h * 0.8),
        np.random.uniform(w * 0.2, w * 0.8),
    ], dtype=np.float32)

    p1 = p0 + (p3 - p0) * np.random.uniform(0.2, 0.5) + \
         np.random.uniform(-h*0.15, h*0.15, 2).astype(np.float32)
    p2 = p0 + (p3 - p0) * np.random.uniform(0.5, 0.8) + \
         np.random.uniform(-h*0.15, h*0.15, 2).astype(np.float32)

    pts = _cubic_bezier(p0, p1, p2, p3, n_points=300)
    # clip to image bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, h-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, w-1)

    thickness = np.random.randint(4, 10)
    intensity = np.random.uniform(0.02, 0.15)
    img = _draw_smooth_curve(img, pts, thickness, intensity, blur_sigma=1.0)

    if np.random.random() > 0.5:
        cy, cx = int(p3[0]), int(p3[1])
        r = np.random.randint(3, 7)
        cv2.circle(img, (cx, cy), r, intensity * 0.5, -1, lineType=cv2.LINE_AA)

    return img


def draw_guidewire(img, h, w):

    p0 = _random_edge_point(h, w)
    p3 = _random_edge_point(h, w)

    while np.linalg.norm(p3 - p0) < min(h, w) * 0.4:
        p3 = _random_edge_point(h, w)

    mid = (p0 + p3) / 2
    p1 = mid + np.random.uniform(-h*0.25, h*0.25, 2).astype(np.float32)
    p2 = mid + np.random.uniform(-h*0.25, h*0.25, 2).astype(np.float32)

    pts = _cubic_bezier(p0, p1, p2, p3, n_points=400)
    pts[:, 0] = np.clip(pts[:, 0], 0, h-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, w-1)

    thickness = np.random.randint(1, 4)
    intensity = np.random.uniform(0.05, 0.20)
    img = _draw_smooth_curve(img, pts, thickness, intensity, blur_sigma=0.5)

    return img


def _draw_single_sternal_wire(img, cy, cx, h, w):

    size = np.random.randint(12, 35)
    angle = np.random.uniform(0, 2 * np.pi)
    intensity = np.random.uniform(0.0, 0.08)  # 極暗
    thickness = np.random.randint(2, 5)

    wire_type = np.random.choice(["clip", "loop", "twist"])

    if wire_type == "clip":
        for sign in [-1, 1]:
            dy = np.cos(angle + sign * 0.4) * size
            dx = np.sin(angle + sign * 0.4) * size
            p0 = np.array([cy, cx], dtype=np.float32)
            p1 = np.array([cy + dy * 0.5, cx + dx * 0.5], dtype=np.float32)
            p2 = np.array([cy + dy, cx + dx], dtype=np.float32)
            pts = _quadratic_bezier(p0, p1, p2, n_points=30)
            pts[:, 0] = np.clip(pts[:, 0], 0, h-1)
            pts[:, 1] = np.clip(pts[:, 1], 0, w-1)
            img = _draw_smooth_curve(img, pts, thickness, intensity, blur_sigma=0.3)

    elif wire_type == "loop":
        for t_frac in np.linspace(0, 2*np.pi, 40):
            pass
        axes = (size // 2, size // 3)
        angle_deg = np.degrees(angle)
        canvas = np.zeros_like(img)
        cv2.ellipse(canvas, (int(cx), int(cy)), axes,
                    angle_deg, 0, 360, 1.0, thickness, lineType=cv2.LINE_AA)
        ksize = 3
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0.5)
        mask_l = canvas > 0.01
        blend = canvas / canvas.max().clip(min=1e-6)
        img[mask_l] = img[mask_l] * (1 - blend[mask_l]) + intensity * blend[mask_l]

    elif wire_type == "twist":
        for sign_y, sign_x in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            dy = sign_y * np.cos(angle) * size * 0.7
            dx = sign_x * np.sin(angle) * size * 0.7
            p0 = np.array([cy, cx], dtype=np.float32)
            p1 = np.array([cy + dy, cx + dx], dtype=np.float32)
            pts_line = np.stack([
                np.linspace(p0[0], p1[0], 20),
                np.linspace(p0[1], p1[1], 20),
            ], axis=1).astype(np.float32)
            pts_line[:, 0] = np.clip(pts_line[:, 0], 0, h-1)
            pts_line[:, 1] = np.clip(pts_line[:, 1], 0, w-1)
            img = _draw_smooth_curve(img, pts_line, thickness, intensity, blur_sigma=0.3)

    return img


def draw_sternal_wires(img, h, w):

    n_wires = np.random.randint(3, 9)

    base_x = np.random.uniform(w * 0.25, w * 0.75)
    y_positions = np.sort(np.random.uniform(h * 0.1, h * 0.9, n_wires))

    for y_pos in y_positions:
        cx = int(base_x + np.random.uniform(-w * 0.05, w * 0.05))
        cy = int(y_pos)
        img = _draw_single_sternal_wire(img, cy, cx, h, w)

    return img


def bezier_artifact_augmentation(img, artifact_prob=0.35):
    """
    - 導管 (40% of triggered)
    - 導絲 (50% of triggered)
    - 胸骨縫線 (40% of triggered)
    - 至少會觸發一種
    """
    if np.random.random() > artifact_prob:
        return img

    h, w = img.shape[:2]
    img = img.copy()

    choices = []
    if np.random.random() < 0.4:
        choices.append("catheter")
    if np.random.random() < 0.5:
        choices.append("guidewire")
    if np.random.random() < 0.4:
        choices.append("sternal")
    if not choices:
        choices.append(np.random.choice(["catheter", "guidewire", "sternal"]))

    for c in choices:
        if c == "catheter":
            img = draw_catheter(img, h, w)
        elif c == "guidewire":
            img = draw_guidewire(img, h, w)
        elif c == "sternal":
            img = draw_sternal_wires(img, h, w)

    return np.clip(img, 0.0, 1.0)

class ArcadeDataset(Dataset):
    def __init__(
        self,
        root: str,
        vessels: List[str] = ("LAD", "LCx", "RCA"),
        split="train",
        img_size: Optional[Tuple[int, int]] = (512, 512),
        augment: bool = False,
        cache_ram: bool = True,
        artifact_prob: float = 0.35,
    ):
        self.root     = Path(root)
        splits        = [split] if isinstance(split, str) else list(split)
        self.split    = "+".join(splits)
        self.img_size = img_size
        self.augment  = augment
        self.cache_ram = cache_ram
        self.artifact_prob = artifact_prob
        self.samples: List[Tuple[Path, Path, int]] = []

        for vessel in vessels:
            vid = VESSEL_TO_ID.get(vessel, len(VESSEL_TO_ID))
            for sp in splits:
                img_dirs = [
                    self.root / vessel / sp / f"{vessel}_image",
                    self.root / vessel / sp / "images",
                    self.root / vessel / sp / "image",
                ]
                mask_dirs = [
                    self.root / vessel / sp / f"{vessel}_mask",
                    self.root / vessel / sp / "masks",
                    self.root / vessel / sp / "mask",
                ]
                img_dir  = next((d for d in img_dirs  if d.exists()), None)
                mask_dir = next((d for d in mask_dirs if d.exists()), None)
                if img_dir is None or mask_dir is None:
                    print(f"[WARNING] 找不到 {vessel}/{sp} 目錄，略過。")
                    continue
                for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
                    stem = img_path.stem
                    mask_path = mask_dir / (stem + ".png")
                    if not mask_path.exists():
                        mask_path = mask_dir / (stem + ".jpg")
                    if mask_path.exists():
                        self.samples.append((img_path, mask_path, vid))
                    else:
                        print(f"[WARNING] 找不到對應遮罩：{mask_path}")

        self._img_cache:  List[np.ndarray] = []
        self._mask_cache: List[np.ndarray] = []

        if self.cache_ram and len(self.samples) > 0:
            self._build_cache()

    def _build_cache(self):
        n = len(self.samples)
        if self.img_size:
            h, w = self.img_size
            mb = n * h * w * 2 / 1024 / 1024
            print(f"[DataLoader] 預載 {n} 張到 RAM（約 {mb:.0f} MB）...")
        for img_p, mask_p, _ in tqdm(self.samples, desc=f"  快取 {self.split}",
                                     unit="img", dynamic_ncols=True, leave=False):
            img_arr, mask_arr = self._read_and_resize(img_p, mask_p)
            self._img_cache.append(img_arr)
            self._mask_cache.append(mask_arr)
        print(f"[DataLoader] {self.split} 快取完成，共 {n} 張。")

    def _read_and_resize(self, img_path, mask_path):
        img  = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        if self.img_size:
            img  = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        img_arr  = np.array(img, dtype=np.uint8)
        mask_arr = np.array(mask, dtype=np.uint8)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_arr  = clahe.apply(img_arr)
        return img_arr, mask_arr

    def _augment(self, img_f32: np.ndarray,
                 mask_i64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_pil  = Image.fromarray((img_f32 * 255).clip(0, 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask_i64.astype(np.uint8) * 255)

        if torch.rand(1).item() > 0.5:
            img_pil  = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)
        if torch.rand(1).item() > 0.5:
            img_pil  = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)
        angle   = (torch.rand(1).item() - 0.5) * 60
        img_pil  = TF.rotate(img_pil,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

        img_pil = TF.adjust_brightness(img_pil, 0.8 + 0.4 * torch.rand(1).item())
        img_pil = TF.adjust_contrast(img_pil,   0.8 + 0.4 * torch.rand(1).item())

        img_f32  = np.array(img_pil,  dtype=np.float32) / 255.0
        mask_i64 = (np.array(mask_pil) > 127).astype(np.int64)

        if torch.rand(1).item() > 0.5:
            img_f32, mask_tmp = elastic_transform(img_f32, mask_i64.astype(np.float32))
            mask_i64 = (mask_tmp > 0.5).astype(np.int64)

        if torch.rand(1).item() > 0.6:
            img_f32 = vessel_cutout(img_f32, mask_i64, n_holes=3, hole_range=(15, 40))

        img_f32 = bezier_artifact_augmentation(img_f32, self.artifact_prob)

        return img_f32, mask_i64

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, vessel_id = self.samples[idx]

        if self.cache_ram:
            img_u8  = self._img_cache[idx]
            mask_u8 = self._mask_cache[idx]
        else:
            img_u8, mask_u8 = self._read_and_resize(
                self.samples[idx][0], self.samples[idx][1])

        img_f32  = img_u8.astype(np.float32) / 255.0
        mask_i64 = (mask_u8 > 127).astype(np.int64)

        if self.augment:
            img_f32, mask_i64 = self._augment(img_f32, mask_i64)

        img_t  = torch.from_numpy(img_f32).unsqueeze(0)
        mask_t = torch.from_numpy(mask_i64)

        return img_t, mask_t, vessel_id, str(self.samples[idx][0])

class UnlabeledDataset(Dataset):

    def __init__(
        self,
        root: str,
        img_size: Optional[Tuple[int, int]] = (512, 512),
        cache_ram: bool = True,
        artifact_prob: float = 0.5,
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self.cache_ram = cache_ram
        self.artifact_prob = artifact_prob

        self.paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]:
            self.paths.extend(sorted(self.root.rglob(ext)))

        print(f"[Unlabeled] 找到 {len(self.paths)} 張無標籤影像 from {root}")

        self._cache: List[np.ndarray] = []
        if cache_ram and len(self.paths) > 0:
            self._build_cache()

    def _build_cache(self):
        print(f"[Unlabeled] 預載 {len(self.paths)} 張...")
        for p in tqdm(self.paths, desc="  快取 unlabeled",
                      unit="img", dynamic_ncols=True, leave=False):
            arr = self._read_and_resize(p)
            self._cache.append(arr)
        print(f"[Unlabeled] 快取完成，共 {len(self.paths)} 張。")

    def _read_and_resize(self, img_path):
        img = Image.open(img_path).convert("L")
        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.uint8)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_arr = clahe.apply(img_arr)
        return img_arr

    def _weak_augment(self, img_f32):
        img_pil = Image.fromarray((img_f32 * 255).clip(0, 255).astype(np.uint8))
        if torch.rand(1).item() > 0.5:
            img_pil = TF.hflip(img_pil)
        img_pil = TF.adjust_brightness(img_pil, 0.9 + 0.2 * torch.rand(1).item())
        return np.array(img_pil, dtype=np.float32) / 255.0

    def _strong_augment(self, img_f32):
        img_pil = Image.fromarray((img_f32 * 255).clip(0, 255).astype(np.uint8))

        if torch.rand(1).item() > 0.5:
            img_pil = TF.hflip(img_pil)
        if torch.rand(1).item() > 0.5:
            img_pil = TF.vflip(img_pil)
        angle = (torch.rand(1).item() - 0.5) * 60
        img_pil = TF.rotate(img_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
        img_pil = TF.adjust_brightness(img_pil, 0.7 + 0.6 * torch.rand(1).item())
        img_pil = TF.adjust_contrast(img_pil,   0.7 + 0.6 * torch.rand(1).item())

        img_f32 = np.array(img_pil, dtype=np.float32) / 255.0

        if torch.rand(1).item() > 0.5:
            dummy_mask = np.zeros_like(img_f32)
            img_f32, _ = elastic_transform(img_f32, dummy_mask)

        img_f32 = bezier_artifact_augmentation(img_f32, self.artifact_prob)

        return np.clip(img_f32, 0.0, 1.0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache_ram:
            img_u8 = self._cache[idx]
        else:
            img_u8 = self._read_and_resize(self.paths[idx])

        img_f32 = img_u8.astype(np.float32) / 255.0

        img_weak   = self._weak_augment(img_f32)
        img_strong = self._strong_augment(img_f32)

        img_weak_t   = torch.from_numpy(img_weak).unsqueeze(0)
        img_strong_t = torch.from_numpy(img_strong).unsqueeze(0)

        return img_weak_t, img_strong_t, -1, str(self.paths[idx])

def get_loader(root, split, vessels=("LAD", "LCx", "RCA"),
               img_size=(512, 512), batch_size=4, num_workers=4,
               cache_ram=True, artifact_prob=0.0):
    is_train   = (split == "train")
    ds         = ArcadeDataset(root, vessels, split, img_size, is_train,
                               cache_ram, artifact_prob=artifact_prob if is_train else 0.0)
    persistent = (num_workers > 0)
    prefetch   = 4 if num_workers > 0 else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=is_train,
        num_workers=num_workers, pin_memory=True, drop_last=is_train,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )


class _SplitSubset(Dataset):
    def __init__(self, parent: ArcadeDataset, indices: List[int], augment: bool):
        self.parent  = parent
        self.indices = indices
        self.augment = augment
        self.samples = [parent.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        saved = self.parent.augment
        self.parent.augment = self.augment
        item = self.parent[self.indices[idx]]
        self.parent.augment = saved
        return item


def _stratified_split(samples, val_ratio: float = 0.15, seed: int = 42):
    rng    = random.Random(seed)
    groups = defaultdict(list)
    for i, (_, _, vid) in enumerate(samples):
        groups[vid].append(i)
    train_idx, val_idx = [], []
    for vid in sorted(groups.keys()):
        idxs  = groups[vid][:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_train_val_loaders(root, vessels=("LAD", "LCx", "RCA"),
                          img_size=(512, 512), batch_size=4, num_workers=4,
                          cache_ram=True,
                          merge_split=False, val_ratio=0.15, split_seed=42,
                          artifact_prob=0.35):
    persistent = (num_workers > 0)
    prefetch   = 4 if num_workers > 0 else None

    if not merge_split:
        tr = ArcadeDataset(root, vessels, "train", img_size, augment=True,
                           cache_ram=cache_ram, artifact_prob=artifact_prob)
        va = ArcadeDataset(root, vessels, "val",   img_size, augment=False,
                           cache_ram=cache_ram, artifact_prob=0.0)
        print(f"[Split] 固定分割  train={len(tr)}  val={len(va)}")
    else:
        merged = ArcadeDataset(root, vessels, split=["train", "val"],
                               img_size=img_size, augment=False,
                               cache_ram=cache_ram, artifact_prob=artifact_prob)
        train_idx, val_idx = _stratified_split(merged.samples, val_ratio, split_seed)
        tr = _SplitSubset(merged, train_idx, augment=True)
        va = _SplitSubset(merged, val_idx,   augment=False)
        tr_vids    = Counter(merged.samples[i][2] for i in train_idx)
        va_vids    = Counter(merged.samples[i][2] for i in val_idx)
        id_to_name = {v: k for k, v in VESSEL_TO_ID.items()}
        print(f"[Split] 合併重切  total={len(merged)}  "
              f"train={len(tr)}  val={len(va)}")
        for vid in sorted(set(list(tr_vids) + list(va_vids))):
            name = id_to_name.get(vid, f"V{vid}")
            print(f"        {name}: train={tr_vids[vid]}  val={va_vids[vid]}")


    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent, prefetch_factor=prefetch)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False,persistent_workers=persistent, prefetch_factor=prefetch)
    return train_loader, val_loader


def get_unlabeled_loader(root, img_size=(512, 512), batch_size=4, num_workers=4, cache_ram=True, artifact_prob=0.5):
    ds = UnlabeledDataset(root, img_size, cache_ram, artifact_prob)
    persistent = (num_workers > 0)
    prefetch   = 4 if num_workers > 0 else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )
