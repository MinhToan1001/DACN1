import os
import json
import random
import shutil
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# BƯỚC 1: PHÂN TÍCH DATASET
# ============================================================

class DatasetAnalyzer:
    """
    Phân tích phân phối dữ liệu và mức độ mất cân bằng.
    Dựa trên: PNAS 2021 - "A framework for wildlife image recognition"
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.class_counts: Dict[str, int] = {}
        self.tier_config = {
            "critical": (0, 10),     # Cực hiếm: 0-10 ảnh
            "rare": (10, 50),        # Hiếm: 10-50 ảnh
            "medium": (50, 200),     # Trung bình: 50-200 ảnh
            "abundant": (200, 9999)  # Nhiều: >200 ảnh
        }

    def analyze(self) -> Dict:
        """Thống kê số lượng ảnh mỗi class."""
        logger.info("Đang phân tích dataset...")

        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
                self.class_counts[class_dir.name] = len(images)

        # Tính toán thống kê
        counts = list(self.class_counts.values())
        stats = {
            "total_classes": len(self.class_counts),
            "total_images": sum(counts),
            "mean": np.mean(counts),
            "median": np.median(counts),
            "std": np.std(counts),
            "min": min(counts),
            "max": max(counts),
            "imbalance_ratio": max(counts) / max(min(counts), 1),
            "class_tiers": self._classify_tiers()
        }

        logger.info(f"Tổng classes: {stats['total_classes']}")
        logger.info(f"Tổng ảnh: {stats['total_images']}")
        logger.info(f"Imbalance ratio: {stats['imbalance_ratio']:.1f}x")
        logger.info(f"Phân tầng: {stats['class_tiers']}")

        return stats

    def _classify_tiers(self) -> Dict[str, List[str]]:
        """Phân loại lớp theo số lượng ảnh."""
        tiers = {tier: [] for tier in self.tier_config}
        for cls, count in self.class_counts.items():
            for tier, (lo, hi) in self.tier_config.items():
                if lo <= count < hi:
                    tiers[tier].append(cls)
                    break
        return {k: v for k, v in tiers.items() if v}

    def plot_distribution(self, save_path: str = "class_distribution.png"):
        """
        Vẽ biểu đồ phân phối dữ liệu (long-tail visualization).
        Sắp xếp theo thứ tự giảm dần để thấy rõ long-tail.
        """
        sorted_items = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_items)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Phân Phối Dữ Liệu - Động Vật Quý Hiếm", fontsize=14, fontweight='bold')

        # --- Biểu đồ 1: Long-tail distribution ---
        colors = []
        for c in counts:
            if c < 10:
                colors.append('#E74C3C')      # Đỏ: cực hiếm
            elif c < 50:
                colors.append('#E67E22')      # Cam: hiếm
            elif c < 200:
                colors.append('#3498DB')      # Xanh: trung bình
            else:
                colors.append('#27AE60')      # Xanh lá: nhiều

        axes[0].bar(range(len(classes)), counts, color=colors, width=0.8)
        axes[0].set_xlabel("Loài (sắp xếp giảm dần)")
        axes[0].set_ylabel("Số lượng ảnh")
        axes[0].set_title("Long-tail Distribution")
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels([c[:15] for c in classes], rotation=45, ha='right', fontsize=8)
        axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Ngưỡng cực hiếm (<10)')
        axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Ngưỡng hiếm (<50)')

        legend_patches = [
            mpatches.Patch(color='#E74C3C', label='Cực hiếm (<10 ảnh)'),
            mpatches.Patch(color='#E67E22', label='Hiếm (10-50 ảnh)'),
            mpatches.Patch(color='#3498DB', label='Trung bình (50-200 ảnh)'),
            mpatches.Patch(color='#27AE60', label='Nhiều (>200 ảnh)'),
        ]
        axes[0].legend(handles=legend_patches, fontsize=8)

        # --- Biểu đồ 2: Histogram phân phối ---
        axes[1].hist(counts, bins=20, color='#3498DB', edgecolor='white', alpha=0.8)
        axes[1].set_xlabel("Số lượng ảnh")
        axes[1].set_ylabel("Số loài")
        axes[1].set_title("Histogram Số Lượng Ảnh Mỗi Loài")
        axes[1].axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.0f}')
        axes[1].axvline(np.median(counts), color='green', linestyle='--', label=f'Median: {np.median(counts):.0f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Đã lưu biểu đồ phân phối: {save_path}")
        plt.close()


# ============================================================
# BƯỚC 2: XỬ LÝ MẤT CÂN BẰNG - AUGMENTATION
# ============================================================

class TieredAugmentationStrategy:
    """
    Chiến lược augmentation khác nhau theo từng tier.
    
    Triết lý thiết kế:
    - Tier "critical" (<10 ảnh): Augmentation mạnh + xem xét GAN
    - Tier "rare" (10-50): Augmentation trung bình + oversampling
    - Tier "medium" (50-200): Augmentation nhẹ + class weighting
    - Tier "abundant" (>200): Chỉ augmentation cơ bản, có thể undersampling
    
    Tham khảo: He et al. (2020) - "Momentum Contrast for Unsupervised Visual Representation Learning"
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def get_transforms(self, tier: str, is_train: bool = True) -> T.Compose:
        """Trả về transform phù hợp với tier và phase."""

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        if not is_train:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])

        base = [
            T.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            T.RandomCrop(self.image_size),
        ]
        normalize = [T.ToTensor(), T.Normalize(mean=imagenet_mean, std=imagenet_std)]

        if tier == "critical":
            # Cực hiếm: augment tối đa để tạo diversity
            augments = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=45),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
                T.RandomGrayscale(p=0.1),
                T.RandomPerspective(distortion_scale=0.3, p=0.4),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ]
        elif tier == "rare":
            # Hiếm: augment trung bình
            augments = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=25),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ]
        elif tier == "medium":
            # Trung bình: augment nhẹ
            augments = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            ]
        else:  # abundant
            # Nhiều ảnh: chỉ augment cơ bản
            augments = [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]

        return T.Compose(base + augments + normalize)


class MixUpAugmentation:
    """
    MixUp - trộn 2 ảnh theo tỷ lệ beta để tăng generalization.
    Đặc biệt hiệu quả cho các lớp ít dữ liệu.
    Tham khảo: Zhang et al. (2018) - "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple:
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed, y1, y2, lam


class CutMixAugmentation:
    """
    CutMix - cắt và dán vùng ảnh để tăng spatial diversity.
    Hiệu quả hơn MixUp cho bài toán nhận diện chi tiết.
    Tham khảo: Yun et al. (2019) - "CutMix: Training Strategy that Makes Use of Sample Mixing"
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x1: torch.Tensor, y1: torch.Tensor,
                 x2: torch.Tensor, y2: torch.Tensor) -> Tuple:
        lam = np.random.beta(self.alpha, self.alpha)
        _, H, W = x1.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cx, cy = np.random.randint(W), np.random.randint(H)

        x1_off = np.clip(cx - cut_w // 2, 0, W)
        y1_off = np.clip(cy - cut_h // 2, 0, H)
        x2_off = np.clip(cx + cut_w // 2, 0, W)
        y2_off = np.clip(cy + cut_h // 2, 0, H)

        x_mixed = x1.clone()
        x_mixed[:, y1_off:y2_off, x1_off:x2_off] = x2[:, y1_off:y2_off, x1_off:x2_off]
        lam_actual = 1 - (x2_off - x1_off) * (y2_off - y1_off) / (H * W)
        return x_mixed, y1, y2, lam_actual


# ============================================================
# BƯỚC 3: SINH THÊM DỮ LIỆU (GAN/Diffusion - Skeleton)
# ============================================================

class SyntheticDataGenerator:
    """
    Skeleton cho việc sinh ảnh tổng hợp bằng GAN/Diffusion.
    Chỉ sử dụng khi tier "critical" (<10 ảnh/class).
    
    Quy trình kiểm tra chất lượng ảnh sinh ra:
    1. FID Score (Frechet Inception Distance) < 50
    2. Diversity score (tránh mode collapse)
    3. Expert visual inspection (Grad-CAM so sánh)
    
    Tham khảo: Karras et al. (2020) - "Training Generative Adversarial Networks with Limited Data"
    """

    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        logger.info("Lưu ý: GAN/Diffusion cần GPU mạnh và tập dữ liệu nguồn đủ lớn.")
        logger.info("Khuyến nghị: Dùng pretrained Stable Diffusion với ControlNet để ổn định hơn.")

    def check_when_to_use_gan(self, class_count: int) -> str:
        """Hướng dẫn khi nào nên dùng GAN."""
        if class_count < 5:
            return "CRITICAL: Cần GAN hoặc Diffusion ngay. Xem xét thu thập thêm ảnh thực."
        elif class_count < 10:
            return "WARNING: Nên dùng Few-shot learning + augmentation mạnh trước. GAN là backup."
        elif class_count < 30:
            return "INFO: Augmentation trung bình + transfer learning đủ. GAN không cần thiết."
        else:
            return "OK: Dataset đủ tốt. Chỉ cần augmentation tiêu chuẩn."

    def compute_fid_guidance(self) -> str:
        """Hướng dẫn đánh giá chất lượng ảnh sinh."""
        return """
        Hướng dẫn đánh giá ảnh GAN/Diffusion:
        - FID < 50: Chất lượng tốt (dùng được cho training)
        - FID 50-100: Chất lượng trung bình (test cẩn thận)
        - FID > 100: Chất lượng kém (không nên dùng)
        
        Ngoài FID, kiểm tra thủ công:
        - Ảnh có đúng loài không? (Expert verification)
        - Có artifact không? (checkerboard, blur)
        - Có đa dạng không? (tránh mode collapse)
        
        Dùng thư viện: pip install torch-fidelity
        Command: fidelity --gpu 0 --fid --input1 real/ --input2 fake/
        """


# ============================================================
# BƯỚC 4: DATASET CLASS (PyTorch)
# ============================================================

class RareAnimalDataset(Dataset):
    """
    Dataset class với hỗ trợ:
    - Tiered augmentation theo số lượng ảnh
    - Balanced sampling
    - Tích hợp knowledge base (JSON) cho post-processing
    """

    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 class_names: List[str],
                 tier_map: Dict[str, str],
                 augmentation_strategy: TieredAugmentationStrategy,
                 knowledge_base: Optional[Dict] = None,
                 is_train: bool = True):

        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.tier_map = tier_map
        self.aug_strategy = augmentation_strategy
        self.knowledge_base = knowledge_base or {}
        self.is_train = is_train

        # Cache transforms theo tier để tránh tạo lại
        self._transform_cache: Dict[str, T.Compose] = {}

    def _get_transform(self, class_name: str) -> T.Compose:
        tier = self.tier_map.get(class_name, "medium")
        cache_key = f"{tier}_{self.is_train}"
        if cache_key not in self._transform_cache:
            self._transform_cache[cache_key] = self.aug_strategy.get_transforms(tier, self.is_train)
        return self._transform_cache[cache_key]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[label]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Lỗi đọc ảnh {img_path}: {e}. Dùng ảnh trắng.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        transform = self._get_transform(class_name)
        tensor = transform(image)

        # Lấy đặc điểm sinh học từ knowledge base (dùng cho Expert System)
        bio_features = self.knowledge_base.get(class_name, {})

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "image_path": str(img_path),
            "tier": self.tier_map.get(class_name, "medium"),
            "bio_features": bio_features
        }


# ============================================================
# BƯỚC 5: TÁCH DATASET (STRATIFIED SPLIT)
# ============================================================

class StratifiedDataSplitter:
    """
    Tách dataset theo tỷ lệ, đảm bảo phân phối lớp đồng đều.
    Xử lý đặc biệt cho lớp cực hiếm (<10 ảnh).
    """

    def __init__(self,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 min_samples_per_class: int = 3,
                 random_seed: int = 42):

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Tổng tỷ lệ phải bằng 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_samples = min_samples_per_class
        self.seed = random_seed

    def split(self, data_dir: str) -> Tuple[List, List, List, List]:
        """
        Thực hiện stratified split.
        Returns: (train_paths, val_paths, test_paths, class_names)
        """
        all_paths, all_labels, class_names = [], [], []
        class_to_idx = {}

        data_dir = Path(data_dir)
        sorted_classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

        for idx, class_name in enumerate(sorted_classes):
            class_to_idx[class_name] = idx
            class_names.append(class_name)
            class_dir = data_dir / class_name
            images = list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png"))

            if len(images) < self.min_samples:
                logger.warning(f"Lớp '{class_name}' chỉ có {len(images)} ảnh - ít hơn ngưỡng {self.min_samples}")
                # Vẫn giữ lại nhưng chỉ vào train set
                all_paths.extend([str(p) for p in images])
                all_labels.extend([idx] * len(images))
            else:
                all_paths.extend([str(p) for p in images])
                all_labels.extend([idx] * len(images))

        # Stratified split: train vs temp
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=all_labels,
            random_state=self.seed
        )

        # Stratified split: val vs test
        val_ratio_of_temp = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1.0 - val_ratio_of_temp),
            stratify=temp_labels,
            random_state=self.seed
        )

        logger.info(f"Train: {len(train_paths)} ảnh | Val: {len(val_paths)} ảnh | Test: {len(test_paths)} ảnh")
        logger.info(f"Tổng classes: {len(class_names)}")

        return (
            list(zip(train_paths, train_labels)),
            list(zip(val_paths, val_labels)),
            list(zip(test_paths, test_labels)),
            class_names
        )


# ============================================================
# BƯỚC 6: LOSS FUNCTIONS (FOCAL LOSS + CLASS WEIGHTING)
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss - tập trung vào những sample khó phân loại.
    Đặc biệt hiệu quả cho long-tail distribution.
    
    Tham khảo: Lin et al. (2017) - "Focal Loss for Dense Object Detection" (RetinaNet)
    
    Khi nào dùng:
    - Imbalance ratio > 10x → dùng Focal Loss thay CrossEntropy
    - Kết hợp với class weighting để tăng hiệu quả
    
    Công thức: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - gamma=2: giảm trọng số sample dễ (đã phân loại đúng)
    - alpha: cân bằng positive/negative
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha  # Tensor shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(class_counts: Dict[str, int],
                          method: str = "inverse_freq") -> torch.Tensor:
    """
    Tính class weights để cân bằng loss.
    
    method options:
    - "inverse_freq": w_i = N / (K * n_i) - chuẩn sklearn
    - "effective_num": w_i = (1-beta)/(1-beta^n_i) - Cui et al. 2019
    """
    counts = torch.tensor(list(class_counts.values()), dtype=torch.float)
    N = counts.sum()
    K = len(counts)

    if method == "inverse_freq":
        weights = N / (K * counts)
    elif method == "effective_num":
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"method phải là 'inverse_freq' hoặc 'effective_num'")

    # Normalize để tổng = K
    weights = weights / weights.sum() * K
    logger.info(f"Class weights: min={weights.min():.3f}, max={weights.max():.3f}")
    return weights


# ============================================================
# BƯỚC 7: BALANCED BATCH SAMPLER
# ============================================================

class BalancedBatchSampler:
    """
    Tạo WeightedRandomSampler đảm bảo mỗi batch có đủ đại diện từ các lớp thiểu số.
    Quan trọng hơn class weighting đơn thuần trong trường hợp extreme imbalance.
    """

    @staticmethod
    def create_sampler(labels: List[int],
                       class_counts: Dict[str, int],
                       class_names: List[str]) -> WeightedRandomSampler:
        """
        Tạo sampler với xác suất tỷ lệ nghịch với class frequency.
        Mỗi sample trong lớp thiểu số có xác suất được chọn cao hơn.
        """
        counts_array = torch.tensor(
            [class_counts.get(class_names[l], 1) for l in labels],
            dtype=torch.float
        )
        # Xác suất chọn tỷ lệ nghịch với số lượng
        weights = 1.0 / counts_array
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(labels),
            replacement=True
        )
        return sampler


# # ============================================================
# # BƯỚC 8: KNOWLEDGE BASE INTEGRATION (EXPERT SYSTEM)
# # ============================================================

# class KnowledgeBaseLoader:
#     """
#     Load và validate file JSON chứa đặc điểm sinh học của các loài.
#     Dùng cho Expert System khi model không chắc chắn (<50% confidence).
#     """

#     REQUIRED_FIELDS = ["scientific_name", "color_description",
#                        "distinctive_features", "legal_group", "habitat"]

#     def __init__(self, json_path: str):
#         self.json_path = json_path
#         self.kb: Dict = {}

#     def load(self) -> Dict:
#         """Load và validate knowledge base."""
#         with open(self.json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         validated = {}
#         for species, info in data.items():
#             missing = [f for f in self.REQUIRED_FIELDS if f not in info]
#             if missing:
#                 logger.warning(f"Loài '{species}' thiếu trường: {missing}")
#             else:
#                 validated[species] = info

#         self.kb = validated
#         logger.info(f"Đã load {len(self.kb)}/{len(data)} loài từ knowledge base")
#         return self.kb

#     def get_identification_questions(self, species_candidates: List[str]) -> List[str]:
#         """
#         Sinh câu hỏi để xác minh loài khi model không chắc chắn.
#         Được gọi bởi RL Agent khi confidence < threshold.
        
#         Tích hợp với luồng: DL → RL Agent → Hỏi User → Expert System → Kết luận
#         """
#         questions = []
#         for species in species_candidates[:2]:  # Top-2 candidates
#             if species in self.kb:
#                 features = self.kb[species].get("distinctive_features", [])
#                 for feat in features[:2]:  # Tối đa 2 câu hỏi mỗi loài
#                     questions.append(f"Con vật này có đặc điểm: {feat} không?")
#         return questions

#     @staticmethod
#     def generate_example_json(output_path: str = "species_knowledge_base.json"):
#         """Tạo file JSON mẫu với cấu trúc đúng."""
#         example = {
#             "Rua_Trung_Bo": {
#                 "scientific_name": "Mauremys annamensis",
#                 "vietnamese_name": "Rùa Trung Bộ",
#                 "legal_group": "IB",
#                 "color_description": "Mai màu nâu đen, phần bụng vàng nhạt",
#                 "distinctive_features": [
#                     "đầu có 3 sọc vàng",
#                     "mai hình ovan dẹt",
#                     "cổ có sọc vàng đối xứng"
#                 ],
#                 "habitat": "Sông suối miền Trung Việt Nam",
#                 "size_cm": {"length": "15-22", "weight_kg": "0.5-1.5"},
#                 "min_fine_vnd": 500000000,
#                 "max_penalty_years": 15
#             },
#             "Ho_Dong_Duong": {
#                 "scientific_name": "Panthera tigris corbetti",
#                 "vietnamese_name": "Hổ Đông Dương",
#                 "legal_group": "IB",
#                 "color_description": "Lông vàng cam với sọc đen, bụng trắng",
#                 "distinctive_features": [
#                     "sọc đen dày hơn hổ Bengal",
#                     "đầu nhỏ hơn, thân mảnh hơn",
#                     "màu lông đậm hơn"
#                 ],
#                 "habitat": "Rừng nhiệt đới Đông Nam Á",
#                 "size_cm": {"length": "220-285", "weight_kg": "150-200"},
#                 "min_fine_vnd": 1000000000,
#                 "max_penalty_years": 15
#             }
#         }

#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(example, f, ensure_ascii=False, indent=2)
#         logger.info(f"Đã tạo file JSON mẫu: {output_path}")
#         return output_path


# ============================================================
# BƯỚC 9: FEW-SHOT LEARNING (Prototypical Network)
# ============================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network cho Few-shot learning.
    Dùng khi class có <10 ảnh và GAN chưa sẵn sàng.
    
    Cách hoạt động:
    - Tính prototype (centroid) của mỗi class trong embedding space
    - Classify dựa trên khoảng cách Euclidean đến prototype
    - Không cần fine-tune khi gặp loài mới (zero-shot capable)
    
    Tham khảo: Snell et al. (2017) - "Prototypical Networks for Few-shot Learning"
    """

    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 512):
        super().__init__()

        if backbone == "resnet50":
            base = models.resnet50(pretrained=True)
            base.fc = nn.Identity()
            self.encoder = base
            feature_dim = 2048
        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(pretrained=True)
            base.classifier = nn.Identity()
            self.encoder = base
            feature_dim = 1536
        else:
            raise ValueError(f"Backbone không hỗ trợ: {backbone}")

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.projection(features)

    def compute_prototypes(self, support_images: torch.Tensor,
                           support_labels: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
        """Tính prototype (trung bình embedding) của mỗi class."""
        embeddings = self.encode(support_images)
        prototypes = torch.zeros(num_classes, embeddings.shape[-1],
                                 device=embeddings.device)
        counts = torch.zeros(num_classes, device=embeddings.device)

        for i, label in enumerate(support_labels):
            prototypes[label] += embeddings[i]
            counts[label] += 1

        # Tránh chia cho 0
        counts = counts.clamp(min=1)
        return prototypes / counts.unsqueeze(1)

    def forward(self, query_images: torch.Tensor,
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                num_classes: int) -> torch.Tensor:
        """Classify query images dựa trên khoảng cách đến prototypes."""
        prototypes = self.compute_prototypes(support_images, support_labels, num_classes)
        query_embeddings = self.encode(query_images)

        # Tính khoảng cách âm (để dùng với cross-entropy)
        distances = torch.cdist(query_embeddings, prototypes)
        return -distances  # Logits: closer = higher score


# ============================================================
# BƯỚC 10: DATA QUALITY CHECK
# ============================================================

class DataQualityChecker:
    """
    Kiểm tra chất lượng data trước khi training.
    Phát hiện: label noise, duplicate, corrupted images, size anomalies.
    """

    def __init__(self, min_size: int = 50, max_aspect_ratio: float = 5.0):
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio

    def check_image(self, img_path: str) -> Dict:
        """Kiểm tra một ảnh đơn lẻ."""
        result = {"path": img_path, "valid": True, "issues": []}

        try:
            img = Image.open(img_path)
            w, h = img.size

            if w < self.min_size or h < self.min_size:
                result["issues"].append(f"Ảnh quá nhỏ: {w}x{h}")
                result["valid"] = False

            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect_ratio:
                result["issues"].append(f"Tỷ lệ khung hình bất thường: {aspect:.1f}:1")

            if img.mode not in ("RGB", "RGBA", "L"):
                result["issues"].append(f"Color mode lạ: {img.mode}")

        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Lỗi đọc file: {str(e)}")

        return result

    def check_dataset(self, data_dir: str) -> Dict:
        """Kiểm tra toàn bộ dataset."""
        data_dir = Path(data_dir)
        results = {"total": 0, "valid": 0, "invalid": 0, "issues_summary": []}

        for class_dir in data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*.[jp][pn][gg]*"):
                check = self.check_image(str(img_path))
                results["total"] += 1
                if check["valid"]:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["issues_summary"].extend(check["issues"])

        logger.info(f"Quality check: {results['valid']}/{results['total']} ảnh hợp lệ")
        if results["invalid"] > 0:
            logger.warning(f"Phát hiện {results['invalid']} ảnh có vấn đề")

        return results


# ============================================================
# BƯỚC 11: PIPELINE CHÍNH
# ============================================================

class RareAnimalPipeline:
    """
    Pipeline tổng hợp: kết nối tất cả các bước từ phân tích đến DataLoader.
    """

    def __init__(self,
                 data_dir: str,
                 knowledge_base_path: Optional[str] = None,
                 image_size: int = 224,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 use_focal_loss: bool = True,
                 use_balanced_sampler: bool = True):

        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_focal_loss = use_focal_loss
        self.use_balanced_sampler = use_balanced_sampler

        # Sub-components
        self.analyzer = DatasetAnalyzer(data_dir)
        self.splitter = StratifiedDataSplitter()
        self.aug_strategy = TieredAugmentationStrategy(image_size)
        self.quality_checker = DataQualityChecker()

        self.knowledge_base = {}
        if knowledge_base_path and Path(knowledge_base_path).exists():
            kb_loader = KnowledgeBaseLoader(knowledge_base_path)
            self.knowledge_base = kb_loader.load()

        self.stats = None
        self.class_names = None
        self.tier_map = None

    def run(self) -> Dict:
        """Thực thi toàn bộ pipeline."""
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU")
        logger.info("=" * 60)

        # Bước 1: Phân tích
        self.stats = self.analyzer.analyze()
        self.analyzer.plot_distribution("output_class_distribution.png")

        # Bước 2: Kiểm tra chất lượng
        quality_result = self.quality_checker.check_dataset(self.data_dir)

        # Bước 3: Xây dựng tier map
        self.class_names = list(self.analyzer.class_counts.keys())
        self.tier_map = self._build_tier_map()
        self._log_tier_recommendations()

        # Bước 4: Tách dataset
        train_data, val_data, test_data, class_names = self.splitter.split(self.data_dir)
        self.class_names = class_names

        train_paths = [p for p, _ in train_data]
        train_labels = [l for _, l in train_data]
        val_paths   = [p for p, _ in val_data]
        val_labels  = [l for _, l in val_data]
        test_paths  = [p for p, _ in test_data]
        test_labels = [l for _, l in test_data]

        # Bước 5: Tạo Datasets
        train_dataset = RareAnimalDataset(
            train_paths, train_labels, class_names, self.tier_map,
            self.aug_strategy, self.knowledge_base, is_train=True
        )
        val_dataset = RareAnimalDataset(
            val_paths, val_labels, class_names, self.tier_map,
            self.aug_strategy, self.knowledge_base, is_train=False
        )
        test_dataset = RareAnimalDataset(
            test_paths, test_labels, class_names, self.tier_map,
            self.aug_strategy, self.knowledge_base, is_train=False
        )

        # Bước 6: Tạo DataLoaders
        train_sampler = None
        if self.use_balanced_sampler:
            train_sampler = BalancedBatchSampler.create_sampler(
                train_labels, self.analyzer.class_counts, class_names
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        # Bước 7: Tính loss function
        loss_fn = self._build_loss_function()

        # Bước 8: Gợi ý Few-shot cho lớp cực hiếm
        self._recommend_fewshot()

        output = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "loss_fn": loss_fn,
            "class_names": class_names,
            "num_classes": len(class_names),
            "tier_map": self.tier_map,
            "stats": self.stats,
            "quality": quality_result,
            "knowledge_base": self.knowledge_base
        }

        logger.info("=" * 60)
        logger.info("PIPELINE HOÀN THÀNH")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches:   {len(val_loader)}")
        logger.info(f"Test batches:  {len(test_loader)}")
        logger.info("=" * 60)

        return output

    def _build_tier_map(self) -> Dict[str, str]:
        """Ánh xạ mỗi class vào tier tương ứng."""
        tier_map = {}
        for cls, count in self.analyzer.class_counts.items():
            if count < 10:
                tier_map[cls] = "critical"
            elif count < 50:
                tier_map[cls] = "rare"
            elif count < 200:
                tier_map[cls] = "medium"
            else:
                tier_map[cls] = "abundant"
        return tier_map

    def _log_tier_recommendations(self):
        """Log khuyến nghị xử lý theo từng tier."""
        tier_counts = Counter(self.tier_map.values())
        logger.info("\n--- KHUYẾN NGHỊ THEO TIER ---")
        if tier_counts.get("critical", 0) > 0:
            logger.warning(f"  CRITICAL ({tier_counts['critical']} lớp): Xem xét GAN/Diffusion hoặc Few-shot learning")
        if tier_counts.get("rare", 0) > 0:
            logger.info(f"  RARE ({tier_counts['rare']} lớp): Augmentation mạnh + oversampling")
        if tier_counts.get("medium", 0) > 0:
            logger.info(f"  MEDIUM ({tier_counts['medium']} lớp): Augmentation trung bình + Focal Loss")
        if tier_counts.get("abundant", 0) > 0:
            logger.info(f"  ABUNDANT ({tier_counts['abundant']} lớp): Augmentation cơ bản, theo dõi overfitting")

    def _build_loss_function(self) -> nn.Module:
        """Chọn loss function phù hợp."""
        if self.use_focal_loss and self.stats["imbalance_ratio"] > 10:
            weights = compute_class_weights(
                self.analyzer.class_counts, method="effective_num"
            )
            loss_fn = FocalLoss(alpha=weights, gamma=2.0)
            logger.info(f"Dùng Focal Loss (imbalance ratio={self.stats['imbalance_ratio']:.1f}x)")
        else:
            weights = compute_class_weights(
                self.analyzer.class_counts, method="inverse_freq"
            )
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            logger.info("Dùng CrossEntropyLoss với class weighting")
        return loss_fn

    def _recommend_fewshot(self):
        """Gợi ý few-shot learning cho lớp cực hiếm."""
        critical_classes = [cls for cls, tier in self.tier_map.items() if tier == "critical"]
        if critical_classes:
            logger.warning(f"\n--- FEW-SHOT RECOMMENDATION ---")
            logger.warning(f"Các lớp cần Few-shot: {critical_classes}")
            logger.warning("Khuyến nghị: Dùng PrototypicalNetwork với backbone EfficientNet-B3")
            logger.warning("Training strategy: Episodic training với N-way K-shot")


# ============================================================
# EXAMPLE USAGE
# ============================================================

def main():
    """Demo pipeline với dataset mẫu."""

    # 1. Tạo knowledge base mẫu nếu chưa có
    kb_path = "species_knowledge_base.json"
    if not Path(kb_path).exists():
        KnowledgeBaseLoader.generate_example_json(kb_path)

    # 2. Khởi tạo và chạy pipeline
    # Thay "path/to/your/dataset" bằng đường dẫn thực của bạn
    # Cấu trúc thư mục cần như sau:
    # dataset/
    #   ├── Ho_Dong_Duong/
    #   │   ├── image001.jpg
    #   │   └── ...
    #   ├── Rua_Trung_Bo/
    #   │   └── ...
    #   └── ...

    DATA_DIR = "dataset/images"

    if not Path(DATA_DIR).exists():
        logger.info("Thư mục dataset chưa tồn tại. Vui lòng cung cấp đường dẫn đúng.")
        logger.info("Cấu trúc thư mục: dataset/images/<class_name>/<image.jpg>")
        return

    pipeline = RareAnimalPipeline(
        data_dir=DATA_DIR,
        knowledge_base_path=kb_path,
        image_size=224,        # 224 cho CNN; đổi sang 640 cho YOLOv8
        batch_size=32,
        num_workers=4,
        use_focal_loss=True,
        use_balanced_sampler=True
    )

    result = pipeline.run()

    # 3. Sử dụng trong training loop
    train_loader = result["train_loader"]
    loss_fn = result["loss_fn"]
    num_classes = result["num_classes"]

    logger.info(f"\nSẵn sàng training với {num_classes} classes!")
    logger.info("Gợi ý model: EfficientNet-B3 hoặc ConvNeXt-Small với pretrained ImageNet")
    logger.info("Optimizer: AdamW với lr=1e-4, weight_decay=1e-4")
    logger.info("Scheduler: CosineAnnealingLR hoặc OneCycleLR")


if __name__ == "__main__":
    main()