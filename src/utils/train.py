import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion, optimizer, device, class_names=None,
                 scheduler=None):
        """
        Args:
            scheduler: ReduceLROnPlateau (hoặc bất kỳ scheduler nào).
                       Nếu truyền vào, sẽ được step sau mỗi epoch với val_loss.
        """
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler   # <-- ReduceLROnPlateau từ app.py
        self.device       = device
        self.class_names  = class_names

        self.model_dir   = Path("D:/hoctap/HK6/DACN1/models")
        self.plot_dir    = Path("D:/hoctap/HK6/DACN1/results/plots")
        self.results_dir = Path("D:/hoctap/HK6/DACN1/results")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # HELPER: MixUp
    # ----------------------------------------------------------------

    @staticmethod
    def mixup_data(inputs: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.2):
        """
        [YÊU CẦU 5] MixUp Augmentation (Zhang et al., 2018).

        Nguyên lý:
            x_mixed = λ * x_i + (1-λ) * x_j
            Không có label y_mixed rõ ràng → loss = λ*L(x_mixed, y_i) + (1-λ)*L(x_mixed, y_j)

        Tại sao MixUp trị được memorization trên class hiếm?
        - Model nhìn thấy các "ảnh pha trộn" thay vì ảnh thực → không thể học
          thuộc lòng pixel-level pattern của 3-9 ảnh gốc.
        - Buộc model học decision boundary tuyến tính giữa các classes
          → generalize tốt hơn thay vì overfit vào ảnh cụ thể.

        Args:
            inputs:  Tensor [B, C, H, W]
            targets: Tensor [B] - ground truth labels
            alpha:   Tham số phân phối Beta. alpha=0.2 → λ thường gần 0 hoặc 1
                     (tức là pha trộn nhẹ, bảo toàn phần lớn ảnh gốc)

        Returns:
            inputs_mixed: Tensor [B, C, H, W]
            targets_a:    Tensor [B] - labels của ảnh gốc
            targets_b:    Tensor [B] - labels của ảnh pha trộn
            lam:          float      - tỷ lệ pha trộn
        """
        if alpha > 0:
            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0

        batch_size = inputs.size(0)
        # Tạo index ngẫu nhiên để ghép cặp ảnh trong cùng batch
        rand_index = torch.randperm(batch_size, device=inputs.device)

        targets_a = targets
        targets_b = targets[rand_index]
        inputs_mixed = lam * inputs + (1 - lam) * inputs[rand_index]

        return inputs_mixed, targets_a, targets_b, lam

    @staticmethod
    def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
        """
        Tính loss MixUp: loss = λ * L(pred, y_a) + (1-λ) * L(pred, y_b)
        Đây là unbiased estimate của loss trên không gian pha trộn.
        """
        return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)

    # ----------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------

    def _calculate_metrics(self, labels, preds, probs):
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average='macro')

        try:
            probs_array = np.array(probs)
            num_classes = probs_array.shape[1] if len(probs_array.shape) > 1 else 1

            if num_classes == 2:
                auc = roc_auc_score(labels, probs_array[:, 1])
            else:
                auc = roc_auc_score(labels, probs_array, multi_class='ovr')
        except ValueError:
            auc = 0.0

        return acc, f1, auc

    # ----------------------------------------------------------------
    # TRAIN EPOCH (với MixUp)
    # ----------------------------------------------------------------

    def train_epoch(self):
        """
        [YÊU CẦU 5] MixUp được áp dụng ngay sau khi load batch.

        Flow:
            1. Load batch (inputs, targets) từ DataLoader
            2. Áp dụng MixUp → inputs_mixed, targets_a, targets_b, lam
            3. Forward pass với inputs_mixed
            4. Tính loss = λ*L(pred, y_a) + (1-λ)*L(pred, y_b)
            5. Backward + optimize

        Lưu ý metrics:
            - Loss được tính với MixUp loss (chính xác)
            - Acc/F1/AUC được tính trên targets_a (ảnh gốc) để có thể
              so sánh với val/test metrics (val/test không dùng MixUp)
        """
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        progress_bar = tqdm(self.train_loader, desc="Training Batch", leave=False)

        for inputs, targets in progress_bar:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            # ------------------------------------------------------------
            # [YÊU CẦU 5] MixUp: trộn ảnh ngay sau khi load batch
            # alpha=0.2 được khuyến nghị từ paper gốc cho classification
            # ------------------------------------------------------------
            inputs_mixed, targets_a, targets_b, lam = self.mixup_data(
                inputs, targets, alpha=0.2
            )

            self.optimizer.zero_grad()

            # Forward với ảnh đã pha trộn
            outputs = self.model(inputs_mixed)

            # Loss MixUp: tổ hợp lồi của 2 losses
            loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Lấy prediction và probability để tính train metrics
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Dùng targets_a (label ảnh gốc) để metrics có ý nghĩa thực tế
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(targets_a.detach().cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        acc, f1, auc = self._calculate_metrics(all_labels, all_preds, all_probs)

        return epoch_loss, acc, f1, auc, all_labels, all_preds

    # ----------------------------------------------------------------
    # EVALUATE (Val / Test - KHÔNG dùng MixUp)
    # ----------------------------------------------------------------

    def evaluate(self, dataloader, desc="Evaluating"):
        """Ham dung chung de danh gia tren tap Val va Test"""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        progress_bar = tqdm(dataloader, desc=desc, leave=False)

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader.dataset)
        acc, f1, auc = self._calculate_metrics(all_labels, all_preds, all_probs)

        return epoch_loss, acc, f1, auc, all_labels, all_preds

    # ----------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------

    def fit(self, epochs: int):
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
            'val_loss':   [], 'val_acc':   [], 'val_f1':   [], 'val_auc':   [],
            'test_loss':  [], 'test_acc':  [], 'test_f1':  [], 'test_auc':  []
        }
        best_f1    = 0.0
        best_epoch = 0

        print(f"🚀 Bat dau huan luyen tren: {self.device} trong {epochs} epochs")

        for epoch in range(epochs):
            # 1. Train (với MixUp)
            t_loss, t_acc, t_f1, t_auc, _, _ = self.train_epoch()

            # 2. Validation
            v_loss, v_acc, v_f1, v_auc, _, _ = self.evaluate(self.val_loader, desc="Validating")

            # 3. Test
            test_loss, test_acc, test_f1, test_auc, _, _ = self.evaluate(self.test_loader, desc="Testing")

            # Luu vao history
            history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
            history['train_f1'].append(t_f1);     history['train_auc'].append(t_auc)
            history['val_loss'].append(v_loss);   history['val_acc'].append(v_acc)
            history['val_f1'].append(v_f1);       history['val_auc'].append(v_auc)
            history['test_loss'].append(test_loss); history['test_acc'].append(test_acc)
            history['test_f1'].append(test_f1);     history['test_auc'].append(test_auc)

            # Log
            print(f"\nEpoch [{epoch+1}/{epochs}]:")
            print(f"  Train - Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, AUC: {t_auc:.4f}, F1: {t_f1:.4f}")
            print(f"  Val   - Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, AUC: {v_auc:.4f}, F1: {v_f1:.4f}")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")

            # ----------------------------------------------------------------
            # [YÊU CẦU 4c] Step scheduler với val_loss
            # ReduceLROnPlateau giảm LR khi val_loss không cải thiện sau
            # patience=3 epochs → tự động điều chỉnh LR, không cần can thiệp thủ công
            # ----------------------------------------------------------------
            if self.scheduler is not None:
                self.scheduler.step(v_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR hiện tại: {current_lr:.2e}")

            # Luu best model theo Val F1-Score
            if v_f1 > best_f1:
                best_f1    = v_f1
                best_epoch = epoch + 1
                save_path  = self.model_dir / "best_animal_model.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"   -> 💾 Da luu model tot nhat tai Epoch {best_epoch} vao: {save_path}")

        print(f"\n🎉 Huan luyen hoan tat! Best Epoch: {best_epoch} voi Val F1: {best_f1:.4f}")

        # --- SAU KHI TRAIN XONG ---
        self.plot_metrics(history, epochs)
        self.save_results_json(history, best_epoch, best_f1)

        print("\n⏳ Dang tao Confusion Matrix tu Best Model...")
        self.model.load_state_dict(torch.load(self.model_dir / "best_animal_model.pt"))

        _, _, _, _, y_true_train, y_pred_train = self.evaluate(self.train_loader, desc="CM Train")
        self.plot_and_save_cm(y_true_train, y_pred_train, "train")

        _, _, _, _, y_true_val, y_pred_val = self.evaluate(self.val_loader, desc="CM Val")
        self.plot_and_save_cm(y_true_val, y_pred_val, "val")

        _, _, _, _, y_true_test, y_pred_test = self.evaluate(self.test_loader, desc="CM Test")
        self.plot_and_save_cm(y_true_test, y_pred_test, "test")

    # ----------------------------------------------------------------
    # PLOTTING & SAVING
    # ----------------------------------------------------------------

    def plot_metrics(self, history: dict, epochs: int):
        epochs_range = range(1, epochs + 1)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics_setup = [
            ('loss', 'Loss',      axes[0, 0]),
            ('acc',  'Accuracy',  axes[0, 1]),
            ('auc',  'ROC-AUC',   axes[1, 0]),
            ('f1',   'F1-Score',  axes[1, 1])
        ]

        for key, title, ax in metrics_setup:
            ax.plot(epochs_range, history[f'train_{key}'], label='Train', marker='o')
            ax.plot(epochs_range, history[f'val_{key}'],   label='Val',   marker='s')
            ax.plot(epochs_range, history[f'test_{key}'],  label='Test',  marker='^')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plot_path = self.plot_dir / "training_metrics_full.png"
        plt.savefig(plot_path)
        print(f"📊 Da luu bieu do huan luyen tai: {plot_path}")
        plt.close()

    def plot_and_save_cm(self, y_true, y_pred, set_name: str):
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix {set_name.upper()}:")
        print(cm)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        labels = self.class_names if self.class_names is not None else "auto"
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45, ax=ax, values_format='.2f')

        plt.title(f"Normalized Confusion Matrix - {set_name.capitalize()}")
        plt.tight_layout()

        cm_path = self.results_dir / f"confusion_matrix_{set_name}.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"🖼️ Da luu Confusion Matrix {set_name} tai: {cm_path}")

    def save_results_json(self, history: dict, best_epoch: int, best_f1: float):
        data = {
            "best_epoch":  best_epoch,
            "best_val_f1": best_f1,
            "history":     history
        }
        json_path = self.results_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"📁 Da luu lich su metrics (JSON) tai: {json_path}")