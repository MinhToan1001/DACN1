import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


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
        self.scheduler    = scheduler
        self.device       = device
        self.class_names  = class_names

        self.model_dir   = Path("models")
        self.plot_dir    = Path("results/plots")
        self.results_dir = Path("results")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    # ----------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------

    def _calculate_metrics(self, labels, preds, probs=None):
            """
            Tính toán các chỉ số đánh giá thực chất cho mô hình Multi-class.
            - Dùng Macro-average để đánh giá công bằng cho cả những class hiếm.
            - Tham số probs=None được giữ lại để không làm lỗi các hàm cũ nếu có truyền vào.
            """
            # 1. Accuracy: Tỷ lệ đoán trúng trên tổng số ảnh
            acc = accuracy_score(labels, preds)
            
            # 2. F1-Score: Trung bình điều hòa của Precision và Recall
            f1 = f1_score(labels, preds, average='macro', zero_division=0)
            
            # 3. Precision: Khi model tự tin kêu "Đây là con X", thì tỷ lệ đúng là bao nhiêu?
            precision = precision_score(labels, preds, average='macro', zero_division=0)
            
            # 4. Recall: Trong tổng số con X ngoài đời thực, model tìm ra được bao nhiêu %?
            recall = recall_score(labels, preds, average='macro', zero_division=0)

            return acc, f1, precision, recall

    def inspect_model_with_gradcam(self, num_species=10, samples_per_species=5):
        print(f"👁️ Đang tạo {num_species*samples_per_species} ảnh Grad-CAM để bạn kiểm tra...")
        self.model.eval()

        class_to_images = {}
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                for i in range(inputs.size(0)):
                    lbl = labels[i].item()
                    if lbl not in class_to_images: class_to_images[lbl] = []
                    if len(class_to_images[lbl]) < samples_per_species:
                        class_to_images[lbl].append(inputs[i])

        available_classes = list(class_to_images.keys())
        selected_classes = random.sample(available_classes, min(num_species, len(available_classes)))

        # MobileNetV2: target layer cuối trong features
        target_layers = [self.model.features[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)

        inspect_dir = self.results_dir / "inspection_gradcam"
        inspect_dir.mkdir(parents=True, exist_ok=True)

        for cls_idx in selected_classes:
            cls_name = self.class_names[cls_idx].replace(" ", "_").replace(",", "")
            cls_path = inspect_dir / cls_name
            cls_path.mkdir(exist_ok=True)

            for i, img_tensor in enumerate(class_to_images[cls_idx]):
                input_batch = img_tensor.unsqueeze(0).to(self.device)
                grayscale_cam = cam(input_tensor=input_batch, targets=None)[0, :]

                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

                vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                plt.imsave(cls_path / f"sample_{i+1}.png", vis)

        print(f"✅ Đã lưu ảnh kiểm tra tại: {inspect_dir}")

    def save_gradcam_sample(self, epoch: int, num_samples: int = 5):
            """
            Lấy ngẫu nhiên num_samples ảnh từ tập Validation và lưu Grad-CAM.
            """
            self.model.eval()
            
            # Lấy 1 batch ngẫu nhiên từ Val Loader
            all_batches = list(self.val_loader)
            batch_data = random.choice(all_batches)
            inputs, labels = batch_data
            
            # Chọn ngẫu nhiên num_samples chỉ số trong batch đó
            max_idx = len(inputs)
            indices = random.sample(range(max_idx), min(num_samples, max_idx))

            cam_dir = self.results_dir / "gradcam_epochs"
            cam_dir.mkdir(parents=True, exist_ok=True)

            for i, idx in enumerate(indices):
                input_tensor = inputs[idx:idx+1].to(self.device)
                label_idx = labels[idx].item()
                class_name = self.class_names[label_idx] if self.class_names else str(label_idx)

                # Grad-CAM cho MobileNetV2
                target_layers = [self.model.layer4[-1]]
                cam = GradCAM(model=self.model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

                # Denormalize ảnh để hiển thị
                img_np = input_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                # Lưu ảnh với tên file phân biệt bằng index i
                save_path = cam_dir / f"epoch_{epoch:02d}_sample_{i+1}_{class_name}.png"

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(visualization)
                ax.set_title(f"Epoch: {epoch} | Sample: {i+1} | Truth: {class_name}")
                ax.axis('off')

                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
            print(f"👁️ Đã lưu {len(indices)} ảnh Grad-CAM tại: {cam_dir}")
    # ----------------------------------------------------------------
    # TRAIN EPOCH — có MixUp (alpha=0.2)
    # ----------------------------------------------------------------
    #cutmix 
    def rand_bbox(self, size, lam):
        """
        Hàm hỗ trợ cho CutMix: Tính toán tọa độ hộp (bounding box) để cắt dán.
        """
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Chọn ngẫu nhiên tâm của hộp
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Tính toán tọa độ 4 góc của hộp (có giới hạn không vượt quá ảnh)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    def train_epoch(self, epoch, use_aug=False):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        # Hiển thị trạng thái trên thanh tiến trình
        status = "[CUTMIX ON]" if use_aug else "[CLEAN DATA]"
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} {status}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if use_aug:
                # -----------------------------------------------------
                # ÁP DỤNG CUTMIX
                # -----------------------------------------------------
                # 1. Lấy tỷ lệ random (CutMix thường dùng alpha=1.0)
                lam = np.random.beta(0.3, 0.3)
                index = torch.randperm(inputs.size(0)).to(self.device)
                
                labels_a = labels
                labels_b = labels[index]
                
                # 2. Tính toán hộp và tạo ảnh CutMix
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
                inputs_mixed = inputs.clone()
                # Cắt vùng ảnh từ index khác đè lên ảnh gốc
                inputs_mixed[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
                
                # 3. Tính lại lamda thực tế dựa trên diện tích hộp đã cắt
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                # 4. Forward và tính Loss
                outputs = self.model(inputs_mixed)
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                # -----------------------------------------------------
                # HỌC TRÊN ẢNH GỐC
                # -----------------------------------------------------
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # Tracking metrics (luôn dựa trên nhãn gốc để theo dõi độ chính xác)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        acc, f1, precision, recall = self._calculate_metrics(all_labels, all_preds)
        return epoch_loss, acc, f1, precision, recall, all_labels, all_preds

    # ----------------------------------------------------------------
    # EVALUATE (Val / Test — KHÔNG dùng MixUp, dữ liệu sạch)
    # ----------------------------------------------------------------

    def evaluate(self, dataloader, desc="Evaluating"):
            """
            [YÊU CẦU 3] Hàm evaluate chạy trên dữ liệu sạch, KHÔNG có MixUp/CutMix.
            Dùng chung cho Val (trong quá trình train) và Test (sau khi train xong).
            """
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
            
            # SỬA DÒNG NÀY: Hứng đúng 4 chỉ số mới (bỏ all_probs đi vì hàm mới không cần nữa)
            acc, f1, precision, recall = self._calculate_metrics(all_labels, all_preds)

            # SỬA DÒNG NÀY: Return đúng 7 giá trị để khớp với hàm fit
            return epoch_loss, acc, f1, precision, recall, all_labels, all_preds

    # ----------------------------------------------------------------
    # FIT — chỉ Train + Val trong vòng lặp; Test một lần sau khi xong
    # ----------------------------------------------------------------

    def fit(self, epochs: int):
        """
        [YÊU CẦU 5] Execution Flow:
        - Vòng lặp epoch: chỉ chạy train_epoch() + evaluate(val_loader)
        - Áp dụng Curriculum Learning: 5 Epoch đầu TẮT MixUp, từ Epoch 6 BẬT MixUp.
        - Sau khi kết thúc huấn luyện: load best_model.pth → evaluate(test_loader) 1 lần
        """
        history = {
                    'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_prec': [], 'train_rec': [],
                    'val_loss':   [], 'val_acc':   [], 'val_f1':   [], 'val_prec':   [], 'val_rec':   [],
                }
        best_f1    = 0.0
        best_epoch = 0
        best_model_path = self.model_dir / "best_animal_model.pt"

        print(f"🚀 Bắt đầu huấn luyện trên: {self.device} trong {epochs} epochs")

        for epoch in range(epochs):
            # ----------------------------------------------------------------
            # CHIẾN THUẬT CURRICULUM LEARNING (Tự động chuyển đổi)
            # - Epoch 0 đến 4 (< 5): False (Học trên ảnh gốc)
            # - Epoch 5 trở đi (>= 5): True (Bật MixUp tạo nhiễu)
            # ----------------------------------------------------------------
            is_aug = True
            
            # [QUAN TRỌNG] Đã thêm tham số use_mixup=is_mixup vào đây
            t_loss, t_acc, t_f1, t_prec, t_rec, _, _ = self.train_epoch(epoch, use_aug=is_aug)
            v_loss, v_acc, v_f1, v_prec, v_rec, _, _ = self.evaluate(self.val_loader, desc="Validating")

# Lưu vào history (Đã thay AUC bằng Precision và Recall)
            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['train_f1'].append(t_f1)
            history['train_prec'].append(t_prec)
            history['train_rec'].append(t_rec)

            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            history['val_f1'].append(v_f1)
            history['val_prec'].append(v_prec)
            history['val_rec'].append(v_rec)

            # Log in ra màn hình
            aug_status = "[CUTMIX ON]" if is_aug else "[CLEAN DATA]"
            print(f"\nEpoch [{epoch+1}/{epochs}] {aug_status}:")
            print(f"  Train - Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, Prec: {t_prec:.4f}, Rec: {t_rec:.4f}, F1: {t_f1:.4f}")
            print(f"  Val   - Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, Prec: {v_prec:.4f}, Rec: {v_rec:.4f}, F1: {v_f1:.4f}")

            # Step scheduler với val_loss
            if self.scheduler is not None:
                self.scheduler.step(v_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR hiện tại: {current_lr:.2e}")

            # Lưu best model theo Val F1-Score
            if v_f1 > best_f1:
                best_f1    = v_f1
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), best_model_path)
                print(f"   -> 💾 Đã lưu model tốt nhất tại Epoch {best_epoch} → {best_model_path}")

            # Grad-CAM theo dõi mỗi epoch
            self.save_gradcam_sample(epoch + 1)

        print(f"\n🎉 Huấn luyện hoàn tất! Best Epoch: {best_epoch} với Val F1: {best_f1:.4f}")

        # ----------------------------------------------------------------
        # [YÊU CẦU 5] Sau khi train xong: load best model → Test 1 lần duy nhất
        # ----------------------------------------------------------------
        print(f"\n⏳ Đang load best model từ Epoch {best_epoch} để đánh giá Test Set...")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        test_loss, test_acc, test_f1, test_auc, y_true_test, y_pred_test = \
            self.evaluate(self.test_loader, desc="Testing (Final)")

        print(f"\n{'='*55}")
        print(f"  KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST (Best Epoch {best_epoch})")
        print(f"{'='*55}")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Acc:  {test_acc:.4f}")
        print(f"  AUC:  {test_auc:.4f}")
        print(f"  F1:   {test_f1:.4f}")
        print(f"{'='*55}")

        # ----------------------------------------------------------------
        # VẼ BIỂU ĐỒ KÈM TẬP TEST Ở CUỐI CÙNG 
        # (Truyền test_metrics vào plot_metrics)
        # ----------------------------------------------------------------
        test_metrics = {'acc': test_acc, 'loss': test_loss}
        self.plot_metrics(history, epochs, test_metrics=test_metrics)
        
        self.save_results_json(history, best_epoch, best_f1, test_acc, test_f1, test_auc)

        # Confusion Matrix cho 3 tập (dùng best model đã load)
        print("\n⏳ Đang tạo Confusion Matrix từ Best Model...")

        _, _, _, _, y_true_train, y_pred_train = self.evaluate(self.train_loader, desc="CM Train")
        self.plot_and_save_cm(y_true_train, y_pred_train, "train")

        _, _, _, _, y_true_val, y_pred_val = self.evaluate(self.val_loader, desc="CM Val")
        self.plot_and_save_cm(y_true_val, y_pred_val, "val")

        self.plot_and_save_cm(y_true_test, y_pred_test, "test")
    # ----------------------------------------------------------------
    # PLOTTING & SAVING
    # ----------------------------------------------------------------

    def plot_metrics(self, history: dict, epochs: int, test_metrics=None):
            """Vẽ biểu đồ Train vs Val (không có Test trong vòng lặp)."""
            epochs_range = range(1, epochs + 1)
            
            # Mở rộng thành lưới 2x3 để vẽ đủ 5 biểu đồ (Loss, Acc, F1, Prec, Rec)
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))

            metrics_setup = [
                ('loss', 'Loss',      axes[0, 0]),
                ('acc',  'Accuracy',  axes[0, 1]),
                ('f1',   'F1-Score',  axes[0, 2]),
                ('prec', 'Precision', axes[1, 0]),
                ('rec',  'Recall',    axes[1, 1])
            ]

            for key, title, ax in metrics_setup:
                ax.plot(epochs_range, history[f'train_{key}'], label='Train', marker='o')
                ax.plot(epochs_range, history[f'val_{key}'],   label='Val',   marker='s')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)

            # Xóa khung đồ thị thứ 6 (vị trí 1,2) vì bị dư
            fig.delaxes(axes[1, 2])

            plt.tight_layout()
            plot_path = self.plot_dir / "training_metrics.png"
            plt.savefig(plot_path)
            print(f"📊 Đã lưu biểu đồ huấn luyện tại: {plot_path}")
            plt.close()

    def plot_and_save_cm(self, y_true, y_pred, set_name: str):
        labels_idx = list(range(len(self.class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        print(f"\nConfusion Matrix {set_name.upper()} đã được tạo.")

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        labels = self.class_names if self.class_names is not None else "auto"
        
        # Tăng kích thước ảnh lớn hơn cho 103 class
        fig, ax = plt.subplots(figsize=(24, 20))
        
        # Dùng include_values=False để không in số đè lên nhau (giúp hình trong trẻo hơn)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, include_values=False)

        plt.title(f"Normalized Confusion Matrix - {set_name.capitalize()}")
        plt.tight_layout()

        cm_path = self.results_dir / f"confusion_matrix_{set_name}.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"🖼️ Đã lưu Confusion Matrix {set_name} tại: {cm_path}")

    def save_results_json(self, history: dict, best_epoch: int, best_val_f1: float,
                          test_acc: float = 0.0, test_f1: float = 0.0, 
                          test_prec: float = 0.0, test_rec: float = 0.0):
        data = {
            "best_epoch":   best_epoch,
            "best_val_f1":  best_val_f1,
            "test_acc":     test_acc,
            "test_f1":      test_f1,
            "test_prec":    test_prec,    # Đổi từ auc sang prec
            "test_rec":     test_rec,     # Đổi từ auc sang rec
            "history":      history
        }
        json_path = self.results_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"📁 Đã lưu lịch sử metrics (JSON) tại: {json_path}")