import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
from tqdm import tqdm  # <-- Đã thêm thư viện tạo thanh tiến trình

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Đường dẫn lưu trữ
        self.model_dir = Path("D:/hoctap/HK6/DACN1/models")
        self.plot_dir = Path("D:/hoctap/HK6/DACN1/config")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        # Thêm tqdm để tạo thanh tiến trình cho tập Train
        progress_bar = tqdm(self.train_loader, desc="Training Batch", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Cập nhật thông số loss hiển thị trực tiếp trên thanh tiến trình
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        # Thêm tqdm để tạo thanh tiến trình cho tập Validation
        progress_bar = tqdm(self.val_loader, desc="Validating Batch", leave=False)

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Hiển thị loss của tập val trên thanh tiến trình
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        # Tính toán Metrics
        val_loss = val_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            auc = 0.0

        return val_loss, acc, f1, auc

    def fit(self, epochs):
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
        best_f1 = 0.0

        print(f"🚀 Bắt đầu huấn luyện trên thiết bị: {self.device} trong {epochs} epochs")
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, acc, f1, auc = self.validate_epoch()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(acc)
            history['val_f1'].append(f1)
            history['val_auc'].append(auc)

            # In ra kết quả tổng quát sau khi chạy xong cả 1 Epoch
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

            # Lưu best model theo F1-Score
            if f1 > best_f1:
                best_f1 = f1
                save_path = self.model_dir / "best_animal_model.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"   -> 💾 Đã lưu model tốt nhất tại: {save_path}")

        self.plot_metrics(history, epochs)

    def plot_metrics(self, history, epochs):
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(18, 5))

        # Loss
        plt.subplot(1, 4, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train')
        plt.plot(epochs_range, history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 4, 2)
        plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='orange')
        plt.title('Accuracy')
        plt.legend()

        # F1 Score
        plt.subplot(1, 4, 3)
        plt.plot(epochs_range, history['val_f1'], label='Val F1', color='green')
        plt.title('F1 Score')
        plt.legend()

        # AUC
        plt.subplot(1, 4, 4)
        plt.plot(epochs_range, history['val_auc'], label='Val AUC', color='red')
        plt.title('ROC-AUC')
        plt.legend()

        plot_path = self.plot_dir / "training_metrics.png"
        plt.savefig(plot_path)
        print(f"📊 Đã lưu biểu đồ huấn luyện tại: {plot_path}")
        plt.close()