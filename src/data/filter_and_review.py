# filter_and_move_fixed.py

import os
import shutil
import uuid
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import pipeline

# ================= CONFIG =================
IMAGES_ROOT = '../images/'   # dùng nếu CSV KHÔNG có 'images/'
CSV_INPUT = 'dataset/dataset_cleaned.csv'
CSV_OUTPUT = 'dataset/filter_move_results.csv'

SUSPICIOUS_ROOT = 'data/suspicious/'
UNCERTAIN_ROOT = 'data/uncertain/' # Folder mới cho ảnh không chắc chắn

os.makedirs(SUSPICIOUS_ROOT, exist_ok=True)
os.makedirs(UNCERTAIN_ROOT, exist_ok=True)

# Ngưỡng chênh lệch điểm số (Margin) để quyết định độ tự tin
CONFIDENCE_MARGIN = 0.15 

# ================= MODEL =================
print("Loading CLIP model...")

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    framework="pt",
    device=0   # GPU
)

# 🔥 TỐI ƯU LABELS: Càng chi tiết càng tốt, tránh dùng từ phủ định (như "no animal")
POSITIVE_LABELS = [
    "a clear photo of a wild animal",
    "an animal standing in the forest",
    "a close up of an animal"
]

NEGATIVE_LABELS = [
    "an empty forest with no animals",
    "only trees, grass, and plants",
    "animal footprints in the dirt",
    "animal feces or scat",
    "a blurry photo of nothing",
    "a photo of the ground or dirt"
]

ALL_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

# ================= PROCESS =================
def get_full_path(rel_path):
    rel_path = str(rel_path).strip()
    rel_path = rel_path.replace('\\', os.sep).replace('/', os.sep)

    if rel_path.startswith("images"):
        return rel_path
    else:
        return os.path.join(IMAGES_ROOT, rel_path)

def move_file_safe(src_path, dest_root, scientific_name):
    """Hàm hỗ trợ move file an toàn, giữ nguyên sub-folder và tránh trùng lặp"""
    target_dir = os.path.join(dest_root, str(scientific_name).replace(' ', '_'))
    os.makedirs(target_dir, exist_ok=True)
    
    filename = os.path.basename(src_path)
    target_path = os.path.join(target_dir, filename)

    # Xử lý trùng tên file
    if os.path.exists(target_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
        target_path = os.path.join(target_dir, filename)

    shutil.move(src_path, target_path)
    return target_path

def process_image(full_path, scientific_name):
    try:
        image = Image.open(full_path).convert('RGB')
        results = classifier(image, candidate_labels=ALL_LABELS)

        # Tính tổng điểm cho nhóm Positive và Negative
        pos_score = sum(r["score"] for r in results if r["label"] in POSITIVE_LABELS)
        neg_score = sum(r["score"] for r in results if r["label"] in NEGATIVE_LABELS)
        
        top_label = results[0]["label"]

        # ===== KEEP: Điểm Positive cao hơn Negative rõ rệt =====
        if pos_score > neg_score + CONFIDENCE_MARGIN:
            return 'keep', pos_score, 'animal_detected', full_path

        # ===== MOVE SUSPICIOUS: Điểm Negative cao hơn Positive rõ rệt =====
        elif neg_score > pos_score + CONFIDENCE_MARGIN:
            new_path = move_file_safe(full_path, SUSPICIOUS_ROOT, scientific_name)
            return 'moved_to_suspicious', neg_score, top_label, new_path

        # ===== MOVE UNCERTAIN: Không rõ ràng (điểm suýt soát nhau) =====
        else:
            new_path = move_file_safe(full_path, UNCERTAIN_ROOT, scientific_name)
            return 'moved_to_uncertain', max(pos_score, neg_score), 'uncertain', new_path

    except Exception as e:
        print(f"Lỗi {full_path}: {e}")
        # Lỗi đọc file (hỏng, corrupt) -> Cho vào Suspicious
        new_path = move_file_safe(full_path, SUSPICIOUS_ROOT, scientific_name)
        return 'moved_error', 0.0, str(e), new_path


# ================= MAIN =================
df = pd.read_csv(CSV_INPUT)

results = []
count_exist = 0
count_missing = 0

print("Processing images...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    full_path = get_full_path(row['file_path'])
    sci_name = row['scientific_name']

    if not os.path.exists(full_path):
        count_missing += 1
        results.append({
            **row.to_dict(),
            'action': 'missing_file',
            'score': 0.0,
            'top_label': 'not_found',
            'new_path': None
        })
        continue

    count_exist += 1

    action, score, label, final_path = process_image(full_path, sci_name)

    results.append({
        **row.to_dict(),
        'action': action,
        'score': round(score, 4),
        'top_label': label,
        'new_path': final_path
    })


# ================= SAVE =================
pd.DataFrame(results).to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')

print("\n===== DONE =====")
print(f"Tổng ảnh xử lý: {count_exist}")
print(f"Ảnh missing: {count_missing}")
print(f"Folder nghi ngờ (Empty, Feces...): {SUSPICIOUS_ROOT}")
print(f"Folder chưa chắc chắn: {UNCERTAIN_ROOT}")
print(f"File kết quả: {CSV_OUTPUT}")