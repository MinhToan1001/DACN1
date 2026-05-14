import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Rescaling
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import image_dataset_from_directory
import os

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa ảnh (Phải khớp với file generate_animals.py)
DATA_DIR = 'animal_dataset/images' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # Tăng lên 1 chút để model học kỹ hơn vì có 100 loài

print(f"--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")
print(f"Đang kiểm tra dữ liệu tại: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy chạy file generate_animals.py và scrape_animal_images.py trước!")
    exit()

# --- 1. CHUẨN BỊ DỮ LIỆU ---
# Load dữ liệu training (80%)
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int' # Label dạng số nguyên (0, 1, 2... 99)
)

# Load dữ liệu validation (20%)
validation_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Lấy danh sách tên loài (Class Names) từ tên thư mục
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)

print(f"\n✅ Đã tìm thấy {NUM_CLASSES} loài động vật.")
print("---------------------------------")
print("!!! QUAN TRỌNG: LIST CLASS NAMES !!!")
print("Copy list này để dùng khi dự đoán (Predict):")
print(class_names)
print("---------------------------------")

# Tối ưu hóa hiệu năng nạp dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. XÂY DỰNG MÔ HÌNH (MobileNetV2) ---
print("\nĐang xây dựng mô hình...")

# Tải MobileNetV2 (đã pre-train trên ImageNet), bỏ lớp đầu ra cũ
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# Đóng băng các lớp cơ sở để không train lại từ đầu (Transfer Learning)
base_model.trainable = False 

# Tạo Input layer
inputs = Input(shape=(224, 224, 3))

# Chuẩn hóa ảnh từ [0, 255] về [-1, 1] theo yêu cầu của MobileNetV2
x = Rescaling(1./127.5, offset=-1)(inputs)

# Đưa qua base model
x = base_model(x, training=False)

# Thêm các lớp classification mới
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x) # Giảm overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Hoàn thiện model
model = Model(inputs=inputs, outputs=predictions)

# --- 3. BIÊN DỊCH MÔ HÌNH ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. HUẤN LUYỆN ---
print(f"\nBắt đầu huấn luyện {EPOCHS} epochs...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# --- 5. LƯU MÔ HÌNH ---
model_filename = 'animal_classifier.keras'
model.save(model_filename)

print("\n---------------------------------")
print(f"✅ ĐÃ HUẤN LUYỆN XONG! Model lưu tại: {model_filename}")
print("Bạn có thể dùng model này để nhận diện tên loài vật.")
print("Để biết loài đó có QUÝ HIẾM hay không, hãy tra cứu kết quả dự đoán vào file 'animal_labels.csv'.")
print("---------------------------------")