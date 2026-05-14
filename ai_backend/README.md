# 🐾 Đồ án: Hệ thống Nhận diện Động vật & Tư vấn Pháp lý (Hybrid AI)

## 📌 Giới thiệu dự án
Dự án này là một hệ thống Trí tuệ nhân tạo lai (Hybrid AI) tham vọng, kết hợp 3 nhánh kiến thức chuyên sâu để giải quyết bài toán định danh động vật hoang dã và tư vấn khung hình phạt pháp lý tại Việt Nam:
1. **Deep Learning (Perception):** Đóng vai trò là "Mắt thần" giúp nhận diện và trích xuất đặc trưng sinh học của loài động vật từ hình ảnh.
2. **Hệ chuyên gia - Expert System (Reasoning):** Đóng vai trò là "Bộ não pháp lý", sử dụng tập luật để suy diễn từ dữ liệu sinh học ra khung hình phạt chuẩn xác.

---

## 🚀 Tiến độ hiện tại (Roadmap)
- [x] **Bước 1:** Thu thập dữ liệu hình ảnh (Data Crawling).
- [x] **Bước 2:** Làm sạch dữ liệu (Data Cleaning & Filtering).
- [x] **Bước 3:** Xây dựng Cơ sở tri thức pháp lý (Knowledge Base Extraction).
- [ ] **Bước 4:** Tiền xử lý & Tăng cường dữ liệu ảnh (Image Preprocessing & Augmentation).
- [ ] **Bước 5:** Huấn luyện mô hình Deep Learning  ResNet.
- [ ] **Bước 6:** Tích hợp Hệ chuyên gia

---

## 🛠️ Hướng dẫn cài đặt và chạy dữ liệu

Dưới đây là các bước quy trình chuẩn bị dữ liệu (Data Preparation Pipeline) đã hoàn thiện trong dự án. Vui lòng chạy các lệnh theo đúng thứ tự.

### Bước 1: Thu thập dữ liệu thô (Scraping)
Script này chịu trách nhiệm cào dữ liệu hình ảnh của các loài động vật hoang dã từ các nguồn trên Internet và lưu vào thư mục chứa dữ liệu thô.
\:hoctap\HK6\DACN1\dataset\dataset.py
---
## Bước 2: làm sạch dữ liệu
Script này chịu trách nhiệm loại bỏ các hình mờ hình không có động vật hình lỗi 
D\:hoctap\HK6\DACN1\src\data\filter_and_review.py
file này để matching lượng ảnh sau khi lọc bỏ với file csv để hoàn thiện bộ dataset cuôi
D\:hoctap\HK6\DACN1\dataset\matching.py

### Bước 3 xây dụng cơ sở pháp lý 
file này để trích xuất file json các thông tin luật pháp về các loài động vật quý hiếm và không quý hiếm 
D\:hoctap\HK6\DACN1\src\legal\generate_rulebase.py
file này để trích xuất file json các thông tin về đặc điểm nhận dạng, thức ăn , tập tính , sinh thái
D\:hoctap\HK6\DACN1\src\knowledge\extract_species_info.py
kết quả 2 file json tại thư mục D\:hoctap\HK6\DACN1\rules
### Bước 4 tiền xử lí dữ liệu 

Đoạn code trong file D\:hoctap\HK6\DACN1\src\data\preprocess.py là một Pipeline Tiền xử lý dữ liệu:
Phân tích & Phân loại dữ liệu (DatasetAnalyzer): Quét thư mục ảnh, đếm số lượng ảnh của từng loài và chia chúng thành 4 nhóm (tier): critical (<10 ảnh), rare (10-50 ảnh), medium (50-200 ảnh), và abundant (>200 ảnh).
Tăng cường dữ liệu phân tầng (TieredAugmentationStrategy): Áp dụng các kỹ thuật biến đổi ảnh (xoay, lật, chỉnh màu, MixUp, CutMix) với cường độ khác nhau dựa trên nhóm của loài. Loài càng hiếm (critical) thì càng bị biến đổi mạnh để ép mô hình học.
Chia tập dữ liệu (StratifiedDataSplitter): Chia bộ dữ liệu thành tập Huấn luyện (Train), Xác thực (Val), và Kiểm tra (Test) theo tỷ lệ cấu hình (mặc định 70-15-15), đảm bảo tỷ lệ các loài được phân phối đều giữa các tập.
Kiểm soát mất cân bằng (BalancedBatchSampler & Focal Loss): Thay vì lấy ảnh ngẫu nhiên, hệ thống dùng WeightedRandomSampler để đảm bảo trong mỗi batch (lô ảnh) luôn có sự xuất hiện của các loài thiểu số. Đồng thời, code định nghĩa hàm FocalLoss để phạt nặng mô hình nếu đoán sai các loài hiếm.
Kiểm tra chất lượng ảnh (DataQualityChecker): Lọc bỏ các ảnh bị lỗi, ảnh quá nhỏ hoặc có tỷ lệ khung hình bất thường trước khi đưa vào huấn luyện.

### Bước 5: Huấn luyện & Đánh giá mô hình Deep Learning
Chạy app.py gọi đên preprocess và train luôn 
Giai đoạn này giữ nguyên, tập trung huấn luyện mô hình thị giác máy tính nhận diện từ ảnh.
D:\hoctap\HK6\DACN1\src\model\ chứa 3 model để so sánh resnet, efficientnet và cnn truyền thống
Script huấn luyện: D:\hoctap\HK6\DACN1\src\utils\train.py
Nhận đầu vào là ảnh, trả ra xác suất dự đoán (Ví dụ: {"Rùa Trung Bộ": 0.45, "Rùa Hộp": 0.30}).

Script giải thích (Grad-CAM): D:\hoctap\HK6\DACN1\src\utils\gradcam.py
Xuất ra ảnh bôi đỏ vùng đặc trưng để tăng tính thuyết phục.
mô hình sau huấn luyện lưu ở D:\hoctap\HK6\DACN1\models

### Bước 6: Xây dựng Động cơ suy diễn Hệ chuyên gia 
Biến file JSON thành cỗ máy logic để tra cứu luật và đối chiếu đặc điểm.

Script Inference Engine: D:\hoctap\HK6\DACN1\src\expert_system\inference_engine.py
Chạy suy diễn tiến (Forward Chaining). Nhận các từ khóa chuẩn (như "mai_mau_den", "dau_co_soc") để tăng dần Hệ số chắc chắn (Certainty Factor - CF). Khi CF > 80%, tự động xuất ra khung hình phạt.

### Bước 7: Tích hợp Hệ thống & Xử lý Ngôn ngữ bằng Gemini API
Đây là bước kết nối tất cả và giao tiếp với người dùng. Gemini API được gọi trong phần Backend làm cầu nối (Middleware) để dịch qua lại giữa "Ngôn ngữ máy" và "Ngôn ngữ người".

Script Backend / API Chính: D:\hoctap\HK6\DACN1\src\app\main_controller.py
Quy trình tích hợp diễn ra theo luồng khép kín sau:

Nhận diện (DL): User tải ảnh mờ lên giao diện bên trái. YOLO dự đoán 45%.

Ra quyết định (RL): Độ tin cậy thấp, DQN quyết định phải hỏi về đặc điểm số 2 (Action = ask_shell_color).

Prompt sinh câu hỏi (Gemini API - Chiều đi): Backend gửi một Prompt ngầm cho Gemini: "Đóng vai chuyên gia bảo tồn động vật. Hãy hỏi người dùng một cách thân thiện, ngắn gọn xem mai con rùa trong ảnh của họ có màu gì. Không tự bịa thông tin."
Gemini trả về UI bên phải: "Chào bạn, ảnh hơi mờ nên hệ thống chưa chắc chắn. Bạn có thể cho mình biết mai bé rùa này có màu gì không?"

Prompt trích xuất từ khóa (Gemini API - Chiều về):
User gõ vào ô chat: "Mình thấy nó đen thui à, có vài vệt vàng nhạt".
Backend không ném nguyên câu này cho Hệ chuyên gia (vì hệ chuyên gia không hiểu), mà gọi Gemini API lần 2: "Trích xuất màu sắc mai rùa từ câu sau của người dùng và định dạng thành JSON {"color": "[màu sắc]"}: 'Mình thấy nó đen thui à, có vài vệt vàng nhạt'".
Gemini trả về chuẩn xác: {"color": "đen, vệt vàng"}.

Chốt kết quả (Hệ chuyên gia): Nhận được từ khóa "đen, vệt vàng", Hệ chuyên gia đẩy độ tự tin lên 90%, chốt kết quả là Rùa Trung Bộ, sau đó hiển thị toàn bộ luật pháp lên nửa màn hình bên trái.