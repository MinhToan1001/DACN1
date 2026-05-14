// backend/data/legal_dict.js

const legalData = {
    // --- MẪU 1 LOÀI (Key phải trùng tên thư mục ảnh) ---
    "ho_mang_chua": {
        vietnamese_name: "Rắn Hổ Mang Chúa",
        legal_group: "Nhóm IB - Nghiêm cấm khai thác",
        decree: "Nghị định 06/2019/NĐ-CP",
        penalty_warning: "Phạt tù từ 1-5 năm hoặc phạt tiền tới 2 tỷ đồng.",
        farming_advice: "Cấm nuôi thương mại. Chỉ phục vụ bảo tồn.",
        status_code: "danger", // Các màu: 'danger' (đỏ), 'warning' (vàng), 'success' (xanh)
        description: "Loài rắn độc dài nhất thế giới, cực kỳ nguy hiểm."
    },

    "khi_vang": {
        vietnamese_name: "Khỉ Vàng",
        legal_group: "Nhóm IIB - Hạn chế khai thác",
        decree: "Nghị định 06/2019/NĐ-CP",
        penalty_warning: "Phạt hành chính 50-300 triệu đồng.",
        farming_advice: "Cần đăng ký mã số cơ sở nuôi với Chi cục Kiểm lâm.",
        status_code: "warning",
        description: "Loài linh trưởng phổ biến, sống bầy đàn."
    },

    "lon_rung": {
        vietnamese_name: "Lợn Rừng",
        legal_group: "Động vật hoang dã thông thường",
        decree: "Luật Chăn nuôi 2018",
        penalty_warning: "Không có hình phạt nếu chứng minh nguồn gốc.",
        farming_advice: "Khuyến khích chăn nuôi kinh tế.",
        status_code: "success",
        description: "Loài vật nuôi phổ biến lấy thịt."
    },
    
    // ... Bạn copy paste thêm các loài khác vào đây ...
};

module.exports = legalData;