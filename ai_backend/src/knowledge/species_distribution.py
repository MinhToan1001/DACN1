import re
import unicodedata
from typing import Dict, List


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFD", value or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.lower()).strip()


class SpeciesDistributionService:
    """Suy diễn khu vực có thể bắt gặp loài từ knowledge sinh học hiện có."""

    LOCATION_POINTS = [
        {"keywords": ["toan quoc", "ca nuoc", "khap nuoc"], "name": "Phân bố rộng trên toàn quốc", "lat": 16.0471, "lng": 108.2068, "type": "Vùng rộng"},
        {"keywords": ["tay nguyen", "gia lai", "dak lak", "dak nong", "kon tum"], "name": "Tây Nguyên", "lat": 13.8079, "lng": 108.1094, "type": "Vùng sinh cảnh"},
        {"keywords": ["dong nam bo", "dong nai", "binh phuoc", "tay ninh"], "name": "Đông Nam Bộ", "lat": 11.1079, "lng": 106.1362, "type": "Vùng sinh cảnh"},
        {"keywords": ["nam bo", "dong bang song cuu long", "mien tay", "u minh"], "name": "Nam Bộ và Đồng bằng sông Cửu Long", "lat": 10.0452, "lng": 105.7469, "type": "Vùng sinh cảnh"},
        {"keywords": ["mien trung", "nam trung bo", "bac trung bo"], "name": "Miền Trung", "lat": 15.8801, "lng": 108.3380, "type": "Vùng sinh cảnh"},
        {"keywords": ["tay bac", "son la", "lai chau", "dien bien"], "name": "Tây Bắc", "lat": 21.3270, "lng": 103.9141, "type": "Vùng sinh cảnh"},
        {"keywords": ["dong bac", "ha giang", "tuyen quang", "bac kan", "quang ninh"], "name": "Đông Bắc", "lat": 22.1470, "lng": 105.8348, "type": "Vùng sinh cảnh"},
        {"keywords": ["day truong son", "truong son"], "name": "Dãy Trường Sơn", "lat": 16.6000, "lng": 106.8000, "type": "Hành lang sinh cảnh"},
        {"keywords": ["cat tien", "vuon quoc gia cat tien", "vqg cat tien"], "name": "Vườn quốc gia Cát Tiên", "lat": 11.4244, "lng": 107.4286, "type": "Vườn quốc gia"},
        {"keywords": ["yok don", "vqg yok don"], "name": "Vườn quốc gia Yok Đôn", "lat": 12.8382, "lng": 107.7430, "type": "Vườn quốc gia"},
        {"keywords": ["cuc phuong", "vqg cuc phuong"], "name": "Vườn quốc gia Cúc Phương", "lat": 20.3167, "lng": 105.6083, "type": "Vườn quốc gia"},
        {"keywords": ["phong nha", "ke bang"], "name": "Vườn quốc gia Phong Nha - Kẻ Bàng", "lat": 17.5903, "lng": 106.2830, "type": "Vườn quốc gia"},
        {"keywords": ["bach ma"], "name": "Vườn quốc gia Bạch Mã", "lat": 16.1938, "lng": 107.8534, "type": "Vườn quốc gia"},
        {"keywords": ["tram chim"], "name": "Vườn quốc gia Tràm Chim", "lat": 10.7081, "lng": 105.5169, "type": "Vườn quốc gia"},
        {"keywords": ["xuan thuy"], "name": "Vườn quốc gia Xuân Thủy", "lat": 20.2447, "lng": 106.5583, "type": "Vườn quốc gia"},
        {"keywords": ["con dao"], "name": "Côn Đảo", "lat": 8.6942, "lng": 106.6114, "type": "Khu biển đảo"},
        {"keywords": ["phu quoc"], "name": "Phú Quốc", "lat": 10.2899, "lng": 103.9840, "type": "Khu biển đảo"},
        {"keywords": ["son tra"], "name": "Bán đảo Sơn Trà", "lat": 16.1196, "lng": 108.2730, "type": "Khu bảo tồn"},
        {"keywords": ["lang biang", "da lat", "lam dong"], "name": "Lang Biang - Lâm Đồng", "lat": 12.0499, "lng": 108.4419, "type": "Vùng núi cao"},
        {"keywords": ["tam dao"], "name": "Tam Đảo", "lat": 21.4569, "lng": 105.6442, "type": "Vùng núi"},
        {"keywords": ["ba den"], "name": "Núi Bà Đen", "lat": 11.3717, "lng": 106.1667, "type": "Vùng núi"},
        {"keywords": ["ninh thuan", "binh thuan"], "name": "Ninh Thuận - Bình Thuận", "lat": 11.3155, "lng": 108.7690, "type": "Vùng khô hạn"},
        {"keywords": ["khanh hoa"], "name": "Khánh Hòa", "lat": 12.2585, "lng": 109.0526, "type": "Ven biển"},
        {"keywords": ["soc trang", "bac lieu", "tra vinh"], "name": "Rừng ngập mặn Nam Bộ", "lat": 9.7984, "lng": 105.9739, "type": "Rừng ngập mặn"},
    ]

    HABITAT_POINTS = [
        {"keywords": ["rung ngap man", "cua song", "bai bun", "vung trieu"], "name": "Vùng cửa sông, bãi bùn và rừng ngập mặn", "lat": 20.2447, "lng": 106.5583, "type": "Sinh cảnh"},
        {"keywords": ["song suoi", "suoi", "ho", "dam lay", "nuoc ngot"], "name": "Sông suối, hồ và vùng đất ngập nước", "lat": 10.7081, "lng": 105.5169, "type": "Sinh cảnh"},
        {"keywords": ["rung nhiet doi", "rung thuong xanh", "rung nguyen sinh"], "name": "Rừng nhiệt đới thường xanh", "lat": 11.4244, "lng": 107.4286, "type": "Sinh cảnh"},
        {"keywords": ["nui da voi", "rung nui da"], "name": "Rừng núi đá vôi", "lat": 20.3167, "lng": 105.6083, "type": "Sinh cảnh"},
        {"keywords": ["ran san ho", "bien", "co bien", "dao"], "name": "Vùng biển, đảo và rạn san hô", "lat": 12.2585, "lng": 109.0526, "type": "Sinh cảnh"},
    ]

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def _match_points(self, text: str, candidates: List[Dict]) -> List[Dict]:
        normalized = _normalize_text(text)
        matched = []
        seen = set()
        for point in candidates:
            for keyword in point["keywords"]:
                if keyword in normalized and point["name"] not in seen:
                    seen.add(point["name"])
                    matched.append({
                        "name": point["name"],
                        "lat": point["lat"],
                        "lng": point["lng"],
                        "type": point["type"],
                        "matched_keyword": keyword,
                    })
                    break
        return matched

    def get_distribution(self, scientific_name: str) -> Dict:
        bio_info, _ = self.kb.get_species_data(scientific_name)
        features = self.kb.get_identification_features(scientific_name)
        distribution_text = (
            features.get("phan_bo_viet_nam")
            or bio_info.get("phan_bo_viet_nam")
            or bio_info.get("distribution")
            or ""
        )
        habitat_text = (
            features.get("sinh_thai")
            or bio_info.get("sinh_thai")
            or bio_info.get("habitat")
            or ""
        )

        areas = self._match_points(distribution_text, self.LOCATION_POINTS)
        if not areas:
            areas = self._match_points(habitat_text, self.HABITAT_POINTS)

        return {
            "species": scientific_name,
            "has_distribution": bool(areas),
            "distribution_text": distribution_text or "Chưa có dữ liệu phân bố tại Việt Nam.",
            "habitat_text": habitat_text or "Chưa có dữ liệu sinh cảnh.",
            "areas": areas[:8],
            "note": (
                "Các điểm trên bản đồ là khu vực tham khảo được suy diễn từ mô tả phân bố/sinh cảnh trong cơ sở tri thức, "
                "không phải tọa độ ghi nhận cá thể chính xác."
            ),
        }
