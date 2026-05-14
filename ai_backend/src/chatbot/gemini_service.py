import google.generativeai as genai
import os

class ForestryChatbot:
    """Hệ thống Chatbot đóng vai Kiểm lâm hỗ trợ người dân"""

    def __init__(self, api_key: str = None):
        # Ưu tiên lấy key từ tham số, nếu không có thì lấy từ biến môi trường
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        # System Prompt để định hình vai trò của bot (Giống file cũ của bạn)
        self.system_instruction = (
            "Bạn là một chuyên gia Kiểm lâm Việt Nam, am hiểu về luật bảo vệ động vật hoang dã, "
            "đặc điểm sinh học và quy trình pháp lý xử lý vi phạm. "
            "Hãy trả lời người dân một cách chuyên nghiệp, dễ hiểu, ngắn gọn và chính xác."
        )

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Dùng model Gemini hiện tại được hỗ trợ bởi API
                self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                print("[INFO] Đã khởi tạo thành công Gemini Chatbot với models/gemini-2.5-flash.")
            except Exception as e:
                print(f"[ERROR] Không thể khởi tạo Gemini: {e}")
        else:
            print("[WARNING] Chưa có GEMINI_API_KEY. Chatbot sẽ không hoạt động.")

    def get_response(self, user_message: str) -> str:
        if not self.model:
            return "Chức năng Chatbot AI đang bảo trì (Thiếu API Key hoặc lỗi kết nối). Vui lòng thử lại sau."
        
        try:
            # Gép prompt hệ thống và câu hỏi người dùng
            full_prompt = f"{self.system_instruction}\n\nNgười dân hỏi: {user_message}\nKiểm lâm trả lời:"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Hệ thống đang bận. Lỗi: {str(e)}"