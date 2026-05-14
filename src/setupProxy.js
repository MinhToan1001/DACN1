const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // 1. Gửi các request bắt đầu bằng /api tới Node.js (Port 5001)
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5001',
      changeOrigin: true,
    })
  );

  // 2. Gửi các request bắt đầu bằng /ai (hoặc đường dẫn bạn quy định) tới Python (Port 5002)
  // Lưu ý: Nếu Flask của bạn chưa có tiền tố /ai, bạn có thể map lại ở đây
  app.use(
    '/ai',
    createProxyMiddleware({
      target: 'http://localhost:5002',
      changeOrigin: true,
      pathRewrite: {
        '^/ai': '', // Xóa chữ /ai khi gửi sang Flask nếu Flask không dùng prefix này
      },
    })
  );
};