// routes/payment.js
const express = require('express');
const router = express.Router();
const { createPaymentUrl, vnpayReturn } = require('../controllers/paymentController');

// Tạo URL thanh toán
router.post('/create_payment_url', createPaymentUrl);

// VNPay trả kết quả về (callback)
router.get('/vnpay_return', vnpayReturn);


module.exports = router;
