const express = require('express');
const router = express.Router();
const contactController = require('../controllers/contactController');

// Gửi email xác nhận liên hệ
router.post('/verify', contactController.sendContactVerification);

// Xác nhận liên hệ (lưu vào DB)
router.get('/confirm', contactController.confirmContact);

router.get('/', contactController.getAllContacts)


router.post("/send-email", contactController.sendEmail);
module.exports = router;
