const db = require('../db');
const moment = require('moment');
const crypto = require('crypto');
const qs = require('qs');

exports.createPaymentUrl = (req, res) => {
  const ipAddr = req.headers['x-forwarded-for'] || req.socket.remoteAddress || null;
  const tmnCode = process.env.VNP_TMN_CODE || process.env.VNP_TMNCODE;
  const secretKey = process.env.VNP_HASH_SECRET;
  let vnpUrl = process.env.VNP_URL;
  const returnUrl = process.env.VNP_RETURN_URL || process.env.VNP_RETURNURL;

  if (!tmnCode || !secretKey || !vnpUrl || !returnUrl) {
    return res.status(500).json({ error: 'Missing VNPay configuration' });
  }

  const date = new Date();
  const createDate = moment(date).format('YYYYMMDDHHmmss');
  const orderId = moment(date).format('HHmmss');
  const amount = Number(req.body.amount || req.query.amount);
  const projectId = req.body.projectId || req.query.projectId;

  if (!amount || !projectId) {
    return res.status(400).json({ error: 'Missing amount or projectId' });
  }

  let vnpParams = {
    vnp_Version: '2.1.0',
    vnp_Command: 'pay',
    vnp_TmnCode: tmnCode,
    vnp_Locale: req.body.locale || req.query.locale || 'vn',
    vnp_CurrCode: 'VND',
    vnp_TxnRef: orderId,
    vnp_OrderInfo: `Thanh toan cho du an ${projectId}`,
    vnp_OrderType: 'donation',
    vnp_Amount: amount * 100,
    vnp_ReturnUrl: returnUrl,
    vnp_IpAddr: ipAddr,
    vnp_CreateDate: createDate,
  };

  vnpParams = sortObject(vnpParams);

  const signData = qs.stringify(vnpParams, { encode: false });
  const hmac = crypto.createHmac('sha512', secretKey);
  vnpParams.vnp_SecureHash = hmac.update(Buffer.from(signData, 'utf-8')).digest('hex');

  vnpUrl += '?' + qs.stringify(vnpParams, { encode: false });
  return res.json({ paymentUrl: vnpUrl, redirectUrl: vnpUrl });
};

exports.vnpayReturn = (req, res) => {
  let vnpParams = { ...req.query };
  const secureHash = vnpParams.vnp_SecureHash;

  delete vnpParams.vnp_SecureHash;
  delete vnpParams.vnp_SecureHashType;

  vnpParams = sortObject(vnpParams);
  const signData = qs.stringify(vnpParams, { encode: false });
  const hmac = crypto.createHmac('sha512', process.env.VNP_HASH_SECRET);
  const signed = hmac.update(Buffer.from(signData, 'utf-8')).digest('hex');

  if (secureHash !== signed) {
    return res.redirect('/payment-error');
  }

  if (vnpParams.vnp_ResponseCode !== '00') {
    return res.redirect('/payment-failed');
  }

  db.query(
    `INSERT INTO donors (user_id, name, email, amount, project_id, transaction_id)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [
      req.query.userId,
      req.query.userName,
      req.query.userEmail,
      Number(vnpParams.vnp_Amount) / 100,
      req.query.projectId,
      vnpParams.vnp_TxnRef,
    ],
    (err) => {
      if (err) {
        console.error('Error saving donation:', err);
        return res.redirect('/payment-error');
      }

      return res.redirect('/payment-success');
    }
  );
};

function sortObject(obj) {
  const sorted = {};
  Object.keys(obj)
    .sort()
    .forEach((key) => {
      sorted[key] = obj[key];
    });
  return sorted;
}
