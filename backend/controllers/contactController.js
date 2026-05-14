require('dotenv').config()
const db = require('../db')
const nodemailer = require('nodemailer')

// API POST gửi yêu cầu xác nhận liên hệ (gửi email xác nhận)
exports.sendContactVerification = (req, res) => {
  const { firstName, lastName, email, message } = req.body;

  if (!firstName || !lastName || !email || !message) {
    return res.status(400).json({ error: 'Vui lòng điền đầy đủ thông tin' });
  }

  const tokenData = { firstName, lastName, email, message };
  const token = Buffer.from(JSON.stringify(tokenData)).toString('base64');
  const confirmLink = `http://localhost:${process.env.PORT || 5001}/api/contact/confirm?token=${token}`;

  // Cấu hình gửi email
  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASSWORD, // App Password
    },
  });

  const mailOptions = {
    from: `"GreenLand" <${process.env.EMAIL_USER}>`,
    to: email,
    subject: 'Xác nhận liên hệ GreenLand',
    html: `
      <p>Xin chào ${firstName} ${lastName},</p>
      <p>Bạn đã gửi yêu cầu liên hệ với GreenLand. Vui lòng xác nhận bằng cách bấm vào liên kết sau:</p>
      <a href="${confirmLink}">Xác nhận liên hệ</a>
      <p>Nếu bạn không thực hiện hành động này, vui lòng bỏ qua email.</p>
    `,
  };

  transporter.sendMail(mailOptions, (error) => {
    if (error) {
      console.error('❌ Lỗi gửi email:', error);
      return res.status(500).json({ error: 'Gửi email thất bại' });
    }
    res.json({ message: 'Đã gửi email xác nhận' });
  });
};

// API xác nhận liên hệ (lưu vào DB)
exports.confirmContact = (req, res) => {
  const token = req.query.token;
  if (!token) return res.status(400).send('Thiếu token xác nhận');

  try {
    const json = Buffer.from(token, 'base64').toString('utf-8');
    const data = JSON.parse(json);
    const { firstName, lastName, email, message } = data;

    const query = 'INSERT INTO contacts (first_name, last_name, email, message) VALUES (?, ?, ?, ?)';
    db.query(query, [firstName, lastName, email, message], (err) => {
      if (err) {
        console.error('❌ Lỗi khi lưu contact:', err);
        return res.status(500).send('Lỗi khi lưu thông tin liên hệ');
      }
      res.send(`<h2 style="color:green">✅ Xác nhận thành công!</h2><p>Chúng tôi đã nhận được thông tin của bạn.</p>`);
    });
  } catch (err) {
    console.error('❌ Token không hợp lệ:', err);
    return res.status(400).send('❌ Token không hợp lệ');
  }
};

// Lấy tất cả liene heej
exports.getAllContacts = (req, res) => {
  db.query('SELECT * FROM contacts', (err, results) => {
    if (err) return res.status(500).json({ error: 'Lỗi truy vấn CSDL' });
    res.json(results);
  });
};

// Gửi email phản hồi
exports.sendEmail = async (req, res) => {
  const { to, subject, content } = req.body;

  if (!to || !subject || !content) {
    return res.status(400).json({ message: "Thiếu thông tin gửi email" });
  }

  try {
    // Cấu hình transporter (thay đổi theo mail bạn dùng)
    let transporter = nodemailer.createTransport({
      host: "smtp.gmail.com",
      port: 465,
      secure: true, // SSL
      auth: {
        user: process.env.EMAIL_USER, // email gửi đi
        pass: process.env.EMAIL_PASSWORD, // mật khẩu hoặc app password
      },
    });

    // Tạo mail options
    let mailOptions = {
      from: `"Support" <${process.env.EMAIL_USER}>`,
      to,
      subject,
      html: content,
    };

    // Gửi mail
    await transporter.sendMail(mailOptions);

    res.json({ message: "Email đã được gửi" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Lỗi khi gửi email" });
  }
};