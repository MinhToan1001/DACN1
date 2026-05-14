const db = require("../db");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

const SECRET_KEY = "ban-nen-thay-key-nay"; // thay bằng key bảo mật riêng

// Đăng ký
exports.register = async (req, res) => {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
        return res.status(400).json({ message: "Vui lòng nhập đầy đủ thông tin." });
    }

    db.query("SELECT * FROM users WHERE email = ?", [email], async (err, results) => {
        if (err) return res.status(500).json({ message: "Lỗi server" });

        if (results.length > 0) {
            return res.status(400).json({ message: "Email đã tồn tại." });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        db.query(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            [name, email, hashedPassword],
            (err) => {
                if (err) return res.status(500).json({ message: "Lỗi lưu dữ liệu." });
                return res.status(201).json({ message: "Đăng ký thành công!" });
            }
        );
    });
};

// Đăng nhập
exports.login = (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ message: "Vui lòng nhập email và mật khẩu." });
    }

    db.query("SELECT * FROM users WHERE email = ?", [email], async (err, results) => {
        if (err) return res.status(500).json({ message: "Lỗi server" });

        if (results.length === 0) {
            return res.status(401).json({ message: "Sai email hoặc mật khẩu." });
        }

        const user = results[0];
        const isMatch = await bcrypt.compare(password, user.password_hash);

        if (!isMatch) {
            return res.status(401).json({ message: "Sai email hoặc mật khẩu." });
        }

        const token = jwt.sign(
            { id: user.id, email: user.email, role: user.role },
            SECRET_KEY,
            { expiresIn: "1h" }
        );

        return res.json({
            message: "Đăng nhập thành công!",
            token,
            user: { id: user.id, name: user.name, email: user.email, role: user.role }
        });
    });
};
