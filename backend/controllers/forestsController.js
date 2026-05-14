const db = require('../db');

// 📌 Lấy danh sách tất cả rừng
exports.getAllForests = (req, res) => {
  const query = 'SELECT * FROM forests ORDER BY id DESC';
  db.query(query, (err, results) => {
    if (err) {
      console.error('❌ Lỗi truy vấn:', err);
      return res.status(500).json({ error: 'Lỗi khi lấy danh sách rừng' });
    }
    res.json(results);
  });
};

// 📌 Lấy chi tiết 1 rừng theo ID
exports.getForestById = (req, res) => {
  const { id } = req.params;
  const query = 'SELECT * FROM forests WHERE id = ? LIMIT 1';
  db.query(query, [id], (err, results) => {
    if (err) {
      console.error('❌ Lỗi truy vấn:', err);
      return res.status(500).json({ error: 'Lỗi khi lấy chi tiết rừng' });
    }
    if (results.length === 0) {
      return res.status(404).json({ error: 'Không tìm thấy rừng' });
    }
    res.json(results[0]);
  });
};

// 📌 Thêm rừng mới
exports.createForest = (req, res) => {
  const { name, lat, lng, square, description, info, image_url } = req.body;
  if (!name || !lat || !lng) {
    return res.status(400).json({ error: 'Vui lòng nhập đầy đủ tên và địa điểm' });
  }
  const query = 'INSERT INTO forests (name, lat, lng, square, description, info, image_url) VALUES (?, ?, ?, ?, ?, ?, ?)';
  db.query(query, [name, lat, lng, square, description, info, image_url], (err, result) => {
    if (err) {
      console.error('❌ Lỗi thêm rừng:', err);
      return res.status(500).json({ error: 'Không thể thêm rừng mới' });
    }
    res.json({ success: true, message: 'Thêm rừng thành công', id: result.insertId });
  });
};

// 📌 Cập nhật thông tin rừng
exports.updateForest = (req, res) => {
  const { id } = req.params;
  const { name, lat, lng, square, description, info, image_url } = req.body;
  const query = 'UPDATE forests SET name=?, lat=?, lng=?, square=?, description=?, info=?, image_url=? WHERE id=?';
  db.query(query, [name, lat, lng, square, description, info, image_url, id], (err) => {
    if (err) {
      console.error('❌ Lỗi cập nhật rừng:', err);
      return res.status(500).json({ error: 'Không thể cập nhật rừng' });
    }
    res.json({ success: true, message: 'Cập nhật rừng thành công' });
  });
};

// 📌 Xóa rừng
exports.deleteForest = (req, res) => {
  const { id } = req.params;
  const query = 'DELETE FROM forests WHERE id = ?';
  db.query(query, [id], (err) => {
    if (err) {
      console.error('❌ Lỗi xóa rừng:', err);
      return res.status(500).json({ error: 'Không thể xóa rừng' });
    }
    res.json({ success: true, message: 'Xóa rừng thành công' });
  });
};
