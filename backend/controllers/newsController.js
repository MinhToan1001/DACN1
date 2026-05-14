const db = require('../db');

// Lấy tất cả tin tức
exports.getAllNews = (req, res) => {
  db.query("SELECT * FROM news", (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(results);
  });
};

// Lấy tin tức theo ID
exports.getNewsById = (req, res) => {
  db.query("SELECT * FROM news WHERE id = ?", [req.params.id], (err, rows) => {
    if (err) return res.status(500).json({ error: err.message });
    if (rows.length === 0) return res.status(404).json({ error: "Không tìm thấy" });
    res.json(rows[0]);
  });
};

// Thêm mới tin tức
exports.createNews = (req, res) => {
  const { title, author, date, content, thumbnail } = req.body;
  db.query(
    "INSERT INTO news (title, author, date, content, thumbnail) VALUES (?, ?, ?, ?, ?)",
    [title, author, date, content, thumbnail],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ id: result.insertId, title, author, date, content, thumbnail });
    }
  );
};

// Cập nhật tin tức
exports.updateNews = (req, res) => {
  const { title, author, date, content, thumbnail } = req.body;
  db.query(
    "UPDATE news SET title = ?, author = ?, date = ?, content = ?, thumbnail = ? WHERE id = ?",
    [title, author, date, content, thumbnail, req.params.id],
    (err) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ id: req.params.id, title, author, date, content, thumbnail });
    }
  );
};

// Xóa tin tức
exports.deleteNews = (req, res) => {
  db.query("DELETE FROM news WHERE id = ?", [req.params.id], (err) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ success: true });
  });
};
