const db = require('../db');

//Lấy tất cả dự án
exports.getAllProjects = (req, res) => {
  const query = 'SELECT * FROM projects';
  db.query(query, (err, results) => {
    if (err) {
      console.error('❌ Lỗi truy vấn projects:', err);
      return res.status(500).json({ error: 'Lỗi truy vấn CSDL' });
    }
    res.json(results);
  });
};

// Lấy dự án theo ID
exports.getProjectById = (req, res) => {
  db.query("SELECT * FROM projects WHERE id = ?", [req.params.id], (err, rows) => {
    if (err) return res.status(500).json({ error: err.message });
    if (rows.length === 0) return res.status(404).json({ error: "Không tìm thấy" });
    res.json(rows[0]);
  });
};

// Thêm dự án
exports.createProject = (req, res) => {
  const { title, description, image_url, progress, goal } = req.body;
  db.query(
    "INSERT INTO projects (title, description, image_url, progress, goal) VALUES (?, ?, ?, ?, ?)",
    [title, description, image_url, progress, goal],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ id: result.insertId, title, description, image_url, progress, goal });
    }
  );
};

// Cập nhật dự án
exports.updateProject = (req, res) => {
  const { title, description, image_url, progress, goal } = req.body;
  db.query(
    "UPDATE projects SET title = ?, description = ?, image_url = ?, progress = ?, goal = ? WHERE id = ?",
    [title, description, image_url, progress, goal],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ id: result.insertId, title, description, image_url, progress, goal });
    }
  );
};

// Xóa dự án
exports.deleteProject = (req, res) => {
  db.query("DELETE FROM projects WHERE id = ?", [req.params.id], (err) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ success: true });
  });
};