const db = require('../db');

// Lấy tất cả động vật
exports.getAllAnimals = (req, res) => {
  db.query('SELECT * FROM animals', (err, results) => {
    if (err) return res.status(500).json({ error: 'Lỗi truy vấn CSDL' });
    res.json(results);
  });
};

// Lấy 1 động vật theo id
exports.getAnimalById = (req, res) => {
  db.query('SELECT * FROM animals WHERE id = ?', [req.params.id], (err, results) => {
    if (err) return res.status(500).json({ error: 'Lỗi truy vấn CSDL' });
    if (results.length === 0) return res.status(404).json({ error: 'Không tìm thấy' });
    res.json(results[0]);
  });
};

// Thêm động vật
exports.createAnimal = (req, res) => {
  const { name, image_url, link, red_list, red_level } = req.body;
  db.query(
    'INSERT INTO animals (name, image_url, link, red_list, red_level) VALUES (?, ?, ?, ?, ?)',
    [name, image_url, link, red_list, red_level],
    (err) => {
      if (err) return res.status(500).json({ error: 'Lỗi khi thêm' });
      res.json({ message: 'Thêm thành công' });
    }
  );
};

// Cập nhật động vật
exports.updateAnimal = (req, res) => {
  const { name, image_url, link, red_list, red_level } = req.body;
  db.query(
    'UPDATE animals SET name=?, image_url=?, link=?, red_list=?, red_level=? WHERE id=?',
    [name, image_url, link, red_list, red_level, req.params.id],
    (err) => {
      if (err) return res.status(500).json({ error: 'Lỗi khi cập nhật' });
      res.json({ message: 'Cập nhật thành công' });
    }
  );
};

// Xóa động vật
exports.deleteAnimal = (req, res) => {
  db.query('DELETE FROM animals WHERE id=?', [req.params.id], (err) => {
    if (err) return res.status(500).json({ error: 'Lỗi khi xóa' });
    res.json({ message: 'Xóa thành công' });
  });
};
