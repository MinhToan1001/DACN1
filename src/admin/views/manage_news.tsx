import React, { useState, useEffect } from "react";
import { Table, Button, Modal, Form, Spinner } from "react-bootstrap";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";
import AdminLayout from "../../layouts/AdminLayout";
import moment from "moment";
interface News {
  id?: number;
  title: string;
  author: string;
  date: string;
  content: string;
  thumbnail: string;
}

export default function ManageNews() {
  const [newsList, setNewsList] = useState<News[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [editId, setEditId] = useState<number | null>(null);
  const [form, setForm] = useState<News>({
    title: "",
    author: "",
    date: "",
    content: "",
    thumbnail: "",
  });

  useEffect(() => {
    fetchNews();
  }, []);

  // Lấy dữ liệu
  const fetchNews = () => {
    setLoading(true);
    fetch('/api/news')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setNewsList(data);
        } else if (Array.isArray(data.data)) { 
          setNewsList(data.data); // nếu API bọc mảng trong key "data"
        } else {
          setNewsList([]); // fallback tránh lỗi
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  const handleAdd = () => {
    setEditId(null);
    setForm({
      title: "",
      author: "",
      date: "",
      content: "",
      thumbnail: "",
    });
    setShowModal(true);
  };

  const handleEdit = (n: News) => {
    setEditId(n.id || null);
    setForm(n);
    setShowModal(true);
  };

  // Lưu dữ liệu
  const handleSave = () => {
    const method = editId ? 'PUT' : 'POST';
    const url = editId ? `/api/news/${editId}` : '/api/news';

    const dataToSend = {
      ...form,
    };

    fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dataToSend),
    })
      .then(res => {
        if (!res.ok) throw new Error('Lỗi khi lưu dữ liệu');
        return res.json();
      })
      .then(() => {
        setShowModal(false);
        fetchNews();
      })
      .catch(err => alert(err.message));
  };

  // Xóa
  const handleDelete = (id?: number) => {
    if (!id) return;
    if (window.confirm('Bạn có chắc muốn xóa?')) {
      fetch(`/api/news/${id}`, { method: 'DELETE' }).then(() => {
        fetchNews();
      });
    }
  };

  return (
    <AdminLayout>
      <div className="p-3">
        <h2>Quản lý tin tức</h2>
        <Button onClick={handleAdd} className="mb-3">
          Thêm tin tức
        </Button>

        {loading ? (
          <div className="text-center py-5">
            <Spinner animation="border" />
          </div>
        ) : (
          <Table bordered hover>
            <thead>
              <tr>
                <th>ID</th>
                <th>Tiêu đề</th>
                <th>Tác giả</th>
                <th>Ngày</th>
                <th>Hành động</th>
              </tr>
            </thead>
            <tbody>
              {Array.isArray(newsList) && newsList.map((n) => (
                <tr key={n.id}>
                  <td>{n.id}</td>
                  <td>{n.title}</td>
                  <td>{n.author}</td>
                  <td>{moment(n.date).format("D MMMM, YYYY")}</td>
                  <td>
                    <Button
                      variant="warning"
                      size="sm"
                      onClick={() => handleEdit(n)}
                      className="me-2"
                    >
                      Sửa
                    </Button>
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={() => handleDelete(n.id)}
                    >
                      Xóa
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}

        <Modal
          show={showModal}
          onHide={() => setShowModal(false)}
          size="lg"
          centered
        >
          <Modal.Header closeButton>
            <Modal.Title>
              {editId ? "Sửa tin tức" : "Thêm tin tức"}
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <div style={{ display: "flex", gap: "20px" }}>
              {/* Cột trái */}
              <div style={{ flex: 1 }}>
                <Form.Group className="mb-3">
                  <Form.Label>Tiêu đề</Form.Label>
                  <Form.Control
                    value={form.title}
                    onChange={(e) => {
                      setForm({...form, title: e.target.value,});
                    }}
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Tác giả</Form.Label>
                  <Form.Control
                    value={form.author}
                    onChange={(e) =>
                      setForm({ ...form, author: e.target.value })
                    }
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Ngày</Form.Label>
                  <Form.Control
                    type="date"
                    value={form.date}
                    onChange={(e) =>
                      setForm({ ...form, date: e.target.value })
                    }
                  />
                </Form.Group>
              </div>

              {/* Cột phải */}
              <div style={{ flex: 1, textAlign: "center" }}>
                {form.thumbnail && (
                  <img
                    src={form.thumbnail}
                    alt="Preview"
                    style={{
                      width: "100%",
                      height: 150,
                      objectFit: "cover",
                      marginBottom: 10,
                    }}
                  />
                )}
                <Form.Group className="mb-3">
                  <Form.Label>URL ảnh thumbnail</Form.Label>
                  <Form.Control
                    value={form.thumbnail}
                    onChange={(e) =>
                      setForm({ ...form, thumbnail: e.target.value })
                    }
                  />
                </Form.Group>
              </div>
            </div>

            <Form.Group className="mb-3">
              <Form.Label>Nội dung</Form.Label>
              <ReactQuill
                theme="snow"
                value={form.content}
                onChange={(value) => setForm({ ...form, content: value })}
                style={{ height: 180 }}
              />
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowModal(false)}>
              Hủy
            </Button>
            <Button variant="primary" onClick={handleSave}>
              Lưu
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    </AdminLayout>
  );
}
