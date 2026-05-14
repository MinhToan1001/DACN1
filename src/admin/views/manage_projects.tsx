import React, { useState, useEffect } from "react";
import { Table, Button, Modal, Form, Spinner, Image } from "react-bootstrap";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";
import AdminLayout from "../../layouts/AdminLayout";
import moment from "moment";

interface Projects {
  id?: number;
  title: string;
  description: string;
  image_url: string;
  progress: number;
  goal: string;
}

export default function ManageProjects() {
  const [project, setProject] = useState<Projects[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [editId, setEditId] = useState<number | null>(null);
  const [form, setForm] = useState<Projects>({
    title: "",
    description: "",
    image_url: "",
    progress: 0,
    goal: "",
  });

  useEffect(() => {
    fetchProject();
  }, []);

  // Lấy dữ liệu
  const fetchProject = () => {
    setLoading(true);
    fetch('/api/projects')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setProject(data);
        } else if (Array.isArray(data.data)) { 
          setProject(data.data); // nếu API bọc mảng trong key "data"
        } else {
          setProject([]); // fallback tránh lỗi
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  const handleAdd = () => {
    setEditId(null);
    setForm({
        title: "",
        description: "",
        image_url: "",
        progress: 0,
        goal: "",
    });
    setShowModal(true);
  };

  const handleEdit = (p: Projects) => {
    setEditId(p.id || null);
    setForm(p);
    setShowModal(true);
  };

  // Lưu dữ liệu
  const handleSave = () => {
    const method = editId ? 'PUT' : 'POST';
    const url = editId ? `/api/projects/${editId}` : '/api/projects';

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
        fetchProject();
      })
      .catch(err => alert(err.message));
  };

  // Xóa
  const handleDelete = (id?: number) => {
    if (!id) return;
    if (window.confirm('Bạn có chắc muốn xóa?')) {
      fetch(`/api/projects/${id}`, { method: 'DELETE' }).then(() => {
        fetchProject();
      });
    }
  };

  return (
    <AdminLayout>
      <div className="p-3">
        <h2>Quản lý dự án</h2>
        <Button onClick={handleAdd} className="mb-3">
          Thêm dự án
        </Button>

        {loading ? (
          <div className="text-center py-5">
            <Spinner animation="border" />
          </div>
        ) : (
          <Table bordered hover>
            <thead>
              <tr>
                <th>Tên dự án</th>
                <th>Mô tả</th>
                <th>Ảnh</th>
                <th>Tiến trình</th>
                <th>Mục tiêu</th>
                <th>Hành động</th>
              </tr>
            </thead>
            <tbody>
              {Array.isArray(project) && project.map((p) => (
                <tr key={p.id}>
                  <td>{p.title}</td>
                  <td>{p.description}</td>
                  <td>
                    <Image src={p.image_url} alt={p.title} width={100} height={80} rounded />
                  </td>
                  <td>{p.progress + "%"}</td>
                  <td>{p.goal}</td>
                  <td>
                    <Button
                      variant="warning"
                      size="sm"
                      onClick={() => handleEdit(p)}
                      className="me-2"
                    >
                      Sửa
                    </Button>
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={() => handleDelete(p.id)}
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
              {editId ? "dự án" : "Thêm dự án"}
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <div style={{ display: "flex", gap: "20px" }}>
              {/* Cột trái */}
              <div style={{ flex: 1 }}>
                <Form.Group className="mb-3">
                  <Form.Label>Tên dự án</Form.Label>
                  <Form.Control
                    value={form.title}
                    onChange={(e) => {
                      setForm({...form, title: e.target.value,});
                    }}
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Mô tả</Form.Label>
                  <Form.Control
                    value={form.description}
                    onChange={(e) =>
                      setForm({ ...form, description: e.target.value })
                    }
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Tiến trình</Form.Label>
                  <Form.Control
                    type="number"
                    value={form.progress}
                    onChange={(e) =>
                      setForm({ ...form, progress: parseFloat(e.target.value) || 0 })
                    }
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Mục tiêu</Form.Label>
                  <Form.Control
                    value={form.goal}
                    onChange={(e) =>
                      setForm({ ...form, goal: e.target.value })
                    }
                  />
                </Form.Group>
              </div>

              {/* Cột phải */}
              <div style={{ flex: 1, textAlign: "center" }}>
                {form.image_url && (
                  <img
                    src={form.image_url}
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
                  <Form.Label>URL ảnh </Form.Label>
                  <Form.Control
                    value={form.image_url}
                    onChange={(e) =>
                      setForm({ ...form, image_url: e.target.value })
                    }
                  />
                </Form.Group>
              </div>
            </div>
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
