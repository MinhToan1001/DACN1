import React, { useState, useEffect } from "react";
import { Table, Button, Modal, Form, Spinner } from "react-bootstrap";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";
import AdminLayout from "../../layouts/AdminLayout";
import moment from "moment";
interface Contacts {
  id?: number;
  first_name: string;
  last_name: string;
  email: string;
  message: string;
  created_at: string;
}

export default function ManageContacts() {
  const [contact, setContact] = useState<Contacts[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [editId, setEditId] = useState<number | null>(null);
  const [form, setForm] = useState<Contacts>({
    first_name: '',
    last_name: '',
    email: '',
    message: '',
    created_at: '',
  });

  useEffect(() => {
    fetchContact();
  }, []);

  // Lấy dữ liệu
  const fetchContact = () => {
    setLoading(true);
    fetch('/api/contact')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setContact(data);
        } else if (Array.isArray(data.data)) { 
          setContact(data.data); // nếu API bọc mảng trong key "data"
        } else {
          setContact([]); // fallback tránh lỗi
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  const handleAdd = () => {
    setEditId(null);
    setForm({
        first_name: '',
        last_name: '',
        email: '',
        message: '',
        created_at: '',
    });
    setShowModal(true);
  };

  // Lưu dữ liệu
  const handleSave = () => {
    const method = editId ? 'PUT' : 'POST';
    const url = editId ? `/api/contact/${editId}` : '/api/contact';

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
        fetchContact();
      })
      .catch(err => alert(err.message));
  };

  // Xóa
  const handleDelete = (id?: number) => {
    if (!id) return;
    if (window.confirm('Bạn có chắc muốn xóa?')) {
      fetch(`/api/contact/${id}`, { method: 'DELETE' }).then(() => {
        fetchContact();
      });
    }
  };

    const [emailSending, setEmailSending] = useState(false);
  const [emailSubject, setEmailSubject] = useState("");
  const [emailContent, setEmailContent] = useState("");

  // Khi mở modal gửi email (handleEdit), reset subject + content
  const handleEdit = (c: Contacts) => {
    setEditId(c.id || null);
    setForm(c);
    setEmailSubject(`Phản hồi cho câu hỏi của bạn, ${c.first_name}`);
    setEmailContent("");
    setShowModal(true);
  };

  // Gửi email phản hồi
  const handleSendEmail = () => {
    if (!form.email) {
      alert("Email người nhận không hợp lệ");
      return;
    }
    setEmailSending(true);

    fetch("/api/contact/send-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        to: form.email,
        subject: emailSubject,
        content: emailContent,
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Gửi email thất bại");
        return res.json();
      })
      .then(() => {
        alert("Email đã được gửi!");
        setShowModal(false);
      })
      .catch((err) => alert(err.message))
      .finally(() => setEmailSending(false));
  };

  return (
    <AdminLayout>
      <div className="p-3">
        <h2>Tin nhắn người dùng </h2>

        {loading ? (
          <div className="text-center py-5">
            <Spinner animation="border" />
          </div>
        ) : (
          <Table bordered hover>
            <thead>
              <tr>
                <th>Tên</th>
                <th>Họ đệm</th>
                <th>Email</th>
                <th>Nội dung</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {Array.isArray(contact) && contact.map((c) => (
                <tr key={c.id}>
                  <td>{c.first_name}</td>
                  <td>{c.last_name}</td>
                  <td>{c.email}</td>
                  <td>{c.message}</td>
                  <td>
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => handleEdit(c)}
                      className="center"
                    >
                      Gửi phản hồi
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}

      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
        <Modal.Header closeButton>
          <Modal.Title>Gửi phản hồi Email</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form.Group className="mb-3">
            <Form.Label>Email người nhận</Form.Label>
            <Form.Control type="email" value={form.email} readOnly />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Chủ đề</Form.Label>
            <Form.Control
              type="text"
              value={emailSubject}
              onChange={(e) => setEmailSubject(e.target.value)}
              placeholder="Nhập chủ đề email"
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Nội dung</Form.Label>
            <ReactQuill
              theme="snow"
              value={emailContent}
              onChange={setEmailContent}
              style={{ height: 180 }}
            />
          </Form.Group>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)} disabled={emailSending}>
            Hủy
          </Button>
          <Button variant="primary" onClick={handleSendEmail} disabled={emailSending}>
            {emailSending ? "Đang gửi..." : "Gửi email"}
          </Button>
        </Modal.Footer>
      </Modal>
      </div>
    </AdminLayout>
  );
}
