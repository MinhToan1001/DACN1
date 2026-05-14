import React, { useState, useEffect } from "react";
import SearchBar from "../../components/search_bar";
import axios from "axios";
import AdminLayout from "../../layouts/AdminLayout";
import { Button, Col, Form, Modal, Row } from "react-bootstrap";
import 'react-quill/dist/quill.snow.css';
import ReactQuill from 'react-quill';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import TextEditor from "../../components/text_editor";

interface Forest {
    id?: number;
    name: string;
    lat: number;
    lng: number;
    square: number;
    description: string;
    info: string;
    image_url: string;
}

// Component chọn vị trí
function LocationPicker({ onSelect }: { onSelect: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onSelect(e.latlng.lat, e.latlng.lng);
    }
  });
  return null;
}

export default function ManageForests() {
    const [forests, setForests] = useState<Forest[]>([]);
    const [loading, setLoading] = useState(true);
    const [form, setForm] = useState<Forest>({
        name: "",
        lat: 0,
        lng: 0,
        square: 0,
        description: "",
        info: "",
        image_url: ""
    });

    const [showModal, setShowModal] = useState(false);
    const [showMap, setShowMap] = useState(false);
    const [editId, setEditId] = useState<number | null>(null);
    const [searchTerm, setSearchTerm] = useState("");

    useEffect(() => {
        fetchForests();
    }, []);

    const fetchForests = async () => {
        try {
        setLoading(true);
        const res = await axios.get("/api/forests-map");
        setForests(res.data);
        } catch (error) {
        console.error("Lỗi tải rừng:", error);
        } finally {
        setLoading(false);
        }
    };

    const filteredForests = forests.filter((forest) => {
        const matchSearch = forest.name
            .toLowerCase()
            .includes(searchTerm.toLowerCase());
        return matchSearch;
    });
    // Mở modal thêm
    const handleAdd = () => {
        setEditId(null);
        setForm({ name: "",lat: 0,lng: 0,square: 0,description: "",info: "",image_url: "" });
        setShowModal(true);
    };

    // Mở modal sửa
    const handleEdit = (forest: Forest) => {
        setEditId(forest.id || null);
        setForm(forest);
        setShowModal(true);
    };
    // Lưu dữ liệu
    const handleSave = () => {
        const method = editId ? 'PUT' : 'POST';
        const url = editId ? `/api/forests-map/${editId}` : '/api/forests-map';

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
                fetchForests();
            })
            .catch(err => alert(err.message));
  };

    // Xóa
    const handleDelete = (id?: number) => {
    if (!id) return;
    if (window.confirm('Bạn có chắc muốn xóa?')) {
      fetch(`/api/forests-map/${id}`, { method: 'DELETE' }).then(() => {
        fetchForests();
        });
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
            setForm({ ...form, image_url: reader.result as string });
        };
        reader.readAsDataURL(file);
        }
    };


  return (
    <AdminLayout>
        <div className="p-4">
        <h1 className="text-xl font-bold mb-4">Quản lý Rừng</h1>
        <Button variant="success" className="mb-3" onClick={handleAdd}>
          + Thêm mới
        </Button>
        {/* Bảng danh sách */}
        {loading ? (
            <p>Đang tải...</p>
        ) : (
            <table className="w-full border">
            <thead>
                <tr className="bg-gray-100">
                <th className="p-2 border">Tên rừng</th>
                <th className="p-2 border">Vĩ độ</th>
                <th className="p-2 border">Kinh độ</th>
                <th className="p-2 border">Diện tích</th>
                <th className="p-2 border">Mô tả</th>
                <th className="p-2 border">Ảnh</th>
                <th className="p-2 border">Hành động</th>
                </tr>
            </thead>
            <tbody>
                {filteredForests.map(forest => (
                <tr key={forest.id} className="border">
                    <td className="p-2 border">{forest.name}</td>
                    <td className="p-2 border">{forest.lat}</td>
                    <td className="p-2 border">{forest.lng}</td>
                    <td className="p-2 border">{forest.square}</td>
                    <td className="p-2 border w-25">{forest.description}</td>
                    <td className="p-2 border">
                    {forest.image_url && (
                        <img src={forest.image_url}
                             alt={forest.name}
                             style={{width: '300px',height: '150px',objectFit: 'cover',borderRadius: '6px'}}/>
                    )}
                    </td>
                    <td className="p-2 border">
                    <Button size="sm" variant="primary" onClick={() => handleEdit(forest)}>
                      Sửa
                    </Button>{' '}
                    <Button size="sm" variant="danger" onClick={() => handleDelete(forest.id)}>
                      Xóa
                    </Button>
                    </td>
                </tr>
                ))}
            </tbody>
            </table>
        )}
        {/* Modal thêm/sửa */}
        <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
            <Modal.Header closeButton>
                <Modal.Title>{editId ? 'Sửa' : 'Thêm mới'} Rừng</Modal.Title>
            </Modal.Header>
            <Modal.Body>
            <Form>
                <Row>
                {/* Cột trái: Form nhập dữ liệu */}
                <Col md={7}>
                    <Form.Group className="mb-3">
                    <Form.Label>Tên rừng</Form.Label>
                    <Form.Control
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                    />
                    </Form.Group>

                    <Form.Group className="mb-3">
                        <Row>
                            <Col md={6}>
                                <Form.Group className="mb-3">
                                <Form.Label>Vĩ độ (Latitude)</Form.Label>
                                <Form.Control
                                    type="number"
                                    step="any"
                                    value={form.lat}
                                    onChange={(e) =>
                                    setForm({ ...form, lat: parseFloat(e.target.value) || 0 })
                                    }
                                />
                                </Form.Group>
                            </Col>
                            <Col md={6}>
                                <Form.Group className="mb-3">
                                <Form.Label>Kinh độ (Longitude)</Form.Label>
                                <Form.Control
                                    type="number"
                                    step="any"
                                    value={form.lng}
                                    onChange={(e) =>
                                    setForm({ ...form, lng: parseFloat(e.target.value) || 0 })
                                    }
                                />
                                </Form.Group>
                            </Col>
                            <Button variant="outline-primary" onClick={() => setShowMap(true)}>
                                Chọn trên bản đồ
                            </Button>
                        </Row>
                    </Form.Group>

                    <Form.Group className="mb-3">
                    <Form.Label>Mô tả ngắn</Form.Label>
                    <Form.Control
                        as="textarea"
                        rows={1}
                        value={form.description}
                        onChange={(e) => setForm({ ...form, description: e.target.value })}
                    />
                    </Form.Group>

                    <Form.Group className="mb-3">
                    <Form.Label>Giới thiệu chi tiết</Form.Label>
                    <TextEditor
                        value={form.info}
                        onChange={(val) => setForm({ ...form, info: val })}
                    />
                    </Form.Group>
                </Col>

                {/* Cột phải: Preview ảnh + chọn ảnh */}
                <Col md={5} className="">
                    <Form.Group className="mb-2">
                    <Form.Label>Diện tích (km²)</Form.Label>
                    <Form.Control
                        type="number"
                        step="any"
                        value={form.square}
                        onChange={(e) => setForm({ ...form, square: parseFloat(e.target.value) || 0 })}
                    />
                    </Form.Group>
                    {/* Preview ảnh */}
                    <div
                    style={{
                        border: "1px solid #ddd",
                        padding: "10px",
                        borderRadius: "8px",
                        minHeight: "200px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        marginBottom: "10px",
                        backgroundColor: "#f8f9fa"
                    }}
                    >
                    {form.image_url ? (
                        <img
                        src={form.image_url}
                        alt="Preview"
                        style={{ maxWidth: "100%", maxHeight: "180px", objectFit: "cover" }}
                        />
                    ) : (
                        <span className="text-muted">Chưa có ảnh</span>
                    )}
                    </div>

            <Form.Group className="mt-3">
              <Form.Label>Chọn ảnh</Form.Label>
              <Form.Control type="file" accept="image/*" onChange={handleFileChange} />
            </Form.Group>

            <Form.Group className="mt-2">
              <Form.Label>Hoặc nhập URL ảnh</Form.Label>
              <Form.Control
                value={form.image_url}
                onChange={(e) => setForm({ ...form, image_url: e.target.value })}
              />
            </Form.Group>
                    <Button variant="secondary" onClick={() => setShowModal(false)}>
                        Hủy
                    </Button>{' '}
                    <Button variant="primary" onClick={handleSave}>
                        Lưu
                    </Button>
                </Col>
                </Row>
            </Form>
            </Modal.Body>
        </Modal>

        <Modal show={showMap} onHide={() => setShowMap(false)} size="lg">
            <Modal.Header closeButton>
                <Modal.Title>Chọn vị trí trên bản đồ</Modal.Title>
            </Modal.Header>
            <Modal.Body style={{ height: '500px' }}>
                <MapContainer center={[21.0285, 105.8542]} zoom={6} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationPicker
                    onSelect={(lat, lng) => {
                    setForm({ ...form, lat, lng });
                    setShowMap(false);
                    }}
                />
                {form.lat && form.lng && (
                    <Marker position={[form.lat, form.lng]} />
                )}
                </MapContainer>
            </Modal.Body>
        </Modal>
        
        </div>
    </AdminLayout>
  );
}
