import React, { useState } from 'react';
import { Modal, Button, Row, Col, Form, Spinner } from 'react-bootstrap';
import axios from 'axios';
import DocumentPreview from './DocumentPreview';

interface FormModalProps {
    show: boolean;
    onHide: () => void;
    prediction: any;
}

const FormGenerationModal: React.FC<FormModalProps> = ({ show, onHide, prediction }) => {
    const [formData, setFormData] = useState({
        city: '',
        fullName: '',
        idCard: '',
        idCardDate: '',
        idCardPlace: '',
        address: '',
        phone: '',
        department: '',
        healthStatus: 'Bình thường'
    });
    const [loading, setLoading] = useState(false);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async () => {
        setLoading(true);
        try {
            // Gửi dữ liệu để tạo đơn theo logic main.py
            const res = await axios.post(`http://localhost:5000/preview_legal_form`, {
                species_name: prediction.result,
                vietnamese_name: prediction.vietnamese_name,
                legal_group: prediction.legal.legal_group,
                ...formData // các thông tin cá nhân từ form
            });

            if (res.data.status === "success") {
                // Logic tải file hoặc hiển thị preview
                alert("Tạo đơn thành công!");
                // Bạn có thể mở window.open để tải file nếu có endpoint download
            }
        } catch (error) {
            alert("Lỗi khi tạo văn bản pháp lý");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Modal show={show} onHide={onHide} size="xl" fullscreen="lg-down" centered>
            <Modal.Header closeButton>
                <Modal.Title>Điền thông tin và xem trước đơn</Modal.Title>
            </Modal.Header>
            <Modal.Body className="p-0">
                <Row g={0} style={{ minHeight: '80vh' }}>
                    {/* ===== FORM INPUTS ===== */}
                    <Col lg={4} className="p-4 border-end">
                        <h4 className='mb-4'>Thông tin người bàn giao</h4>
                        <Form>
                            <Form.Group className="mb-3">
                                <Form.Label>Họ và Tên</Form.Label>
                                <Form.Control type="text" name="fullName" value={formData.fullName} onChange={handleInputChange} />
                            </Form.Group>
                            <Row>
                                <Col md={7}>
                                    <Form.Group className="mb-3">
                                        <Form.Label>Số CCCD/CMND</Form.Label>
                                        <Form.Control type="text" name="idCard" value={formData.idCard} onChange={handleInputChange} />
                                    </Form.Group>
                                </Col>
                                <Col md={5}>
                                    <Form.Group className="mb-3">
                                        <Form.Label>Ngày cấp</Form.Label>
                                        <Form.Control type="date" name="idCardDate" value={formData.idCardDate} onChange={handleInputChange} />
                                    </Form.Group>
                                </Col>
                            </Row>
                            <Form.Group className="mb-3">
                                <Form.Label>Nơi cấp</Form.Label>
                                <Form.Control type="text" name="idCardPlace" value={formData.idCardPlace} onChange={handleInputChange} />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Địa chỉ thường trú</Form.Label>
                                <Form.Control as="textarea" rows={2} name="address" value={formData.address} onChange={handleInputChange} />
                            </Form.Group>
                             <Form.Group className="mb-3">
                                <Form.Label>Số điện thoại</Form.Label>
                                <Form.Control type="tel" name="phone" value={formData.phone} onChange={handleInputChange} />
                            </Form.Group>
                            <hr className='my-4'/>
                            <h4 className='mb-3'>Thông tin đơn</h4>
                             <Form.Group className="mb-3">
                                <Form.Label>Nơi làm đơn (Tỉnh/Thành phố)</Form.Label>
                                <Form.Control type="text" name="city" value={formData.city} onChange={handleInputChange} />
                            </Form.Group>
                             <Form.Group className="mb-3">
                                <Form.Label>Gửi đến Chi cục Kiểm lâm (Tỉnh/Thành phố)</Form.Label>
                                <Form.Control type="text" name="department" value={formData.department} onChange={handleInputChange} />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Tình trạng sức khỏe cá thể</Form.Label>
                                <Form.Control type="text" name="healthStatus" value={formData.healthStatus} onChange={handleInputChange} />
                            </Form.Group>
                        </Form>
                    </Col>

                    {/* ===== DOCUMENT PREVIEW ===== */}
                    <Col lg={8}>
                        <DocumentPreview animalData={prediction} formData={formData} />
                    </Col>
                </Row>
            </Modal.Body>
            <Modal.Footer className="border-top-0 pt-0 p-3">
                <Button variant="secondary" onClick={onHide}>
                    Hủy
                </Button>
                <Button variant="success" onClick={handleSubmit} disabled={loading}>
                    {loading ? <Spinner as="span" animation="border" size="sm" /> : <i className="fas fa-download me-2"></i>}
                    {loading ? 'Đang tạo...' : 'Tạo và Tải xuống'}
                </Button>
            </Modal.Footer>
        </Modal>
    );
};

export default FormGenerationModal;