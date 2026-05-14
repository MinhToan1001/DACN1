import React, { useState } from 'react';
import { Container, Row, Col, Card, Form, Button, Spinner, Alert } from 'react-bootstrap';
import axios from 'axios';
import Layout from '../../layouts/Layout';
import './OCRExtraction.css';

const OCRExtraction = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [extractedText, setExtractedText] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setExtractedText('');
            setError(null);
        }
    };

    const handleExtract = async () => {
        if (!selectedFile) return;
        
        setLoading(true);
        setError(null);
        setExtractedText('');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await axios.post('/ai/api/ocr', formData);
            setExtractedText(res.data.extracted_text);
        } catch (err: any) {
            console.error(err);
            const errorMessage = err.response?.data?.error || 'Lỗi trong quá trình trích xuất.';
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Layout>
            <div className="ocr-extraction-page">
                <Container className="py-5 mt-5">
                    {/* TITLE */}
                    <div className="text-center mb-5">
                        <h1 className="fw-bold text-success display-5">
                            TRÍCH XUẤT DỮ LIỆU VĂN BẢN
                        </h1>
                        <p className="text-muted">
                            Tải lên hình ảnh của một tài liệu để tự động nhận dạng và trích xuất nội dung.
                        </p>
                    </div>

                    {/* UPLOAD & RESULT */}
                    <Row className="justify-content-center">
                        <Col lg={10} xl={10}>
                            <Card className="shadow-sm border-0">
                                <Card.Body className="p-4">
                                    <Row>
                                        <Col md={5}>
                                            <h5 className="fw-bold mb-3">1. Tải ảnh lên</h5>
                                            <div className="upload-rect-ocr">
                                                <Form.Control
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={handleFileChange}
                                                    className="position-absolute opacity-0 top-0 start-0 w-100 h-100"
                                                    style={{ zIndex: 10, cursor: 'pointer' }}
                                                />
                                                {!previewUrl ? (
                                                    <div className="upload-prompt text-center">
                                                        <i className="fas fa-file-image fa-2x text-success mb-3"></i>
                                                        <h6 className="fw-bold">Chọn hoặc kéo thả ảnh</h6>
                                                        <p className="small text-muted">Hỗ trợ các định dạng JPG, PNG, WEBP...</p>
                                                    </div>
                                                ) : (
                                                    <img src={previewUrl} alt="Preview" className="upload-preview-ocr" />
                                                )}
                                            </div>
                                            <div className="d-grid mt-3">
                                                <Button variant="success" onClick={handleExtract} disabled={!selectedFile || loading}>
                                                    {loading ? <Spinner size="sm" /> : <i className="fas fa-magic me-2"></i>}
                                                    {loading ? 'Đang xử lý...' : '2. Bắt đầu trích xuất'}
                                                </Button>
                                            </div>
                                        </Col>
                                        <Col md={7}>
                                             <h5 className="fw-bold mb-3">3. Kết quả</h5>
                                             <div className='result-wrapper'>
                                                {error && <Alert variant="danger">{error}</Alert>}
                                                <Form.Control
                                                    as="textarea"
                                                    rows={15}
                                                    readOnly={loading}
                                                    value={loading ? 'Đang chờ kết quả từ AI...' : extractedText}
                                                    placeholder="Nội dung văn bản được trích xuất sẽ xuất hiện ở đây..."
                                                    className="result-textarea"
                                                />
                                             </div>
                                        </Col>
                                    </Row>
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                </Container>
            </div>
        </Layout>
    );
};

export default OCRExtraction;
