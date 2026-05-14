import React from 'react';
import './DocumentPreview.css';

interface DocumentPreviewProps {
    animalData: any;
    formData: any;
}

const DocumentPreview: React.FC<DocumentPreviewProps> = ({ animalData, formData }) => {
    const today = new Date();
    const {
        city = '...',
        fullName = '...',
        idCard = '...',
        idCardDate = '...',
        idCardPlace = '...',
        address = '...',
        phone = '...',
        department = '...',
        healthStatus = '...'
    } = formData;

    return (
        <div className="document-preview-a4-wrapper">
            <div className="document-preview-a4">
                <div className="text-center">
                    <p className="font-bold">CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</p>
                    <p className="font-bold">Độc lập - Tự do - Hạnh phúc</p>
                    <p>--------------------</p>
                </div>

                <p className="text-right" style={{ marginTop: '2rem' }}>
                    {city || '...'}, ngày {today.getDate()} tháng {today.getMonth() + 1} năm {today.getFullYear()}
                </p>

                <h2 className="text-center font-bold" style={{ marginTop: '2rem', fontSize: '16pt' }}>
                    ĐƠN TỰ NGUYỆN BÀN GIAO ĐỘNG VẬT HOANG DÃ
                </h2>

                <p style={{ marginTop: '2rem' }}>
                    <span className="font-bold">Kính gửi:</span> Chi cục Kiểm lâm tỉnh/thành phố {department || '...'}
                </p>

                <p style={{ marginTop: '1rem' }}>
                    Tôi tên là: <span className="font-bold">{fullName || '...'}</span>
                </p>
                <p>
                    Số CCCD/CMND: {idCard || '...'} Ngày cấp: {idCardDate || '...'} Nơi cấp: {idCardPlace || '...'}
                </p>
                <p>Địa chỉ thường trú: {address || '...'}</p>
                <p>Số điện thoại liên lạc: {phone || '...'}</p>

                <p style={{ marginTop: '1rem' }}>
                    Nay tôi làm đơn này tự nguyện bàn giao cho quý cơ quan cá thể động vật sau:
                </p>

                <table className="preview-table">
                    <thead>
                        <tr>
                            <th>Tên loài</th>
                            <th>Tên khoa học</th>
                            <th>Phân loại pháp lý</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{animalData?.result}</td>
                            <td><em>{animalData?.raw_name.replace(/_/g, ' ')}</em></td>
                            <td>{animalData?.legal?.legal_group}</td>
                        </tr>
                    </tbody>
                </table>

                <p>
                    Tình trạng sức khỏe hiện tại: {healthStatus || '...'}
                </p>
                <p style={{ marginTop: '1rem' }}>
                    Lý do bàn giao: Tôi nhận thức được đây là loài động vật nguy cấp quý hiếm cần được bảo tồn và tự nguyện bàn giao cho cơ quan chức năng để cứu hộ, tái thả về tự nhiên.
                </p>
                <p style={{ marginTop: '1rem' }}>
                    Tôi cam đoan thông tin trên là đúng sự thật và hoàn toàn chịu trách nhiệm trước pháp luật.
                </p>

                <div className="text-right" style={{ marginTop: '4rem' }}>
                    <p className="font-bold">NGƯỜI LÀM ĐƠN</p>
                    <p>(Ký, ghi rõ họ tên)</p>
                    <div style={{ height: '80px' }}></div> {/* Spacer for signature */}
                    <p className="font-bold">{fullName || '...'}</p>
                </div>
            </div>
        </div>
    );
};

export default DocumentPreview;