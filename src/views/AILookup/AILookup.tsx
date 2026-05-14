import React, { useState, useRef, useEffect, useCallback } from 'react';
import Layout from '../../layouts/Layout';
import { Container, Row, Col, Card, Button, Spinner, Modal } from 'react-bootstrap';
import axios from 'axios';
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet';
import './AILookup.css';

/* ================================================================
   TYPES
   ================================================================ */
interface ChatMessage {
    role: 'user' | 'bot';
    text: string;
}

interface Question {
    id: string;
    label: string;
    question: string;
    detail?: string;
    icon?: string;
    cf_yes: number;
    cf_no: number;
    cf_unknown?: number;
}

interface InferenceTraceItem {
    message?: string;
    phase?: string;
    level?: 'info' | 'success' | 'warning' | 'danger';
}

interface PenaltyFrame {
    id: string;
    title: string;
    category: string;
    items: string[];
}

interface SpeciesArea {
    name: string;
    lat: number;
    lng: number;
    type: string;
    matched_keyword?: string;
}

interface SpeciesDistribution {
    species: string;
    has_distribution: boolean;
    distribution_text: string;
    habitat_text: string;
    note: string;
    areas: SpeciesArea[];
}

interface LegalScenario {
    status: 'OBSERVE' | 'CAPTIVITY' | 'ORDINARY_OBSERVE' | 'ORDINARY_CAPTIVITY';
    legal_group?: string;
    group_name?: string;
    quantity?: number;
    message?: string;
    selected_frame?: PenaltyFrame | null;
    all_frames?: PenaltyFrame[];
    inference_trace?: Array<string | InferenceTraceItem>;
}

interface PredictData {
    status: 'SUCCESS' | 'ASKING' | 'REJECTED' | 'UNCERTAIN';
    species?: string;
    vietnamese_name?: string;
    confidence: number;
    raw_confidence?: number;
    message?: string;
    is_final?: boolean;
    result?: string;
    description?: string;
    biology?: Record<string, any>;
    legal?: Record<string, any>;
    inferred_legal_group?: string;
    questions?: Question[];
    inference_trace?: Array<string | InferenceTraceItem>;
    has_map?: boolean;
    raw_name?: string;
}

type AnswerValue = boolean | 'unknown';

/* ================================================================
   HELPERS
   ================================================================ */
function cfColor(pct: number): string {
    if (pct >= 90) return '#1e8050';
    if (pct >= 60) return '#f0a500';
    if (pct >= 31) return '#1565c0';
    return '#dc3545';
}

function mycin(cfOld: number, cfE: number): number {
    let cf1 = cfOld * 2 - 1, cf2 = cfE, cfN: number;
    if (cf1 >= 0 && cf2 >= 0)          cfN = cf1 + cf2 * (1 - cf1);
    else if (cf1 < 0 && cf2 < 0)       cfN = cf1 + cf2 * (1 + cf1);
    else                                cfN = (cf1 + cf2) / (1 - Math.min(Math.abs(cf1), Math.abs(cf2)));
    return Math.max(0.01, Math.min(0.99, +((cfN + 1) / 2).toFixed(4)));
}

function getField(obj: Record<string, any>, ...keys: string[]): string {
    for (const k of keys) {
        const v = obj[k];
        if (v && typeof v === 'string' && v.trim() && !v.includes('Chưa có')) return v.trim();
    }
    return '';
}

/* ================================================================
   SUB-COMPONENTS
   ================================================================ */

/* ── Confidence Bar ── */
const CfBar: React.FC<{ pct: number; light?: boolean }> = ({ pct, light = false }) => {
    const bg   = light ? 'rgba(255,255,255,.2)' : '#dde8f5';
    const fill = cfColor(pct);
    const lblColor = light ? 'rgba(255,255,255,.75)' : '#5a7a66';
    return (
        <>
            <div className="cf-bar-wrap" style={{ background: bg }}>
                <div className="cf-bar-fill" style={{ width: `${Math.round(pct)}%`, background: fill }} />
            </div>
            <div className="cf-label" style={{ color: lblColor }}>
                {Math.round(pct)}% confidence
            </div>
        </>
    );
};

/* ── Info Block ── */
const InfoBlock: React.FC<{ icon: string; label: string; value: string; full?: boolean }> = ({ icon, label, value, full }) => (
    <div className={`info-block ${full ? 'full' : ''}`}>
        <div className="lbl">{icon} {label}</div>
        <div className={`val ${value ? '' : 'empty'}`}>{value || 'Chưa có dữ liệu trong hệ thống'}</div>
    </div>
);

const normalizeTraceText = (step: string | InferenceTraceItem): string => (
    typeof step === 'string' ? step : (step.message || JSON.stringify(step))
);

const normalizeTracePhase = (step: string | InferenceTraceItem): string => (
    typeof step === 'string' ? 'Log' : (step.phase || 'Log')
);

const normalizeTraceLevel = (step: string | InferenceTraceItem): string => (
    typeof step === 'string' ? 'info' : (step.level || 'info')
);

const InferenceTracePanel: React.FC<{ steps?: Array<string | InferenceTraceItem>; title?: string }> = ({ steps = [], title = 'Quá trình suy diễn' }) => {
    if (!steps.length) return null;
    return (
        <div className="trace-panel">
            <div className="trace-title">
                <i className="fas fa-project-diagram"></i>{title}
            </div>
            <div className="trace-list">
                {steps.map((step, idx) => (
                    <div key={`${idx}-${normalizeTraceText(step)}`} className={`trace-step ${normalizeTraceLevel(step)}`}>
                        <span className="trace-index">{idx + 1}</span>
                        <span className="trace-phase">{normalizeTracePhase(step)}</span>
                        <span className="trace-message">{normalizeTraceText(step)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const PenaltyFrameCard: React.FC<{ frame: PenaltyFrame; selected?: boolean }> = ({ frame, selected = false }) => (
    <div className={`penalty-frame ${selected ? 'selected' : ''}`}>
        <div className="penalty-frame-head">
            <span>{frame.category}</span>
            <strong>{frame.title}</strong>
        </div>
        <ul>
            {(frame.items || []).map((item, idx) => <li key={idx}>{item}</li>)}
        </ul>
    </div>
);

/* ── Result SUCCESS ── */
const MapResize: React.FC = () => {
    const map = useMap();
    useEffect(() => {
        const timer = window.setTimeout(() => map.invalidateSize(), 250);
        return () => window.clearTimeout(timer);
    }, [map]);
    return null;
};

const SpeciesDistributionMap: React.FC<{ distribution: SpeciesDistribution | null; loading: boolean }> = ({ distribution, loading }) => {
    const areas = distribution?.areas || [];
    if (loading) {
        return (
            <div className="species-map-section">
                <h6><i className="fas fa-map-location-dot me-2"></i>PHÂN BỐ & KHU VỰC CÓ THỂ BẮT GẶP</h6>
                <div className="species-map-loading">Đang tải dữ liệu phân bố...</div>
            </div>
        );
    }
    if (!distribution) return null;

    return (
        <div className="species-map-section">
            <h6><i className="fas fa-map-location-dot me-2"></i>PHÂN BỐ & KHU VỰC CÓ THỂ BẮT GẶP</h6>
            <div className="species-map-text">
                <strong>Phân bố:</strong> {distribution.distribution_text}
                <br />
                <strong>Sinh cảnh:</strong> {distribution.habitat_text}
            </div>
            {areas.length > 0 ? (
                <div className="species-map-box">
                    <MapContainer center={[16.0471, 108.2068]} zoom={5} scrollWheelZoom={false} style={{ height: '100%', width: '100%' }}>
                        <MapResize />
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />
                        {areas.map((area) => (
                            <CircleMarker
                                key={`${area.name}-${area.lat}-${area.lng}`}
                                center={[area.lat, area.lng]}
                                radius={9}
                                pathOptions={{ color: '#145c36', fillColor: '#1e8050', fillOpacity: 0.72, weight: 2 }}
                            >
                                <Popup>
                                    <strong>{area.name}</strong>
                                    <br />
                                    {area.type}
                                </Popup>
                            </CircleMarker>
                        ))}
                    </MapContainer>
                </div>
            ) : (
                <div className="species-map-empty">Chưa suy diễn được địa điểm cụ thể từ dữ liệu phân bố hiện có.</div>
            )}
            <div className="species-map-note">{distribution.note}</div>
        </div>
    );
};

const ResultSuccess: React.FC<{
    data: PredictData;
    imageUrl: string | null;
    onPreview: () => void;
    onGenerate: () => void;
    formLoading: boolean;
    legalScenario: LegalScenario | null;
    legalLoading: boolean;
    legalMode: 'observe' | 'captivity' | '';
    quantity: number;
    distribution: SpeciesDistribution | null;
    distributionLoading: boolean;
    onLegalModeChange: (mode: 'observe' | 'captivity') => void;
    onQuantityChange: (quantity: number) => void;
    onInferLegal: () => void;
}> = ({ data, imageUrl, onPreview, onGenerate, formLoading, legalScenario, legalLoading, legalMode, quantity, distribution, distributionLoading, onLegalModeChange, onQuantityChange, onInferLegal }) => {
    const bio   = data.biology || {};
    const feats = (bio as any).dac_diem_nhan_dang || bio;

    const ngoaiHinh = getField(feats, 'mo_ta_ngoai_hinh', 'ngoai_hinh', 'appearance', 'morphology', 'mo_ta');
    const thucAn    = getField(feats, 'thuc_an', 'diet', 'food');
    const tapTinh   = getField(feats, 'tap_tinh', 'behavior', 'behaviour');
    const sinhThai  = getField(feats, 'sinh_thai', 'habitat', 'ecology', 'moi_truong');
    const phanBo    = getField(feats, 'phan_bo_viet_nam', 'phan_bo', 'distribution', 'range');
    const dacDiem   = getField(feats, 'dac_diem_phan_biet', 'distinguishing_features', 'key_features', 'dac_trung');

    const legal      = data.legal || {};
    const legalGroup = data.inferred_legal_group
        || (legal as any).nhom_phap_ly || (legal as any).legal_group
        || (legal as any).legal_advice?.group_name
        || 'Không thuộc danh mục bảo tồn';

    const lgCls  = /IA|IB/.test(legalGroup)  ? 'lg-ib'  : /IIA|IIB|II/.test(legalGroup) ? 'lg-iia' : 'lg-none';
    const lgIcon = /IA|IB/.test(legalGroup)  ? 'fa-shield-alt' : /II/.test(legalGroup)   ? 'fa-exclamation-circle' : 'fa-check-circle';

    const showForm    = /I/.test(legalGroup);
    const selectedFrame = legalScenario?.selected_frame || null;
    const allFrames = legalScenario?.all_frames || [];
    const penEntries: [string, any][] = [];
    const vn_name     = data.vietnamese_name || 'Chưa rõ';

    return (
        <div className="result-card mb-4">
            {/* HERO */}
            <div className="result-hero">
                {imageUrl && <img src={imageUrl} alt="specimen" />}
                <div className="hero-info">
                    <div className="d-flex align-items-center gap-2 flex-wrap mb-1">
                        <span className="sbadge sb-success">
                            <i className="fas fa-check-circle"></i> XÁC NHẬN THÀNH CÔNG
                        </span>
                        <span className="sbadge sb-cf">{Math.round(data.confidence)}%</span>
                    </div>
                    <h2>{vn_name}</h2>
                    <div className="sci">{data.species}</div>
                    <CfBar pct={data.confidence} light />
                    <div style={{ fontSize: '.78rem', opacity: .75, marginTop: 5 }}>{data.message}</div>
                </div>
            </div>

            {/* NHÓM PHÁP LÝ */}
            <div className="legal-section" style={{ paddingTop: 22 }}>
                <h6><i className="fas fa-gavel me-2"></i>NHÓM PHÁP LÝ</h6>
                <div className={`legal-group-box ${lgCls}`}>
                    <i className={`fas ${lgIcon}`}></i>
                    {legalGroup}
                </div>
            </div>

            {/* ĐẶC ĐIỂM SINH HỌC */}
            <div className="info-section" style={{ paddingTop: 6 }}>
                <h6><i className="fas fa-leaf me-2"></i>ĐẶC ĐIỂM SINH HỌC & SINH THÁI</h6>
                <div className="info-grid">
                    <InfoBlock icon="🔍" label="Ngoại hình"              value={ngoaiHinh} />
                    <InfoBlock icon="🍃" label="Thức ăn"                  value={thucAn} />
                    <InfoBlock icon="🐾" label="Tập tính"                 value={tapTinh} />
                    <InfoBlock icon="🌿" label="Sinh thái"                value={sinhThai} />
                    <InfoBlock icon="🌍" label="Phân bố tại Việt Nam"     value={phanBo} full />
                    {dacDiem && <InfoBlock icon="⚡" label="Đặc điểm nhận dạng đặc trưng" value={dacDiem} full />}
                </div>
            </div>

            <SpeciesDistributionMap distribution={distribution} loading={distributionLoading} />

            {/* CHẾ TÀI */}
            <div className="legal-section">
                <div className="penalty-title">
                    <i className="fas fa-exclamation-triangle"></i>CẢNH BÁO HÀNH VI & CHẾ TÀI
                </div>
                <div className="legal-inquiry">
                    <div className="legal-question">Bạn chỉ quan sát loài này hay đang nuôi/giữ cá thể?</div>
                    <div className="legal-mode-grid">
                        <button className={legalMode === 'observe' ? 'active' : ''} onClick={() => onLegalModeChange('observe')}>
                            <i className="fas fa-eye"></i> Chỉ quan sát
                        </button>
                        <button className={legalMode === 'captivity' ? 'active danger' : ''} onClick={() => onLegalModeChange('captivity')}>
                            <i className="fas fa-box"></i> Đang nuôi/nhốt
                        </button>
                    </div>
                    {legalMode === 'captivity' && (
                        <div className="quantity-row">
                            <label>Số lượng cá thể</label>
                            <input type="number" min={1} value={quantity} onChange={(e) => onQuantityChange(Math.max(1, Number(e.target.value) || 1))} />
                        </div>
                    )}
                    <button className="btn-infer-legal" onClick={onInferLegal} disabled={!legalMode || legalLoading}>
                        {legalLoading
                            ? <><span className="spinner-border spinner-border-sm me-2"></span>Đang suy diễn...</>
                            : <><i className="fas fa-gavel me-2"></i>Suy diễn khung pháp lý</>
                        }
                    </button>
                </div>

                {legalScenario && (
                    <div className="legal-scenario-result">
                        <div className="scenario-message">{legalScenario.message}</div>
                        {selectedFrame && <PenaltyFrameCard frame={selectedFrame} selected />}
                        {allFrames.length > 0 && (
                            <div className="penalty-frame-list">
                                <div className="penalty-subtitle">
                                    {legalScenario.status === 'OBSERVE' ? 'Toàn bộ khung pháp lý tham khảo' : 'Các khung khác để đối chiếu'}
                                </div>
                                {allFrames
                                    .filter(frame => !selectedFrame || frame.id !== selectedFrame.id)
                                    .map(frame => <PenaltyFrameCard key={`${frame.category}-${frame.id}`} frame={frame} />)}
                            </div>
                        )}
                        <InferenceTracePanel steps={legalScenario.inference_trace} title="Log suy diễn pháp lý" />
                    </div>
                )}

                <div className="penalty-wrap legacy-penalty-wrap">
                    {penEntries.length === 0 ? (
                        <div className="penalty-item">
                            <p style={{ color: '#666', fontStyle: 'italic' }}>Không có dữ liệu mức phạt cụ thể.</p>
                        </div>
                    ) : penEntries.map(([k, v]) => {
                        const txt = Array.isArray(v) ? (v as string[]).join('\n• ') : String(v);
                        return (
                            <div className="penalty-item" key={k}>
                                <strong>{k.replace(/_/g, ' ')}</strong>
                                <p>• {txt}</p>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* NÚT TẠO ĐƠN */}
            <InferenceTracePanel steps={data.inference_trace} />

            {showForm && (
                <div className="form-actions">
                    <button className="btn-outline-green" onClick={onPreview} disabled={formLoading}>
                        {formLoading
                            ? <><span className="spinner-border spinner-border-sm me-2"></span>Đang tạo...</>
                            : <><i className="fas fa-eye me-2"></i>XEM TRƯỚC MẪU ĐƠN</>
                        }
                    </button>
                    <button className="btn-solid-green" onClick={onGenerate} disabled={formLoading}>
                        <i className="fas fa-file-signature me-2"></i>TẠO ĐƠN XIN PHÉP / BÀN GIAO
                    </button>
                </div>
            )}
        </div>
    );
};

/* ── Result ASKING ── */
const ResultAsking: React.FC<{
    data: PredictData;
    answered: Record<string, AnswerValue>;
    allQuestions: Question[];
    onAnswer: (id: string, answer: AnswerValue) => void;
    loading: boolean;
}> = ({ data, answered, allQuestions, onAnswer, loading }) => {
    const conf     = data.confidence ?? 0;
    const vn_name  = data.vietnamese_name || 'Chưa rõ';
    const questions: Question[] = data.questions || allQuestions;

    const dots = questions.map(q => {
        const cls = !(q.id in answered) ? 'cur' : answered[q.id] === 'unknown' ? 'unknown' : answered[q.id] ? 'yes' : 'no';
        return <div key={q.id} className={`qdot ${cls}`} />;
    });

    return (
        <div className="qa-card mb-4">
            <div className="qa-header">
                <div className="d-flex align-items-start justify-content-between gap-3 flex-wrap">
                    <div style={{ flex: 1 }}>
                        <h5><i className="fas fa-question-circle me-2"></i>Xác nhận đặc điểm sinh học</h5>
                        <p>{data.message || ''}</p>
                        <div className="qa-species">
                            <strong>{vn_name}</strong>
                            &nbsp;·&nbsp;
                            <em>{data.species}</em>
                        </div>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                        <span className="sbadge sb-cf" style={{ fontSize: '.8rem' }}>{Math.round(conf)}%</span>
                        <CfBar pct={conf} light />
                    </div>
                </div>
            </div>

            {/* Progress dots */}
            <div className="qdots">{dots}</div>

            {/* CF live bar */}
            <div className="cf-live-bar" style={{ margin: '0 26px 8px' }}>
                <div className="cf-live-fill" style={{ width: `${Math.round(conf)}%`, background: cfColor(conf) }} />
            </div>

            {/* Câu hỏi */}
            {questions.map(q => {
                const done   = q.id in answered;
                const ansYes = answered[q.id] === true;
                const ansNo  = answered[q.id] === false;
                const ansUnknown = answered[q.id] === 'unknown';
                return (
                    <div key={q.id} className={`q-item ${done ? 'done' : ''}`}>
                        <div className="q-lbl">{q.icon || '❓'} {q.label}</div>
                        <div className="q-text">{q.question || 'Mẫu vật có đặc điểm này không?'}</div>
                        <div className="q-detail">
                            {q.detail || <em style={{ color: '#aaa' }}>Không có mô tả chi tiết</em>}
                        </div>
                        <div className="q-btns">
                            <button
                                className={`btn-yes ${ansYes ? 'active' : ''}`}
                                onClick={() => !done && !loading && onAnswer(q.id, true)}
                                disabled={done || loading}
                            >
                                <i className="fas fa-check"></i> CÓ, có đặc điểm này
                            </button>
                            <button
                                className={`btn-no ${ansNo ? 'active' : ''}`}
                                onClick={() => !done && !loading && onAnswer(q.id, false)}
                                disabled={done || loading}
                            >
                                <i className="fas fa-times"></i> KHÔNG có
                            </button>
                        </div>
                        <div className="q-unknown-row">
                            <button
                                className={`btn-unknown ${ansUnknown ? 'active' : ''}`}
                                onClick={() => !done && !loading && onAnswer(q.id, 'unknown')}
                                disabled={done || loading}
                            >
                                <i className="fas fa-question"></i> Không biết
                            </button>
                        </div>
                        <div className="cf-hint">
                            <span className="plus">▲ Nếu CÓ: +{Math.round(q.cf_yes * 100)}% tin cậy</span>
                            &nbsp;|&nbsp;
                            <span className="minus">▼ Nếu KHÔNG: {Math.round(q.cf_no * 100)}%</span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

/* ── Result REJECTED ── */
const ResultRejected: React.FC<{ data: PredictData }> = ({ data }) => {
    const conf = data.confidence || 0;
    return (
        <div className="rejected-card">
            <div className="rejected-icon"><i className="fas fa-ban"></i></div>
            <div className="rejected-title">Không thể nhận dạng</div>
            <div className="rejected-msg">{data.message || 'Loài này chưa có trong dữ liệu hệ thống.'}</div>
            {conf > 0 && (
                <div style={{ marginTop: 16, maxWidth: 320, marginLeft: 'auto', marginRight: 'auto' }}>
                    <div style={{ fontSize: '.72rem', color: '#aaa', marginBottom: 4, fontFamily: 'JetBrains Mono, monospace' }}>
                        Độ tin cậy cuối: {Math.round(conf)}%
                    </div>
                    <div style={{ background: '#f5dada', borderRadius: 99, height: 8, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${Math.round(conf)}%`, background: '#dc3545', borderRadius: 99 }} />
                    </div>
                </div>
            )}
        </div>
    );
};

const ResultUncertain: React.FC<{ data: PredictData }> = ({ data }) => {
    const conf = data.confidence || 0;
    return (
        <div className="rejected-card uncertain-card">
            <div className="rejected-icon"><i className="fas fa-search"></i></div>
                    <div className="rejected-title">Chưa thể kết luận chắc chắn</div>
            <div className="rejected-msg">{data.message}</div>
            <div style={{ marginTop: 16, maxWidth: 320, marginLeft: 'auto', marginRight: 'auto' }}>
                <div style={{ fontSize: '.72rem', color: '#777', marginBottom: 4, fontFamily: 'JetBrains Mono, monospace' }}>
                    Độ tin cậy hiện tại: {Math.round(conf)}%
                </div>
                <div style={{ background: '#f9e7bd', borderRadius: 99, height: 8, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.round(conf)}%`, background: '#f0a500', borderRadius: 99 }} />
                </div>
            </div>
            <InferenceTracePanel steps={data.inference_trace} />
        </div>
    );
};

/* ================================================================
   MAIN COMPONENT
   ================================================================ */
const AILookup = () => {
    /* ── File / Preview ── */
    const [selectedFile, setSelectedFile]   = useState<File | null>(null);
    const [previewUrl,   setPreviewUrl]     = useState<string | null>(null);

    /* ── Kết quả ── */
    const [currentData, setCurrentData]     = useState<PredictData | null>(null);

    /* ── Hệ chuyên gia state ── */
    const [allQuestions, setAllQuestions]   = useState<Question[]>([]);
    const [answered,     setAnswered]       = useState<Record<string, AnswerValue>>({});
    const [currentConf,  setCurrentConf]    = useState(0);

    /* ── Loading ── */
    const [loading,     setLoading]         = useState(false);
    const [formLoading, setFormLoading]     = useState(false);
    const [legalLoading, setLegalLoading]   = useState(false);
    const [legalMode, setLegalMode]         = useState<'observe' | 'captivity' | ''>('');
    const [quantity, setQuantity]           = useState(1);
    const [legalScenario, setLegalScenario] = useState<LegalScenario | null>(null);
    const [speciesDistribution, setSpeciesDistribution] = useState<SpeciesDistribution | null>(null);
    const [distributionLoading, setDistributionLoading] = useState(false);

    /* ── Preview Modal ── */
    const [showModal,     setShowModal]     = useState(false);
    const [previewContent, setPreviewContent] = useState('');

    /* ── Chatbot ── */
    const [isChatOpen,  setIsChatOpen]      = useState(false);
    const [messages,    setMessages]        = useState<ChatMessage[]>([
        { role: 'bot', text: 'Chào bạn! Tôi là Trợ lý AI Kiểm lâm. Tôi có thể giúp gì cho bạn về động vật hoang dã hoặc thủ tục pháp lý hôm nay?' }
    ]);
    const [userInput,   setUserInput]       = useState('');
    const [chatLoading, setChatLoading]     = useState(false);
    const chatEndRef    = useRef<HTMLDivElement>(null);
    const chatWindowRef = useRef<HTMLDivElement>(null);
    const chatInputRef  = useRef<HTMLInputElement>(null);

    /* ── Auto scroll chat ── */
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, chatLoading]);

    /* ── Đóng chat khi click ngoài ── */
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (
                isChatOpen &&
                chatWindowRef.current &&
                !chatWindowRef.current.contains(e.target as Node) &&
                !(e.target as HTMLElement).closest('.chat-toggle-btn')
            ) setIsChatOpen(false);
        };
        if (isChatOpen) document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isChatOpen]);

    useEffect(() => {
        const species = currentData?.species || currentData?.result;
        if (currentData?.status !== 'SUCCESS' || !species) {
            setSpeciesDistribution(null);
            return;
        }

        setDistributionLoading(true);
        axios.get<SpeciesDistribution>('http://localhost:5000/species_distribution', { params: { species } })
            .then(res => setSpeciesDistribution(res.data))
            .catch(err => {
                console.error('[SPECIES DISTRIBUTION ERROR]', err);
                setSpeciesDistribution(null);
            })
            .finally(() => setDistributionLoading(false));
    }, [currentData?.status, currentData?.species, currentData?.result]);

    /* ================================================================
       FILE CHANGE
       ================================================================ */
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setCurrentData(null);
            setAllQuestions([]);
            setAnswered({});
            setCurrentConf(0);
            setLegalMode('');
            setQuantity(1);
            setLegalScenario(null);
            setSpeciesDistribution(null);
        }
    };

    /* ================================================================
       PREDICT
       ================================================================ */
    const handleUpload = async () => {
        if (!selectedFile) return;
        setLoading(true);
        setCurrentData(null);
        setAllQuestions([]);
        setAnswered({});
        setLegalMode('');
        setQuantity(1);
        setLegalScenario(null);
        setSpeciesDistribution(null);

        const formData = new FormData();
        formData.append('file', selectedFile);
        try {
            const res  = await axios.post<PredictData>('http://localhost:5000/predict', formData);
            const data = res.data;
            console.log('[PREDICT]', data);

            // Chuẩn hoá confidence (backend có thể trả 0-1 hoặc 0-100)
            if (data.confidence <= 1) data.confidence = data.confidence * 100;

            const rawConf = data.raw_confidence ?? (data.confidence / 100);
            setCurrentConf(rawConf);
            setAllQuestions(data.questions || []);
            setCurrentData(data);
        } catch (err) {
            console.error('[PREDICT ERROR]', err);
            alert('Không thể kết nối đến Server AI (Port 5000). Vui lòng kiểm tra Flask đang chạy.');
        } finally {
            setLoading(false);
        }
    };

    /* ================================================================
       ANSWER QUESTION
       ================================================================ */
    const handleAnswer = useCallback(async (questionId: string, answer: AnswerValue) => {
        if (!currentData) return;
        const q = allQuestions.find(q => q.id === questionId);
        if (!q) return;

        // Cập nhật local ngay
        const newAnswered = { ...answered, [questionId]: answer };
        setAnswered(newAnswered);

        // Client-side MYCIN optimistic update
        const evidence   = answer === 'unknown' ? (q.cf_unknown || 0) : answer ? q.cf_yes : q.cf_no;
        const newConf    = mycin(currentConf, evidence);
        setCurrentConf(newConf);

        const species = currentData.species || currentData.result;
        setLoading(true);
        try {
            const res  = await axios.post<PredictData>('http://localhost:5000/answer_question', {
                species,
                current_confidence: currentConf,
                answered: newAnswered,
            });
            const data = res.data;
            console.log('[ANSWER]', data);

            if (data.confidence <= 1) data.confidence = data.confidence * 100;

            // Merge câu hỏi
            if (data.status === 'ASKING' && data.questions) {
                const merged = [...allQuestions];
                for (const nq of data.questions) {
                    if (!merged.find(q => q.id === nq.id)) merged.push(nq);
                }
                data.questions = merged;
                setAllQuestions(merged);
            }

            // Bổ sung tên từ state nếu server không trả về
            if (!data.vietnamese_name) data.vietnamese_name = currentData.vietnamese_name;
            if (!data.species)         data.species         = currentData.species;

            if (data.confidence) setCurrentConf(data.confidence / 100);
            setCurrentData(data);
        } catch (err) {
            console.error('[ANSWER ERROR]', err);
            alert('Lỗi khi gửi câu trả lời. Vui lòng thử lại.');
        } finally {
            setLoading(false);
        }
    }, [currentData, allQuestions, answered, currentConf]);

    /* ================================================================
       LEGAL FORM HELPERS
       ================================================================ */
    const handleLegalModeChange = (mode: 'observe' | 'captivity') => {
        setLegalMode(mode);
        setLegalScenario(null);
        if (mode === 'observe') setQuantity(0);
        if (mode === 'captivity' && quantity < 1) setQuantity(1);
    };

    const handleInferLegal = async () => {
        if (!currentData || !legalMode) return;
        setLegalLoading(true);
        try {
            const species = currentData.species || currentData.result;
            const res = await axios.post<LegalScenario>('http://localhost:5000/infer_legal_scenario', {
                species,
                possession_status: legalMode,
                quantity: legalMode === 'captivity' ? quantity : 0,
            });
            setLegalScenario(res.data);
        } catch (err) {
            console.error('[LEGAL SCENARIO ERROR]', err);
            alert('Không thể suy diễn tình huống pháp lý. Vui lòng kiểm tra Flask server.');
        } finally {
            setLegalLoading(false);
        }
    };

    const legalPayload = () => {
        const d = currentData || {};
        const legal = (d as any).legal || {};
        return {
            species_name:    (d as any).species,
            vietnamese_name: (d as any).vietnamese_name,
            legal_group:     (d as any).inferred_legal_group
                             || legal.nhom_phap_ly || legal.legal_group
                             || legal?.legal_advice?.group_name,
        };
    };

    const handlePreview = async () => {
        setFormLoading(true);
        try {
            const res = await axios.post('http://localhost:5000/preview_legal_form', legalPayload());
            if (res.data.status === 'success') {
                setPreviewContent(res.data.content);
                setShowModal(true);
            } else alert('Lỗi: ' + (res.data.message || 'Không thể tạo mẫu đơn'));
        } catch { alert('Lỗi kết nối server.'); }
        finally { setFormLoading(false); }
    };

    const handleGenerate = async () => {
        setFormLoading(true);
        try {
            const res = await axios.post('http://localhost:5000/generate_legal_doc', legalPayload());
            if (res.data.status === 'success') window.location.href = res.data.download_url;
            else alert('Lỗi: ' + res.data.message);
        } catch { alert('Không thể tạo đơn.'); }
        finally { setFormLoading(false); }
    };

    /* ================================================================
       CHATBOT
       ================================================================ */
    const handleSendMessage = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!userInput.trim()) return;

        const text       = userInput.trim();
        const newMsgs: ChatMessage[] = [...messages, { role: 'user', text }];
        setMessages(newMsgs);
        setUserInput('');
        setChatLoading(true);

        try {
            const res = await axios.post('http://localhost:5000/chat', { message: text });
            setMessages([...newMsgs, { role: 'bot', text: res.data.reply }]);
        } catch {
            setMessages([...newMsgs, { role: 'bot', text: 'Xin lỗi, đã có lỗi xảy ra với dịch vụ chat.' }]);
        } finally {
            setChatLoading(false);
        }
    };

    /* ================================================================
       RENDER
       ================================================================ */
    const status = currentData?.status;

    return (
        <Layout>
            <div className="ai-lookup-page">
                {/* ── HEADER GRADIENT ── */}
                <div className="app-header" style={{
                    background: 'linear-gradient(120deg, #0d3320 0%, #145c36 55%, #1e8050 100%)',
                    padding: '22px 0 18px',
                    boxShadow: '0 3px 18px rgba(0,0,0,.22)',
                    marginBottom: 0,
                }}>
                    <Container>
                        <div className="d-flex align-items-center justify-content-between flex-wrap gap-2">
                            <div>
                                <h1 style={{ fontSize: '1.45rem', fontWeight: 800, color: '#fff', letterSpacing: '.04em', margin: 0 }}>
                                    <i className="fas fa-paw me-2"></i>Pháp Lý & Nhận Dạng Động Vật AI
                                </h1>
                                <div style={{ fontSize: '.76rem', color: 'rgba(255,255,255,.65)', textTransform: 'uppercase', letterSpacing: '.1em', marginTop: 3 }}>
                                    Hệ chuyên gia · Fuzzy Logic · MYCIN CF
                                </div>
                            </div>
                            <span style={{ background: 'rgba(255,255,255,.12)', border: '1px solid rgba(255,255,255,.22)', color: '#fff', fontSize: '.68rem', fontWeight: 700, padding: '3px 11px', borderRadius: 99, letterSpacing: '.06em' }}>
                                Expert System v2
                            </span>
                        </div>
                    </Container>
                </div>

                <Container className="py-4">
                    <Row className="justify-content-center">
                        <Col lg={9}>

                            {/* ══════════════════════════════════════════
                                UPLOAD CARD
                               ══════════════════════════════════════════ */}
                            <Card className="upload-wrapper mb-4">
                                <div className="upload-rect" onClick={() => document.getElementById('fileInput')?.click()}>
                                    <input
                                        type="file"
                                        id="fileInput"
                                        accept="image/*"
                                        style={{ display: 'none' }}
                                        onChange={handleFileChange}
                                    />

                                    {!previewUrl ? (
                                        <div className="upload-prompt text-center">
                                            <i className="fas fa-camera-retro"></i>
                                            <div style={{ color: '#5a7a66', fontWeight: 500, fontSize: '.88rem' }}>
                                                Bấm để tải ảnh con vật lên
                                            </div>
                                            <div style={{ fontSize: '.75rem', color: '#aaa', marginTop: 4 }}>
                                                JPG · PNG · WEBP
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="preview-container">
                                            <img
                                                src={previewUrl}
                                                alt="preview"
                                                className="upload-preview"
                                                style={{ maxHeight: 260, objectFit: 'contain' }}
                                            />
                                            <div className="change-image-overlay" style={{ zIndex: 11, pointerEvents: 'none' }}>
                                                <i className="fas fa-camera me-2"></i> Nhấp để đổi ảnh khác
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div className="text-center py-4 bg-white">
                                    <button
                                        className="btn-predict-main"
                                        onClick={handleUpload}
                                        disabled={loading || !selectedFile}
                                    >
                                        {loading
                                            ? <><span className="spinner-border spinner-border-sm me-2"></span>Đang phân tích...</>
                                            : <><i className="fas fa-search me-2"></i>BẮT ĐẦU NHẬN DIỆN</>
                                        }
                                    </button>
                                </div>
                            </Card>

                            {/* ══════════════════════════════════════════
                                DYNAMIC AREA
                               ══════════════════════════════════════════ */}
                            {status === 'SUCCESS' && currentData && (
                                <ResultSuccess
                                    data={currentData}
                                    imageUrl={previewUrl}
                                    onPreview={handlePreview}
                                    onGenerate={handleGenerate}
                                    formLoading={formLoading}
                                    legalScenario={legalScenario}
                                    legalLoading={legalLoading}
                                    legalMode={legalMode}
                                    quantity={quantity}
                                    distribution={speciesDistribution}
                                    distributionLoading={distributionLoading}
                                    onLegalModeChange={handleLegalModeChange}
                                    onQuantityChange={setQuantity}
                                    onInferLegal={handleInferLegal}
                                />
                            )}

                            {status === 'ASKING' && currentData && (
                                <ResultAsking
                                    data={currentData}
                                    answered={answered}
                                    allQuestions={allQuestions}
                                    onAnswer={handleAnswer}
                                    loading={loading}
                                />
                            )}

                            {status === 'REJECTED' && currentData && (
                                <ResultRejected data={currentData} />
                            )}

                            {status === 'UNCERTAIN' && currentData && (
                                <ResultUncertain data={currentData} />
                            )}

                            {/* ── LOADING SKELETON ── */}
                            {loading && !currentData && (
                                <div className="text-center py-5">
                                    <Spinner animation="border" variant="success" />
                                    <p className="text-muted mt-3">Hệ thống AI đang phân tích ảnh…</p>
                                </div>
                            )}

                        </Col>
                    </Row>
                </Container>

                {/* ══════════════════════════════════════════
                    CHATBOT FLOATING
                   ══════════════════════════════════════════ */}
                <button
                    className="chat-toggle-btn"
                    onClick={() => setIsChatOpen(v => !v)}
                    title="Trợ lý Kiểm lâm AI"
                >
                    <i className={`fas ${isChatOpen ? 'fa-times' : 'fa-robot'}`}></i>
                </button>

                <div className={`chat-window-fixed ${isChatOpen ? 'open' : ''}`} ref={chatWindowRef}>
                    <div className="chat-header">
                        <span><i className="fa-solid fa-leaf me-2"></i>Trợ lý Kiểm lâm</span>
                        <button onClick={() => setIsChatOpen(false)}>
                            <i className="fa-solid fa-xmark"></i>
                        </button>
                    </div>

                    <div className="chat-body" id="chat-body">
                        {messages.map((m, i) => (
                            <div key={i} className={`chat-message ${m.role}`}>
                                {m.text}
                            </div>
                        ))}
                        <div className="typing-indicator" style={{ display: chatLoading ? 'block' : 'none' }}>
                            Đang trả lời...
                        </div>
                        <div ref={chatEndRef} />
                    </div>

                    <div className="chat-footer">
                        <input
                            ref={chatInputRef}
                            type="text"
                            placeholder="Nhập câu hỏi của bạn..."
                            value={userInput}
                            onChange={e => setUserInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
                            disabled={chatLoading}
                        />
                        <button onClick={() => handleSendMessage()} disabled={chatLoading}>
                            <i className="fa-solid fa-paper-plane"></i>
                        </button>
                    </div>
                </div>

                {/* ══════════════════════════════════════════
                    MODAL XEM TRƯỚC ĐƠN
                   ══════════════════════════════════════════ */}
                <Modal show={showModal} onHide={() => setShowModal(false)} size="xl">
                    <Modal.Header style={{ background: '#145c36', color: '#fff' }}>
                        <Modal.Title>
                            <i className="fas fa-file-alt me-2"></i>Mẫu đơn đã điền tự động
                        </Modal.Title>
                        <button
                            type="button"
                            className="btn-close btn-close-white"
                            onClick={() => setShowModal(false)}
                        />
                    </Modal.Header>
                    <Modal.Body className="p-4">
                        <pre id="previewContent" className="bg-light p-4 border rounded shadow-sm">
                            {previewContent}
                        </pre>
                    </Modal.Body>
                    <Modal.Footer>
                        <Button variant="secondary" onClick={() => setShowModal(false)}>Đóng</Button>
                        <Button
                            variant="success"
                            onClick={() => { setShowModal(false); handleGenerate(); }}
                        >
                            <i className="fas fa-download me-2"></i>Tạo file & Tải xuống
                        </Button>
                    </Modal.Footer>
                </Modal>
            </div>
        </Layout>
    );
};

export default AILookup;
