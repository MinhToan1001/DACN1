import { useState, useEffect, SetStateAction } from 'react';
import { NavLink, useNavigate, Link } from 'react-router-dom';
import './project.css';
import { useParams } from 'react-router-dom';

import axios from 'axios';
import Layout from '../../layouts/Layout';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faFacebookF, faInstagram, faTwitter, faLinkedin } from '@fortawesome/free-brands-svg-icons'
import { faAngleLeft } from '@fortawesome/free-solid-svg-icons'

import { Container, Row, Col, } from 'react-bootstrap';

import Alert from 'react-bootstrap/Alert';
import vnpay from '../../assets/image/vnpay.png'
import flagEn from '../../assets/image/en.svg';
import flagVi from '../../assets/image/vi.svg';

import { useTranslation } from 'react-i18next';
import i18n from '../../i18n';

interface Props { }

function Project(props: Props) {
    const { } = props;
    const [showAlert, setShowAlert] = useState(false);

    // Change language
    const [language, setLanguage] = useState<string>(localStorage.getItem('language') || 'en');
    const [flag, setFlag] = useState<string>(localStorage.getItem('flag') || flagEn);

    useEffect(() => {
        i18n.changeLanguage(language);
        setFlag(language === 'en' ? flagEn : flagVi);
    }, [language]);

    const changeLanguage = () => {
        const newLanguage = language === 'en' ? 'vi' : 'en';
        setLanguage(newLanguage);
        localStorage.setItem('language', newLanguage);
        localStorage.setItem('flag', newLanguage === 'en' ? flagEn : flagVi);
    };
    const { t } = useTranslation();

    const { id } = useParams();
    const [project, setProject] = useState<any>(null);

    // Lấy dữ liệu project theo id
    useEffect(() => {
    if (id) {
        axios.get(`/api/projects/${id}`)
        .then(res => setProject(res.data))
        .catch(err => console.error('Lỗi tải project:', err));
    }
    }, [id]);

    // Donate Method
    const [donateMethod, setDonateMethod] = useState('');

    useEffect(() => {
        setDonateMethod('online');
    }, []);

    const handleDonateMethodChange = (method: SetStateAction<string>) => {
        setDonateMethod(method);
    };

    const [amount, setAmount] = useState<number>(10000); // số tiền mặc định
    // Gọi API thanh toán
    const handleDonate = async () => {
        const user = JSON.parse(localStorage.getItem('user') || 'null');
        if (!user || !user.id || !id) {
            console.error("Thiếu thông tin user hoặc project");
            return;
        }

        try {
            const url = '/api/payment/create_payment_url';

            const res = await axios.post(url, {
            name: user.name,
            email: user.email,
            amount,
            projectId: id
            }, {
            headers: { 'Content-Type': 'application/json' }
            });

            if (res.data.paymentUrl) {
            window.location.href = res.data.paymentUrl;
            }
        } catch (err) {
            console.error("Lỗi khi tạo URL thanh toán:", err);
        }
    };



    return (
        <Layout>
            {/* Hero section */}
            <div id="hero">
                <Container>
                    <h1>{t('make_donation')}</h1>
                </Container>
            </div>

            {/* Donate */}
            <div id="main">
                <Container>
                    <Row>
                        <Col xl={3} className='image'>
                            {project && project.image_url && (
                                <img src={project.image_url} alt={project.title} />
                            )}
                        </Col>
                        <Col xl={4} className='content'>
                            <h6>{t('donatingto')}:</h6>
                            <h4>{project?.title}</h4>
                            <p>{project?.description}</p>
                            <ul className='socials-list'>
                                <li>
                                    <FontAwesomeIcon className='icon' icon={faFacebookF} />
                                </li>
                                <li>
                                    <FontAwesomeIcon className='icon' icon={faInstagram} />
                                </li>
                                <li>
                                    <FontAwesomeIcon className='icon' icon={faLinkedin} />
                                </li>
                                <li>
                                    <FontAwesomeIcon className='icon' icon={faTwitter} />
                                </li>
                            </ul>
                        </Col>
                        <Col xl={5}>
                            <div className="donate-box">
                                <div>
                                    <label>{t('donation_amount')}: </label>
                                    <br />
                                    <input
                                        className='amount'
                                        type="number"
                                        value={amount}
                                        onChange={(e) => setAmount(Number(e.target.value))}/>
                                </div>
                                <Row>
                                    <Col md={6}>
                                        <div className="form-check form-check-radio">
                                            <input
                                                type="radio"
                                                name='donate-method'
                                                className="form-check-input"
                                                required={true}
                                                checked={donateMethod === 'online'}
                                                onChange={() => handleDonateMethodChange('online')} />
                                            <label
                                                className="form-check-label"
                                                htmlFor="online"
                                            >
                                                {t('online')}
                                            </label>
                                        </div>
                                    </Col>
                                    <Col md={6}>
                                        <div className="form-check form-check-radio">
                                            <input
                                                type="radio"
                                                name='donate-method'
                                                className="form-check-input"
                                                required={true}
                                                onChange={() => handleDonateMethodChange('offline')} />
                                            <label
                                                className="form-check-label"
                                                htmlFor="offline"
                                            >
                                                {t('offline')}
                                            </label>
                                        </div>
                                    </Col>
                                </Row>
                                {donateMethod === 'offline' && (
                                    <div className="offline-content">
                                        <div className='title'>{t('method')}</div>
                                        <div className='main-content'>{t('donate_des')}</div>
                                        <br />
                                        <button className='button button-left'>{t('donate_now')}</button>
                                    </div>
                                )}
                                {donateMethod === 'online' && (
                                    <div className="online-content">
                                        <div className='title'>{t('method')}</div>
                                        <img src={vnpay} alt="" />
                                        <br />
                                        <button className='button button-left' onClick={handleDonate}>{t('donate_now')}</button>
                                    </div>
                                )}
                                {showAlert && (
                                    <Alert className="alert-position" variant="success" onClose={() => setShowAlert(false)} dismissible>
                                        Donation successful! Thank you for your contribution.
                                    </Alert>
                                )}
                            </div>
                        </Col>
                    </Row>
                </Container>
            </div>

            {/* Project's description */}
            <div id="des">
                <Container>
                    <Row>
                        <Col xl={7} className='descript'>
                            <h3>{t('introduce')} THE NATURE'S KEEPERS</h3>
                            <p>{t('pr1_1')}</p>
                            <p>{t('pr1_2')}</p>
                            <p>{t('pr1_3')}</p>
                            <hr />
                        </Col>
                        <Col md={7} >
                            <h4>{t('promise')}</h4>
                            <p className='promise'>{t('promise_des')}</p>
                        </Col>
                    </Row>
                </Container>
            </div >

            {/* End section Donate */}

        </Layout>
    );
}

export default Project;
