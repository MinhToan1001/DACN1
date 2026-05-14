import { Container, Row, Col } from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFacebookF, faInstagram, faTwitter, faLinkedin } from '@fortawesome/free-brands-svg-icons';
import { Link } from 'react-router-dom';
import footer_head from '../assets/image/footer_head.png';
import { useTranslation } from 'react-i18next';

const Footer = () => {
  const { t } = useTranslation();

  return (
    <div id="footer">
      <div className="footer-head">
        <img src={footer_head} alt="Footer Top" />
      </div>
      <div className="footer_main">
        <Container>
          <div className="inner-main">
            <Row>
              {/* Cột 1: Tên & mô tả */}
              <Col lg={4} sm={12}>
                <h2>GREEN LAND</h2>
                <p>{t('footer_des')}</p>
                <div className="contact-list">
                  <a href="https://www.facebook.com/" target="_blank" rel="noreferrer">
                    <FontAwesomeIcon className="icon" icon={faFacebookF} />
                  </a>
                  <a href="https://www.instagram.com/" target="_blank" rel="noreferrer">
                    <FontAwesomeIcon className="icon" icon={faInstagram} />
                  </a>
                  <a href="https://x.com/?lang=vi" target="_blank" rel="noreferrer">
                    <FontAwesomeIcon className="icon" icon={faTwitter} />
                  </a>
                  <a href="https://www.linkedin.com/" target="_blank" rel="noreferrer">
                    <FontAwesomeIcon className="icon" icon={faLinkedin} />
                  </a>
                </div>
              </Col>

              {/* Cột 2: Menu điều hướng */}
              <Col lg={2} xs={6}>
                <ul>
                  <li className="head"><b>{t('nav')}</b></li>
                  <li><Link to="/home">{t('home')}</Link></li>
                  <li><Link to="/about-us">{t('about_us')}</Link></li>
                  <li><Link to="/reality">{t('reality')}</Link></li>
                  <li><Link to="/contact">{t('contact')}</Link></li>
                  <li><Link to="/donate">{t('donate')}</Link></li>
                </ul>
              </Col>

              {/* Cột 3: Liên hệ */}
              <Col lg={3} xs={6}>
                <ul>
                  <li className="head"><b>{t('contact')}</b></li>
                  <li>{t('phone')}: 0236 3667 111</li>
                  <li>Email: greenland@gmail.com</li>
                  <li>{t('address')}</li>
                </ul>
              </Col>

              {/* Cột 4: Mailbox */}
              <Col lg={3} sm={12}>
                <ul>
                  <li className="head"><b>Mailbox</b></li>
                  <li>{t('mail_box_des')}</li>
                  <li className="email-input">
                    <form onSubmit={(e) => e.preventDefault()}>
                      <input type="email" placeholder={t('your_email')} />
                      <button>{t('send')}</button>
                    </form>
                  </li>
                </ul>
              </Col>
            </Row>

            <hr />

            <div className="copyright-box">
              <div className="content">
                Copyright &copy; Green Land since 2014
              </div>
            </div>
          </div>
        </Container>
      </div>
    </div>
  );
};

export default Footer;
