// AboutPage.tsx
import Layout from '../../layouts/Layout';
import './about.css';

import { useEffect, useRef, useState } from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import AOS from 'aos';
import 'aos/dist/aos.css';
import { useTranslation } from 'react-i18next';

import journey_1 from '../../assets/image/journey-1.jpg';
import journey_2 from '../../assets/image/journey-2.jpg';
import leader_1 from '../../assets/image/leader1.png';
import leader_2 from '../../assets/image/leader2.png';
import leader_3 from '../../assets/image/leader3.png';
import end_1 from '../../assets/image/about_end_1.png';

const About = () => {
  const { t } = useTranslation();

    // Animation on Scroll
    useEffect(() => {
        AOS.init({
            duration: 1000,
            once: true,
        });
    }, []);

    // Journey
    const spaceHolderRef = useRef<HTMLDivElement | null>(null);
    const horizontalRef = useRef<HTMLDivElement | null>(null);
    const stickyRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        const spaceHolder = spaceHolderRef.current;
        const horizontal = horizontalRef.current;
        const sticky = stickyRef.current;

        if (spaceHolder && horizontal && sticky) {
            spaceHolder.style.height = `${calcDynamicHeight(horizontal)}px`;

            const handleScroll = () => {
                horizontal.style.transform = `translateX(-${sticky.offsetTop}px)`;
            };

            const handleResize = () => {
                spaceHolder.style.height = `${calcDynamicHeight(horizontal)}px`;
            };

            window.addEventListener('scroll', handleScroll);
            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('scroll', handleScroll);
                window.removeEventListener('resize', handleResize);
            };
        }
    }, []);

    const calcDynamicHeight = (ref: HTMLDivElement) => {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const objectWidth = ref.scrollWidth;
        return objectWidth - vw + vh + 150;
    };

  return (
    <Layout>
      {/* Hero Section */}
      <div id="hero-section">
        <Container>
          <p style={{ color: '#315740', fontWeight: '800', marginLeft: '10px' }}>{t('explore')}</p>
          <h1>
            <span style={{ color: '#315740' }}>{t('a')}</span> <span style={{ color: '#232323' }}>{t('new')}</span>{' '}
            <span style={{ color: '#315740' }}>{t('journey')}</span> <br />
            <span style={{ color: '#232323' }}>{t('in')}</span>{' '}
            <span style={{ color: '#315740' }}>{t('wildlife')}</span> <br />
            <span style={{ color: '#232323' }}>{t('experience')}</span>
          </h1>
          <button className="button button-left">
            <a href="#about">{t('let_explore')}</a>
          </button>
        </Container>
      </div>

      {/* About Section */}
      <div id="about">
        <Container>
          <Row>
            <Col lg={6}>
              <div className="image-box" data-aos="flip-left">
                <img
                  src="https://fastwpdemo.com/newwp/weldlfe/wp-content/uploads/2022/05/about-3.jpg"
                  alt=""
                />
              </div>
            </Col>
            <Col lg={6}>
              <div className="content-box" data-aos="flip-left">
                <h2>
                  {t('get_to_know1')}<br />
                  {t('get_to_know2')}
                </h2>
                <h4>{t('dedicated')}</h4>
                <p>{t('our_org')}</p>
              </div>
            </Col>
          </Row>
        </Container>
      </div>

      {/* Journey Section */}
      <div id="journey">
          <div className="container">
              <div className="space-holder" ref={spaceHolderRef}>
                  <h2>{t('our_journey')}</h2>
                  <div className="sticky" ref={stickyRef}>
                      <div className="horizontal" ref={horizontalRef}>
                          <section role="feed" className="cards">
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={journey_1} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>03</p>
                                          <span>2014</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('foundating')}
                                      </a>
                                  </div>
                              </div>
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={journey_2} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>05</p>
                                          <span>2015</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('launch')}
                                      </a>
                                  </div>
                              </div>
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={'https://c8.alamy.com/comp/2KYHG7F/world-wildlife-day-on-march-3rd-to-raise-animal-awareness-plant-and-preserve-their-habitat-in-forest-in-flat-cartoon-hand-drawn-template-illustration-2KYHG7F.jpg'} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>06</p>
                                          <span>2016</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('campaign')}
                                      </a>
                                  </div>
                              </div>
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={'https://mediaproxy.salon.com/width/1200/https://media2.salon.com/2022/07/fist-bump-teamwork-hands-in-0725221.jpg'} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>03</p>
                                          <span>2017</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('cooperate')}
                                      </a>
                                  </div>
                              </div>
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={'https://365give.ca/wp-content/uploads/2023/08/Small-Steps-Big-Impact.png'} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>12</p>
                                          <span>2018</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('earthday')}
                                      </a>
                                  </div>
                              </div>
                              <div className="sample-card">
                                  <div className="box-header">
                                      <div className="main-img">
                                          <a href="#" target='_blank'>
                                              <img src={'https://www.balisafarimarinepark.com/wp-content/uploads/2022/09/64149287_medium.jpg'} alt="" />
                                          </a>
                                      </div>
                                      <div className="date-box">
                                          <p>03</p>
                                          <span>2024</span>
                                      </div>
                                  </div>
                                  <div className="title">
                                      <a href="#">
                                          {t('anniversary')}
                                      </a>
                                  </div>
                              </div>
                          </section>
                      </div>
                  </div>
              </div>
          </div>
      </div>

      {/* Team Section */}
      {/* <div id="team">
        <Container>
            <h2>{t('leader')}</h2>
            <div className='team-member'>
                <div className='leader-infor'>
                    <Row className="content_leader">
                        <Col lg={1} xs={2}>
                            <ul className="list_icon">
                                <li><a href="https://www.instagram.com/" ><i className="fa-brands fa-instagram" id="icon"></i></a></li>
                                <li><a href="https://www.facebook.com/"><i className="fa-brands fa-facebook-f" id="icon"></i></a></li>
                                <li><a href="https://x.com/?lang=vi" ><i className="fa-brands fa-twitter" id="icon"></i></a></li>
                            </ul>
                        </Col>
                        <Col lg={1} xs={1}><p></p></Col>
                        <Col lg={3} xs={5}>
                            <img src={leader_1} className="image_ld" />
                        </Col>
                        <Col lg={7} xs={12}>
                            <div className="text-ld">
                                <p className="name">{t('name_1')}</p>
                                <p className="posi">{t('pos_1')}</p>
                                <p className="mail"><b>Email: </b>triennd.23ai@vku.udn.vn</p>
                                <p className="intro-text">{t('des_1')}</p>
                            </div>
                        </Col>
                    </Row>
                </div>
                <hr />
                <div className='leader-infor'>
                    <Row className="content_leader">
                        <Col lg={7} xs={12}>
                            <div className="text-ld">
                                <p className="name">{t('name_2')}</p>
                                <p className="posi">{t('pos_2')}</p>
                                <p className="mail"><b>Email: </b>tridt.23ai@vku.udn.vn</p>
                                <p className="intro-text">{t('des_2')}</p>
                            </div>
                        </Col>
                        <Col lg={4} xs={6}>
                            <img src={leader_2} className="image_ld" />
                        </Col>
                        <Col lg={1} xs={2}>
                            <ul className="list_icon">
                                <li><a href="https://www.instagram.com/" ><i className="fa-brands fa-instagram" id="icon"></i></a></li>
                                <li><a href="https://www.facebook.com/"><i className="fa-brands fa-facebook-f" id="icon"></i></a></li>
                                <li><a href="https://x.com/?lang=vi" ><i className="fa-brands fa-twitter" id="icon"></i></a></li>
                            </ul>
                        </Col>
                    </Row>
                </div>
                <hr />
                <div className='leader-infor'>
                    <Row className="content_leader">
                        <Col lg={1} xs={2}>
                            <ul className="list_icon">
                                <li><a href="https://www.instagram.com/" ><i className="fa-brands fa-instagram" id="icon"></i></a></li>
                                <li><a href="https://www.facebook.com/"><i className="fa-brands fa-facebook-f" id="icon"></i></a></li>
                                <li><a href="https://x.com/?lang=vi" ><i className="fa-brands fa-twitter" id="icon"></i></a></li>
                            </ul>
                        </Col>
                        <Col lg={1} xs={1}><p></p></Col>
                        <Col lg={3} xs={5}>
                            <img src={leader_3} className="image_ld" />
                        </Col>
                        <Col lg={7} xs={12}>
                            <div className="text-ld">
                                <p className="name">{t('name_3')}</p>
                                <p className="posi">{t('pos_3')}</p>
                                <p className="mail"><b>Email: </b>quynhnnx.23ai@vku.udn.vn</p>
                                <p className="intro-text">{t('des_3')}</p>
                            </div>
                        </Col>
                    </Row>
                </div>
            </div>
        </Container>
      </div> */}

      {/* Join Section */}
      <div id="join">
          <Container>
              <Row>
                  <Col lg={7}>
                      <div className="content-box" data-aos="flip-left">
                          <p className='head-title'>{t('volunteer_campaign')}</p>
                          <h2>{t('join_team1')} <br />
                              {t('join_team2')}</h2>
                          <p className='infor'>{t('join_des_1')}
                              <br />{t('join_des_2')} <br />
                              {t('join_des_3')}
                          </p>
                          <Link to='/contact' className='button button-left'>
                              {t('join_us')}
                          </Link>
                      </div>
                  </Col>
                  <Col lg={1}></Col>
                  <Col lg={4}>
                      <div className="image-list">
                          <img src={'https://ktmt.vnmediacdn.com/stores/news_dataimages/svtt/072020/11/10/in_article/5221_1594377305-dong-vat-hoang-da.jpg'} alt="" />
                          <img src={end_1} alt="" />
                          <img src={'https://vcdn1-giadinh.vnecdn.net/2024/04/10/image002-3224-1712748148.jpg?w=460&h=0&q=100&dpr=2&fit=crop&s=Ber-r6cyiK-4hQmVfhDMbA'} alt="" />
                      </div>
                  </Col>
              </Row>
          </Container>
      </div>

    </Layout>
  );
};

export default About;
