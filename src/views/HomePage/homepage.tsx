import Layout from '../../layouts/Layout';
import './homepage.css';
import AOS from 'aos';
import 'aos/dist/aos.css';
import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import i18n from '../../i18n';
import { Container, Row, Col } from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleCheck, faCircleDollarToSlot, faHourglassStart, faCheck, faLocationDot, faPhone } from '@fortawesome/free-solid-svg-icons';
import { Link, NavLink } from 'react-router-dom';
import Slider from 'react-slick';

import about_1 from '../../assets/image/about-1.jpg';
import project_1 from '../../assets/image/project_1.jpg';
import project_2 from '../../assets/image/project_2.jpg';
import project_3 from '../../assets/image/project_3.jpg';
import Nav from 'react-bootstrap/Nav'
import { toast } from 'react-toastify';

import axios from 'axios';
import { useFetchNews } from '../../hooks/useFetchNews';
import { useFetchProject } from '../../hooks/useFetchProject';

const HomePage = () => {
  const { t } = useTranslation();
  //Effect for hero section
  useEffect(() => {
    document.documentElement.style.setProperty('--cursor-size', '42px');
    const handleMove = (e: MouseEvent) => {
      document.documentElement.style.setProperty('--cursor-x', `${e.clientX}px`);
      document.documentElement.style.setProperty('--cursor-y', `${e.clientY}px`);
    };
    window.addEventListener('mousemove', handleMove);
    return () => window.removeEventListener('mousemove', handleMove);
  }, []);

  const onMouseEnter = () => document.documentElement.style.setProperty('--cursor-size', '300px');
  const onMouseLeave = () => document.documentElement.style.setProperty('--cursor-size', '16px');

   // React Slick
   var settings = {
      dots: true,
      infinite: false,
      speed: 500,
      slidesToShow: 3,
      slidesToScroll: 1,
      responsive: [
         {
            breakpoint: 798,
            settings: {
               slidesToShow: 2,
               slidesToScroll: 2,
               initialSlide: 2,
            }
         },
         {
            breakpoint: 576,
            settings: {
               slidesToShow: 1,
               slidesToScroll: 1,
            }
         },
      ],
      // prevArrow: <FontAwesomeIcon icon={faAngleLeft} />,
      // nextArrow: <FontAwesomeIcon icon={faAngleRight} />,
   };

   // Animation on Scroll
   useEffect(() => {
      AOS.init({
         duration: 1000,
         once: true,
      });
   }, []);

   // Impact effect
   useEffect(() => {
      let countTriggered = false;

      const handleScroll = () => {
         if (!countTriggered && isInViewport(document.getElementById('impact') as HTMLElement)) {
            countTriggered = true;
            countUp('yearCount', 9);
            countUp('partnerCount', 53);
            countUp('projectCount', 48);
            countUp('memberCount', 5);
         }
      };

      window.addEventListener('scroll', handleScroll);

      return () => {
         window.removeEventListener('scroll', handleScroll);
      };
   }, []);

   const isInViewport = (element: HTMLElement) => {
      const rect = element.getBoundingClientRect();
      const windowHeight = window.innerHeight || document.documentElement.clientHeight;
      const windowWidth = window.innerWidth || document.documentElement.clientWidth;

      const verticallyVisible = rect.top <= windowHeight && rect.bottom >= 0;
      const horizontallyVisible = rect.left <= windowWidth && rect.right >= 0;

      return verticallyVisible && horizontallyVisible;
   };

   const countUp = (id: string, target: number) => {
      let current = 0;
      const increment = Math.ceil(target / 10);
      const interval = setInterval(() => {
         if (current >= target) {
            clearInterval(interval);
         } else {
            current += increment;
            const element = document.getElementById(id);
            if (element) {
               element.textContent = current.toLocaleString();
            }
         }
      }, 100);
   };

   // Form Contact
   const [firstName, setFirstName] = useState('');
   const [lastName, setLastName] = useState('');
   const [email, setEmail] = useState('');
   const [message, setMessage] = useState('');

   const handleFirstNameChange = (e: React.ChangeEvent<HTMLInputElement>) => setFirstName(e.target.value);
   const handleLastNameChange = (e: React.ChangeEvent<HTMLInputElement>) => setLastName(e.target.value);
   const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value);
   const handleMessageChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => setMessage(e.target.value);

   const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();

      // Kiểm tra cơ bản
      if (!firstName.trim() || !lastName.trim()) {
         toast.error('Vui lòng nhập đầy đủ họ tên');
         return;
      }

      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!email.trim() || !emailRegex.test(email)) {
         toast.error('Email không hợp lệ');
         return;
      }

      const message = (document.getElementById("message") as HTMLTextAreaElement)?.value.trim();
      if (!message) {
         toast.error('Vui lòng nhập câu hỏi hoặc nội dung thắc mắc');
         return;
      }

      try {
         const res = await axios.post('/api/contact/verify', {
            firstName,
            lastName,
            email,
            message,
         });
         toast.success('Vui lòng kiểm tra email để xác nhận liên hệ!');
         
         // Reset form nếu muốn
         setFirstName('');
         setLastName('');
         setEmail('');
         (document.getElementById("message") as HTMLTextAreaElement).value = '';
      } catch (error) {
         console.error(error);
         toast.error('Gửi email xác nhận thất bại. Vui lòng thử lại!');
      }
   };

   const { newsList, loading} = useFetchNews();
   const { project, loadding} = useFetchProject();

  return (
    <Layout>
         {/* Hero Section */}
         <div className="hero-section">
            <div className="layer layer-01">
               <div className="text-container text-01" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
                  <div>{t('variety')}</div>
                  <div>{t('is_the')}</div>
                  <div>{t('spice')}</div>
                  <div>{t('of')}</div>
                  <div>{t('life')}</div>
               </div>
            </div>
            <div className="layer layer-02">
               <div className="text-container">
                  <div>{t('keep')} </div>
                  <div>{t('variety')}</div>
                  <div>{t('and')}</div>
                  <div>{t('prevent')}</div>
                  <div>{t('extinction')}!</div>
               </div>
            </div>
         </div>

         {/* Introduce Section */}
         <div id="introduce">
            <Container>
               <Row>
                  <Col lg={6}>
                     <div className="image-box">
                        <div className="shape"></div>
                        <div className="image">
                           <img src={about_1} alt="" />
                        </div>
                     </div>
                  </Col>
                  <Col lg={6}>
                     <div className="content-box">
                        <h2>{t('welcome')}</h2>
                        <h4>{t('help_us')}</h4>
                        <p>{t('descript_1')}</p>
                        <ul>
                           <li>
                              <FontAwesomeIcon className='icon' icon={faCircleCheck} />
                              <span>{t('key_1')}</span>
                           </li>
                           <li>
                              <FontAwesomeIcon className='icon' icon={faCircleCheck} />
                              <span>{t('key_2')}</span>
                           </li>
                           <li>
                              <FontAwesomeIcon className='icon' icon={faCircleCheck} />
                              <span>{t('key_3')}</span>
                           </li>
                        </ul>
                        <Link to="/about-us" className='button button-left'>{t('discover_more')}</Link>
                     </div>
                  </Col>
               </Row>
            </Container>
         </div>

         {/* Impact Section */}
         <div id="impact">
            <Container>
               <div className="inner-content">
                  <h2>{t('impact')}</h2>
                  <ul className="inner-infor">
                     <li>
                        <h3 id="yearCount">1</h3>
                        <p>{t('exp')}</p>
                     </li>
                     <li>
                        <h3 id="partnerCount">1</h3>
                        <p>{t('partner')}</p>
                     </li>
                     <li>
                        <h3 id="projectCount">1</h3>
                        <p>{t('project')}</p>
                     </li>
                     <li>
                        <div className="box d-flex">
                           <h3 id="memberCount">1</h3>
                           <h3> K</h3>
                        </div>
                        <p>{t('mem')}</p>
                     </li>
                  </ul>
               </div>
            </Container>
         </div>

         {/* Recent-project Section */}
         <div id="recent-proj">
            <div className="head">
               <h2>{t('recent_prj')}</h2>
            </div>
            <Container>
               <div className="card__container">
                  <Row>
                     {project.slice(0, 3).map((project, index) => (
                        <Col lg={4} className="card__article" key={index}>
                           <img src={project.image_url} alt="image" className='card__img' />
                           <div className="image-title">{project.title}</div>
                           <div className="card__data">
                           <h2 className="card__title">{project.title} Project</h2>
                           <Link to={`/projects/${project.id}`} className='card__button button button-left'>
                              {t('donate_now')}
                           </Link>
                           </div>
                        </Col>
                     ))}
                  </Row>
               </div>
            </Container>
         </div>

         {/* How we work Section */}
         <div id="work">
            <Container>
               <div className="title">
                  <h4>{t('working_process')}</h4>
                  <h2>{t('you_donate')}</h2>
               </div>
               <Row>
                  <Col lg={4} className='step step-1'>
                     <div className="step-content">
                        <div className="step-icon">
                           <FontAwesomeIcon icon={faCircleDollarToSlot} />
                        </div>
                        <div className="step-name">{t('donating')}</div>
                        <div className="step-description">{t('d_des')}</div>
                     </div>
                     <div className="step-line">
                        <img src="https://demo.bravisthemes.com/nonta/wp-content/uploads/2023/07/line1.png" alt="" />
                     </div>
                  </Col>
                  <Col lg={4} className='step step-2'>
                     <div className="step-content">
                        <div className="step-icon">
                           <FontAwesomeIcon icon={faHourglassStart} />
                        </div>
                        <div className="step-name">{t('process')}</div>
                        <div className="step-description">{t('p_des')}</div>
                     </div>
                     <div className="step-line">
                        <img src="https://demo.bravisthemes.com/nonta/wp-content/uploads/2023/07/line2.png" alt="" />
                     </div>
                  </Col>
                  <Col lg={4} className='step step-3'>
                     <div className="step-content">
                        <div className="step-icon">
                           <FontAwesomeIcon icon={faCheck} />
                        </div>
                        <div className="step-name">{t('complete')}</div>
                        <div className="step-description">{t('c_des')}</div>
                     </div>
                  </Col>
               </Row>
            </Container>
         </div>

         {/* Latest new Section */}
         <div id="latest-new" >
            <Container>
               <h2>{t('latest_new')}</h2>
               <div>
                  <Slider {...settings}>
                  {newsList.map(news => (
                     <div className="inner-box" key={news.id}>
                        <div className="box-header">
                        <div className="main-img">
                           <NavLink to={`/news/${news.id}`}>
                              <img src={news.thumbnail} alt={news.title} />
                           </NavLink>
                        </div>
                        <div className="date-box">
                           <p>{new Date(news.date).getDate()}</p>
                           <span>{new Date(news.date).toLocaleString('default', { month: 'short' })}</span>
                        </div>
                        </div>
                        <div className="title">
                        <NavLink to={`/news/${news.id}`}>{news.title}</NavLink>
                        </div>
                        <div className="description" dangerouslySetInnerHTML={{ __html: news.content.slice(0, 80)}}
                        ></div>
                     </div>
                  ))}
                  </Slider>
               </div>
            </Container>
         </div>

         {/* Subscribe Section */}
         <div id="subscribe">
            <Container>
               <div className="inner-main">
                  <div className="inner-content">
                     <h2>{t('volunteer')}</h2>
                     <p>{t('volunteer_des')}</p>
                  </div>
               </div>
            </Container>
         </div>

         {/* Contact */}
         <div className="contact-section">
            <Container>
               <Row>
                  <div className="col-xl-5 col-12 ms-auto mb-5 ">
                     <div className="contact-main">
                        <p className='address'>
                           <FontAwesomeIcon icon={faLocationDot} />
                           <span>470 Tran Dai Nghia,
                              <br /> Da Nang</span>
                        </p>
                        <p className='phone'>
                           <FontAwesomeIcon icon={faPhone} />
                           <span>0236 3667 111</span> <br />
                           <span>greenland@gmail.com</span>
                        </p>
                     </div>
                  </div>
                  <div className="col-xl-5 col-12 mx-auto">
                     <form action="#" className="contact-form custom-form" onSubmit={handleSubmit}>
                        <p style={{ color: "#315740" }}>{t('our_contact')}</p>
                        <h2>{t('getin')}</h2>
                        <div className="row">
                           <div className="col-lg-6 col-md-6 col-12">
                              <input
                                 type="text"
                                 className="form-control"
                                 name="first-name"
                                 id="first-name"
                                 placeholder={t('fname')}
                                 value={firstName}
                                 onChange={handleFirstNameChange}
                              />
                           </div>
                           <div className="col-lg-6 col-md-6 col-12">
                              <input
                                 type="text"
                                 className="form-control"
                                 name="last-name"
                                 id="last-name"
                                 placeholder={t('lname')}
                                 value={lastName}
                                 onChange={handleLastNameChange}
                              />
                           </div>
                        </div>
                        <input
                           type="email"
                           className="form-control"
                           name="email"
                           id="email"
                           placeholder="example@gmail.com"
                           value={email}
                           onChange={handleEmailChange}
                        />
                        <textarea
                           name="message"
                           id="message"
                           className="form-control"
                           rows={5}
                           placeholder={t('help')}
                           value={message}
                           onChange={handleMessageChange}
                        />
                        <button type='submit' className="button button-left">{t('mess')}</button>
                     </form>
                  </div>
               </Row>
            </Container>
         </div>
         {/* End contact */}
    </Layout>
  );
};

export default HomePage;