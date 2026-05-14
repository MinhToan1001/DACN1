import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Breadcrumb } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHouse, faUser, faClock, faEnvelope, faAngleLeft, faAngleRight } from '@fortawesome/free-solid-svg-icons';
import { faFacebookF, faInstagram, faTwitter, faLinkedin } from '@fortawesome/free-brands-svg-icons';

import Layout from '../../layouts/Layout';
import './NewsDetail.css'

import moment from 'moment';
import 'moment/locale/vi'

function NewsDetail() {
  const { id } = useParams();
  const [news, setNews] = useState<any>(null);

  useEffect(() => {
    console.log("Slug đang lấy từ URL:", id);  // thêm dòng này
    axios.get(`/api/news/${id}`)
      .then(res => setNews(res.data))
      .catch(err => console.error('❌ API lỗi:', err));
  }, [id]);


  if (!news) return <div>Loading...</div>;

  return (
    <Layout>
      <div id="content">
        <div className="head">
          <div className="header">{news.title}</div>
          <Breadcrumb>
            <Breadcrumb.Item>
              <Link to="/home"><FontAwesomeIcon icon={faHouse} /></Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item active>{news.title}</Breadcrumb.Item>
          </Breadcrumb>
        </div>
        <Container>
          <div className="top">
            <FontAwesomeIcon icon={faUser} /> {news.author} -
            <FontAwesomeIcon icon={faClock} /> {moment(news.date).locale('vi').format('D MMMM, YYYY')}
          </div>
          <div className="inner_main_content">
            {news.thumbnail && <img src={news.thumbnail} alt={news.title} />}
            <div className="description" dangerouslySetInnerHTML={{ __html: news.content}}></div>
            <p className='share'>Share this article</p>
            <ul className='socials-list'>
              <li><a href="#"><FontAwesomeIcon className='icon' icon={faFacebookF} /></a></li>
              <li><a href="#"><FontAwesomeIcon className='icon' icon={faInstagram} /></a></li>
              <li><a href="#"><FontAwesomeIcon className='icon' icon={faLinkedin} /></a></li>
              <li><a href="#"><FontAwesomeIcon className='icon' icon={faTwitter} /></a></li>
              <li><a href="#"><FontAwesomeIcon className='icon' icon={faEnvelope} /></a></li>
            </ul>
          </div>
        </Container>
      </div>
    </Layout>
  );
}

export default NewsDetail;
