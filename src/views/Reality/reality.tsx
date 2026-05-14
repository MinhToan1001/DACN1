import './reality.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import React, { useEffect, useState } from 'react';

import climatechange from '../../assets/image/climatechange.jpg';
import habitatloss from '../../assets/image/habitatlosss.jpg';
import polution from '../../assets/image/polution.jpg';
import polarbear from '../../assets/image/polarbear.jpeg';
import header_bg from '../../assets/image/img1.jpg';
import news_vid from '../../assets/videos/news-vid.mp4';

import Layout from '../../layouts/Layout';
import { useTranslation } from 'react-i18next';

const Reality = () => {
  const { t } = useTranslation();

  // effect for play video & open modal video
  useEffect(() => {
    const videoThumbs = document.querySelectorAll<HTMLVideoElement>(".thumb-video");

    videoThumbs.forEach((videoThumb) => {
      const handleMouseOver = () => {
        videoThumb.style.opacity = '1';
        videoThumb.play();
        videoThumb.playbackRate = 2.0;
      };

      const handleMouseLeave = () => {
        videoThumb.style.opacity = '0';
        videoThumb.pause();
      };

      videoThumb.addEventListener("mouseover", handleMouseOver);
      videoThumb.addEventListener("mouseleave", handleMouseLeave);

      return () => {
        videoThumb.removeEventListener("mouseover", handleMouseOver);
        videoThumb.removeEventListener("mouseleave", handleMouseLeave);
      };
    });

    const videoModalScreens = document.querySelectorAll<HTMLElement>(".video-modal-screen");
    const videoCards = document.querySelectorAll<HTMLElement>(".video-card-body");
    const modalCloseBtns = document.querySelectorAll<HTMLElement>(".vid-modal-close-btn");

    var videoModal = (modalClick: number) => {
      videoModalScreens[modalClick].classList.add("active");
    };

    videoCards.forEach((videoCard, i) => {
      const handleClick = () => {
        videoModal(i);
      };

      videoCard.addEventListener("click", handleClick);

      return () => {
        videoCard.removeEventListener("click", handleClick);
      };
    });

    modalCloseBtns.forEach((modalCloseBtn) => {
      const handleClose = () => {
        videoModalScreens.forEach((videoModalScreen) => {
          videoModalScreen.classList.remove("active");
        });
      };

      modalCloseBtn.addEventListener("click", handleClose);

      return () => {
        modalCloseBtn.removeEventListener("click", handleClose);
      };
    });
  }, []);
  // effect for see more & see less card
  useEffect(() => {
    const cards = document.querySelectorAll('.card');

    cards.forEach((card) => {
      const seeMoreBtn = card.querySelector('.see-more-btn') as HTMLButtonElement;
      const textContent = card.querySelector('.card-content .text') as HTMLDivElement;

      const toggleCard = () => {
        card.classList.toggle('active');
        card.classList.toggle('gradient');

        if (card.classList.contains('active')) {
          seeMoreBtn.innerHTML = t('See Less');
          textContent.style.height = `${textContent.scrollHeight}px`;
        } else {
          seeMoreBtn.innerHTML = t('See More');
          textContent.style.height = '100px';
        }
      };

      seeMoreBtn.addEventListener('click', toggleCard);

      // Store the toggleCard function on the button element so we can remove it later
      (seeMoreBtn as any).toggleCard = toggleCard;

      if (textContent.scrollHeight <= 100) {
        seeMoreBtn.style.display = 'none';
        textContent.style.height = 'fit-content';
      } else {
        textContent.style.height = '100px';
        card.classList.add('gradient');
      }
    });

    const handleWindowLoad = () => {
      cards.forEach((card) => {
        const seeMoreBtn = card.querySelector('.see-more-btn') as HTMLButtonElement;
        const textContent = card.querySelector('.card-content .text') as HTMLDivElement;

        if (textContent.scrollHeight === textContent.clientHeight) {
          seeMoreBtn.style.display = 'none';
          textContent.style.height = 'fit-content';
        } else {
          card.classList.add('gradient');
        }
      });
    };

    window.addEventListener('load', handleWindowLoad);

    // Cleanup event listeners on unmount
    return () => {
      cards.forEach((card) => {
        const seeMoreBtn = card.querySelector('.see-more-btn') as HTMLButtonElement;
        seeMoreBtn.removeEventListener('click', (seeMoreBtn as any).toggleCard);
      });
      window.removeEventListener('load', handleWindowLoad);
    };
  }, []);

  return (
    <Layout>
      <div className="homepage-poster">
        <figure className="homepage-poster__image">
          <picture>
            <source srcSet={header_bg} media="(min-width: 1258px)" />
            <img alt="" src="https://getwallpapers.com/wallpaper/full/7/5/1/1016772-large-lion-wallpaper-1920x1200-computer.jpg" />
          </picture>
        </figure>
        <div className="homepage-news-large homepage-poster__content-container -left">
          <p className="homepage-poster__content">
            <a className="homepage-news-large__link" href="/">
              <span className="homepage-news-large__headline">
                {t('question_1_1')} <span className="text-danger">{t('threats')}</span> {t('question_1_2')}
              </span>
              <span className="homepage-news-large__subhead">{t('desc_1')}</span>
            </a>
          </p>
        </div>
      </div>

      <section className="homepage-news">
        <div className="homepage-news__header">
          <h2 className="homepage__heading mb-3">{t('nands')}</h2>
          <a className="homepage-button" href="/">
            {t('viewall')}
            <i className="fa-solid fa-angle-right ms-2"></i>
          </a>
        </div>
        <div className="homepage-news__featured">
          <div className="video-card-container">
            <div className="video-card">
              <div className="video-card-body">
                <div className="video-thumbnail">
                  <img src={polarbear} alt="" className="thumb-img" />
                  <video src={news_vid} className="thumb-video" muted loop></video>
                </div>
                <p className="video-card-title">{t('title_video')}</p>
                <div className="video-card-info">
                  <a href="/" className="user">
                    <i className="fa-regular fa-user me-2"></i>
                    CBS This Morning
                    <i className="fa-solid fa-circle-check ms-2"></i>
                  </a>
                  <span className="uploaded-time">
                    <i className="fa-regular fa-clock me-2"></i>5 {t('years')}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="video-modal-screen">
            <i className="fa-solid fa-xmark vid-modal-close-btn"></i>
            <div className="video-modal-content">
              <video src={news_vid} className="modal-video" controls autoPlay loop></video>
              <p className="modal-video-title">
                Million animals, plants at risk of extinction due to human activities, U.N report says
              </p>
              <div className="modal-video-info">
                <a href="/" className="user">
                  <i className="fa-regular fa-user me-2"></i>
                  CBS This Morning
                  <i className="fa-solid fa-circle-check ms-2"></i>
                </a>
                <span className="uploaded-time">
                  <i className="fa-regular fa-clock me-2"></i>5 years ago
                </span>
              </div>
            </div>
          </div>
          <div className="homepage-news__featured__content">
            <h3 className="homepage-news__featured__title">{t('question_2')}</h3>
            <p>
              {t('desc_2_1')} <span className="text-success fw-bold">{t('desc_2_2')}</span> {t('desc_2_3')}
            </p>
          </div>
        </div>
      </section>

      <section className="wildlife-threats">
        <div className="threats-title">
          <hr className="threats-line" />
          <h2>{t('desc_3')}</h2>
        </div>
        <div className="card-container">
          {[{
            img: habitatloss,
            title: 'Habitat Loss',
            text: t('des_habitatloss'),
          }, {
            img: climatechange,
            title: 'Climate Change',
            text: t('des_climatechange'),
          }, {
            img: polution,
            title: 'Pollution',
            text: t('des_pollution'),
          }].map((item, index) => (
            <div className="card" key={index}>
              <div className="card-img">
                <img src={item.img} alt="" />
              </div>
              <div className="card-content">
                <h2 className="reality-title">{t(item.title)}</h2>
                <p className="text">{item.text}</p>
                <span className="see-more-btn">{t('See More')}</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </Layout>
  );
}

export default Reality;
