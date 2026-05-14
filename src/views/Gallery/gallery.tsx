import './gallery.css';
import React, { useEffect, useState } from 'react';
import Layout from '../../layouts/Layout';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

function Gallery() {

  const { t } = useTranslation();
  const [animals, setAnimals] = useState<any[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;
  const [selectedLevel, setSelectedLevel] = useState('');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    axios.get('/api/animals')
      .then(res => setAnimals(res.data))
      .catch(err => console.error('Lỗi tải animals:', err));
  }, []);

  const handlePageChange = (page: number) => setCurrentPage(page);

  const filteredAnimals = animals.filter(animal => {
    const matchesLevel = selectedLevel ? animal.red_level === selectedLevel : true;
    const matchesSearch = animal.name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  const totalPages = Math.ceil(filteredAnimals.length / itemsPerPage);

  const currentAnimals = filteredAnimals.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <Layout>
      <section className="animals-gallery">
        <hr className="threats-line" />
        <h2 className="gallery-header">
          {t('animals')} <span className="text-success fw-lighter">{t('Gallery')}</span>
        </h2>
        <div className="search-container">
          <input
            type="text"
            placeholder="Search by animal name..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1); // quay về trang đầu khi search
            }}
          />
        </div>

        <div className="filter-container">
          <label htmlFor="filter">Filter by Red List Level:</label>
          <select
            id="filter"
            value={selectedLevel}
            onChange={(e) => setSelectedLevel(e.target.value)}
          >
            <option value="">All</option>
            <option value="EN">Endangered (EN)</option>
            <option value="VU">Vulnerable (VU)</option>
            <option value="R">Rare (R)</option>
            <option value="T">Threatened (T)</option>
            <option value="NT">Near Threatened (NT)</option>
          </select>
        </div>

        <div className="gallery pt-2">
          {currentAnimals.map((animal) => (
            <div className="gallery__item gallery__item--1" key={animal.id}>
              <a href={animal.link} className="gallery__link" target='_blank' rel="noreferrer">
                <img src={animal.image_url} className="gallery__image" alt={animal.name} />
                <div className="gallery__overlay">
                  <span>
                    {animal.name}
                    {animal.red_list ? ` 🔴 (${animal.red_level})` : ''}
                  </span>
                </div>
              </a>
            </div>
          ))}
        </div>

        <div className="pagination">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
            <button
              key={page}
              className={`page-button ${page === currentPage ? 'active' : ''}`}
              onClick={() => handlePageChange(page)}
            >
              {page}
            </button>
          ))}
        </div>
      </section>
    </Layout>
  );
}

export default Gallery;
