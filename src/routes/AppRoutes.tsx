import { Routes, Route } from 'react-router-dom';

import HomePage from '../views/HomePage/homepage';
import About from '../views/About/about';
import Reality from '../views/Reality/reality';
import Gallery from '../views/Gallery/gallery';
import Donation from '../views/Donation/donation';
import VnpayReturn from '../VNpayReturn';
import NewsDetail from '../views/NewsDetail/NewsDetail';
import Project from '../views/Projects/project';
import ForestsMap from '../views/ForestMap/ForestMap';
import AILookup from '../views/AILookup/AILookup';
import OCRExtraction from '../views/OCRExtraction/OCRExtraction';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/about-us" element={<About />} />
      <Route path="/reality" element={<Reality />} />
      <Route path="/gallery" element={<Gallery />} />
      <Route path="/donate" element={<Donation />} />
      <Route path="/news/:id" element={<NewsDetail />} />
      <Route path="/projects/:id" element={<Project />} />
      <Route path="/forests-map" element={<ForestsMap />} />
      <Route path="/ai-lookup" element={<AILookup />} />
      <Route path="/ocr" element={<OCRExtraction />} />

      <Route path="/vnpay_return" element={<VnpayReturn />} />
    </Routes>
  );
};

export default AppRoutes;
