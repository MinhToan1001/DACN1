import { Routes, Route } from 'react-router-dom';

import AdminDashboard from '../admin/views/dashboard';
import ManageAnimals from '../admin/views/manage_animals';
import ManageForests from '../admin/views/manage_forests';
import ManageNews from '../admin/views/manage_news';
import ManageProjects from '../admin/views/manage_projects';
import ManageContacts from '../admin/views/manage_contacts';

const AdminRoutes = () => {
  return (
    <Routes>
      <Route path="/admin/dashboard" element={<AdminDashboard />} />
      <Route path="/admin/manage-animals" element={<ManageAnimals />} />
      <Route path="/admin/manage-forests" element={<ManageForests />} />
      <Route path="/admin/manage-news" element={<ManageNews />} />
      <Route path="/admin/manage-projects" element={<ManageProjects/>} />
      <Route path="/admin/manage-contacts" element={<ManageContacts/>} />
    </Routes>
  );
};

export default AdminRoutes;
