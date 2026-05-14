import { Container } from 'react-bootstrap';
import { ReactNode } from 'react';
import Sidebar from '../admin/Sidebar';
import Header from '../admin/Header';

interface AdminLayoutProps {
  children: ReactNode;
}

function AdminLayout({ children }: AdminLayoutProps) {
  return (
    <div className="admin-layout d-flex" style={{ minHeight: '100vh' }}>
      <Sidebar />

      <main className="flex-grow-1 d-flex flex-column">
        <Header />  
        <Container fluid className="p-4 flex-grow-1">
          {children}
        </Container>
      </main>
    </div>
  );
}

export default AdminLayout
