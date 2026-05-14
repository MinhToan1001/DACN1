import { Container, Row, Col, Card, Button } from 'react-bootstrap';
import { faUser, faUserCheck, faShoppingCart, faClipboardList } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import AdminLayout from '../../layouts/AdminLayout';


const AdminDashboard = () => {
  return (
    <AdminLayout>
      <Container fluid className="p-4 bg-light">
      <Row className="mb-4">
        <Col>
          <h3 className="fw-bold">Dashboard</h3>
          <p className="text-muted">Free Bootstrap 5 Admin Dashboard</p>
        </Col>
        <Col className="text-end">
          <Button variant="outline-primary" className="me-2">Manage</Button>
          <Button variant="primary">Add Customer</Button>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={3}>
          <Card className="shadow-sm border-0 p-3">
            <div className="d-flex align-items-center">
              <FontAwesomeIcon className='icon' icon={faUser} />
              <div>
                <small className="text-muted">Visitors</small>
                <h5 className="mb-0">1,294</h5>
              </div>
            </div>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="shadow-sm border-0 p-3">
            <div className="d-flex align-items-center">
              <FontAwesomeIcon className='icon' icon={faUserCheck} />
              <div>
                <small className="text-muted">Subscribers</small>
                <h5 className="mb-0">1,303</h5>
              </div>
            </div>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="shadow-sm border-0 p-3">
            <div className="d-flex align-items-center">
              <FontAwesomeIcon className='icon' icon={faShoppingCart} />
              <div>
                <small className="text-muted">Sales</small>
                <h5 className="mb-0">$1,345</h5>
              </div>
            </div>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="shadow-sm border-0 p-3">
            <div className="d-flex align-items-center">
              <FontAwesomeIcon className='icon' icon={faClipboardList} />
              <div>
                <small className="text-muted">Order</small>
                <h5 className="mb-0">576</h5>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={8}>
          <Card className="shadow-sm border-0 p-3">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h5 className="mb-0">User Statistics</h5>
              <div>
                <Button variant="outline-success" size="sm" className="me-2">Export</Button>
                <Button variant="outline-primary" size="sm">Print</Button>
              </div>
            </div>
            <div style={{ height: '250px', backgroundColor: '#f1f1f1' }}>
              {/* Chart Placeholder */}
              <p className="text-center mt-5">[Chart Here]</p>
            </div>
          </Card>
        </Col>

        <Col md={4}>
          <Card className="shadow-sm border-0 p-3 bg-primary text-white">
            <div className="d-flex justify-content-between mb-2">
              <h6>Daily Sales</h6>
              <Button variant="light" size="sm">Export</Button>
            </div>
            <p className="mb-1">March 25 - April 02</p>
            <h4 className="fw-bold">$4,578.58</h4>
            <div style={{ height: '100px', backgroundColor: '#ffffff33' }}>
              {/* Chart Placeholder */}
              <p className="text-center mt-4">[Line Chart]</p>
            </div>
            <div className="mt-3 d-flex justify-content-between">
              <p className="mb-0">17 Users online</p>
              <p className="mb-0">+5%</p>
            </div>
          </Card>
        </Col>
      </Row>
    </Container>
    </AdminLayout>
  );
};

export default AdminDashboard;
