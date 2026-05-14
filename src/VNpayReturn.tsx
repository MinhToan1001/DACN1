import React, { useEffect, useState } from 'react';
import { Container, Nav, Navbar, Offcanvas } from 'react-bootstrap';
import { NavLink, useLocation } from 'react-router-dom';

import './VnpayReturn.css'

import logo from './assets/image/logo.png';

interface PaymentData {
    vnp_TxnRef: string;
    vnp_Amount: number;
}

const VnpayReturn = () => {
    const location = useLocation();
    const [message, setMessage] = useState('');
    const [data, setData] = useState<PaymentData | null>(null);

    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const responseCode = params.get('vnp_ResponseCode');

        if (responseCode === '00') {
            setMessage('Giao dịch thành công');
            const paymentData: PaymentData = {
                vnp_TxnRef: params.get('vnp_TxnRef') || '',
                vnp_Amount: parseInt(params.get('vnp_Amount') || '0', 10),
            };
            setData(paymentData);
        } else {
            setMessage('Giao dịch không thành công');
        }
    }, [location.search]);

    const [isScrolledPast, setIsScrolledPast] = useState(false);

    // Effect for header
    useEffect(() => {
        function handleScroll() {
            const isPast = window.scrollY > 0;
            setIsScrolledPast(isPast);
        }

        window.addEventListener('scroll', handleScroll);

        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);
    return (
        <div>
            {/* Header*/}
            {['lg'].map((expand) => (
                <Navbar className={`fixed-top ${isScrolledPast ? 'scrolled-past' : ''}`} key={expand} expand={expand}>
                    <Container>
                        <Navbar.Brand className='logo' href="#">
                            <img src={logo} alt="" />
                            <div className='brand-name'>GREEN LAND</div>
                        </Navbar.Brand>
                        <Navbar.Toggle aria-controls={`offcanvasNavbar-expand-${expand}`} />
                        <Navbar.Offcanvas
                            id={`offcanvasNavbar-expand-${expand}`}
                            aria-labelledby={`offcanvasNavbarLabel-expand-${expand}`}
                            placement="end"
                        >
                            <Offcanvas.Header closeButton>
                                <Offcanvas.Title className='logo' id={`offcanvasNavbarLabel-expand-${expand}`}>
                                    <img src={logo} alt="" />
                                    <div className='brand-name'>GREEN LAND</div>
                                </Offcanvas.Title>
                            </Offcanvas.Header>
                            <Offcanvas.Body>
                                <Nav className="justify-content-end flex-grow-1">
                                    <Nav.Link as={NavLink} to="/home">Home</Nav.Link>
                                    <Nav.Link as={NavLink} to="/about-us" >About us</Nav.Link>
                                    <Nav.Link as={NavLink} to="/reality">Reality</Nav.Link>
                                    <Nav.Link as={NavLink} to="/contact">Contact</Nav.Link>
                                    <Nav.Link as={NavLink} to="/donate">
                                        <button className='button button-left'>Donate</button>
                                    </Nav.Link>
                                </Nav>
                            </Offcanvas.Body>
                        </Navbar.Offcanvas>
                    </Container>
                </Navbar>
            ))}
            {/* End Header*/}
            {/* Main */}
            <div id="content">
                <Container>
                    <h2>KẾT QUẢ GIAO DỊCH</h2>
                    <p>{message}</p>
                    {data && (
                        <div>
                            <p>Mã giao dịch: {data.vnp_TxnRef}</p>
                            <p>Số tiền: {data.vnp_Amount / 100} VND</p>
                        </div>
                    )}
                    <Nav.Link as={NavLink} to='/donate'>
                        <button className='button button-left'>
                            Back to Donate Page
                        </button>
                    </Nav.Link>
                </Container>
            </div>

        </div>
    );
};

export default VnpayReturn;