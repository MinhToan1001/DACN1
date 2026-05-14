import { useEffect, useState } from 'react';
import { Nav } from 'react-bootstrap';
import { NavLink } from 'react-router-dom';
import sidebarMenu from '../menu-item.json';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faChartLine, faInbox,
  faPaw, faTree,
  faNewspaper, faBriefcase,
  faCircleDollarToSlot, faRightFromBracket,
  faChevronUp, faChevronRight
} from '@fortawesome/free-solid-svg-icons';

interface MenuItem {
  section: string;
  subtitle?: string;
  items: {
    label: string;
    href?: string;
    icon?: string;
    disabled?: boolean;
    children?: {
      label: string;
      href: string;
      disabled?: boolean;
    }[];
  }[];
}

const iconMap: { [key: string]: any } = {
  dashboard: faChartLine,
  inbox: faInbox,
  paw: faPaw,
  forest: faTree,
  news: faNewspaper,
  project: faBriefcase,
  fund: faCircleDollarToSlot,
  logout: faRightFromBracket
};

export default function Sidebar() {
  const [menuSections, setMenuSections] = useState<MenuItem[]>([]);
  const [openMenus, setOpenMenus] = useState<{ [key: string]: boolean }>({});

  useEffect(() => {
    setMenuSections(sidebarMenu);
  }, []);

  const toggleMenu = (key: string) => {
    setOpenMenus((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <aside
      className="bg-dark text-white p-3"
      style={{ width: '300px', height: '100vh',}}
    >
      <h4 className="text-white mb-4">kaiadmin</h4>

      <Nav className="flex-column">
        {menuSections.map((section, secIndex) => (
          <div key={secIndex} className="mb-4">
            <div className="text-uppercase text-primary small fw-bold mb-1">{section.section}</div>
            {section.subtitle && (
              <div className="mb-2" style={{ fontSize: '12px' }}>
                {section.subtitle}
              </div>
            )}

            {section.items.map((item, idx) => {
              const key = `${secIndex}-${idx}`;
              const hasChildren = !!item.children?.length;
              const IconComponent = item.icon && iconMap[item.icon];

              return (
                <div key={key}>
                  {hasChildren ? (
                    // Menu cha có toggle
                    <Nav.Link
                      onClick={() => toggleMenu(key)}
                      className={`d-flex justify-content-between align-items-center text-white px-2 py-1 ${item.disabled ? 'disabled' : ''}`}
                      style={{
                        fontWeight: 500,
                        cursor: item.disabled ? 'not-allowed' : 'pointer',
                        opacity: item.disabled ? 0.5 : 1,
                      }}
                    >
                      <div className="d-flex align-items-center gap-2">
                        {IconComponent && <FontAwesomeIcon icon={IconComponent} className="me-2" />}
                        {item.label}
                      </div>
                      <FontAwesomeIcon
                        icon={openMenus[key] ? faChevronUp : faChevronRight}
                        className="ms-2"
                      />
                    </Nav.Link>
                  ) : (
                    // Menu con dẫn link
                    <Nav.Link
                      as={NavLink}
                      to={item.href || '#'}
                      className={`d-flex align-items-center text-white px-2 py-1 ${item.disabled ? 'disabled' : ''}`}
                      style={{
                        fontWeight: 500,
                        cursor: item.disabled ? 'not-allowed' : 'pointer',
                        opacity: item.disabled ? 0.5 : 1,
                      }}
                    >
                      {IconComponent && <FontAwesomeIcon icon={IconComponent} className="me-2" />}
                      {item.label}
                    </Nav.Link>
                  )}

                  {/* Submenu */}
                  {hasChildren && openMenus[key] && (
                    <div className="ms-4">
                      {item.children?.map((sub, subIdx) => (
                        <Nav.Link
                          key={subIdx}
                          as={NavLink}
                          to={!sub.disabled ? sub.href : '#'}
                          className={`text-white px-2 py-1 ${sub.disabled ? 'disabled' : ''}`}
                          style={{
                            fontSize: '0.9rem',
                            cursor: sub.disabled ? 'not-allowed' : 'pointer',
                            opacity: sub.disabled ? 0.5 : 1,
                          }}
                        >
                          {sub.label}
                        </Nav.Link>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </Nav>
    </aside>
  );
}
