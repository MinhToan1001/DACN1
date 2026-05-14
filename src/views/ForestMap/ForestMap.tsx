import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';
import { Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import './ForestMap.css';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface Forest {
  id: number;
  name: string;
  lat: number | string;
  lng: number | string;
  square: number | null;
  description: string;
  info: string;
  image_url: string;
}

const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const MapEffects = ({ forest }: { forest: Forest | null }) => {
  const map = useMap();

  useEffect(() => {
    const resizeTimer = window.setTimeout(() => map.invalidateSize(), 300);

    if (forest) {
      map.flyTo([Number(forest.lat), Number(forest.lng)], 13, { duration: 1.5 });
    }

    return () => window.clearTimeout(resizeTimer);
  }, [forest, map]);

  return null;
};

const ForestsMap = () => {
  const [forests, setForests] = useState<Forest[]>([]);
  const [selectedForest, setSelectedForest] = useState<Forest | null>(null);
  const [loadError, setLoadError] = useState('');
  const markerRefs = useRef<Record<number, L.Marker>>({});
  const navigate = useNavigate();

  useEffect(() => {
    axios.get(`${apiBaseUrl}/api/forests-map`)
      .then((res) => {
        setForests(res.data);
        setLoadError('');
      })
      .catch((err) => {
        console.error('Error loading forests:', err);
        setLoadError('Không tải được dữ liệu khu bảo tồn. Hãy kiểm tra backend ở http://localhost:5001.');
      });
  }, []);

  const handleOpenModal = (forest: Forest) => {
    setSelectedForest(forest);
    markerRefs.current[forest.id]?.closePopup();
  };

  const validForests = forests.filter(
    (forest) => Number.isFinite(Number(forest.lat)) && Number.isFinite(Number(forest.lng))
  );

  return (
    <div style={{ height: '100vh', width: '100%', display: 'flex' }}>
      <div style={{ width: selectedForest ? '45%' : '100%', height: '100%', position: 'relative' }}>
        <button
          onClick={() => navigate('/')}
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            zIndex: 1000,
            padding: '8px 12px',
            backgroundColor: '#ffffff',
            border: '1px solid #ccc',
            borderRadius: '4px',
            cursor: 'pointer',
            boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
          }}
        >
          Quay lại
        </button>

        {loadError && <div className="map-error">{loadError}</div>}

        <MapContainer center={[16.5, 107.5]} zoom={6} style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <MapEffects forest={selectedForest} />
          {validForests.map((forest) => (
            <Marker
              key={forest.id}
              position={[Number(forest.lat), Number(forest.lng)]}
              ref={(ref) => {
                if (ref && ref instanceof L.Marker) {
                  markerRefs.current[forest.id] = ref;
                }
              }}
            >
              <Popup>
                <strong>{forest.name}</strong><br />
                <img
                  src={forest.image_url}
                  alt={forest.name}
                  style={{
                    width: '100%',
                    maxHeight: '120px',
                    objectFit: 'cover',
                    margin: '5px 0',
                  }}
                />
                <div>{forest.description}</div>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleOpenModal(forest)}
                  style={{ marginTop: '5px' }}
                >
                  Xem chi tiết
                </Button>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {selectedForest && (
        <div className="detail-panel show">
          <h3>{selectedForest.name}</h3>
          <img
            src={selectedForest.image_url}
            alt={selectedForest.name}
            style={{
              width: '100%',
              maxHeight: '300px',
              objectFit: 'cover',
              marginBottom: '15px',
            }}
          />
          <p><strong>{selectedForest.description}</strong></p>
          <div dangerouslySetInnerHTML={{ __html: selectedForest.info }} />
          <Button variant="secondary" onClick={() => setSelectedForest(null)}>
            Đóng
          </Button>
        </div>
      )}
    </div>
  );
};

export default ForestsMap;
