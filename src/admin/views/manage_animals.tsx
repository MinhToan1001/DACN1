import { useEffect, useState } from 'react';
import { Table, Badge, Spinner, Image, Button, Modal, Form } from 'react-bootstrap';
import AdminLayout from '../../layouts/AdminLayout';
import SearchBar from '../../components/search_bar';
import FilterSelect from '../../components/filter_select';

interface Animal {
  id?: number;
  name: string;
  image_url: string;
  link: string;
  red_list: boolean;
  red_level: string;
}

export default function ManageAnimal() {
  const [animals, setAnimals] = useState<Animal[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState<Animal>({ name: '', image_url: '', link: '', red_list: false, red_level: '' });
  const [editId, setEditId] = useState<number | null>(null);

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedLevel, setSelectedLevel] = useState("");

  const filteredAnimals = animals.filter((animal) => {
    const matchSearch = animal.name
      .toLowerCase()
      .includes(searchTerm.toLowerCase());
    const matchLevel = selectedLevel
      ? animal.red_level === selectedLevel
      : true;
    return matchSearch && matchLevel;
  });

  // Lấy dữ liệu
  const fetchAnimals = () => {
    setLoading(true);
    fetch('/api/animals')
      .then((res) => res.json())
      .then((data) => {
        setAnimals(data);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchAnimals();
  }, []);

  // Mở modal thêm
  const handleAdd = () => {
    setEditId(null);
    setForm({ name: '', image_url: '', link: '', red_list: false, red_level: '' });
    setShowModal(true);
  };

  // Mở modal sửa
  const handleEdit = (animal: Animal) => {
    setEditId(animal.id || null);
    setForm(animal);
    setShowModal(true);
  };

  // Lưu dữ liệu
  const handleSave = () => {
    const method = editId ? 'PUT' : 'POST';
    const url = editId ? `/api/animals/${editId}` : '/api/animals';

    const dataToSend = {
      ...form,
      red_list: form.red_list ? 1 : 0
    };

    fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dataToSend),
    })
      .then(res => {
        if (!res.ok) throw new Error('Lỗi khi lưu dữ liệu');
        return res.json();
      })
      .then(() => {
        setShowModal(false);
        fetchAnimals();
      })
      .catch(err => alert(err.message));
  };

  // Xóa
  const handleDelete = (id?: number) => {
    if (!id) return;
    if (window.confirm('Bạn có chắc muốn xóa?')) {
      fetch(`/api/animals/${id}`, { method: 'DELETE' }).then(() => {
        fetchAnimals();
      });
    }
  };

  return (
    <AdminLayout>
      <div>
        <h3 className="mb-4">Quản lý Động vật</h3>
        <SearchBar
          value={searchTerm}
          onChange={(val) => {
            setSearchTerm(val);
          }}
          placeholder="Search by animal name..."
          delay={400} // debounce 400ms
        />
        {/* Filter */}
        <FilterSelect
          label="Filter by Red List Level:"
          value={selectedLevel}
          onChange={(val) => setSelectedLevel(val)}
          options={[
            { value: "", label: "All" },
            { value: "EX", label: "Extinct (EX)" },
            { value: "EW", label: "Extinct in the Wild (EW)" },
            { value: "CR", label: "Critically Endangered (CR)" },
            { value: "EN", label: "Endangered (EN)" },
            { value: "VU", label: "Vulnerable (VU)" },
            { value: "NT", label: "Near-threatened (NT)" },
            { value: "LC", label: "Least concern (LC)" },
          ]}
        />
        <Button variant="success" className="mb-3" onClick={handleAdd}>
          + Thêm mới
        </Button>

        {loading ? (
          <Spinner animation="border" />
        ) : (
          <Table striped bordered hover responsive>
            <thead>
              <tr>
                <th>Ảnh</th>
                <th>Tên loài</th>
                <th>Sách đỏ</th>
                <th>Mức độ nguy cấp</th>
                <th>Wikipedia</th>
                <th>Hành động</th>
              </tr>
            </thead>
            <tbody>
              {filteredAnimals.map((animal) => (
                <tr key={animal.id}>
                  <td>
                    <Image src={animal.image_url} alt={animal.name} width={80} height={60} rounded />
                  </td>
                  <td>{animal.name}</td>
                  <td>
                    {animal.red_list ? <Badge bg="danger">Có</Badge> : <Badge bg="secondary">Không</Badge>}
                  </td>
                  <td>{animal.red_level}</td>
                  <td>
                    <a href={animal.link} target="_blank" rel="noopener noreferrer">
                      Xem
                    </a>
                  </td>
                  <td>
                    <Button size="sm" variant="primary" onClick={() => handleEdit(animal)}>
                      Sửa
                    </Button>{' '}
                    <Button size="sm" variant="danger" onClick={() => handleDelete(animal.id)}>
                      Xóa
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}

        {/* Modal thêm/sửa */}
        <Modal show={showModal} onHide={() => setShowModal(false)}>
          <Modal.Header closeButton>
            <Modal.Title>{editId ? 'Sửa' : 'Thêm mới'} Động vật</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Tên loài</Form.Label>
                <Form.Control
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Label>Ảnh URL</Form.Label>
                <Form.Control
                  value={form.image_url}
                  onChange={(e) => setForm({ ...form, image_url: e.target.value })}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Label>Wikipedia Link</Form.Label>
                <Form.Control
                  value={form.link}
                  onChange={(e) => setForm({ ...form, link: e.target.value })}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Check
                  type="checkbox"
                  label="Thuộc Sách đỏ"
                  checked={form.red_list}
                  onChange={(e) => setForm({ ...form, red_list: e.target.checked })}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Label>Mức độ nguy cấp</Form.Label>
                <Form.Control
                  value={form.red_level}
                  onChange={(e) => setForm({ ...form, red_level: e.target.value })}
                />
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowModal(false)}>
              Hủy
            </Button>
            <Button variant="primary" onClick={handleSave}>
              Lưu
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    </AdminLayout>
  );
}
