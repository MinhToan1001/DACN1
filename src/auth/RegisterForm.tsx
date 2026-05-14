import { ChangeEvent, FormEvent, useState } from 'react';

interface Props {
  switchToLogin: () => void;
}

export default function RegisterForm({ switchToLogin }: Props) {
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });
  const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    try {
      const res = await fetch(`${apiBaseUrl}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await res.json();

      if (res.ok) {
        alert(data.message);
        switchToLogin();
      } else {
        alert(data.message || 'Đăng ký thất bại.');
      }
    } catch (error) {
      console.error('Register error:', error);
      alert('Không kết nối được server đăng ký. Hãy kiểm tra backend ở http://localhost:5001.');
    }
  };

  return (
    <div>
      <h5 className="text-center">Đăng ký</h5>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label>Tên</label>
          <input type="text" name="name" className="form-control" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label>Email</label>
          <input type="email" name="email" className="form-control" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label>Mật khẩu</label>
          <input type="password" name="password" className="form-control" onChange={handleChange} required />
        </div>
        <button className="btn btn-success w-100" type="submit">Đăng ký</button>
      </form>
      <p className="mt-3 text-center" style={{ cursor: 'pointer' }} onClick={switchToLogin}>
        Đã có tài khoản? Đăng nhập
      </p>
    </div>
  );
}
