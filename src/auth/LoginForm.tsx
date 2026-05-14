import { ChangeEvent, FormEvent, useState } from 'react';
import { saveUser } from '../utils/authStorage';

interface Props {
  switchToRegister: () => void;
  onLoginSuccess: (user: any) => void;
}

export default function LoginForm({ switchToRegister, onLoginSuccess }: Props) {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const apiBaseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    try {
      const res = await fetch(`${apiBaseUrl}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await res.json();

      if (res.ok) {
        alert(data.message);
        saveUser(data.user);
        onLoginSuccess(data.user);
      } else {
        alert(data.message || 'Đăng nhập thất bại.');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('Không kết nối được server đăng nhập. Hãy kiểm tra backend ở http://localhost:5001.');
    }
  };

  return (
    <div>
      <h5 className="text-center">Đăng nhập</h5>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label>Email</label>
          <input type="email" name="email" className="form-control" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label>Mật khẩu</label>
          <input type="password" name="password" className="form-control" onChange={handleChange} required />
        </div>
        <button className="btn btn-primary w-100" type="submit">Đăng nhập</button>
      </form>
      <p className="mt-3 text-center" style={{ cursor: 'pointer' }} onClick={switchToRegister}>
        Chưa có tài khoản? Đăng ký
      </p>
    </div>
  );
}
