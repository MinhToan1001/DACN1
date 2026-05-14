import { clearUser } from '../utils/authStorage';

interface Props {
  user: any;
  onLogout: () => void;
}

export default function UserMenu({ user, onLogout }: Props) {
  const handleLogout = () => {
    clearUser();
    onLogout();
  };

  return (
    <div>
      <p><strong>{user.name}</strong></p>
      <p>{user.email}</p>
      <button className="btn btn-info w-100 mb-2">Thông tin cá nhân</button>
      <button className="btn btn-danger w-100" onClick={handleLogout}>Đăng xuất</button>
    </div>
  );
}
