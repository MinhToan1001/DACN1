// Lưu user vào localStorage
export const saveUser = (user: any) => {
  localStorage.setItem('user', JSON.stringify(user));
};

// Lấy user từ localStorage
export const getUser = () => {
  const data = localStorage.getItem('user');
  return data ? JSON.parse(data) : null;
};

// Xoá user khỏi localStorage
export const clearUser = () => {
  localStorage.removeItem('user');
};
