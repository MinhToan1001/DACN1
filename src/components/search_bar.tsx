import React, { useState, useEffect } from "react";

interface SearchBarProps {
  value: string; // giá trị thực tế ở parent
  onChange: (value: string) => void; // callback khi search term thay đổi
  placeholder?: string;
  delay?: number; // thời gian debounce (ms)
}

const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChange,
  placeholder,
  delay = 300, // mặc định 300ms
}) => {
  const [internalValue, setInternalValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      onChange(internalValue); // gọi callback sau khi ngừng gõ
    }, delay);

    return () => {
      clearTimeout(handler); // hủy nếu user vẫn đang gõ
    };
  }, [internalValue, delay, onChange]);

  useEffect(() => {
    setInternalValue(value); // đồng bộ khi value từ ngoài thay đổi
  }, [value]);

  return (
    <div className="search-container" style={{ marginBottom: "16px" }}>
      <input
        type="text"
        placeholder={placeholder || "Search..."}
        value={internalValue}
        onChange={(e) => setInternalValue(e.target.value)}
        style={{
          padding: "8px",
          borderRadius: "4px",
          border: "1px solid #ccc",
          width: "250px",
        }}
      />
    </div>
  );
};

export default SearchBar;
