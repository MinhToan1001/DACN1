import React from "react";

interface Option {
  value: string;
  label: string;
}

interface FilterSelectProps {
  label?: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
}

const FilterSelect: React.FC<FilterSelectProps> = ({
  label,
  options,
  value,
  onChange,
}) => {
  return (
    <div className="filter-container" style={{ marginBottom: "16px" }}>
      {label && (
        <label htmlFor="filter" style={{ marginRight: "8px" }}>
          {label}
        </label>
      )}
      <select
        id="filter"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          padding: "6px 8px",
          borderRadius: "4px",
          border: "1px solid #ccc",
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default FilterSelect;
