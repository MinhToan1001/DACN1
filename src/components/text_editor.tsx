// RichTextEditor.tsx
import React, { useMemo } from "react";
import ReactQuill, { Quill } from "react-quill";
import "react-quill/dist/quill.snow.css";

// Fix lỗi "Font is of type 'unknown'"
const Font = Quill.import("formats/font") as any;
Font.whitelist = ["arial", "times-new-roman", "verdana", "monospace"];
Quill.register(Font, true);

type Props = {
  value: string;
  onChange: (value: string) => void;
};

const TextEditor: React.FC<Props> = ({ value, onChange }) => {
  const modules = useMemo(
    () => ({
      toolbar: [
        [{ font: Font.whitelist }, { size: [] }],
        ["bold", "italic", "underline", "strike"],
        [{ color: [] }, { background: [] }],
        [{ script: "sub" }, { script: "super" }],
        [{ header: "1" }, { header: "2" }, "blockquote", "code-block"],
        [
          { list: "ordered" },
          { list: "bullet" },
          { indent: "-1" },
          { indent: "+1" }
        ],
        [{ direction: "rtl" }, { align: [] }],
        ["link", "image", "video"],
        ["clean"]
      ]
    }),
    []
  );

  return (
    <ReactQuill
      value={value}
      onChange={onChange}
      modules={modules}
      theme="snow"
    />
  );
};

export default TextEditor;
