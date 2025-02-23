import React from "react";

const SourcePreview = ({ title, url, imageUrl, isBelow }) => {
  return (
    <div
      className={`absolute left-1/2 transform -translate-x-1/2 bg-[#333333] text-white rounded-xl p-3 shadow-lg border border-[#444444] w-72 z-10 text-left ${
        isBelow ? "top-14" : "bottom-14"
      }`}
    >
      <p className="text-sm font-montserrat font-semibold mb-1">{title}</p>
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-[#FF5733] text-xs underline hover:text-[#e04e2d] transition-colors duration-300 break-words"
      >
        {url}
      </a>
      <img
        src={imageUrl}
        alt="Source"
        className="w-full h-40 object-cover rounded-xl mt-2"
      />
    </div>
  );
};

export default SourcePreview;
