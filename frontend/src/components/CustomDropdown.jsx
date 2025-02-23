import React, { useState } from "react";
import { RiArrowDropDownLine } from "react-icons/ri";
import { FaHistory } from "react-icons/fa";

const CustomDropdown = ({ videos, onClick }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleDropdown = (e) => {
    e.stopPropagation(); // Prevent panel focus
    setIsOpen(!isOpen);
  };

  const handleOptionClick = (url, e) => {
    e.stopPropagation(); // Prevent panel focus
    if (url) window.open(url, "_blank", "noopener,noreferrer");
    setIsOpen(false); // Close dropdown after selection
  };

  return (
    <div className="relative w-24">
      {/* Toggle Button */}
      <div
        className="text-[#cccccc] text-xs p-2 rounded-xl cursor-pointer hover:border-[#FF5733] transition-colors duration-300 flex items-center justify-end"
        onClick={toggleDropdown}
      >
        <FaHistory
          size={18}
          color="#cccccc"
          className={`mr-1 ${
            isOpen ? "transform rotate-20" : ""
          } transition-transform duration-300`}
        />
      </div>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 w-full bg-[#444444] text-[#cccccc] text-xs rounded-xl border border-[#555555] mt-1 shadow-lg z-10 max-h-32 overflow-y-auto">
          {videos.slice(1).map((video, index) => (
            <div
              key={index}
              className="p-2 hover:bg-[#555555] cursor-pointer transition-colors duration-200"
              onClick={(e) => handleOptionClick(video, e)}
            >
              Video {videos.length - index - 1}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CustomDropdown;
