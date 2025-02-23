import React, { useState } from "react";
import SourcePreview from "./SourcePreview";

const Source = ({ imageUrl, title, url }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isTopRegion, setIsTopRegion] = useState(false);

  const handleMouseEnter = (e) => {
    setIsHovered(true);
    setIsTopRegion(e.clientY < 400); // Check if cursor is in top 200px
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setIsTopRegion(false); // Reset on leave
  };

  return (
    <div
      className="relative inline-block cursor-pointer"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={(e) => {
        e.stopPropagation(); // Prevent panel focus
        window.open(url, "_blank");
      }}
    >
      <img
        src={imageUrl}
        alt={title}
        className="w-6 h-6 rounded-full object-cover  hover:border-[#e04e2d] transition-colors duration-300"
        onError={(e) => {
          e.target.src = "https://placehold.co/400";
        }}
      />
      {isHovered && (
        <SourcePreview
          title={title}
          url={url}
          imageUrl={imageUrl}
          isBelow={isTopRegion} // Pass flag to position below if in top region
        />
      )}
    </div>
  );
};

export default Source;
