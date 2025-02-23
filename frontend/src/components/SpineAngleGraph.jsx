import React from "react";

const SpineAngleGraph = ({ angle = 7 }) => {
  // Graph dimensions
  const width = 100;
  const height = 100;
  const centerX = width / 4;
  const centerY = (height * 3) / 4;
  const radius = 70;

  // Convert angle to radians
  const angleRad = (270 - angle * Math.PI) / 180;
  const endX = centerX + radius * Math.cos(angleRad);
  const endY = centerY - radius * Math.sin(angleRad); // Upward direction

  return (
    <div className="flex items-center justify-center gap-2">
      <svg width={width} height={height} className="bg-card1 rounded-lg p-2 ">
        {/* Reference Line (0°) */}
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + radius}
          y2={centerY}
          className="stroke-[#555555] stroke-1 stroke-dashed"
        />
        {/* Angle Arc */}
        <path
          d={`M ${centerX},${centerY} A ${radius},${radius} 0 0 1 ${endX},${endY}`}
          className="stroke-[#FF5733] stroke-2 fill-none"
        />
        {/* Angle Line */}
        <line
          x1={centerX}
          y1={centerY}
          x2={endX}
          y2={endY}
          className="stroke-[#FF5733] stroke-2"
        />
      </svg>
      <p className="text-[#FF5733] text-lg font-bold ml-2">{angle}°</p>
    </div>
  );
};

export default SpineAngleGraph;
