import React, { useState } from "react";
import {
  FaEye,
  FaEyeSlash,
  FaPlayCircle,
  FaRunning,
  FaFire,
} from "react-icons/fa";

const RunVideo = ({ videoUrl }) => {
  const [isBackgroundVisible, setIsBackgroundVisible] = useState(true);
  const [selectedMode, setSelectedMode] = useState("default"); // Default mode selected

  if (!videoUrl) {
    return (
      <p className="text-[#cccccc] text-sm mt-4">No video uploaded yet.</p>
    );
  }

  // Toggle background visibility
  const toggleBackground = () => setIsBackgroundVisible(!isBackgroundVisible);

  // Handle mode selection
  const handleModeChange = (mode) => setSelectedMode(mode);

  return (
    <div className="mt-4 w-full relative">
      <h3 className="text-[#cccccc] text-sm mb-2">Latest Run</h3>
      <div className="relative">
        <video
          width="100%"
          controls
          className={`rounded-xl ${
            isBackgroundVisible ? "bg-[#333333]" : "bg-transparent"
          }`}
          key={videoUrl}
        >
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>

        {/* Icon Buttons Container */}
        <div className="absolute top-0 left-0 flex gap-2 items-center w-full justify-between p-2">
          {/* Toggle Background Button */}
          <div
            onClick={(e) => {
              e.stopPropagation(); // Prevent video controls interference
              toggleBackground();
            }}
            className="p-2 bg-[#444444] rounded-full text-[#FF5733] hover:bg-[#555555] transition-colors duration-300"
            title={isBackgroundVisible ? "Hide Background" : "Show Background"}
          >
            {isBackgroundVisible ? (
              <FaEyeSlash size={16} className="text-[#FF5733]" />
            ) : (
              <FaEye size={16} className="text-white" />
            )}
          </div>

          {/* Mode Button Group */}
          <div className="flex gap-1 bg-[#444444] rounded-full border border-[#555555]">
            <div
              onClick={(e) => {
                e.stopPropagation();
                handleModeChange("default");
              }}
              className={`p-2 rounded-full ${
                selectedMode === "default"
                  ? "bg-[#FF5733] text-white"
                  : "bg-[#444444] text-[#cccccc] hover:bg-[#555555]"
              } transition-colors duration-300`}
              title="Default View"
            >
              <FaPlayCircle size={16} />
            </div>
            <div
              onClick={(e) => {
                e.stopPropagation();
                handleModeChange("running");
              }}
              className={`p-2 rounded-full ${
                selectedMode === "running"
                  ? "bg-[#FF5733] text-white"
                  : "bg-[#444444] text-[#cccccc] hover:bg-[#555555]"
              } transition-colors duration-300`}
              title="Running Overlay"
            >
              <FaRunning size={16} />
            </div>
            <div
              onClick={(e) => {
                e.stopPropagation();
                handleModeChange("heatmap");
              }}
              className={`p-2 rounded-full ${
                selectedMode === "heatmap"
                  ? "bg-[#FF5733] text-white"
                  : "bg-[#444444] text-[#cccccc] hover:bg-[#555555]"
              } transition-colors duration-300`}
              title="Heat Map"
            >
              <FaFire size={16} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RunVideo;
