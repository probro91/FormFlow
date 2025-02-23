import React, { useState } from "react";
import {
  FaEye,
  FaEyeSlash,
  FaPlayCircle,
  FaRunning,
  FaFire,
} from "react-icons/fa";

const RunVideo = ({ processedVideos }) => {
  console.log(processedVideos);
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
        <div className="absolute top-0 right-0 flex gap-2 items-center justify-between p-2">
          {/* Toggle Background Button */}

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
