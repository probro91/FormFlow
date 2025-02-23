import React, { useState, useEffect } from "react";
import {
  FaEye,
  FaEyeSlash,
  FaPlayCircle,
  FaRunning,
  FaFire,
} from "react-icons/fa";
import ReactPlayer from "react-player";

const RunVideo = ({ processedVideos }) => {
  console.log("PROCESSED VIDEOS:", processedVideos);
  const [selectedMode, setSelectedMode] = useState("skeleton"); // Default mode selected
  const [videoUrl, setVideoUrl] = useState(null);
  useEffect(() => {
    if (!processedVideos) return;

    let newUrl = null;
    Object.entries(processedVideos).forEach(([num, video]) => {
      if (video.includes(selectedMode)) {
        newUrl = video;
      }
    });
    console.log("SELECTED MODE:", selectedMode);
    console.log("NEW VIDEO:", newUrl);
    setVideoUrl(newUrl);
  }, [processedVideos, selectedMode]);
  // useEffect(() => {
  //   if (!processedVideos) return;

  //   let newUrl = null;
  //   for (const key in processedVideos) {
  //     if (key.includes(selectedMode)) {
  //       newUrl = URL.createObjectURL(processedVideos[key]);
  //       const a = document.createElement('a');
  //       a.href = newUrl;
  //       a.download = 'test.mp4';
  //       a.click();
  //       break;
  //     }
  //   }

  //   setVideoUrl(newUrl);
  // }, [processedVideos, selectedMode]);

  console.log("VIDEO URL:", videoUrl);

  if (!videoUrl) {
    return (
      <p className="text-[#cccccc] text-sm mt-4">No video uploaded yet.</p>
    );
  }
  // <video
  //         width="100%"
  //         controls
  //         className={`rounded-xl`}
  //         key={videoUrl}
  //       >
  //         <source key={videoUrl} src={videoUrl} type="video/mp4" />
  //         Your browser does not support the video tag.
  //       </video>

  // Handle mode selection
  const handleModeChange = (mode) => setSelectedMode(mode);

  return (
    <div className="mt-4 w-full relative">
      <h3 className="text-[#cccccc] text-sm mb-2">Latest Run</h3>
      <div className="relative">
        <ReactPlayer
          url={videoUrl}
          controls
          width="100%"
          height="360px"
        />

        {/* Icon Buttons Container */}
        <div className="absolute top-0 right-0 flex gap-2 items-center justify-between p-2">
          {/* Toggle Background Button */}

          {/* Mode Button Group */}
          <div className="flex gap-1 bg-[#444444] rounded-full border border-[#555555]">
            <div
              onClick={(e) => {
                e.stopPropagation();
                handleModeChange("skeleton");
              }}
              className={`p-2 rounded-full ${
                selectedMode === "skeleton"
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
                handleModeChange("overlay");
              }}
              className={`p-2 rounded-full ${
                selectedMode === "overlay"
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
