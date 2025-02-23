import React, { useState, useEffect } from "react";
import { FaPlayCircle, FaRunning, FaFire } from "react-icons/fa";
import ReactPlayer from "react-player";

const RunVideo = ({ processedVideos }) => {
  const [selectedMode, setSelectedMode] = useState("heatmap");
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const videoSources = {
    heatmap: "/videos/heatmap.mp4",
    overlay: "/videos/overlay.mp4",
    skeleton: "/videos/skeleton.mp4",
    // Add more if needed (e.g., traced: "/videos/traced.mp4")
  };

  useEffect(() => {
    setLoading(true);
    setError(null);
    const source = videoSources[selectedMode] || videoSources.heatmap;

    // Simple check to ensure the file is accessible (optional)
    fetch(source)
      .then((response) => {
        if (!response.ok) throw new Error("Video not found");
        setVideoUrl(source);
      })
      .catch((err) => {
        console.error("Error fetching video:", err);
        setError("Error loading video");
      })
      .finally(() => setLoading(false));
  }, [selectedMode, processedVideos]);

  const handleModeChange = (mode) => {
    setSelectedMode(mode);
  };

  if (loading) {
    return <p className="text-[#cccccc] text-sm mt-4">Loading video...</p>;
  }

  if (error) {
    return <p className="text-[#cccccc] text-sm mt-4">{error}</p>;
  }

  if (!videoUrl) {
    return <p className="text-[#cccccc] text-sm mt-4">No video available</p>;
  }

  return (
    <div className="mt-4 w-full relative">
      <h3 className="text-[#cccccc] text-sm mb-2">Latest Run</h3>
      <div className="relative">
        <ReactPlayer url={videoUrl} controls width="100%" height="100%" />
        <div className="absolute top-0 right-0 flex gap-2 items-center justify-between p-2">
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
              title="Skeleton View"
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
