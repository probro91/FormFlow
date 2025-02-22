import React, { useState, useRef } from "react";
import RunVideo from "./RunVideo";
import { MdOutlineFileUpload } from "react-icons/md";

const SourcesPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  videos,
  setVideos,
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null); // Ref to trigger hidden file input

  console.log("SourcesPanel videos:", videos);

  // Handle file selection and preview
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("video/")) {
      setSelectedFile(file);
    } else {
      alert("Please upload a valid video file.");
      setSelectedFile(null); // Reset if invalid
    }
  };

  // Handle video upload
  const handleUpload = () => {
    if (selectedFile) {
      const videoUrl = URL.createObjectURL(selectedFile);
      console.log("Uploading videoUrl:", videoUrl);
      setVideos([videoUrl, ...videos]); // Add to videos state
      setSelectedFile(null); // Clear preview
    }
  };

  // Handle cancel preview
  const handleCancel = (e) => {
    e.stopPropagation(); // Prevent panel focus
    setSelectedFile(null); // Clear selected file
  };

  // Trigger file input click when icon is clicked
  const handleIconClick = (e) => {
    e.stopPropagation(); // Prevent panel focus
    fileInputRef.current.click();
  };

  return (
    <div
      className={`flex-2 bg-[#333333] text-white rounded-lg transition-all duration-300 ease-in-out cursor-pointer flex flex-col items-center text-center p-6 border-2 border-[#444444] ${
        activePanel === id ? "scale-101 border-1 border-[#fff]" : ""
      }`}
      onClick={() => setActivePanel(id)}
    >
      {/* Title and Upload Section */}
      <div className="w-full flex flex-col items-start">
        <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold">
          {title}
        </h2>
        <div className="flex flex-col w-full">
          {/* Hidden File Input */}
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            className="hidden"
          />
          {/* Upload Icon and Buttons */}
          <div className="flex items-center gap-2 mb-4">
            <button onClick={handleIconClick}>
              <MdOutlineFileUpload size={20} color="#fff" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleUpload();
              }}
              disabled={!selectedFile}
              className={`px-4 py-2 rounded-lg text-white font-montserrat ${
                selectedFile
                  ? "bg-[#FF5733] hover:bg-[#e04e2d]"
                  : "bg-gray-500 cursor-not-allowed"
              }`}
            >
              Upload Video
            </button>
            {selectedFile && (
              <button
                onClick={handleCancel}
                className="px-4 py-2 rounded-lg text-white font-montserrat bg-gray-600 hover:bg-gray-700"
              >
                Cancel
              </button>
            )}
          </div>
          {/* Video Preview */}
          {selectedFile ? (
            <div className="w-full mb-4">
              <h3 className="text-[#cccccc] text-sm mb-2">Preview</h3>
              <video
                key={URL.createObjectURL(selectedFile)}
                width="100%"
                controls
                className="rounded-md"
              >
                <source
                  src={URL.createObjectURL(selectedFile)}
                  type="video/mp4"
                />
                Your browser does not support the video tag.
              </video>
            </div>
          ) : (
            <p className="text-[#cccccc] text-sm mb-4">
              No video selected for preview
            </p>
          )}
        </div>
      </div>

      {/* Previous Uploads Dropdown */}
      {videos.length > 1 && (
        <div className="mt-4 w-full">
          <h3 className="text-[#cccccc] text-sm mb-2">Previous Uploads</h3>
          <select
            className="w-full bg-[#444444] text-[#cccccc] text-xs p-2 rounded-md border border-[#555555] focus:outline-none focus:border-[#FF5733]"
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const url = e.target.value;
              if (url) window.open(url, "_blank", "noopener,noreferrer");
            }}
          >
            <option value="">Select a previous video</option>
            {videos.slice(1).map((video, index) => (
              <option key={index} value={video}>
                Video {videos.length - index - 1}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Most Recent Video */}
      <RunVideo videoUrl={videos.length > 0 ? videos[0] : null} />
    </div>
  );
};

export default SourcesPanel;
