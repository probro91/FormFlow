import React, { useState, useRef } from "react";
import RunVideo from "./RunVideo";
import { MdOutlineFileUpload } from "react-icons/md";
import CustomDropdown from "./CustomDropdown";
import colors from "../colors";

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

      // call backend here

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
      className={`flex-2 text-white rounded-xl transition-all duration-300 ease-in-out flex flex-col items-start text-left p-6 border-2 border-[#444444] justify-between hover:border-[#555555] hover:scale-101`}
      style={{ backgroundColor: colors.card1 }}
      onClick={() => setActivePanel(id)}
    >
      {/* Title and Upload Section */}
      <div className="w-full flex flex-col items-start">
        <div className="w-full">
          <h2 className="text-[#FF5733] font-montserrat font-bold mb-2">
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
            <div className="flex items-center gap-2 mb-4 justify-between w-full">
              <div className="flex items-center gap-4">
                <p className="text-[#cccccc] text-sm">Select a video file:</p>
                <div
                  onClick={handleIconClick}
                  className="cursor-pointer px-8 py-2 border border-[#aaa] rounded-xl hover:bg-gray-800 hover:border-[#fff]"
                >
                  <MdOutlineFileUpload size={20} color="#fff" />
                </div>
              </div>
              {videos.length > 1 && (
                <CustomDropdown videos={videos} setVideos={setVideos} />
              )}
            </div>
            {/* Video Preview */}
            {selectedFile && (
              <div className="w-full mb-4">
                <h3 className="text-[#cccccc] text-sm mb-2">Preview</h3>
                <video
                  key={URL.createObjectURL(selectedFile)}
                  width="30%"
                  controls
                  className="rounded-xl"
                >
                  <source
                    src={URL.createObjectURL(selectedFile)}
                    type="video/mp4"
                  />
                  Your browser does not support the video tag.
                </video>
              </div>
            )}
          </div>
        </div>

        {selectedFile && (
          <div className="flex items-center gap-4 w-full">
            <p className="text-[#cccccc] text-sm truncate">
              {selectedFile.name}
            </p>
            <div
              onClick={handleCancel}
              className="font-montserrat cursor-pointer text-[#FF5733] text-sm hover:text-[#e04e2d] transition-colors duration-300"
            >
              Cancel
            </div>
          </div>
        )}

        {selectedFile && (
          <div
            onClick={(e) => {
              e.stopPropagation();
              handleUpload();
            }}
            className={`mt-4 px-4 py-2 rounded-xl text-white font-montserrat cursor-pointer transition-colors duration-300 ${
              selectedFile
                ? "bg-[#FF5733] hover:bg-[#e04e2d]"
                : "bg-gray-500 cursor-not-allowed"
            }`}
          >
            Analyze Run
          </div>
        )}
      </div>

      {/* Most Recent Video */}
      <RunVideo videoUrl={videos.length > 0 ? videos[0] : null} />
    </div>
  );
};

export default SourcesPanel;
