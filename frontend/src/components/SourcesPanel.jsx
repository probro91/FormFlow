import React, { useState, useRef } from "react";
import RunVideo from "./RunVideo";
import { MdOutlineFileUpload } from "react-icons/md";
import CustomDropdown from "./CustomDropdown";
import colors from "../colors";
import axios from "axios";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";

const s3Client = new S3Client({
  region: import.meta.env.AWS_REGION,
  credentials: {
    accessKeyId: import.meta.env.AWS_ACCESS_KEY,
    secretAccessKey: import.meta.env.AWS_SECRET_KEY,
  },
});

const SourcesPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  videos,
  setVideos,
}) => {
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null); // Ref to trigger hidden file input
  const [processedVideos, setProcessedVideos] = useState({});

  const videoFiles = {
    heatmap: "initalvids/tmp8rm4sxe_.mp4_heatmap.mp4",
    overlay: "initalvids/tmp8rm4sxe_.mp4_overlay.mp4",
    skeleton: "initalvids/tmp8rm4sxe_.mp4_skeleton.mp4",
  };

  const fetchVideos = async () => {
    try {
      const videoPromises = Object.entries(videoFiles).map(
        async ([key, keyPath]) => {
          const params = {
            Bucket: "formflow-videos",
            Key: keyPath,
          };

          const command = new GetObjectCommand(params);
          const { Body } = await s3Client.send(command);
          const blob = await streamToBlob(Body, "video/mp4");

          return { key, blob };
        }
      );

      const results = await Promise.all(videoPromises); // Store the videos in state

      setProcessedVideos(
        results.reduce((acc, { key, blob }) => {
          acc[key] = blob;
          return acc;
        }, {})
      );
    } catch (error) {
      console.error("Error fetching videos:", error);
    }
  }; // Convert ReadableStream to Blob

  const streamToBlob = async (stream, mimeType) => {
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    return new Blob(chunks, { type: mimeType });
  };

  console.log("SourcesPanel videos:", videos); // Handle file selection and preview

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("video/")) {
      setSelectedFile(file);
    } else {
      alert("Please upload a valid video file.");
      setSelectedFile(null); // Reset if invalid
    }
  }; // Handle video upload

  async function handleUpload() {
    console.log("Uploading video...");
    if (!selectedFile) {
      console.error("No file selected");
      return;
    }
    const videoUrl = URL.createObjectURL(selectedFile);
    console.log("Created object URL:", videoUrl);
    try {
      const formData = new FormData();
      formData.append("video", selectedFile);
      console.log("FormData created");
      const response = await axios.post(
        "http://localhost:5001/analyze",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      console.log("Upload successful:", response.data); // Ensure we're using the latest state
      setVideos((prevVideos) => [videoUrl, ...prevVideos]);
      setSelectedFile(null); // Clear after upload completes
      fetchVideos();

      console.log("the processed vidoes" + processedVideos);
    } catch (error) {
      console.error("Error uploading video:", error);
    } // Cleanup the created object URL when the component unmounts
    useEffect(() => {
      return () => {
        URL.revokeObjectURL(videoUrl);
      };
    }, [videoUrl]);
    console.log("Upload process complete");
  } // Handle cancel preview

  const handleCancel = (e) => {
    e.stopPropagation(); // Prevent panel focus
    setSelectedFile(null); // Clear selected file
  }; // Trigger file input click when icon is clicked

  const handleIconClick = (e) => {
    e.stopPropagation(); // Prevent panel focus
    fileInputRef.current.click();
  };

  return (
    <div
      className={`flex-2 text-white rounded-xl transition-all duration-300 ease-in-out flex flex-col items-start text-left p-6 border-[#444444] justify-between hover:border-[#555555] hover:scale-101 relative`}
      style={{ backgroundColor: colors.card1 }}
      onClick={() => setActivePanel(id)}
    >
            {/* Spinner Overlay */}
            
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center rounded-xl z-10">
                    
          <Rings
            height="80"
            width="80"
            radius="9"
            color="white" // Orange accent
            ariaLabel="audio-loading"
            wrapperStyle={{}}
            wrapperClass=""
            visible={true}
          />
                  
        </div>
      )}
            {/* Title and Upload Section */}
            
      <div
        className="w-full flex flex-col items-start gap-4"
        style={{ opacity: loading ? 0.5 : 1 }}
      >
                
        <div className="w-full">
                    
          <h2 className="text-[#FF5733] font-montserrat font-bold mb-2 border-b-1 border-[#FF5733] mb-4">
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
              selectedFile && !loading
                ? "bg-[#FF5733] hover:bg-[#e04e2d]"
                : "bg-gray-500 cursor-not-allowed"
            }`}
          >
                        Analyze Run           
          </div>
        )}
              
      </div>
            {/* Most Recent Video */}
            
      <div style={{ opacity: loading ? 0.5 : 1 }}>
                
        <RunVideo videoUrl={videos.length > 0 ? videos[0] : null} />
              
      </div>
          
    </div>
  );
};

export default SourcesPanel;
