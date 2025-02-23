import React, { useState, useRef } from "react";
import RunVideo from "./RunVideo";
import { MdOutlineFileUpload } from "react-icons/md";
import CustomDropdown from "./CustomDropdown";
import colors from "../colors";
import axios from "axios";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { Rings } from "react-loader-spinner";
import { FaPersonRunning } from "react-icons/fa6";


const s3Client = new S3Client({
  region: import.meta.env.VITE_AWS_REGION,
  credentials: {
    accessKeyId: import.meta.env.VITE_AWS_ACCESS_KEY_ID,
    secretAccessKey: import.meta.env.VITE_AWS_SECRET_ACCESS_KEY,
  },
});

const SourcesPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  videos,
  setVideos,
  overallScoreData,
  setOverallScoreData,
  setStats,
  setChatBot,
  setTips,
  setExercises,
}) => {
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null); // Ref to trigger hidden file input
  const [processedVideos, setProcessedVideos] = useState({});

  const fetchVideos = async (generated_videos) => {
    try {
      console.log("generated_videos:", generated_videos);
      let s3links = [];
      for (const key in generated_videos) {
        const s3file = "https://formflow-videos.s3.us-east-1.amazonaws.com/initalvids/" + key;
        s3links.push(s3file);
      }
      // const videoPromises = Object.entries(generated_videos).map(
      //   async ([key, keyPath]) => {
      //     const s3file = "https://formflow-videos.s3.us-east-1.amazonaws.com/initalvids/" + key;
      //     console.log("s3file:", s3file);
      //     return s3file;
      //   }
      // );
      //   async ([key, keyPath]) => {
      //     console.log("Fetching video:", key, keyPath);
      //     const params = {
      //       Bucket: "formflow-videos",
      //       Key: "https://formflow-videos." + keyPath,
      //     };

      //     const command = new GetObjectCommand(params);
      //     const { Body } = await s3Client.send(command);
      //     const blob = await streamToBlob(Body, "video/mp4");

      //     return { key, blob };
      //   }
      // );

      // const results = await Promise.all(videoPromises); // Store the videos in state
      // console.log("Fetched videos:", results);

      // setProcessedVideos(
      //   results.reduce((acc, { key, blob }) => {
      //     acc[key] = blob;
      //     return acc;
      //   }, {})
      // );
      console.log("s3links:", s3links);
      setProcessedVideos(s3links);
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
  console.log("SourcesPanel processedVideos:", processedVideos);

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
    setLoading(true); // Create an object URL for the selected file
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

      const stats = {
        avgCadence: response.data.body.cadence.steps_per_minute,
        avgStrideLength: response.data.body.stride.average_length,
        footStrike:
          response.data.body.analysis_categories[
            "Foot Strike"
          ].issue_description.split("(")[0],
        overallScore: response.data.body.form_score,
        spineAlignment: response.data.body.spine_alignment.degrees,
      };

      console.log("Stats:", stats);

      // set tips in the following format
      {
        /* 
        {
        id: 1,
        type: "green",
        summary: "Great Cadence",
        details:
          "Your cadence is consistently above 180 SPM, optimizing efficiency.",
        sources: [
          {
            imageUrl: "https://placehold.co/400",
            title: "Benefits of Stretching for Runners",
            url: "https://www.health.harvard.edu/staying-healthy/the-importance-of-stretching",
          },
        ],
      },
      */
      }
      let tips = [];

      const analysis_categories = response.data.body.analysis_categories;

      for (const key in analysis_categories) {
        console.log(key, analysis_categories[key]);

        const category = analysis_categories[key];

        if (category.status === "wrong") {
          const sources = category.articles.map((source) => ({
            imageUrl: source.image,
            title: source.title,
            url: source.url,
          }));

          tips = [
            ...tips,
            {
              id: key,
              type: "red",
              summary: key,
              details: category.issue_description,
              sources,
            },
          ];
        }

        // If this category is "right", produce a green tip
        else if (category.status === "right") {
          tips = [
            ...tips,
            {
              id: key,
              type: "green",
              summary: key,
              details: category.issue_description,
              sources: [],
            },
          ];
        }
      }

      // add yellow tip if cadence is off
      if (response.data.body.cadence.steps_per_minute < 170) {
        tips = [
          ...tips,
          {
            id: "Cadence",
            type: "yellow",
            summary: "Low Cadence",
            details:
              "Your cadence is below 170 SPM. Try to increase your steps per minute to improve efficiency.",
            sources: [],
          },
        ];
      }
      if (response.data.body.cadence.steps_per_minute > 180) {
        tips = [
          ...tips,
          {
            id: "Cadence",
            type: "yellow",
            summary: "High Cadence",
            details:
              "Your cadence is above 180 SPM. Try to decrease your steps per minute to improve efficiency.",
            sources: [],
          },
        ];
      } // add yellow tip if stride length is off

      // add yellow tip if stride length is off
      if (response.data.body.stride.average_length < 1.2) {
        tips = [
          ...tips,
          {
            id: "Stride Length",
            type: "yellow",
            summary: "Short Stride Length",
            details:
              "Your stride length is below 1.2 meters. Try to increase your stride length to improve efficiency.",
            sources: [],
          },
        ];
      }
      if (response.data.body.stride.average_length > 1.4) {
        tips = [
          ...tips,
          {
            id: "Stride Length",
            type: "yellow",
            summary: "Long Stride Length",
            details:
              "Your stride length is above 1.4 meters. Try to decrease your stride length to improve efficiency.",
            sources: [],
          },
        ];
      }

      console.log("Tips:", tips);

      // set exercises in the following format
      {
        /*
        {
      id: 1,
      name: "Dynamic Stretching",
      description:
        "Warm up with leg swings and arm circles to improve flexibility.",
      videoUrl: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
      sources: [
        {
          id: 1,
          imageUrl: "https://placehold.co/400",
          title: "Dynamic Stretching Routine",
          url: "https://www.youtube.com/watch?v=L_jWHffIx5E&list=RDdQw4w9WgXcQ&index=7",
        },
      ],
    },
      */
      }
      let exercises = [];

      // loop over analysis_categories and get the exercises if there are any
      for (const key in analysis_categories) {
        console.log(key, analysis_categories[key]);

        const category = analysis_categories[key];

        if (category.exercises.length > 0) {
          for (const exercise of category.exercises) {
            exercises = [
              ...exercises,
              {
                id: exercise.name,
                name: exercise.name,
                videoUrl: exercise.youtube_link,
              },
            ];
          }
        }
      }

      setStats(stats);
      setTips(tips);
      setExercises(exercises);
      // add over all score data
      setOverallScoreData([...overallScoreData, response.data.body.form_score]);

      const claudeResponse = response.data.body.claude_suggestions;
      setChatBot(claudeResponse);
      setVideos((prevVideos) => [videoUrl, ...prevVideos]);
      setSelectedFile(null); // Clear after upload completes
      fetchVideos(response.data.body.generated_videos);

      setLoading(false);
    } catch (error) {
      console.error("Error uploading video:", error);
    }

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
                    
          <div className="flex items-center gap-2 w-full mb-2 border-b-1 border-[#FF5733] pb-2">
            <FaPersonRunning size={20} color="#FF5733" />
            <h2 className="text-[#FF5733] font-montserrat font-bold">
                          Form Analyzer           
            </h2>
          </div>
                    
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
                            
              <div className="flex flex-col gap-4 mt-2">
                                
                <p className="text-[#cccccc] text-sm">Select a video file:</p>
                                
                <div
                  onClick={handleIconClick}
                  className="cursor-pointer px-14 py-28 border border-[#aaa] rounded-xl hover:bg-gray-800 hover:border-[#fff] items-center justify-center flex gap-2"
                >
                                    
                  <MdOutlineFileUpload size={42} color="#fff" />
                                  
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
            className={`mt-4 px-4 py-2 rounded-xl text-white font-montserrat cursor-pointer transition-colors duration-300 bg-[#FF5733] hover:bg-[#e04e2d]`}
          >
                        Analyze Run           
          </div>
        )}
              
      </div>
            {/* Most Recent Video */}
            
      <div style={{ opacity: loading ? 0.5 : 1 }}>
                
        <RunVideo processedVideos={processedVideos} />
              
      </div>
          
    </div>
  );
};

export default SourcesPanel;
