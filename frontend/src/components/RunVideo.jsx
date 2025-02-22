import React from "react";

const RunVideo = ({ videoUrl }) => {
  if (!videoUrl) {
    return (
      <p className="text-[#cccccc] text-sm mt-4">No video uploaded yet.</p>
    );
  }

  return (
    <div className="mt-4 w-full">
      <h3 className="text-[#cccccc] text-sm mb-2">Latest Run</h3>
      <video width="100%" controls className="rounded-md" key={videoUrl}>
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default RunVideo;
