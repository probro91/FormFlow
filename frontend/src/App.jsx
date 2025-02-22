import React, { useState } from "react";
import ThreePanelUI from "./ThreePanelUI";

function App() {
  const [activePanel, setActivePanel] = useState(null);
  const [videos, setVideos] = useState([]); // Store uploaded video URLs

  return (
    <div className="min-h-screen min-w-screen bg-[#1A2533] text-white">
      <ThreePanelUI
        activePanel={activePanel}
        setActivePanel={setActivePanel}
        videos={videos}
        setVideos={setVideos}
      />
    </div>
  );
}

export default App;
