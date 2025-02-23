import React, { useState } from "react";
import ThreePanelUI from "./ThreePanelUI";
import title from "./assets/title.png";

function App() {
  const [activePanel, setActivePanel] = useState(null);
  const [videos, setVideos] = useState([]); // Store uploaded video URLs

  return (
    <div className="min-h-screen min-w-screen text-white font-montserrat">
      {/* Navigation */}
      <nav className="p-4 pb-0 sflex justify-between items-center">
        <img src={title} alt="title" className="w-48 h-8 object-cover" />
      </nav>
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
