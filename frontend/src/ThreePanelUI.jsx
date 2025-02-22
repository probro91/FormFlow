import React, { useState } from "react";
import SourcesPanel from "./components/SourcesPanel";
import TipsPanel from "./components/TipsPanel";
import WellnessPanel from "./components/WellnessPanel";

const Panel = ({ id, title, description, activePanel, setActivePanel }) => {
  return (
    <div
      className={`flex-1 bg-[#333333] text-white p-5 rounded-lg transition-all duration-300 ease-in-out cursor-pointer flex flex-col items-center justify-center text-center ${
        activePanel === id
          ? "flex-[1.2] scale-105 shadow-[0_0_15px_rgba(255,87,51,0.7)] bg-[#444444]"
          : ""
      }`}
      onClick={() => setActivePanel(id)}
    >
      <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold">{title}</h2>
      <p className="text-[#cccccc] text-base leading-relaxed">{description}</p>
    </div>
  );
};

const ThreePanelUI = () => {
  const [activePanel, setActivePanel] = useState(null);
  const [videos, setVideos] = useState([]);
  const [tips, setTips] = useState([]);
  const [expandedTip, setExpandedTip] = useState(null);
  const [sources, setSources] = useState([]);
  // State for TipsPanel
  const [cadenceData, setCadenceData] = useState(null); // Could be populated from real data
  const [strideLengthData, setStrideLengthData] = useState(null);
  const [overallScoreData, setOverallScoreData] = useState(null);
  const [stats, setStats] = useState(null);

  const panels = [
    {
      id: 1,
      title: "Stats",
      description: "Track your pace, distance, and calories burned.",
      isSources: true,
    },
    {
      id: 2,
      title: "Form Analysis",
      description: "Get AI insights on your running technique.",
      isTips: true,
    },
    {
      id: 3,
      title: "Goals",
      description: "Set and monitor your training targets.",
    },
  ];

  return (
    <div className="flex min-h-screen justify-between gap-5 p-5 bg-[#1A2533]">
      {panels.map((panel) =>
        panel.isSources ? (
          <SourcesPanel
            key={panel.id}
            id={panel.id}
            title={panel.title}
            activePanel={activePanel}
            setActivePanel={setActivePanel}
            videos={videos}
            setVideos={setVideos}
          />
        ) : panel.isTips ? (
          <TipsPanel
            key={panel.id}
            id={panel.id}
            title={panel.title}
            activePanel={activePanel}
            setActivePanel={setActivePanel}
            expandedTip={expandedTip}
            setExpandedTip={setExpandedTip}
            cadenceData={cadenceData}
            strideLengthData={strideLengthData}
            overallScoreData={overallScoreData}
            stats={stats}
          />
        ) : (
          <WellnessPanel
            key={panel.id}
            id={panel.id}
            title={panel.title}
            activePanel={activePanel}
            setActivePanel={setActivePanel}
          />
        )
      )}
    </div>
  );
};

export default ThreePanelUI;
