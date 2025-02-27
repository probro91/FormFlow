import React, { useState } from "react";
import SourcesPanel from "./components/SourcesPanel";
import TipsPanel from "./components/TipsPanel";
import GraphsPanel from "./components/GraphsPanel";
import WellnessPanel from "./components/WellnessPanel";
import AICoach from "./components/AICoach";

const ThreePanelUI = () => {
  const [activePanel, setActivePanel] = useState(null);
  const [videos, setVideos] = useState([]);
  const [tips, setTips] = useState([]);
  // State for TipsPanel
  const [cadenceData, setCadenceData] = useState(null); // Could be populated from real data
  const [strideLengthData, setStrideLengthData] = useState(null);
  const [overallScoreData, setOverallScoreData] = useState([44, 48, 36, 67]);
  const [stats, setStats] = useState(null);
  const [chatBot, setChatBot] = useState(
    "I'm Mr. Flow, your AI running coach. Let's get started!"
  );
  const [exercises, setExercises] = useState([]);

  const panels = [
    {
      id: 1,
      title: "Video Analyzer",
      description: "Track your pace, distance, and calories burned.",
      isSources: true,
    },
    {
      id: 2,
      title: "Tips",
      description: "Get AI insights on your running technique.",
      isTips: true,
    },
    {
      id: 4,
      title: "Wellness",
      description: "Stay healthy with our post-run suggestions.",
    },
  ];

  return (
    <div className="flex justify-between gap-5 p-5 py-16 bg-gradient-to-br from-gray-900 via-indigo-900 to-blue-900 min-h-screen">
      {panels.map((panel) =>
        panel.isSources ? (
          <div className="flex-3 flex flex-col gap-4">
            <SourcesPanel
              key={panel.id}
              id={panel.id}
              title={panel.title}
              activePanel={activePanel}
              setActivePanel={setActivePanel}
              videos={videos}
              setVideos={setVideos}
              setStats={setStats}
              chatBot={chatBot}
              setChatBot={setChatBot}
              overallScoreData={overallScoreData}
              setOverallScoreData={setOverallScoreData}
              setTips={setTips}
              setExercises={setExercises}
            />
          </div>
        ) : panel.isTips ? (
          <div className="flex-3 flex flex-col gap-4">
            <TipsPanel
              key={panel.id}
              id={panel.id}
              title={panel.title}
              activePanel={activePanel}
              setActivePanel={setActivePanel}
              tips={tips}
            />
            <GraphsPanel
              key={panel.id}
              id={3}
              title="Results"
              activePanel={activePanel}
              setActivePanel={setActivePanel}
              cadenceData={cadenceData}
              setCadenceData={setCadenceData}
              strideLengthData={strideLengthData}
              setStrideLengthData={setStrideLengthData}
              overallScoreData={overallScoreData}
              setOverallScoreData={setOverallScoreData}
              stats={stats}
              setStats={setStats}
            />
          </div>
        ) : (
          <div>
            <WellnessPanel
              key={panel.id}
              id={panel.id}
              title={panel.title}
              activePanel={activePanel}
              setActivePanel={setActivePanel}
              exercises={exercises}
            />
          </div>
        )
      )}
    </div>
  );
};

export default ThreePanelUI;
