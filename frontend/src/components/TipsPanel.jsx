import React from "react";
import { RiArrowDropDownLine } from "react-icons/ri";
import Source from "./Source";
import colors from "../colors";

const TipsPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  expandedTip,
  setExpandedTip,
  stats, // Keeping stats prop for circular meters
}) => {
  // Default stats if none provided
  const defaultStats = {
    avgCadence: "180",
    avgStrideLength: "1.2 m",
    footStrike: "Heel",
    overallScore: "85",
  };

  const { avgCadence, avgStrideLength, footStrike, overallScore } =
    stats || defaultStats;

  // Placeholder data for tips
  const tips = [
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
    {
      id: 2,
      type: "yellow",
      summary: "Stride Length Warning",
      details: "Your stride length varies too much, risking overstriding.",
      sources: [
        {
          imageUrl: "https://placehold.co/400",
          title: "Hydration Tips for Athletes",
          url: "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/water/art-20044256",
        },
      ],
    },
    {
      id: 3,
      type: "red",
      summary: "Poor Foot Strike",
      details: "Heavy heel striking detected, increasing injury risk.",
      sources: [
        {
          imageUrl: "https://placehold.co/400",
          title: "Importance of Sleep for Recovery",
          url: "https://www.sleepfoundation.org/physical-health/physical-activity-and-sleep",
        },
      ],
    },
  ];

  return (
    <div
      className={`text-white rounded-xl transition-all duration-300 ease-in-out flex flex-col items-center text-center border-2 hover:scale-101 hover:border-[#555555] border-[#444444]`}
      style={{ backgroundColor: colors.card1 }}
      onClick={() => setActivePanel(id)}
    >
      {/* Scrollable Content Container */}
      <div className="w-full p-5 flex flex-col items-center justify-between gap-8">
        {/* Tips List */}
        <div className="w-full flex flex-col items-start">
          <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold">
            Tips
          </h2>
          <div className="space-y-4 w-full">
            {tips.map((tip) => (
              <div
                key={tip.id}
                className={`p-4 rounded-xl transition-all duration-300 w-full text-left border-1 border-[#444444] 
                 
                `}
                style={{ backgroundColor: colors.card2 }}
              >
                <div className="flex items-center gap-2">
                  {/* Color Indicator */}
                  <div
                    className={`w-4 h-4 rounded-full ${
                      tip.type === "green"
                        ? "bg-green-500"
                        : tip.type === "yellow"
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                  ></div>
                  <p className="text-white font-montserrat">{tip.summary}</p>
                  {/* Expand Icon */}
                  <RiArrowDropDownLine
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent panel focus
                      setExpandedTip(expandedTip === tip.id ? null : tip.id);
                    }}
                    size={20}
                    className={`cursor-pointer transform transition-transform duration-300 ${
                      expandedTip === tip.id ? "rotate-180" : ""
                    }`}
                  />
                </div>
                {expandedTip === tip.id && (
                  <div className="mt-2 text-sm text-[#cccccc]">
                    <p>{tip.details}</p>
                    <div className="flex gap-2 mt-2">
                      {tip.sources.map((source, index) => (
                        <Source
                          key={index}
                          imageUrl={source.imageUrl}
                          title={source.title}
                          url={source.url}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TipsPanel;
