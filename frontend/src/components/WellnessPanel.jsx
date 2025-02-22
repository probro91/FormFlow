import React, { useState, useEffect } from "react";

const WellnessPanel = ({ id, title, activePanel, setActivePanel }) => {
  // Placeholder data for exercise suggestions
  const exercises = [
    {
      id: 1,
      name: "Dynamic Stretching",
      description:
        "Warm up with leg swings and arm circles to improve flexibility.",
    },
    {
      id: 2,
      name: "Core Strengthening",
      description: "Planks and bridges to enhance stability and running form.",
    },
    {
      id: 3,
      name: "Cool-Down Walk",
      description: "A 5-10 minute walk post-run to aid recovery.",
    },
  ];

  // Placeholder data for articles/research
  const articles = [
    {
      id: 1,
      title: "Benefits of Stretching for Runners",
      url: "https://www.health.harvard.edu/staying-healthy/the-importance-of-stretching",
      img: "https://placehold.co/400",
      description: "Learn how stretching can improve your running performance.",
    },
    {
      id: 2,
      title: "Hydration Tips for Athletes",
      url: "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/water/art-20044256",
      img: "https://placehold.co/400",
      description: "Stay hydrated to maintain peak athletic performance.",
    },
    {
      id: 3,
      title: "Importance of Sleep for Recovery",
      url: "https://www.sleepfoundation.org/physical-health/physical-activity-and-sleep",
      img: "https://placehold.co/400",
      description: "Quality sleep is essential for muscle repair and growth.",
    },
  ];

  return (
    <div
      className={`flex-2 bg-[#333333] text-white rounded-lg transition-all duration-300 ease-in-out cursor-pointer flex flex-col items-center text-center border-2 border-[#444444] ${
        activePanel === id ? "scale-101 border-1 border-[#fff]" : ""
      }`}
      onClick={() => setActivePanel(id)}
    >
      {/* Scrollable Content Container */}
      <div className="w-full h-full p-5 overflow-y-auto flex flex-col items-center justify-start gap-8">
        {/* Top Half: Exercise Suggestions */}
        <div className="w-full">
          <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold text-left">
            {title}
          </h2>
          <div className="space-y-4">
            {exercises.map((exercise) => (
              <div
                key={exercise.id}
                className="p-3 bg-[#1A2533] rounded-md text-left"
              >
                <p className="text-white font-montserrat font-semibold">
                  {exercise.name}
                </p>
                <p className="text-[#cccccc] text-sm mt-1">
                  {exercise.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Half: Helpful Articles/Research with Static Previews */}
        <div className="w-full mt-6">
          <h3 className="text-md mb-2 font-montserrat font-bold text-left text-[#FF5733]">
            Resources
          </h3>
          <div className="space-y-4 w-full">
            {articles.map((article) => (
              <div
                key={article.id}
                className="p-4 bg-[#1A2533] rounded-md flex flex-col items-start max-w-[500px]"
              >
                <p className="text-white font-montserrat">{article.title}</p>
                <p className="text-[#cccccc] text-sm mt-1 text-left">
                  {article.description}
                </p>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 w-full flex items-center gap-2"
                >
                  {article.img ? (
                    <img
                      src={article.img}
                      alt="Article Preview"
                      className="w-8 object-cover rounded-full"
                    />
                  ) : null}
                  <p className="text-[#aaa] text-xs mt-2 underline cursor-pointer text-left truncate">
                    {article.url}
                  </p>
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WellnessPanel;
