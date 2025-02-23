import React from "react";
import colors from "../colors";
import Source from "./Source";

const WellnessPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  exercises,
}) => {
  // Placeholder data for exercise suggestions
  const exercisesDummy = [
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
    {
      id: 2,
      name: "Core Strengthening",
      description: "Planks and bridges to enhance stability and running form.",
      videoUrl: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
      sources: [
        {
          id: 1,
          imageUrl: "https://placehold.co/400",
          title: "Core Strengthening Exercises",
          url: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
        },
      ],
    },
    {
      id: 3,
      name: "Cool-Down Walk",
      description: "A 5-10 minute walk post-run to aid recovery.",
      videoUrl: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
      sources: [
        {
          id: 1,
          imageUrl: "https://placehold.co/400",
          title: "Cool-Down Walk Benefits",
          url: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
        },
      ],
    },
  ];

  const exercisesData = exercises.length > 0 ? exercises : exercisesDummy;

  return (
    <div
      className={`flex-2 text-white rounded-xl transition-all duration-300 ease-in-out flex flex-col items-center text-center border-[#444444] hover:border-[#555555] hover:scale-101`}
      style={{ backgroundColor: colors.card1 }}
      onClick={() => setActivePanel(id)}
    >
      {/* Scrollable Content Container */}
      <div className="w-full h-full p-5 flex flex-col items-center justify-start gap-8">
        {/* Top Half: Exercise Suggestions */}
        <div className="w-full">
          <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold text-left flex flex-col items-start gap-2 border-b-1 border-[#FF5733] mb-4">
            {title}
          </h2>
          <div className="space-y-2">
            {exercisesData.map((exercise) => (
              <div
                key={exercise.id}
                className={`p-4 rounded-xl flex flex-col items-start max-w-[500px] gap-2 border-1 border-[#444444] hover:scale-101`}
                style={{ backgroundColor: colors.card2 }}
              >
                <p className="text-white font-montserrat">{exercise.name}</p>
                <p className="text-[#cccccc] text-sm text-left">
                  {exercise.description}
                </p>

                <div className="space-y-4 w-full flex">
                  {exercise.sources.map((source) => (
                    <Source
                      key={source.id}
                      imageUrl={source.imageUrl}
                      title={source.title}
                      url={source.url}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WellnessPanel;
