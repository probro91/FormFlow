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
      videoUrl:
        "https://www.youtube.com/watch?v=DHJupOV_IOA&list=PLT4Yite3Tx5ne9PJTByCLgdAtxo0lhd2s&index=1",
    },
    {
      id: 2,
      name: "Core Strengthening",

      videoUrl: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
    },
    {
      id: 3,
      name: "Cool-Down Walk",

      videoUrl: "https://www.youtube.com/watch?v=1i8Z8u2J1j8",
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
            🏋️ Workouts
          </h2>
          <div className="space-y-2">
            {exercisesData.map((exercise) => (
              <div
                key={exercise.id}
                className={`p-4 rounded-xl flex flex-col items-start max-w-[500px] gap-2 border-1 border-[#444444] hover:scale-101`}
                style={{ backgroundColor: colors.card2 }}
              >
                <p className="text-white font-montserrat">{exercise.name}</p>

                {/* Video */}
                <iframe
                  width={240}
                  height={135}
                  src={`https://www.youtube.com/embed/${
                    exercise.videoUrl.split("v=")[1]?.split("&")[0]
                  }`}
                  title={exercise.name}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="rounded-xl"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WellnessPanel;
