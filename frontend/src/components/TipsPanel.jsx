import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Circular Meter Component
const CircularMeter = ({ value, max, idealRange, label }) => {
  const percentage = Math.min((value / max) * 100, 100); // Cap at 100%
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  // Color logic: Red (low) to Green (high)
  let strokeColor;
  if (idealRange) {
    // For cadence (ideal 170-180 SPM)
    const [minIdeal, maxIdeal] = idealRange;
    if (value < minIdeal - 10) strokeColor = "#EF4444"; // Red below 160
    else if (value > maxIdeal + 10) strokeColor = "#EF4444"; // Red above 190
    else if (value >= minIdeal && value <= maxIdeal)
      strokeColor = "#10B981"; // Green in 170-180
    else strokeColor = "#F59E0B"; // Yellow in transitional zones
  } else {
    // For overall score (0-100%)
    strokeColor = `hsl(${percentage}, 100%, 50%)`; // Red (0) to Green (100)
  }

  return (
    <div className="relative flex items-center justify-center">
      <svg width="100" height="100" className="transform -rotate-90">
        <circle
          cx="50"
          cy="50"
          r={radius}
          stroke="#444444"
          strokeWidth="8"
          fill="none"
        />
        <circle
          cx="50"
          cy="50"
          r={radius}
          stroke={strokeColor}
          strokeWidth="8"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
        />
      </svg>
      <div className="absolute text-center">
        <p className="text-[#FF5733] text-xl font-bold">
          {value}
          {label === "Overall Score" ? "%" : " SPM"}
        </p>
      </div>
    </div>
  );
};

const TipsPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  expandedTip,
  setExpandedTip,
  cadenceData,
  strideLengthData,
  overallScoreData,
  stats,
}) => {
  // Fake data if no state is provided
  const defaultCadenceData = {
    labels: ["0s", "10s", "20s", "30s", "40s", "50s", "60s"],
    datasets: [
      {
        label: "Cadence (SPM)",
        data: [170, 175, 180, 182, 178, 180, 181],
        borderColor: "#FF5733",
        backgroundColor: "rgba(255, 87, 51, 0.2)",
        fill: true,
      },
    ],
  };

  const defaultStrideLengthData = {
    labels: ["0s", "10s", "20s", "30s", "40s", "50s", "60s"],
    datasets: [
      {
        label: "Stride Length (m)",
        data: [1.1, 1.15, 1.2, 1.25, 1.18, 1.22, 1.2],
        borderColor: "#FF5733",
        backgroundColor: "rgba(255, 87, 51, 0.2)",
        fill: true,
      },
    ],
  };

  const defaultOverallScoreData = {
    labels: ["0s", "10s", "20s", "30s", "40s", "50s", "60s"],
    datasets: [
      {
        label: "Overall Score (%)",
        data: [80, 82, 85, 87, 84, 86, 85],
        borderColor: "#FF5733",
        backgroundColor: "rgba(255, 87, 51, 0.2)",
        fill: true,
      },
    ],
  };

  const defaultStats = {
    avgCadence: "180",
    avgStrideLength: "1.2 m",
    footStrike: "Heel",
    overallScore: "85",
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true },
    },
    scales: {
      x: { display: true, grid: { color: "#444" } },
      y: { display: true, grid: { color: "#444" } },
    },
  };

  const cadenceChartData = cadenceData || defaultCadenceData;
  const strideLengthChartData = strideLengthData || defaultStrideLengthData;
  const overallScoreChartData = overallScoreData || defaultOverallScoreData;
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
      sources: ["https://example.com/cadence", "https://example.com/running"],
    },
    {
      id: 2,
      type: "yellow",
      summary: "Stride Length Warning",
      details: "Your stride length varies too much, risking overstriding.",
      sources: ["https://example.com/stride"],
    },
    {
      id: 3,
      type: "red",
      summary: "Poor Foot Strike",
      details: "Heavy heel striking detected, increasing injury risk.",
      sources: ["https://example.com/footstrike", "https://example.com/injury"],
    },
  ];

  return (
    <div
      className={`flex-3 bg-[#333333] text-white rounded-lg transition-all duration-300 ease-in-out cursor-pointer flex flex-col items-center text-center border-2 border-[#444444] ${
        activePanel === id ? " scale-101 border-1 border-[#fff]" : ""
      }`}
      onClick={() => setActivePanel(id)}
    >
      {/* Scrollable Content Container */}
      <div className="w-full h-full p-5 overflow-y-auto flex flex-col items-center justify-start gap-8">
        {/* Tips List */}
        <div className="w-full flex flex-col items-start">
          <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold">
            Tips
          </h2>
          <div className="space-y-2 w-full">
            {tips.map((tip) => (
              <div
                key={tip.id}
                className={`p-2 px-4 rounded-3xl cursor-pointer transition-all duration-300 w-full text-left bg-[#1A2533] border-1 border-[#444444]
                 ${expandedTip === tip.id && "border-1 border-[#fff]"}`}
                onClick={(e) => {
                  e.stopPropagation(); // Prevent panel focus
                  setExpandedTip(expandedTip === tip.id ? null : tip.id);
                }}
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
                </div>
                {expandedTip === tip.id && (
                  <div className="mt-2 text-sm text-[#cccccc]">
                    <p>{tip.details}</p>
                    <div className="flex gap-2 mt-2">
                      {tip.sources.map((source, index) => (
                        <a
                          key={index}
                          href={source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="w-6 h-6 bg-[#FF5733] rounded-full flex items-center justify-center text-white text-xs"
                        >
                          S{index + 1}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Graphs and Stats */}
        <div className="w-full p-4 rounded-md mt-4">
          <h2 className="text-[#FF5733] mb-2 font-montserrat font-bold">
            {title}
          </h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="col-span-2">
              <p className="text-[#cccccc] text-sm">Cadence Over Time</p>
              <div className="h-24">
                <Line data={cadenceChartData} options={chartOptions} />
              </div>
            </div>
            <div>
              <p className="text-[#cccccc] text-sm">Stride Length Over Time</p>
              <div className="h-24">
                <Line data={strideLengthChartData} options={chartOptions} />
              </div>
            </div>
            <div>
              <p className="text-[#cccccc] text-sm">Overall Score Over Time</p>
              <div className="h-24">
                <Line data={overallScoreChartData} options={chartOptions} />
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <p className="text-[#cccccc] text-sm">Overall Score</p>
              <CircularMeter
                value={parseFloat(overallScore)}
                max={100} // Percentage
                label="Overall Score"
              />
            </div>
            <div>
              <p className="text-[#cccccc] text-sm">Avg Cadence</p>
              <CircularMeter
                value={parseFloat(avgCadence)}
                max={200} // Reasonable max for cadence
                idealRange={[170, 180]}
                label="Avg Cadence"
              />
            </div>
            <div>
              <p className="text-[#cccccc] text-sm">Avg Stride Length</p>
              <p className="text-[#FF5733] text-xl font-bold">
                {avgStrideLength}
              </p>
            </div>
            <div>
              <p className="text-[#cccccc] text-sm">Foot Strike</p>
              <p className="text-[#FF5733] text-xl font-bold">{footStrike}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TipsPanel;
