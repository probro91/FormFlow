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
import SpineAngleGraph from "./SpineAngleGraph";
import colors from "../colors";
import { MdOutlineAutoGraph } from "react-icons/md";

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
          {label === "Overall Score" ? "%" : ""}
        </p>
      </div>
    </div>
  );
};

const GraphsPanel = ({
  id,
  title,
  activePanel,
  setActivePanel,
  strideLengthData,
  overallScoreData,
  stats,
}) => {
  // make array labels for the graph
  const labels = Array.from({ length: overallScoreData.length }, (_, i) => i);

  const defaultOverallScoreData = {
    labels,
    datasets: [
      {
        label: "Overall Score (%)",
        data: overallScoreData,
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
    spineAlignment: 8,
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

  const overallScoreChartData = defaultOverallScoreData;
  const {
    avgCadence,
    avgStrideLength,
    footStrike,
    overallScore,
    spineAlignment,
  } = stats || defaultStats;

  return (
    <div
      className={`text-white rounded-xl transition-all duration-300 ease-in-out flex flex-col text-center border-[#444444] p-6 pb-12 hover:border-[#555555] hover:scale-101 `}
      style={{ backgroundColor: colors.card1 }}
      onClick={() => setActivePanel(id)}
    >
      <div className="flex items-center gap-2 w-full mb-2 border-b-1 border-[#FF5733] pb-2">
        <MdOutlineAutoGraph size={20} color="#FF5733" />
        <h2 className="text-[#FF5733] font-montserrat font-bold">
                      Results          
        </h2>
      </div>
      <div className="grid grid-cols-2 gap-4 my-4">
        <div>
          <p className="text-[#cccccc] text-sm">Overall Score Over Time</p>
          <div className="h-24">
            <Line data={overallScoreChartData} options={chartOptions} />
          </div>
        </div>
        <div>
          <p className="text-[#cccccc] text-sm">Spine Alignment</p>
          <SpineAngleGraph spineAlignment={spineAlignment} />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mt-6 text-center">
        <div>
          <p className="text-[#cccccc] text-sm">Overall Score</p>
          <CircularMeter
            value={parseFloat(overallScore)}
            max={100}
            label="Overall Score"
          />
        </div>
        <div>
          <p className="text-[#cccccc] text-sm">Avg Cadence (SPM)</p>
          <CircularMeter
            value={parseFloat(avgCadence)}
            max={240}
            idealRange={[150, 180]}
            label="Avg Cadence"
          />
        </div>
        <div>
          <p className="text-[#cccccc] text-sm">Avg Stride Length</p>
          <p className="text-[#FF5733] text-xl font-bold">{avgStrideLength}</p>
        </div>
        <div>
          <p className="text-[#cccccc] text-sm">Foot Strike</p>
          <p className="text-[#FF5733] text-xl font-bold">{footStrike}</p>
        </div>
      </div>
    </div>
  );
};

export default GraphsPanel;
