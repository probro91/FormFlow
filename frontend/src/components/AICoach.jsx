import React from "react";
import logo from "../assets/logo.png";
// fa arrow-right
import { FaArrowRight } from "react-icons/fa";

const AICoach = ({ chatBot, setChatBot }) => {
  if (!chatBot) return null;
  return (
    <div className="flex items-end gap-4 mt-6 w-full">
      {/* Coach Icon */}
      <img src={logo} alt="AI Coach" className="w-18 h-18 rounded-full -ml-2" />
      {/* Chat Response Card */}
      <div className="bg-[#fff] text-black rounded-xl p-4 shadow-lg w-full text-left -ml-2 w-full">
        <p className="text-md font-montserrat">{`"${chatBot}"`}</p>
        {/* Close Button */}
        <div
          className="flex items-center justify-end gap-2 cursor-pointer mt-2"
          onClick={() => setChatBot("")}
        >
          <FaArrowRight size={16} className="text-[#aaa]" />
        </div>
      </div>
    </div>
  );
};

export default AICoach;
