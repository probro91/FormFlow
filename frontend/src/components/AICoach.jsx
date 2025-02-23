import React from "react";
import logo from "../assets/logo.png";
const AICoach = ({ chatBot, setChatBot }) => {
  if (!chatBot) return null;
  return (
    <div className="fixed bottom-3 right-5 flex flex-col items-end gap-2">
      {/* Chat Response Card */}
      <div className="bg-[#fff] text-black rounded-xl p-4 shadow-lg w-full max-w-118 text-left">
        <p className="text-md font-montserrat">{`"${chatBot}"`}</p>
        <p className="text-md text-[#aaa] mt-2">-Mr. Flow</p>
        {/* Close Button */}
        <div className="flex justify-end mt-2">
          <div
            className="text-[#aaa] text-sm font-montserrat cursor-pointer"
            onClick={() => setChatBot("")}
          >
            Dismiss
          </div>
        </div>
      </div>

      {/* Coach Icon */}
      <img src={logo} alt="AI Coach" className="w-18 h-18 rounded-full" />
    </div>
  );
};

export default AICoach;
