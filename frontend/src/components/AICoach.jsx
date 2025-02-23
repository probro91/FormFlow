import React from "react";
import logo from "../assets/logo.png";
const AICoach = (text) => {
  text = Object.values(text)[0] || "Hello! How can I help you today?";
  return (
    <div className="fixed bottom-3 right-5 flex flex-col items-end gap-2">
      {/* Chat Response Card */}
      <div className="bg-[#fff] text-black rounded-lg p-4 shadow-lg w-full max-w-118">
        <p className="text-sm font-montserrat">{`"${text}"`}</p>
      </div>

      {/* Coach Icon */}
      <img src={logo} alt="AI Coach" className="w-18 h-18 rounded-full" />
    </div>
  );
};

export default AICoach;
