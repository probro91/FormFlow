import React, { useState } from "react";
import ThreePanelUI from "./ThreePanelUI";
import title from "./assets/title.png";
import logo from "./assets/logo.png";
import logoWhite from "./assets/logo-white.png";

function App() {
  return (
    <div className="min-h-screen min-w-screen text-white font-montserrat min-h-screen flex flex-col justify-between">
      <div>
        {/* Navigation */}
        <nav className="p-4 -mb-6 pl-6 flex justify-between items-center">
          <div className="flex items-center">
            <img src={logoWhite} alt="title" className="w-8 object-contain" />
            <p className="text-2xl font-semibold">FormFlow</p>
          </div>
          <p className="flex text-[#aaa] gap-4 pr-4">
            Made by Ethan, Batu, Amir, and Sam
          </p>
        </nav>
        <ThreePanelUI />
      </div>
      <div className="flex justify-center items-center h-2 text-sm text-[#cccccc] pb-6">
        We understand that everyone's running form is unique.
      </div>
    </div>
  );
}

export default App;
