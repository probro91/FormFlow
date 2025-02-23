import React, { useState } from "react";
import ThreePanelUI from "./ThreePanelUI";
import title from "./assets/title.png";

function App() {
  return (
    <div className="min-h-screen min-w-screen text-white font-montserrat">
      {/* Navigation */}
      <nav className="p-4 pb-0 sflex justify-between items-center">
        <img src={title} alt="title" className="w-48 h-8 object-cover" />
      </nav>
      <ThreePanelUI />
      <div className="flex justify-center items-center h-2 text-sm text-[#cccccc] pb-4">
        We understand that everyone's running form is unique.
      </div>
    </div>
  );
}

export default App;
