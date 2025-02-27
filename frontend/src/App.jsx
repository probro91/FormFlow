import React from "react";
import ThreePanelUI from "./ThreePanelUI";
import title from "./assets/title.png";
import logo from "./assets/logo.png";
import logoWhite from "./assets/logo-white.png";
import Homepage from "./Homepage";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

function App() {
  return (
    <Router>
      <div className="min-h-screen min-w-screen text-white font-montserrat min-h-screen flex flex-col justify-between">
        {/* Navigation */}
        <nav className="absolute top-0 left-0 right-0 p-4 -mb-6 pl-6 flex justify-between items-center">
          <Link
            className="flex items-center cursor-pointer opacity-80 hover:opacity-100"
            to="/"
          >
            <img src={logoWhite} alt="title" className="w-8 object-contain" />
            <p className="text-2xl font-semibold text-white">FormFlow</p>
          </Link>
          <p className="flex text-[#aaa] gap-4 pr-4">
            Made by Ethan, Batu, Amir, and Sam
          </p>
        </nav>
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/app" element={<ThreePanelUI />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
