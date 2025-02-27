// src/components/Homepage.jsx
import React from "react";
import { Link } from "react-router-dom";
import logo from "./assets/logo.png";

const Homepage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-indigo-900 to-blue-900 text-white flex flex-col pt-20">
      <header className="text-center py-16 px-4">
        <h1 className="text-5xl md:text-6xl font-bold relative">
          <img src={logo} alt="logo" className="w-16 inline object-contain" />
          <span className="relative z-10">FormFlow</span>
          <span className="absolute inset-0 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 animate-bg-move bg-[length:200%_100%] opacity-20 blur-xl rounded-full -z-10" />
        </h1>
        <p className="text-xl md:text-2xl mt-4 opacity-90">
          Your AI-Powered Running Coach
        </p>
      </header>

      <main className="flex-grow px-4 py-8 flex flex-col items-center gap-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl w-full">
          <div className="bg-white/10 p-6 rounded-xl backdrop-blur-sm hover:-translate-y-1 transition-transform">
            <h3 className="text-cyan-400 text-xl font-semibold">
              Smart Form Analysis
            </h3>
            <p className="mt-2 opacity-90">
              Real-time feedback on your running technique
            </p>
          </div>
          <div className="bg-white/10 p-6 rounded-xl backdrop-blur-sm hover:-translate-y-1 transition-transform">
            <h3 className="text-cyan-400 text-xl font-semibold">
              Personalized Plans
            </h3>
            <p className="mt-2 opacity-90">
              Custom training tailored to your goals
            </p>
          </div>
          <div className="bg-white/10 p-6 rounded-xl backdrop-blur-sm hover:-translate-y-1 transition-transform">
            <h3 className="text-cyan-400 text-xl font-semibold">
              Progress Tracking
            </h3>
            <p className="mt-2 opacity-90">
              Monitor your improvement over time
            </p>
          </div>
        </div>

        <Link
          to="/app"
          className="border-[#FF5733] border-1 px-8 py-3 rounded-full font-semibold text-lg transition-all hover:scale-105 hover:bg-[#FF5733] hover:bg-opacity-20 hoverLfont-semibold"
        >
          <p className="flex items-center gap-2 text-gray-100">Get Started</p>
        </Link>
      </main>

      <footer className="text-center py-4 opacity-70 text-sm">
        <p>© 2025 FormFlow. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Homepage;
