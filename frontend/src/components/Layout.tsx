import type { ReactNode } from "react";
import Navbar from "./Navbar";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
      <footer className="border-t border-brand-100 bg-white/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-brand-400">
            BiteScore v0.2.0 &mdash; Protein Digestibility Intelligence Suite
          </p>
          <p className="text-xs text-brand-300">
            MIT License &middot; Built with science in mind
          </p>
        </div>
      </footer>
    </div>
  );
}
