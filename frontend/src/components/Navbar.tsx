import { Link, useLocation } from "react-router-dom";
import { FlaskConical, Home, BarChart3, Info, Menu, X } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: Home },
  { to: "/predict", label: "Predict", icon: FlaskConical },
  { to: "/about", label: "About", icon: Info },
];

export default function Navbar() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 glass border-b border-brand-100/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-400/20 group-hover:shadow-brand-400/30 transition-shadow">
              <img
                src="/api/assets/logo.png"
                alt="BiteScore"
                className="w-7 h-7 rounded-lg"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            </div>
            <div>
              <span className="text-xl font-bold text-brand-900 tracking-tight">
                BiteScore
              </span>
              <span className="hidden sm:block text-[11px] text-brand-400 -mt-0.5 font-medium">
                Protein Digestibility Intelligence
              </span>
            </div>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const active =
                to === "/"
                  ? location.pathname === "/"
                  : location.pathname.startsWith(to);
              return (
                <Link
                  key={to}
                  to={to}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-150",
                    active
                      ? "bg-brand-100/80 text-brand-700 shadow-sm"
                      : "text-brand-500 hover:bg-brand-50 hover:text-brand-700"
                  )}
                >
                  <Icon size={16} />
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-xl hover:bg-brand-50 text-brand-600"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Mobile nav */}
        {mobileOpen && (
          <nav className="md:hidden pb-4 flex flex-col gap-1 animate-fade-in">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const active =
                to === "/"
                  ? location.pathname === "/"
                  : location.pathname.startsWith(to);
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMobileOpen(false)}
                  className={clsx(
                    "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all",
                    active
                      ? "bg-brand-100/80 text-brand-700"
                      : "text-brand-500 hover:bg-brand-50"
                  )}
                >
                  <Icon size={18} />
                  {label}
                </Link>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
}
