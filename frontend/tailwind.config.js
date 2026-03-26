/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f6ff",
          100: "#dceaff",
          200: "#b8d4ff",
          300: "#84b5ff",
          400: "#4a90e2",
          500: "#377dd5",
          600: "#2563b8",
          700: "#1d4e96",
          800: "#1a3f77",
          900: "#1d3557",
          950: "#0f1f33",
        },
      },
      fontFamily: {
        sans: ['"Inter"', '"Segoe UI"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Fira Code"', "monospace"],
      },
      boxShadow: {
        soft: "0 4px 24px rgba(29, 53, 87, 0.06)",
        card: "0 8px 32px rgba(29, 53, 87, 0.08)",
        elevated: "0 16px 48px rgba(29, 53, 87, 0.12)",
        glow: "0 0 24px rgba(74, 144, 226, 0.15)",
      },
      borderRadius: {
        xl: "16px",
        "2xl": "20px",
        "3xl": "24px",
      },
      animation: {
        "fade-in": "fadeIn 0.4s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
