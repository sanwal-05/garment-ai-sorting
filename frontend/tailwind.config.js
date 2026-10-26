/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f4efe6",
        ink: "#1c1917",
        clay: "#c45c26",
        sage: "#3f6b4a",
        sand: "#e7dccb",
        mist: "#fbf8f3",
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Outfit", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 18px 40px -24px rgba(28, 25, 23, 0.35)",
      },
    },
  },
  plugins: [],
};
