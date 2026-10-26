/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // User requested color accents
        paper: "#F7F5EF",
        bgWarm: "#F7F5EF",
        ink: "#29332D",
        charcoal: "#29332D",
        primary: {
          DEFAULT: "#8EAD91",
          hover: "#7E9D81",
          light: "#EAF1EB",
          dark: "#5A7C5F",
        },
        sage: {
          DEFAULT: "#8EAD91",
          hover: "#7E9D81",
          light: "#EAF1EB",
          dark: "#5A7C5F",
        },
        secondary: {
          DEFAULT: "#9DB7C5",
          hover: "#8BAAB9",
          light: "#EDF3F6",
          dark: "#688494",
        },
        dusty: {
          DEFAULT: "#9DB7C5",
          hover: "#8BAAB9",
          light: "#EDF3F6",
          dark: "#688494",
        },
        warning: {
          DEFAULT: "#D8B56A",
          hover: "#C9A454",
          light: "#FAF5E9",
          dark: "#96752C",
        },
        amber: {
          DEFAULT: "#D8B56A",
          hover: "#C9A454",
          light: "#FAF5E9",
          dark: "#96752C",
        },
        critical: {
          DEFAULT: "#C77B6B",
          hover: "#B66958",
          light: "#F9ECE9",
          dark: "#8E4435",
        },
        terracotta: {
          DEFAULT: "#C77B6B",
          hover: "#B66958",
          light: "#F9ECE9",
          dark: "#8E4435",
        },
        clay: "#C77B6B",
        sand: "#E5DEC9",
        mist: "#FCFBF7",
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Outfit", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 18px 40px -24px rgba(41, 51, 45, 0.22)",
        card: "0 4px 20px -2px rgba(41, 51, 45, 0.06), 0 2px 6px -1px rgba(41, 51, 45, 0.04)",
      },
    },
  },
  plugins: [],
};
