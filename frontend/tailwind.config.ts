import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./features/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Coastal palette
        deep: "#062d3a",
        lagoon: "#0e7490",
        reef: "#0891b2",
        sand: "#f4e9d8",
        alert: "#dc2626",
      },
    },
  },
  plugins: [],
};

export default config;
