import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the PumpkinSpice FastAPI backend (pumpkinspice serve).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8077",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist" },
});
