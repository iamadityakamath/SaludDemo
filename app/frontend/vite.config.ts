import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local only for now: the dev server listens on localhost and forwards /api to the FastAPI backend (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
