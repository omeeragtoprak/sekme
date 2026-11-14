import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the FastAPI backend (sekme serve, port 8765). changeOrigin stays false (the string
// shorthand turns it on): the backend accepts a write only when its Origin matches the Host it receives.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
