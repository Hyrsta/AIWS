import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const BACKEND = process.env.VITE_BACKEND_URL || "http://127.0.0.1:18000";
const proxied = ["/health", "/catalog", "/jobs", "/preview", "/outputs"];

export default defineConfig({
  base: "",                                   // relative asset URLs → served from FastAPI at "/"
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    proxy: Object.fromEntries(proxied.map((p) => [p, { target: BACKEND, changeOrigin: true }])),
  },
  build: { outDir: "dist", emptyOutDir: true },
});
