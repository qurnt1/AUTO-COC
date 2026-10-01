import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const configuredBackend = mode === "development"
    ? loadEnv(mode, process.cwd(), ["AUTO_COC_DEV_BACKEND_URL"]).AUTO_COC_DEV_BACKEND_URL
    : undefined;
  const proxyTarget = configuredBackend ? new URL(configuredBackend) : null;
  if (proxyTarget && (proxyTarget.hostname !== "127.0.0.1" || proxyTarget.protocol !== "http:" || proxyTarget.username || proxyTarget.password || proxyTarget.pathname !== "/" || proxyTarget.search || proxyTarget.hash)) {
    throw new Error("AUTO_COC_DEV_BACKEND_URL doit cibler la racine d’un service HTTP sur 127.0.0.1.");
  }

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      ...(proxyTarget ? { proxy: { "/api": { target: proxyTarget.origin, changeOrigin: true } } } : {}),
    },
    build: { outDir: "dist", emptyOutDir: true },
    test: {
      environment: "node",
      include: ["src/**/*.test.ts"],
    },
  };
});
