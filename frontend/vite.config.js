import path from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
export default defineConfig({
    plugins: [react()],
    base: '/',
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 5173,
        proxy: {
            '/api/v1': {
                target: 'https://app-ttggsmwa.fly.dev',
                changeOrigin: true,
                secure: false,
                rewrite: (path) => path.replace(/^\/api\/v1/, '/api/v1'),
            },
        },
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: undefined,
            },
        },
    },
});
