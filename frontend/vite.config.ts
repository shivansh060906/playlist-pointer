import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies /api to the FastAPI backend, so there is no CORS setup to think about.
// Point it somewhere else with:  VITE_PROXY_TARGET=http://localhost:9000 npm run dev
const target = process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, 'src') } },
  server: { port: 5173, proxy: { '/api': { target, changeOrigin: true } } },
  preview: { port: 5173, proxy: { '/api': { target, changeOrigin: true } } },
})
