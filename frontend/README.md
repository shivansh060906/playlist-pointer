# Playlist Search: frontend

React + TypeScript + Vite, Tailwind and shadcn/ui. See the main README for the full setup.

```bash
npm install
npm run dev        # http://localhost:5173 (the backend must be running on port 8000)
```

The dev server proxies `/api` to `http://localhost:8000`.

| Need | How |
|---|---|
| Backend on another port | `VITE_PROXY_TARGET=http://localhost:8001 npm run dev` |
| Production build | `npm run build`, serve `dist/` |
| Frontend and API on different hosts | build with `VITE_API_URL=https://api.example.com`, and add the frontend origin to `CORS_ORIGINS` in `backend/.env` |