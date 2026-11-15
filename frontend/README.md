# Cherno C++ Search: frontend

React + TypeScript + Vite, styled with Tailwind and shadcn/ui components (the component source lives in
`src/components/ui`, so you can edit it directly). System font stack, light and dark themes.

## Run it

The backend must be running first (from `backend/`: `uvicorn app.main:app --port 8000`).

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` to `http://localhost:8000`, so there is nothing to configure and no CORS to fight.

| need | how |
|---|---|
| backend on another port | `VITE_PROXY_TARGET=http://localhost:9000 npm run dev` |
| deploy frontend and API on different hosts | build with `VITE_API_URL=https://api.example.com npm run build` (and add the frontend origin to `CORS_ORIGINS` in the backend `.env`) |
| production build | `npm run build`, then serve `dist/` (`npm run preview` to try it locally) |

## What it does

- **Find moments**: type what you are looking for; each result is a transcript passage with a timestamp button. Click it
  and the video on the right jumps to that moment. Query words are highlighted in the passages.
- **Ask a question**: the backend writes a short answer from the top passages. The small numbers in the answer
  are citations; click one to play that source. This tab is disabled until an LLM key is set in `backend/.env`.
- **All videos** limits the search to chosen lectures. **Options** holds matching mode, transcript splitting,
  number of results, rerank and glossary expansion. Rerank and expansion are off by default because they did not
  improve accuracy in the evaluation (see the main README).
- `/` focuses the search box. The URL keeps `?q=...&tab=...`, so a search can be shared or reloaded.

## Layout

```
src/
  App.tsx                    page state, API calls, layout
  lib/api.ts                 typed client for /api/health, /api/videos, /api/search, /api/ask
  lib/text.tsx               query-term highlighting, clock formatting
  components/
    hit-card.tsx             one passage + timestamp button
    player.tsx               YouTube embed that starts at the chosen second
    answer-panel.tsx         generated answer with clickable [n] citations
    video-filter.tsx         multi-select of lectures
    options-popover.tsx      advanced settings
    theme-toggle.tsx
    ui/                      shadcn/ui primitives (button, input, select, popover, tabs, ...)
```

## Notes

- The embedded player uses `youtube-nocookie.com`. Some videos disallow embedding; the "YouTube" button always works.
- Adding another shadcn component later: `npx shadcn@latest add <name>` works as normal (`components.json` is included).


## Off-topic queries

When the API returns `out_of_scope: true` the page shows a short refusal instead of results, with example queries and a "Show closest matches anyway" button. The threshold lives on the backend (`MIN_RELEVANCE`).
