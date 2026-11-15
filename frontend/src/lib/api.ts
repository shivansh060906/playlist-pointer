export type Mode = 'hybrid' | 'dense' | 'bm25'

export interface Hit {
  chunk_id: string
  video_id: string
  title: string
  playlist_index: number
  start: number
  end: number
  seek_seconds: number
  timestamp: string
  url: string
  text: string
  score: number
  dense_rank: number | null
  bm25_rank: number | null
  rerank_score: number | null
}

export interface SearchResponse {
  query: string
  strategy: string
  mode: Mode
  rerank_applied: boolean
  expanded_query: string | null
  timings_ms: Record<string, number>
  hits: Hit[]
}

export interface Citation {
  index: number
  hit: Hit
}

export interface AskResponse {
  question: string
  answer: string
  model: string
  citations: Citation[]
  hits: Hit[]
  timings_ms: Record<string, number>
}

export interface Video {
  video_id: string
  title: string
  playlist_index: number
  duration: number
  chunk_count: number
}

export interface Health {
  status: string
  default_strategy: string
  strategies_built: string[]
  llm_configured: boolean
  llm_provider: string
}

export interface SearchOptions {
  strategy: string | null
  mode: Mode
  k: number
  rerank: boolean
  expand: boolean
  videoIds: string[]
}

export const DEFAULT_OPTIONS: SearchOptions = {
  strategy: null, // null = let the server use its default
  mode: 'hybrid',
  k: 6,
  rerank: false,
  expand: false,
  videoIds: [],
}

const BASE: string = import.meta.env.VITE_API_URL ?? ''

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, init)
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError('Could not reach the API.', 0)
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
      else if (Array.isArray(body?.detail)) detail = body.detail.map((d: { msg?: string }) => d.msg).join('; ')
    } catch {
      /* body was not JSON; keep the status text */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export const getHealth = (signal?: AbortSignal) => request<Health>('/api/health', { signal })
export const getVideos = (signal?: AbortSignal) => request<Video[]>('/api/videos', { signal })

export function search(q: string, o: SearchOptions, signal?: AbortSignal) {
  const p = new URLSearchParams({
    q,
    k: String(o.k),
    mode: o.mode,
    rerank: String(o.rerank),
    expand: String(o.expand),
  })
  if (o.strategy) p.set('strategy', o.strategy)
  o.videoIds.forEach((id) => p.append('video_id', id))
  return request<SearchResponse>(`/api/search?${p}`, { signal })
}

export function ask(question: string, o: SearchOptions, signal?: AbortSignal) {
  return request<AskResponse>('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      k: o.k,
      strategy: o.strategy,
      mode: o.mode,
      rerank: o.rerank,
      expand: o.expand,
      video_ids: o.videoIds.length ? o.videoIds : null,
    }),
    signal,
  })
}
