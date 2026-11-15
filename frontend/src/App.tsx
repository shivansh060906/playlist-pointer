import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, CornerDownLeft, Search } from 'lucide-react'

import { AnswerPanel } from '@/components/answer-panel'
import { HitCard } from '@/components/hit-card'
import { OptionsPopover } from '@/components/options-popover'
import { Player } from '@/components/player'
import { ThemeToggle } from '@/components/theme-toggle'
import { VideoFilter } from '@/components/video-filter'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  ApiError,
  DEFAULT_OPTIONS,
  ask,
  getHealth,
  getVideos,
  search,
  type AskResponse,
  type Health,
  type Hit,
  type SearchOptions,
  type SearchResponse,
  type Video,
} from '@/lib/api'
import { queryTerms } from '@/lib/text'

type Tab = 'search' | 'ask'
type Result = { kind: 'search'; query: string; data: SearchResponse } | { kind: 'ask'; query: string; data: AskResponse }

const EXAMPLES = [
  "What's the difference between a reference and a pointer?",
  'unresolved external symbol',
  'Why do I need #pragma once in a header?',
  'How does a for loop actually run?',
  'What is constant folding?',
]

function initialParams(): { q: string; tab: Tab } {
  const p = new URLSearchParams(window.location.search)
  return { q: p.get('q') ?? '', tab: p.get('tab') === 'ask' ? 'ask' : 'search' }
}

export default function App() {
  const [initial] = useState(initialParams)
  const [query, setQuery] = useState(initial.q)
  const [tab, setTab] = useState<Tab>(initial.tab)
  const [options, setOptions] = useState<SearchOptions>(DEFAULT_OPTIONS)

  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [videos, setVideos] = useState<Video[]>([])

  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selected, setSelected] = useState<Hit | null>(null)
  const [autoplay, setAutoplay] = useState(false)
  const [nonce, setNonce] = useState(0)

  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Check the API once on load.
  useEffect(() => {
    const ctrl = new AbortController()
    getHealth(ctrl.signal)
      .then((h) => {
        setHealth(h)
        return getVideos(ctrl.signal)
      })
      .then((v) => setVideos([...v].sort((a, b) => a.playlist_index - b.playlist_index)))
      .catch((e) => {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setHealthError(e instanceof ApiError ? e.message : 'Unexpected error')
      })
    return () => ctrl.abort()
  }, [])

  // Press "/" anywhere to jump to the search box.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement
      if (e.key === '/' && !(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement)) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const run = useCallback(
    async (q: string, which: Tab, opts: SearchOptions) => {
      const text = q.trim()
      if (text.length < 3) return
      abortRef.current?.abort()
      const ctrl = new AbortController()
      abortRef.current = ctrl

      setLoading(true)
      setError(null)
      const url = new URL(window.location.href)
      url.searchParams.set('q', text)
      url.searchParams.set('tab', which)
      window.history.replaceState(null, '', url)

      try {
        const next: Result =
          which === 'ask'
            ? { kind: 'ask', query: text, data: await ask(text, opts, ctrl.signal) }
            : { kind: 'search', query: text, data: await search(text, opts, ctrl.signal) }
        setResult(next)
        setSelected(next.data.hits[0] ?? null)
        setAutoplay(false)
        setNonce((n) => n + 1)
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setResult(null)
        setError(e instanceof ApiError ? e.message : 'Something went wrong.')
      } finally {
        if (abortRef.current === ctrl) setLoading(false)
      }
    },
    [],
  )

  // If the page was opened with ?q=..., run it once the API check has finished.
  const ranInitial = useRef(false)
  useEffect(() => {
    if (health && !ranInitial.current && initial.q.trim().length >= 3) {
      ranInitial.current = true
      void run(initial.q, initial.tab, options)
    }
  }, [health, initial, options, run])

  function submit(e?: React.FormEvent) {
    e?.preventDefault()
    void run(query, tab, options)
  }

  function runExample(text: string) {
    setQuery(text)
    void run(text, tab, options)
  }

  function changeTab(next: string) {
    const t = next as Tab
    setTab(t)
    if (query.trim().length >= 3 && result) void run(query, t, options)
  }

  function changeOptions(next: SearchOptions) {
    setOptions(next)
    if (result) void run(query, tab, next)
  }

  function pick(hit: Hit) {
    setSelected(hit)
    setAutoplay(true)
    setNonce((n) => n + 1)
  }

  function jumpToSource(n: number) {
    const hit = result?.data.hits[n - 1]
    if (hit) pick(hit)
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  const terms = useMemo(() => queryTerms(result?.query ?? ''), [result])
  const hits = result?.data.hits ?? []
  const citedSet = useMemo(
    () => new Set(result?.kind === 'ask' ? result.data.citations.map((c) => c.index) : []),
    [result],
  )
  const noIndex = health !== null && health.strategies_built.length === 0
  const askUnavailable = health !== null && !health.llm_configured
  const timing = result?.data.timings_ms.total

  return (
    <div className="min-h-screen">
      <header className="border-b">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <div className="flex items-baseline gap-2">
            <h1 className="text-base font-semibold tracking-tight">Cherno C++ Search</h1>
            <span className="hidden text-sm text-muted-foreground sm:inline">Find the moment in the lecture</span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        <div className="space-y-3">
          <Tabs value={tab} onValueChange={changeTab}>
            <TabsList>
              <TabsTrigger value="search">Find moments</TabsTrigger>
              <TabsTrigger value="ask" disabled={askUnavailable} title={askUnavailable ? 'Add an LLM key to backend/.env to enable this' : undefined}>
                Ask a question
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <form onSubmit={submit} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={tab === 'ask' ? 'Ask anything about C++ as taught in the series' : 'Describe what you want to find, e.g. why is my linker complaining'}
                className="h-11 pr-16 pl-9 text-base"
                maxLength={tab === 'ask' ? 500 : 300}
                autoFocus
                aria-label="Search the lectures"
              />
              <kbd className="pointer-events-none absolute top-1/2 right-3 hidden -translate-y-1/2 rounded border bg-muted px-1.5 font-mono text-[11px] text-muted-foreground sm:block">
                /
              </kbd>
            </div>
            <Button type="submit" size="lg" className="h-11" disabled={loading || query.trim().length < 3}>
              {tab === 'ask' ? 'Ask' : 'Search'}
              <CornerDownLeft className="hidden sm:block" />
            </Button>
          </form>

          <div className="flex flex-wrap items-center gap-2">
            <VideoFilter videos={videos} value={options.videoIds} onChange={(ids) => changeOptions({ ...options, videoIds: ids })} />
            <OptionsPopover
              value={options}
              onChange={changeOptions}
              strategies={health?.strategies_built ?? []}
              defaultStrategy={health?.default_strategy ?? 'semantic'}
            />
            {result && !loading && (
              <span className="ml-auto text-xs text-muted-foreground tabular-nums">
                {hits.length} passages{timing !== undefined ? ` · ${Math.round(timing)} ms` : ''}
              </span>
            )}
          </div>
        </div>

        {healthError && (
          <Alert variant="destructive" className="mt-6">
            <AlertCircle />
            <AlertTitle>Can't reach the backend</AlertTitle>
            <AlertDescription>
              <p>
                Start it with <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">uvicorn app.main:app --port 8000</code> from the
                backend folder, then reload this page.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {noIndex && (
          <Alert className="mt-6">
            <AlertCircle />
            <AlertTitle>No index built yet</AlertTitle>
            <AlertDescription>
              <p>
                Run <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">python -m scripts.build_index --strategy semantic</code> in the
                backend folder.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert variant="destructive" className="mt-6">
            <AlertCircle />
            <AlertTitle>{tab === 'ask' ? "Couldn't get an answer" : "Couldn't search"}</AlertTitle>
            <AlertDescription>
              <p>{error}</p>
            </AlertDescription>
          </Alert>
        )}

        {!result && !loading && !error && !healthError && (
          <section className="mt-10 space-y-3">
            <p className="text-sm text-muted-foreground">Try one of these</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <Button key={ex} variant="outline" size="sm" className="h-auto whitespace-normal py-1.5 text-left font-normal" onClick={() => runExample(ex)}>
                  {ex}
                </Button>
              ))}
            </div>
            <p className="pt-2 text-sm text-muted-foreground">
              Search finds the exact moment in a video. Ask writes a short answer and links the moments it used.
            </p>
          </section>
        )}

        {(loading || result) && (
          <div className="mt-6 grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
            <div className="order-2 space-y-3 lg:order-1">
              {loading ? (
                <>
                  {tab === 'ask' && <Skeleton className="h-32 w-full rounded-xl" />}
                  {Array.from({ length: 4 }).map((_, i) => (
                    <Skeleton key={i} className="h-28 w-full rounded-xl" />
                  ))}
                </>
              ) : (
                result && (
                  <>
                    {result.kind === 'ask' && <AnswerPanel data={result.data} onCite={jumpToSource} />}
                    {result.kind === 'ask' && <h2 className="pt-2 text-sm font-medium text-muted-foreground">Sources</h2>}
                    {result.kind === 'search' && result.data.expanded_query && (
                      <p className="text-xs text-muted-foreground">Also searched for: {result.data.expanded_query}</p>
                    )}
                    {hits.length === 0 && <p className="py-8 text-center text-sm text-muted-foreground">Nothing matched. Try different words.</p>}
                    {hits.map((h, i) => (
                      <HitCard
                        key={h.chunk_id}
                        hit={h}
                        terms={terms}
                        active={selected?.chunk_id === h.chunk_id}
                        citation={result.kind === 'ask' ? i + 1 : undefined}
                        cited={citedSet.has(i + 1)}
                        onSelect={() => pick(h)}
                      />
                    ))}
                  </>
                )
              )}
            </div>

            <aside className="order-1 lg:order-2">
              <div className="lg:sticky lg:top-4">
                <Player hit={selected} autoplay={autoplay} nonce={nonce} />
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  )
}
