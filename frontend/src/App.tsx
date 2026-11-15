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
  addPlaylist,
  deletePlaylist,
  getPlaylists,
  getVideos,
  isBusy,
  search,
  type AskResponse,
  type Health,
  type Hit,
  type Playlist,
  type SearchOptions,
  type SearchResponse,
  type Video,
} from '@/lib/api'
import { PlaylistSwitcher } from '@/components/playlist-switcher'
import { OutOfScope } from '@/components/out-of-scope'
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

function initialParams(): { q: string; tab: Tab; pl: string | null } {
  const p = new URLSearchParams(window.location.search)
  return { q: p.get('q') ?? '', tab: p.get('tab') === 'ask' ? 'ask' : 'search', pl: p.get('pl') }
}

const PL_KEY = 'playlist'
function savedPlaylist(): string | null {
  try {
    return localStorage.getItem(PL_KEY)
  } catch {
    return null
  }
}

export default function App() {
  const [initial] = useState(initialParams)
  const [query, setQuery] = useState(initial.q)
  const [tab, setTab] = useState<Tab>(initial.tab)
  const [options, setOptions] = useState<SearchOptions>(DEFAULT_OPTIONS)

  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [videos, setVideos] = useState<Video[]>([])
  const [playlists, setPlaylists] = useState<Playlist[] | null>(null)
  const [playlist, setPlaylist] = useState<string | null>(null)

  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selected, setSelected] = useState<Hit | null>(null)
  const [autoplay, setAutoplay] = useState(false)
  const [nonce, setNonce] = useState(0)
  const [showClosest, setShowClosest] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Check the API once on load, then pick a playlist: ?pl= from the URL, else the last one used, else the first ready one.
  useEffect(() => {
    const ctrl = new AbortController()
    getHealth(ctrl.signal)
      .then((h) => {
        setHealth(h)
        return getPlaylists(ctrl.signal)
      })
      .then((list) => {
        setPlaylists(list)
        const ready = list.filter((p) => p.status === 'ready')
        const wanted = [initial.pl, savedPlaylist()].find((id) => id && ready.some((p) => p.id === id))
        setPlaylist(wanted ?? ready[0]?.id ?? null)
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setHealthError(e instanceof ApiError ? e.message : 'Unexpected error')
      })
    return () => ctrl.abort()
  }, [initial])

  // Load the video list whenever the playlist changes.
  useEffect(() => {
    if (!playlist) return
    const ctrl = new AbortController()
    getVideos(playlist, ctrl.signal)
      .then((v) => setVideos([...v].sort((a, b) => a.playlist_index - b.playlist_index)))
      .catch((e) => {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setVideos([])
      })
    return () => ctrl.abort()
  }, [playlist])

  // While a playlist is being indexed, poll its progress; select it when it finishes.
  const busyId = playlists?.find(isBusy)?.id ?? null
  useEffect(() => {
    if (!busyId) return
    const timer = setInterval(() => {
      getPlaylists()
        .then((list) => {
          setPlaylists(list)
          const done = list.find((p) => p.id === busyId)
          if (done?.status === 'ready') choosePlaylist(done.id)
        })
        .catch(() => {})
    }, 1500)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busyId])

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
      if (playlist) url.searchParams.set('pl', playlist)
      window.history.replaceState(null, '', url)

      try {
        const next: Result =
          which === 'ask'
            ? { kind: 'ask', query: text, data: await ask(text, opts, playlist, ctrl.signal) }
            : { kind: 'search', query: text, data: await search(text, opts, playlist, ctrl.signal) }
        setResult(next)
        setShowClosest(false)
        setSelected(next.data.out_of_scope ? null : (next.data.hits[0] ?? null))
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
    [playlist],
  )

  // If the page was opened with ?q=..., run it once the API check has finished.
  const ranInitial = useRef(false)
  useEffect(() => {
    if (health && playlist && !ranInitial.current && initial.q.trim().length >= 3) {
      ranInitial.current = true
      void run(initial.q, initial.tab, options)
    }
  }, [health, playlist, initial, options, run])

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

  function choosePlaylist(id: string) {
    setPlaylist(id)
    setResult(null)
    setError(null)
    setSelected(null)
    setShowClosest(false)
    setVideos([])
    setOptions((o) => ({ ...o, videoIds: [] }))
    try {
      localStorage.setItem(PL_KEY, id)
    } catch {
      /* storage unavailable; the choice just won't be remembered */
    }
    const url = new URL(window.location.href)
    url.searchParams.set('pl', id)
    url.searchParams.delete('q')
    window.history.replaceState(null, '', url)
  }

  async function handleAdd(url: string) {
    const info = await addPlaylist(url)
    setPlaylists((list) => [...(list ?? []).filter((p) => p.id !== info.id), info])
    if (info.status === 'ready') choosePlaylist(info.id)
  }

  async function handleRemove(id: string) {
    await deletePlaylist(id)
    const list = await getPlaylists()
    setPlaylists(list)
    if (id === playlist) {
      const next = list.find((p) => p.status === 'ready')
      if (next) choosePlaylist(next.id)
      else {
        setPlaylist(null)
        setResult(null)
        setVideos([])
      }
    }
  }

  function revealClosest() {
    setShowClosest(true)
    setSelected(result?.data.hits[0] ?? null)
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
  const refused = !!result?.data.out_of_scope && !showClosest
  const noIndex = playlists !== null && !playlists.some((p) => p.status === 'ready') && !playlists.some(isBusy)
  const examples = playlist === 'default' ? EXAMPLES : videos.slice(0, 4).map((v) => v.title)
  const askUnavailable = health !== null && !health.llm_configured
  const timing = result?.data.timings_ms.total

  return (
    <div className="min-h-screen">
      <header className="border-b">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <div className="flex items-baseline gap-2">
            <h1 className="text-base font-semibold tracking-tight">Playlist Search</h1>
            <span className="hidden text-sm text-muted-foreground md:inline">Find the moment in the video</span>
          </div>
          <div className="flex items-center gap-2">
            {playlists && (
              <PlaylistSwitcher
                playlists={playlists}
                selected={playlist}
                onSelect={choosePlaylist}
                onAdd={handleAdd}
                onRemove={(id) => void handleRemove(id)}
              />
            )}
            <ThemeToggle />
          </div>
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
                placeholder={tab === 'ask' ? 'Ask a question about this playlist' : 'Describe what you want to find'}
                className="h-11 pr-16 pl-9 text-base"
                maxLength={tab === 'ask' ? 500 : 300}
                autoFocus
                aria-label="Search the playlist"
              />
              <kbd className="pointer-events-none absolute top-1/2 right-3 hidden -translate-y-1/2 rounded border bg-muted px-1.5 font-mono text-[11px] text-muted-foreground sm:block">
                /
              </kbd>
            </div>
            <Button type="submit" size="lg" className="h-11" disabled={loading || !playlist || query.trim().length < 3}>
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
                {refused ? 'Not related to this playlist' : `${hits.length} passages`}{timing !== undefined ? ` · ${Math.round(timing)} ms` : ''}
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
            <AlertTitle>No playlist yet</AlertTitle>
            <AlertDescription>
              <p>Use the playlist button at the top right and paste a YouTube playlist link. It takes about a second per video.</p>
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

        {!result && !loading && !error && !healthError && playlist && examples.length > 0 && (
          <section className="mt-10 space-y-3">
            <p className="text-sm text-muted-foreground">Try one of these</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((ex) => (
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
          <div
            className={
              refused
                ? 'mt-6 grid grid-cols-[minmax(0,1fr)] gap-6'
                : 'mt-6 grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]'
            }
          >
            <div className="order-2 space-y-3 lg:order-1">
              {loading ? (
                <>
                  {tab === 'ask' && <Skeleton className="h-32 w-full rounded-xl" />}
                  {Array.from({ length: 4 }).map((_, i) => (
                    <Skeleton key={i} className="h-28 w-full rounded-xl" />
                  ))}
                </>
              ) : (
                result && refused ? (
                  <OutOfScope
                    message={result.kind === 'ask' ? result.data.answer : undefined}
                    hasHits={hits.length > 0}
                    onShowClosest={revealClosest}
                    onPickExample={runExample}
                    examples={examples}
                  />
                ) : (
                  result && (
                  <>
                    {result.data.out_of_scope && (
                      <p className="text-xs text-muted-foreground">
                        Closest matches only. These probably don't answer your query.
                      </p>
                    )}
                    {result.kind === 'ask' && !result.data.out_of_scope && <AnswerPanel data={result.data} onCite={jumpToSource} />}
                    {result.kind === 'ask' && !result.data.out_of_scope && <h2 className="pt-2 text-sm font-medium text-muted-foreground">Sources</h2>}
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
                )
              )}
            </div>

            {!refused && (
              <aside className="order-1 lg:order-2">
                <div className="lg:sticky lg:top-4">
                  <Player hit={selected} autoplay={autoplay} nonce={nonce} />
                </div>
              </aside>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
