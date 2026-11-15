import type { ReactNode } from 'react'

const STOP = new Set([
  'the', 'and', 'for', 'are', 'with', 'what', 'when', 'why', 'how', 'does', 'did', 'can', 'could', 'should', 'would',
  'this', 'that', 'there', 'these', 'those', 'from', 'into', 'about', 'have', 'has', 'was', 'were', 'will', 'you', 'your',
  'its', 'not', 'but', 'then', 'than', 'just', 'actually', 'instead', 'use', 'using', 'get', 'got', 'make',
])

/** Words from the query worth highlighting in a snippet. */
export function queryTerms(query: string): string[] {
  const words = query.toLowerCase().match(/[a-z0-9_+#]+/g) ?? []
  return [...new Set(words.filter((w) => w.length >= 3 && !STOP.has(w)))]
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function Highlight({ text, terms }: { text: string; terms: string[] }): ReactNode {
  if (!terms.length) return text
  const re = new RegExp(`(${terms.map(escapeRe).join('|')})`, 'gi')
  return text.split(re).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded-[3px] bg-amber-200/70 px-0.5 text-inherit dark:bg-amber-400/25">
        {part}
      </mark>
    ) : (
      part
    ),
  )
}

export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`
}
