import { ExternalLink, Play } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Hit } from '@/lib/api'

interface Props {
  hit: Hit | null
  autoplay: boolean
  /** Changes every time the user picks a result, so choosing the same one again restarts playback. */
  nonce: number
}

export function Player({ hit, autoplay, nonce }: Props) {
  if (!hit) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-xl border border-dashed bg-muted/40 px-6 text-center text-sm text-muted-foreground">
        <span className="flex flex-col items-center gap-2">
          <Play className="size-5" />
          Pick a result and the video opens at that moment.
        </span>
      </div>
    )
  }

  const src = `https://www.youtube-nocookie.com/embed/${hit.video_id}?start=${hit.seek_seconds}&autoplay=${autoplay ? 1 : 0}&rel=0`

  return (
    <div className="space-y-3">
      <div className="aspect-video overflow-hidden rounded-xl border bg-black">
        <iframe
          key={`${hit.chunk_id}-${nonce}`}
          src={src}
          title={hit.title}
          className="size-full"
          allow="accelerometer; autoplay; encrypted-media; picture-in-picture; fullscreen"
          allowFullScreen
          referrerPolicy="strict-origin-when-cross-origin"
        />
      </div>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-medium">{hit.title}</h2>
          <p className="text-xs text-muted-foreground">Starts at {hit.timestamp}</p>
        </div>
        <Button asChild variant="outline" size="sm" className="shrink-0 font-normal">
          <a href={hit.url} target="_blank" rel="noreferrer">
            YouTube
            <ExternalLink />
          </a>
        </Button>
      </div>
    </div>
  )
}
