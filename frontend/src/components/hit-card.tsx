import { Play } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import type { Hit } from '@/lib/api'
import { Highlight } from '@/lib/text'
import { cn } from '@/lib/utils'

interface Props {
  hit: Hit
  terms: string[]
  active: boolean
  /** The [n] this source carries in a generated answer, if any. */
  citation?: number
  cited?: boolean
  onSelect: () => void
}

function rankNote(hit: Hit): string | null {
  const parts: string[] = []
  if (hit.dense_rank) parts.push(`#${hit.dense_rank} by meaning`)
  if (hit.bm25_rank) parts.push(`#${hit.bm25_rank} by keywords`)
  return parts.length ? parts.join(' · ') : null
}

export function HitCard({ hit, terms, active, citation, cited, onSelect }: Props) {
  const note = rankNote(hit)
  return (
    <Card
      id={citation ? `source-${citation}` : undefined}
      onClick={onSelect}
      className={cn(
        'cursor-pointer gap-3 transition-colors hover:bg-accent/50',
        active && 'border-foreground/40 bg-accent/60 ring-1 ring-foreground/15',
      )}
    >
      <div className="flex items-start justify-between gap-3 px-4">
        <div className="min-w-0 space-y-0.5">
          <div className="flex items-center gap-2">
            {citation !== undefined && (
              <Badge variant={cited ? 'default' : 'outline'} className="tabular-nums">
                {citation}
              </Badge>
            )}
            <h3 className="truncate text-sm font-medium">{hit.title}</h3>
          </div>
          {note && <p className="text-xs text-muted-foreground">{note}</p>}
        </div>
        <Button
          size="sm"
          variant={active ? 'default' : 'secondary'}
          className="shrink-0 tabular-nums"
          onClick={(e) => {
            e.stopPropagation()
            onSelect()
          }}
        >
          <Play className="size-3.5" />
          {hit.timestamp}
        </Button>
      </div>
      <p className="line-clamp-4 px-4 text-sm leading-relaxed text-muted-foreground">
        <Highlight text={hit.text} terms={terms} />
      </p>
    </Card>
  )
}
