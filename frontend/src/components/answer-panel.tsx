import { Fragment, type ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import type { AskResponse } from '@/lib/api'

interface Props {
  data: AskResponse
  onCite: (index: number) => void
}

/** Turn **bold** and [n] markers in the model's answer into real elements; everything else stays plain text. */
function renderInline(line: string, onCite: (n: number) => void, valid: Set<number>): ReactNode[] {
  return line.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
    const cite = /^\[(\d+)\]$/.exec(part)
    if (cite) {
      const n = Number(cite[1])
      return valid.has(n) ? (
        <button
          key={i}
          type="button"
          onClick={() => onCite(n)}
          className="mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded border bg-secondary px-1 text-[11px] font-medium tabular-nums text-secondary-foreground hover:bg-accent"
        >
          {n}
        </button>
      ) : (
        <Fragment key={i}>{part}</Fragment>
      )
    }
    const bold = /^\*\*([^*]+)\*\*$/.exec(part)
    return bold ? <strong key={i}>{bold[1]}</strong> : <Fragment key={i}>{part}</Fragment>
  })
}

export function AnswerPanel({ data, onCite }: Props) {
  const valid = new Set(data.hits.map((_, i) => i + 1))
  const paragraphs = data.answer.split(/\n{2,}/).filter((p) => p.trim())
  return (
    <Card className="gap-3">
      <div className="space-y-3 px-4 text-[15px] leading-relaxed">
        {paragraphs.map((p, i) => (
          <p key={i} className="whitespace-pre-line">
            {p.split('\n').map((line, j) => (
              <Fragment key={j}>
                {j > 0 && '\n'}
                {renderInline(line, onCite, valid)}
              </Fragment>
            ))}
          </p>
        ))}
      </div>
      <div className="flex items-center gap-2 px-4 text-xs text-muted-foreground">
        <Badge variant="outline" className="font-normal">
          {data.model}
        </Badge>
        <span>Generated from the passages below. Check the video before relying on it.</span>
      </div>
    </Card>
  )
}
