import { SlidersHorizontal } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { DEFAULT_OPTIONS, type Mode, type SearchOptions } from '@/lib/api'

interface Props {
  value: SearchOptions
  onChange: (next: SearchOptions) => void
  strategies: string[]
  defaultStrategy: string
}

const STRATEGY_HELP: Record<string, string> = {
  fixed: 'Fixed-size windows',
  sliding: 'Overlapping windows',
  semantic: 'Split at topic changes',
}

const MODE_LABEL: Record<Mode, string> = {
  hybrid: 'Meaning + keywords',
  dense: 'Meaning only',
  bm25: 'Keywords only',
}

export function OptionsPopover({ value, onChange, strategies, defaultStrategy }: Props) {
  const set = (patch: Partial<SearchOptions>) => onChange({ ...value, ...patch })
  const changed =
    value.mode !== DEFAULT_OPTIONS.mode ||
    value.k !== DEFAULT_OPTIONS.k ||
    value.rerank ||
    value.expand ||
    (value.strategy !== null && value.strategy !== defaultStrategy)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="font-normal">
          <SlidersHorizontal />
          Options
          {changed && <span className="size-1.5 rounded-full bg-foreground" aria-label="changed from defaults" />}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 space-y-4">
        <div className="space-y-1.5">
          <Label>Matching</Label>
          <Select value={value.mode} onValueChange={(v) => set({ mode: v as Mode })}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
                <SelectItem key={m} value={m}>
                  {MODE_LABEL[m]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {strategies.length > 1 && (
          <div className="space-y-1.5">
            <Label>Transcript splitting</Label>
            <Select value={value.strategy ?? defaultStrategy} onValueChange={(v) => set({ strategy: v })}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {strategies.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s[0].toUpperCase() + s.slice(1)}
                    {STRATEGY_HELP[s] ? ` · ${STRATEGY_HELP[s]}` : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="space-y-1.5">
          <Label>Results</Label>
          <Select value={String(value.k)} onValueChange={(v) => set({ k: Number(v) })}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[3, 6, 10, 15].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-3 border-t pt-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-0.5">
              <Label htmlFor="opt-rerank">Rerank results</Label>
              <p className="text-xs text-muted-foreground">Second-pass model. Slower, and it did not improve accuracy in testing.</p>
            </div>
            <Switch id="opt-rerank" checked={value.rerank} onCheckedChange={(c) => set({ rerank: c })} />
          </div>
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-0.5">
              <Label htmlFor="opt-expand">Expand C++ terms</Label>
              <p className="text-xs text-muted-foreground">Adds related words (nullptr, ref, vtable...) to the keyword search.</p>
            </div>
            <Switch id="opt-expand" checked={value.expand} onCheckedChange={(c) => set({ expand: c })} />
          </div>
        </div>

        <Button variant="ghost" size="sm" className="w-full" disabled={!changed} onClick={() => onChange({ ...DEFAULT_OPTIONS, videoIds: value.videoIds })}>
          Reset to defaults
        </Button>
      </PopoverContent>
    </Popover>
  )
}
