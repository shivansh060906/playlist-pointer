import { ChevronDown } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { Video } from '@/lib/api'

interface Props {
  videos: Video[]
  value: string[]
  onChange: (ids: string[]) => void
}

export function VideoFilter({ videos, value, onChange }: Props) {
  const label =
    value.length === 0
      ? 'All videos'
      : value.length === 1
        ? (videos.find((v) => v.video_id === value[0])?.title ?? '1 video')
        : `${value.length} videos`

  const toggle = (id: string, on: boolean) => onChange(on ? [...value, id] : value.filter((v) => v !== id))

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="max-w-[16rem] font-normal">
          <span className="truncate">{label}</span>
          <ChevronDown className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 p-0">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <span className="text-sm font-medium">Search in</span>
          <Button variant="ghost" size="sm" className="h-7 px-2" disabled={value.length === 0} onClick={() => onChange([])}>
            Clear
          </Button>
        </div>
        <ul className="max-h-72 overflow-y-auto p-1">
          {videos.map((v) => (
            <li key={v.video_id}>
              <Label className="cursor-pointer items-start gap-2.5 rounded-sm px-2 py-1.5 font-normal leading-snug hover:bg-accent">
                <Checkbox
                  className="mt-0.5"
                  checked={value.includes(v.video_id)}
                  onCheckedChange={(c) => toggle(v.video_id, c === true)}
                />
                <span>
                  <span className="mr-1.5 tabular-nums text-muted-foreground">{v.playlist_index}.</span>
                  {v.title}
                </span>
              </Label>
            </li>
          ))}
          {videos.length === 0 && <li className="px-2 py-3 text-sm text-muted-foreground">No videos indexed yet.</li>}
        </ul>
      </PopoverContent>
    </Popover>
  )
}
