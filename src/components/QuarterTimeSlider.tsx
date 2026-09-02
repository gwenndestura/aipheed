import { useMemo } from "react";
import { Sparkles, Lock, History } from "lucide-react";
import { QUARTER_IDS } from "@/data/quarterData";

export interface Quarter {
  id: string;
  year: number;
  q: 1 | 2 | 3 | 4;
  label: string;
  monthsLabel: string;
  forecast?: boolean;
  locked?: boolean;
  current?: boolean;
  past?: boolean; // a quarter earlier than "now" — an actual, not a forecast
}

interface Props {
  value: string;
  onChange: (id: string) => void;
  quarters?: Quarter[];
  onOpenHistory?: () => void;
}

const MONTHS = ["Jan–Mar", "Apr–Jun", "Jul–Sep", "Oct–Dec"];

// Chronological rank of a "YYYY-Qn" id.
const ord = (id: string) => {
  const [y, q] = id.split("-Q");
  return Number(y) * 4 + (Number(q) - 1);
};

/** The quarter that contains `now` in real time — e.g. September ⇒ Q3. */
export function currentQuarterId(now: Date = new Date()): string {
  return `${now.getFullYear()}-Q${Math.floor(now.getMonth() / 3) + 1}`;
}

/**
 * Timeline quarters, spanning the full available data range. `current` tracks
 * real time: quarters before it are actuals (history), quarters after it are
 * forecasts. `current` is clamped to the data range so it always resolves.
 */
export function buildQuarters(now: Date = new Date()): Quarter[] {
  const ids = QUARTER_IDS as readonly string[];
  const first = ord(ids[0]);
  const last = ord(ids[ids.length - 1]);
  const cur = Math.min(Math.max(ord(currentQuarterId(now)), first), last);

  return ids.map((id) => {
    const [yStr, qStr] = id.split("-Q");
    const q = Number(qStr) as 1 | 2 | 3 | 4;
    const o = ord(id);
    return {
      id,
      year: Number(yStr),
      q,
      label: `Q${q}`,
      monthsLabel: MONTHS[q - 1],
      forecast: o > cur,
      current: o === cur,
      past: o < cur,
      locked: false,
    };
  });
}

export function QuarterTimeSlider({ value, onChange, quarters, onOpenHistory }: Props) {
  const qs = useMemo(() => quarters ?? buildQuarters(), [quarters]);
  const idx = Math.max(0, qs.findIndex((q) => q.id === value));
  const current = qs[idx] ?? qs[0];
  const pct = qs.length > 1 ? (idx / (qs.length - 1)) * 100 : 0;

  return (
    <div className="px-4 pt-2 pb-1 select-none">
      {/* Floating label row */}
      <div className="flex items-center justify-center gap-2 mb-3">
        {current.forecast && (
          <span className="flex items-center gap-1 text-[8px] font-bold uppercase tracking-[0.18em] px-1.5 py-0.5 rounded-full bg-primary/15 text-primary border border-primary/30">
            <Sparkles className="h-2.5 w-2.5" />
            Forecast
          </span>
        )}
        {current.current && (
          <span className="text-[8px] font-bold uppercase tracking-[0.18em] px-1.5 py-0.5 rounded-full bg-risk-low/15 text-risk-low border border-risk-low/30">
            Current
          </span>
        )}
        {current.past && (
          <span className="text-[8px] font-bold uppercase tracking-[0.18em] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground border border-border/50">
            Actual
          </span>
        )}
        <span className="font-mono-num text-[11px] font-semibold tabular-nums text-foreground/90 drop-shadow-[0_2px_8px_rgba(0,0,0,0.7)]">
          {current.label} {current.year}
        </span>
        <span className="font-mono-num text-[10px] text-muted-foreground/80">· {current.monthsLabel}</span>
        {onOpenHistory && (
          <button
            onClick={onOpenHistory}
            title="Browse past forecasts"
            className="ml-1 flex items-center gap-1 text-[8px] font-bold uppercase tracking-[0.15em] px-1.5 py-0.5 rounded-full border border-border/50 text-muted-foreground hover:text-foreground hover:border-foreground/40 transition-colors"
          >
            <History className="h-2.5 w-2.5" />
            History
          </button>
        )}
      </div>

      {/* Track */}
      <div className="relative h-6">
        {/* Glowing base line */}
        <div
          className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-px"
          style={{
            background: "linear-gradient(to right, hsl(var(--border) / 0.4), hsl(var(--border) / 0.6), hsl(var(--border) / 0.4))",
          }}
        />
        {/* Filled glowing portion */}
        <div
          className="absolute top-1/2 -translate-y-1/2 left-0 h-[2px] rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(to right, hsl(var(--primary) / 0.6), hsl(var(--primary)))",
            boxShadow: "0 0 8px hsl(var(--primary) / 0.8)",
          }}
        />
        {qs.map((q, i) => {
          const left = qs.length > 1 ? (i / (qs.length - 1)) * 100 : 0;
          const isActive = i === idx;
          const isPast = i < idx;
          const locked = q.locked;
          return (
            <button
              key={q.id}
              onClick={() => !locked && onChange(q.id)}
              disabled={locked}
              title={locked ? "Data not yet available." : `${q.label} ${q.year}`}
              className="absolute top-0 h-full -translate-x-1/2 group flex flex-col items-center"
              style={{ left: `${left}%`, cursor: locked ? "not-allowed" : "pointer" }}
              aria-label={`${q.label} ${q.year}${locked ? " (locked)" : ""}`}
            >
              <span className="absolute top-1/2 -translate-y-1/2 flex items-center justify-center">
                {locked ? (
                  <span className="w-2.5 h-2.5 rounded-full border border-muted-foreground/40 bg-transparent flex items-center justify-center">
                    <Lock className="h-1.5 w-1.5 text-muted-foreground/60" />
                  </span>
                ) : isActive ? (
                  <span
                    className="w-3 h-3 rounded-full bg-primary"
                    style={{ boxShadow: "0 0 0 3px hsl(var(--primary) / 0.25), 0 0 12px hsl(var(--primary) / 0.9)" }}
                  />
                ) : q.forecast ? (
                  <span className="w-2.5 h-2.5 rounded-full border border-dashed border-primary/70 bg-primary/10" />
                ) : q.current ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-risk-low/70 ring-2 ring-risk-low/25" />
                ) : q.past || isPast ? (
                  <span className="w-2 h-2 rounded-full bg-foreground/60" />
                ) : (
                  <span className="w-2 h-2 rounded-full border border-muted-foreground/50 bg-transparent" />
                )}
              </span>
              <span
                className={`absolute top-[calc(50%+12px)] font-mono-num text-[9px] tabular-nums whitespace-nowrap transition-colors ${
                  isActive
                    ? "text-primary font-bold"
                    : locked
                      ? "text-muted-foreground/40 italic"
                      : q.forecast
                        ? "text-primary/80 italic"
                        : "text-muted-foreground/70 group-hover:text-foreground"
                }`}
              >
                {q.label}'{String(q.year).slice(-2)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
