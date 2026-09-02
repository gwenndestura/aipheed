import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { History, ArrowRight } from "lucide-react";
import { Quarter } from "./QuarterTimeSlider";
import {
  getCalabarzonAverage,
  getRiskLabel,
  getTriggerBreakdown,
  getQuarterMeta,
} from "@/data/quarterData";

interface Props {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  quarters: Quarter[];
  value: string;
  onSelect: (id: string) => void;
}

/**
 * Browse past (and current) quarterly forecasts. Rows carry the verified
 * regional risk index, band, quarter-over-quarter change, dominant trigger and
 * the forecast's issue / verification dates. Selecting a row loads that quarter
 * on the map.
 */
export function ForecastHistoryModal({ open, onOpenChange, quarters, value, onSelect }: Props) {
  const cur = quarters.find((q) => q.current);
  // Actuals + current, most recent first.
  const rows = quarters.filter((q) => !q.forecast).slice().reverse();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="z-[1200] max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-primary" />
            <DialogTitle className="text-sm">Forecast history — CALABARZON</DialogTitle>
          </div>
          <DialogDescription className="text-[11px] leading-relaxed pt-1">
            Past quarterly forecasts and their verified regional risk. The current quarter
            {cur ? ` (${cur.label} ${cur.year})` : ""} is tracked from today's date — later
            quarters remain forecasts. Select a row to load that quarter on the map.
          </DialogDescription>
        </DialogHeader>

        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead>
              <tr className="text-[9px] uppercase tracking-wider text-muted-foreground border-b border-border/50">
                <th className="text-left py-1.5 pr-3">Quarter</th>
                <th className="text-right px-2">Index</th>
                <th className="text-center px-2">Band</th>
                <th className="text-right px-2">QoQ</th>
                <th className="text-left px-2">Top driver</th>
                <th className="text-left px-2 hidden sm:table-cell">Issued → Verified</th>
                <th className="px-1" />
              </tr>
            </thead>
            <tbody>
              {rows.map((q) => {
                const score = getCalabarzonAverage(q.id);
                const prevId = quarters[quarters.findIndex((x) => x.id === q.id) - 1]?.id;
                const delta = prevId ? score - getCalabarzonAverage(prevId) : 0;
                const band = getRiskLabel(score);
                const top = getTriggerBreakdown(null, q.id)[0];
                const meta = getQuarterMeta(q.id);
                const selected = q.id === value;
                return (
                  <tr
                    key={q.id}
                    onClick={() => onSelect(q.id)}
                    className={`border-b border-border/25 cursor-pointer transition-colors hover:bg-secondary/40 ${
                      selected ? "bg-secondary/50" : ""
                    }`}
                  >
                    <td className="py-2 pr-3 font-semibold whitespace-nowrap">
                      {q.label} {q.year}
                      {q.current && (
                        <span className="ml-1.5 text-[8px] font-bold uppercase tracking-wider text-risk-low">now</span>
                      )}
                    </td>
                    <td className="text-right px-2 font-mono-num tabular-nums">{score.toFixed(2)}</td>
                    <td className="text-center px-2">
                      <span
                        className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded-full ${
                          band === "HIGH" ? "bg-risk-high/15 text-risk-high" : "bg-risk-low/15 text-risk-low"
                        }`}
                      >
                        {band}
                      </span>
                    </td>
                    <td
                      className={`text-right px-2 font-mono-num tabular-nums ${
                        delta > 0.001 ? "text-risk-high" : delta < -0.001 ? "text-risk-low" : "text-muted-foreground"
                      }`}
                    >
                      {delta > 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </td>
                    <td className="px-2 whitespace-nowrap">
                      {top?.label ?? "—"}{" "}
                      {top && <span className="text-muted-foreground font-mono-num">{top.pct}%</span>}
                    </td>
                    <td className="px-2 text-[10px] text-muted-foreground whitespace-nowrap hidden sm:table-cell">
                      {meta.generatedOn} → {meta.verificationBy}
                    </td>
                    <td className="px-1 text-right">
                      <ArrowRight className="h-3 w-3 text-muted-foreground inline" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="mt-3 text-[9px] italic text-muted-foreground leading-snug">
          Regional index = mean of the 5 provincial forecasts. Quarter-over-quarter change is versus the
          preceding quarter. AI-generated estimates; not official food-insecurity classifications.
        </p>
      </DialogContent>
    </Dialog>
  );
}
