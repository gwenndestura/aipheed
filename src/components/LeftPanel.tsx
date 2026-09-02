import { useState } from "react";
import { RegionData, MunicipalityData, RISK_COLORS, RISK_LABELS } from "@/data/types";
import { regionsData, municipalitiesByProvince } from "@/data/mockData";
import { X, MapPin, TrendingUp, TrendingDown, BarChart3, ChevronRight, ArrowLeft, Briefcase, Plane, CloudRain, Store, AlertCircle, Info, Fish } from "lucide-react";
import { AboutModal } from "./AboutModal";
import { FeedbackModal } from "./FeedbackModal";
import { useCountUp } from "@/hooks/useCountUp";
import {
  PROVINCE_QUARTER_DATA,
  TRIGGER_CATEGORIES,
  getTriggerComposition,
  getTriggerBreakdown,
  isLimitedSignal,
  getRiskLabel,
  getQuarterMeta,
  getVulnerabilityDriver,
  ALERT_THRESHOLD,
} from "@/data/quarterData";
import { Sparkline } from "./RightAnalyticsPanel";

interface Props {
  selectedRegion: RegionData | null;
  previewRegion: RegionData | null;
  selectedMunicipality: MunicipalityData | null;
  onClose: () => void;
  onCloseMunicipality: () => void;
  onRegionClick: (r: RegionData) => void;
  onMunicipalityClick: (m: MunicipalityData) => void;
  rejectedProvinceIds?: Set<string>;
}

export function LeftPanel({
  selectedRegion,
  previewRegion,
  selectedMunicipality,
  onClose,
  onCloseMunicipality,
  onRegionClick,
  onMunicipalityClick,
  rejectedProvinceIds,
}: Props) {
  const activeRegion = selectedRegion || previewRegion;
  const isRegionRejected = !!activeRegion && !!rejectedProvinceIds?.has(activeRegion.id);
  const isMuniRejected = !!selectedMunicipality && !!rejectedProvinceIds?.has(selectedMunicipality.provinceId);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {selectedMunicipality ? (
          isMuniRejected ? (
            <RejectedForecastPlaceholder
              name={selectedMunicipality.provinceName}
              onBack={onCloseMunicipality}
            />
          ) : (
            <MunicipalityDetail
              municipality={selectedMunicipality}
              onBack={onCloseMunicipality}
            />
          )
        ) : activeRegion ? (
          isRegionRejected ? (
            <RejectedForecastPlaceholder
              name={activeRegion.name}
              isLocked={!!selectedRegion}
              onClose={onClose}
            />
          ) : (
            <RegionDetail
              region={activeRegion}
              isLocked={!!selectedRegion}
              onClose={onClose}
              onMunicipalityClick={onMunicipalityClick}
            />
          )
        ) : (
          <RegionalOverview onRegionClick={onRegionClick} />
        )}
      </div>

      {/* Footer: About + Feedback */}
      <div className="shrink-0 border-t border-border/40 px-4 py-2.5 flex items-center justify-between gap-2 bg-card/60">
        <AboutModal />
        <span className="text-[10px] text-muted-foreground/30">·</span>
        <FeedbackModal />
      </div>
    </div>
  );
}

/* ─── CALABARZON Overview (legacy single-panel rendering — still used as fallback) ─── */
function RegionalOverview({ onRegionClick }: { onRegionClick: (r: RegionData) => void }) {
  return (
    <div className="px-5 py-4 space-y-3">
      <RfiiScoreCardBody />
      <RiskDriversCardBody />
      <ProvinceRankingCardBody onRegionClick={onRegionClick} />
    </div>
  );
}

/* ─── Reusable card bodies (used by floating cards in Index.tsx) ─── */

export function RfiiScoreCardBody({ quarterId = "2026-Q2", quarterLabel }: { quarterId?: string; quarterLabel?: string } = {}) {
  const avgScore =
    PROVINCE_QUARTER_DATA.reduce((s, p) => s + (p.scoresByQuarter[quarterId] ?? 0), 0) /
    PROVINCE_QUARTER_DATA.length;
  const animScore = useCountUp(avgScore, 1000, 2);
  const C = 2 * Math.PI * 40;

  let high = 0, low = 0, limited = 0;
  PROVINCE_QUARTER_DATA.forEach((p) => {
    const s = p.scoresByQuarter[quarterId] ?? 0;
    if (getRiskLabel(s) === "HIGH") high++;
    else low++;
    if (isLimitedSignal(p.id, quarterId)) limited++;
  });

  const R = 70;
  const CIRC = Math.PI * R;
  const pct = Math.min(Math.max(animScore, 0), 1);
  const dash = pct * CIRC;
  const total = high + low;
  const highPct = total ? (high / total) * 100 : 0;

  const band =
    animScore >= 0.5 ? { label: "HIGH", color: "hsl(var(--risk-high))" }
    : { label: "LOW", color: "hsl(var(--risk-low))" };

  const ticks = Array.from({ length: 11 }, (_, i) => i / 10);

  // Position of pointer along 0..1 segmented track
  const trackPct = Math.min(Math.max(animScore, 0), 1) * 100;

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground/80 font-semibold">
          CALABARZON — Regional Average Risk Level
        </p>
        {quarterLabel && (
          <p className="text-[9px] text-muted-foreground/70 font-mono-num tabular-nums shrink-0">{quarterLabel}</p>
        )}
      </div>

      <div
        className="relative rounded-2xl border border-border/40 bg-gradient-to-br from-secondary/35 via-card/60 to-background/40 px-5 pt-5 pb-4 overflow-hidden"
        style={{
          ['--accent-color' as any]: band.color.replace('hsl(', '').replace(')', ''),
          animation: 'rfii-hero-pulse 3.6s ease-in-out infinite',
        }}
      >
        {/* Aurora sweep */}
        <div
          className="pointer-events-none absolute -inset-[30%] opacity-50 blur-2xl"
          style={{
            background: `conic-gradient(from 0deg, transparent 0deg, ${band.color}55 90deg, transparent 180deg, ${band.color}33 270deg, transparent 360deg)`,
            animation: 'rfii-aurora 14s linear infinite',
          }}
        />
        {/* Top edge accent line */}
        <div
          className="pointer-events-none absolute top-0 left-4 right-4 h-px"
          style={{ background: `linear-gradient(90deg, transparent, ${band.color}, transparent)` }}
        />

        <div className="relative flex flex-col">
          {/* Eyebrow */}
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block w-1.5 h-1.5 rounded-full"
              style={{ background: band.color, boxShadow: `0 0 10px ${band.color}, 0 0 20px ${band.color}aa` }}
            />
            <span className="text-[9px] uppercase tracking-[0.28em] text-muted-foreground/80 font-bold">
              Forecast Index
            </span>
          </div>

          {/* Hero score */}
          <div className="flex items-end gap-3">
            <span
              className="font-mono-num font-black tabular-nums leading-[0.85] tracking-tight"
              style={{
                fontSize: '5.5rem',
                color: band.color,
                textShadow: `0 0 32px ${band.color}cc, 0 0 70px ${band.color}66`,
                animation: 'rfii-score-breathe 3.2s ease-in-out infinite',
                transformOrigin: 'left bottom',
              }}
            >
              {animScore.toFixed(2)}
            </span>
            <span className="text-[10px] text-muted-foreground/70 font-mono-num pb-2">/ 1.00</span>
          </div>

          {/* Risk band pill */}
          <div className="mt-2.5">
            <span
              className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.25em] px-3 py-1 rounded-full border"
              style={{
                background: `${band.color}1a`,
                color: band.color,
                borderColor: `${band.color}66`,
                boxShadow: `0 0 14px ${band.color}55`,
              }}
            >
              <span className="w-1 h-1 rounded-full" style={{ background: band.color, boxShadow: `0 0 6px ${band.color}` }} />
              {band.label} · risk
            </span>
          </div>

          <p className="mt-2.5 text-[10px] text-muted-foreground/85 leading-snug">
            Average across all {PROVINCE_QUARTER_DATA.length} provinces
          </p>

          {/* Segmented 0–1 risk scale with pointer */}
          <div className="mt-3.5">
            <div className="relative h-2 w-full rounded-full overflow-visible">
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background: 'linear-gradient(90deg, hsl(var(--risk-low)) 0%, hsl(var(--risk-high)) 100%)',
                  opacity: 0.85,
                }}
              />
              {/* Segment divider at 0.5 */}
              {[50].map((p) => (
                <div key={p} className="absolute top-0 bottom-0 w-px bg-background/60" style={{ left: `${p}%` }} />
              ))}
              {/* Pointer */}
              <div
                className="absolute -top-1 -bottom-1 w-[3px] rounded-full"
                style={{
                  left: `calc(${trackPct}% - 1.5px)`,
                  background: 'hsl(var(--foreground))',
                  boxShadow: `0 0 10px ${band.color}, 0 0 20px ${band.color}cc`,
                  transition: 'left 1.4s cubic-bezier(0.22,1,0.36,1)',
                }}
              />
            </div>
            <div className="flex justify-between text-[8px] uppercase tracking-wider text-muted-foreground/60 font-mono-num mt-1.5">
              <span>0.0 low</span>
              <span>0.5</span>
              <span>1.0 high</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <SummaryChip label="HIGH" value={high} color="hsl(var(--risk-high))" />
        <SummaryChip label="LOW" value={low} color="hsl(var(--risk-low))" />
        <SummaryChip label="LIMITED" value={limited} color="hsl(var(--muted-foreground))" tip="Provinces with < 5 geocoded articles" />
      </div>
    </div>
  );
}

const TRIGGER_ICONS: Record<string, typeof Briefcase> = {
  market: Store,
  climate: CloudRain,
  fishkill: Fish,
  employment: Briefcase,
  ofw: Plane,
};

export function RiskDriversCardBody({
  provinceId = null,
  quarterId = "2026-Q2",
}: { provinceId?: string | null; quarterId?: string } = {}) {
  // Same source of truth as the right-panel SHAP card: all 5 trigger
  // categories, always shown (even at 0%), ranked high → low.
  const drivers = getTriggerBreakdown(provinceId, quarterId);

  return (
    <div key={`${provinceId ?? "region"}-${quarterId}`}>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-4 rounded-full bg-primary" />
        <h3 className="text-[11px] font-bold uppercase tracking-wider">Why is this Province at Risk?</h3>
        <span className="text-[9px] text-muted-foreground/60 ml-auto">Ranked</span>
      </div>
      <div className="space-y-2">
        {drivers.map((d, i) => {
          const Icon = TRIGGER_ICONS[d.key] ?? Store;
          return (
            <div key={d.key} className="grid grid-cols-[18px_1fr_42px] items-center gap-2">
              <Icon className="h-3.5 w-3.5 text-muted-foreground" />
              <div>
                <div className="flex items-center justify-between text-[10px] mb-1">
                  <span className="font-semibold">{d.label}</span>
                </div>
                <div className="h-2 bg-secondary/40 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full animate-bar-fill"
                    style={{
                      animationDelay: `${i * 80}ms`,
                      width: `${d.pct}%`,
                      background: d.color,
                      boxShadow: `0 0 6px ${d.color}80`,
                    }}
                  />
                </div>
              </div>
              <span className="font-mono-num text-[10px] font-bold tabular-nums text-right" style={{ color: d.color }}>
                {d.pct}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ProvinceRankingCardBody({ onRegionClick }: { onRegionClick: (r: RegionData) => void }) {
  const sortedProvinces = regionsData.slice().sort((a, b) => b.riskScore - a.riskScore);
  const maxScore = Math.max(...regionsData.map((r) => r.riskScore));

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-4 rounded-full bg-primary" />
        <h3 className="text-[11px] font-bold uppercase tracking-wider">Province Ranking</h3>
        <span className="text-[9px] text-muted-foreground/60 ml-auto">High → Low</span>
      </div>
      <div className="space-y-1.5">
        {sortedProvinces.map((r) => {
          const widthPct = (r.riskScore / maxScore) * 100;
          return (
            <button
              key={r.id}
              onClick={() => onRegionClick(r)}
              className="w-full text-left group px-1 py-1 rounded-md hover:bg-secondary/40 transition-colors"
            >
              <div className="flex items-center justify-between text-[10px] mb-1">
                <span className="font-semibold flex items-center gap-1.5">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{
                      backgroundColor: RISK_COLORS[r.riskLevel],
                      boxShadow: `0 0 6px ${RISK_COLORS[r.riskLevel]}`,
                    }}
                  />
                  {r.name}
                  <span className="text-[9px] text-muted-foreground ml-1 opacity-60">
                    {RISK_LABELS[r.riskLevel]}
                  </span>
                </span>
                <span className="font-mono-num font-bold tabular-nums" style={{ color: RISK_COLORS[r.riskLevel] }}>
                  {r.riskScore.toFixed(2)}
                </span>
              </div>
              <div className="h-2 bg-secondary/40 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full animate-bar-fill"
                  style={{ width: `${widthPct}%`, backgroundColor: RISK_COLORS[r.riskLevel] }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}


/* ─── Rejected forecast placeholder ─── */
function RejectedForecastPlaceholder({
  name,
  isLocked,
  onClose,
  onBack,
}: {
  name: string;
  isLocked?: boolean;
  onClose?: () => void;
  onBack?: () => void;
}) {
  return (
    <div className="px-5 py-4 space-y-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          {onBack && (
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors group mb-2"
            >
              <ArrowLeft className="h-3 w-3 group-hover:-translate-x-0.5 transition-transform" />
              Back
            </button>
          )}
          <p className="text-[9px] uppercase tracking-widest text-muted-foreground/70 font-semibold">Province of</p>
          <h2 className="text-base font-bold leading-tight">{name}</h2>
        </div>
        {isLocked && onClose && (
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-secondary transition-colors">
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        )}
      </div>
      <div className="rounded-xl border border-border/40 bg-secondary/20 px-4 py-8 flex flex-col items-center text-center space-y-2">
        <span className="text-3xl text-muted-foreground/30 select-none">—</span>
        <p className="text-[11px] font-semibold text-muted-foreground">No forecast available</p>
        <p className="text-[10px] text-muted-foreground/60 leading-snug max-w-[200px]">
          This quarter's forecast has been rejected by the admin and is not available for display.
        </p>
      </div>
    </div>
  );
}

/* ─── Province (Region) Detail ─── */
function RegionDetail({
  region,
  isLocked,
  onClose,
  onMunicipalityClick,
}: {
  region: RegionData;
  isLocked: boolean;
  onClose: () => void;
  onMunicipalityClick: (m: MunicipalityData) => void;
}) {
  const riskColor = RISK_COLORS[region.riskLevel];
  const momPositive = region.momChange > 0;
  const munis = municipalitiesByProvince(region.id).slice().sort((a, b) => b.riskScore - a.riskScore);

  const qid = "2026-Q2";
  const limited = isLimitedSignal(region.id, qid);
  const specLabel = getRiskLabel(region.riskScore);
  const isActiveAlert = region.riskScore >= ALERT_THRESHOLD;
  const meta = getQuarterMeta(qid);

  return (
    <div className="px-5 py-4 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-3">
          <div>
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground/70 font-semibold">Province of</p>
            <h2 className="text-base font-bold leading-tight">{region.name}</h2>
            <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
              <span
                className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full font-mono-num"
                style={{ backgroundColor: `${riskColor}20`, color: riskColor }}
                title="Forecast risk level (0.50 display cutoff)"
              >
                {specLabel} · risk
              </span>
              {isActiveAlert && (
                <span className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-destructive text-destructive-foreground severe-pulse" title={`Active alert: Risk Level ≥ ${ALERT_THRESHOLD}`}>
                  Active alert
                </span>
              )}
              {limited && (
                <span className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-muted text-muted-foreground flex items-center gap-1" title="Province has < 5 geocoded articles (Algorithm Rule 5)">
                  <AlertCircle className="h-2.5 w-2.5" />
                  Limited signal
                </span>
              )}
            </div>
          </div>
        </div>
        {isLocked && (
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-secondary transition-colors">
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        )}
      </div>

      {/* Confidence statement — spec rule 6 */}
      <div className="rounded-lg border border-border/40 bg-secondary/20 px-3 py-2 flex items-start gap-2">
        <Info className="h-3 w-3 text-primary mt-0.5 shrink-0" />
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          This is a <b className="text-foreground">{meta.horizonLabel}</b> estimate generated{" "}
          <b className="text-foreground">{meta.generatedOn}</b>; verification will be available by{" "}
          <b className="text-foreground">{meta.verificationBy}</b>.
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="rounded-xl px-3 py-3 border border-border/50 bg-secondary/30">
          <div className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">Risk Level Score</div>
          <div className="text-3xl font-extrabold mt-1 tracking-tight">{region.riskScore.toFixed(2)}</div>
          <div className="h-1.5 rounded-full mt-2 bg-muted overflow-hidden">
            <div className="h-full rounded-full transition-all" style={{ width: `${region.riskScore * 100}%`, backgroundColor: riskColor }} />
          </div>
        </div>
        <div className="rounded-xl px-3 py-3 border border-border/50 bg-secondary/30">
          <div className="flex items-center gap-1.5">
            {momPositive ? <TrendingUp className="h-3 w-3 text-destructive" /> : <TrendingDown className="h-3 w-3 text-risk-low" />}
            <span className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">QoQ Change</span>
          </div>
          <div className={`text-3xl font-extrabold mt-1 tracking-tight ${momPositive ? "text-destructive" : "text-risk-low"}`}>
            {momPositive ? "+" : ""}{region.momChange}%
          </div>
        </div>
      </div>

      {/* Population */}
      <div className="rounded-xl px-3 py-2.5 border border-border/50 bg-secondary/30 flex items-center justify-between">
        <div>
          <div className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">Population</div>
          <div className="text-sm font-bold mt-0.5 tabular-nums">{region.population.toLocaleString()}</div>
        </div>
        <BarChart3 className="h-4 w-4 text-muted-foreground/40" />
      </div>

      {/* Trigger composition — spec §3 */}
      <TriggerCompositionBar provinceId={region.id} />

      {/* Municipalities list */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-1 h-4 rounded-full bg-primary" />
            <h3 className="text-[11px] font-bold uppercase tracking-wider">
              Cities & Municipalities
            </h3>
          </div>
          <span className="text-[9px] text-muted-foreground tabular-nums">{munis.length}</span>
        </div>
        {/* Disaggregation note — spec rule 1 */}
        <p className="text-[9px] text-muted-foreground/70 italic mb-2 leading-snug">
          Spatially disaggregated estimates derived from the validated province forecast (Algorithm Rule 8).
        </p>
        <div className="space-y-1 max-h-[260px] overflow-y-auto pr-1">
          {munis.map((m) => {
            const povNorm = Math.min(1, m.povertyRate / 50);
            const denNorm = Math.min(1, m.population / 500000);
            const driver = getVulnerabilityDriver(povNorm, denNorm);
            return (
              <button
                key={m.id}
                onClick={() => onMunicipalityClick(m)}
                className="w-full flex items-center justify-between bg-secondary/30 hover:bg-secondary/70 border border-border/40 hover:border-border rounded-lg px-2.5 py-2 transition-all text-left group"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: RISK_COLORS[m.riskLevel] }}
                  />
                  <div className="min-w-0">
                    <div className="text-[11px] font-semibold truncate">{m.name}</div>
                    <div className="text-[9px] text-muted-foreground/70 flex items-center gap-1.5">
                      <span>{m.classification}</span>
                      <span className="opacity-40">·</span>
                      <span title="Vulnerability driver: derived from poverty vs density normalization" className="font-medium text-foreground/60">
                        {driver}
                      </span>
                      {limited && (
                        <span className="px-1 rounded bg-muted text-muted-foreground text-[8px] font-bold uppercase tracking-wider" title="Limited signal propagated from province (Rule 5)">
                          LIMITED
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className="text-[11px] font-bold tabular-nums font-mono-num" style={{ color: RISK_COLORS[m.riskLevel] }}>
                    {m.riskScore.toFixed(2)}
                  </span>
                  <ChevronRight className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </button>
            );
          })}
        </div>
      </div>

    </div>
  );
}

/* ─── Municipality / City Detail ─── */
function MunicipalityDetail({
  municipality,
  onBack,
}: {
  municipality: MunicipalityData;
  onBack: () => void;
}) {
  const riskColor = RISK_COLORS[municipality.riskLevel];
  const momPositive = municipality.momChange > 0;

  return (
    <div className="px-5 py-4 space-y-4">
      {/* Back to province */}
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors group"
      >
        <ArrowLeft className="h-3 w-3 group-hover:-translate-x-0.5 transition-transform" />
        Back to {municipality.provinceName}
      </button>

      {/* Header */}
      <div className="flex items-start gap-3">
        <div>
          <p className="text-[9px] uppercase tracking-widest text-muted-foreground/70 font-semibold">
            {municipality.classification} of
          </p>
          <h2 className="text-base font-bold leading-tight">{municipality.name}</h2>
          <p className="text-[10px] text-muted-foreground mt-0.5">{municipality.provinceName}, CALABARZON</p>
          <div className="flex items-center gap-2 mt-1.5">
            <span
              className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
              style={{ backgroundColor: `${riskColor}20`, color: riskColor }}
            >
              {RISK_LABELS[municipality.riskLevel]}
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="rounded-xl px-3 py-3 border border-border/50 bg-secondary/30">
          <div className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">Risk Level Score</div>
          <div className="text-3xl font-extrabold mt-1 tracking-tight">{municipality.riskScore.toFixed(2)}</div>
          <div className="h-1.5 rounded-full mt-2 bg-muted overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${municipality.riskScore * 100}%`, backgroundColor: riskColor }}
            />
          </div>
        </div>
        <div className="rounded-xl px-3 py-3 border border-border/50 bg-secondary/30">
          <div className="flex items-center gap-1.5">
            {momPositive ? (
              <TrendingUp className="h-3 w-3 text-destructive" />
            ) : (
              <TrendingDown className="h-3 w-3 text-risk-low" />
            )}
            <span className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">QoQ Change</span>
          </div>
          <div className={`text-3xl font-extrabold mt-1 tracking-tight ${momPositive ? "text-destructive" : "text-risk-low"}`}>
            {momPositive ? "+" : ""}{municipality.momChange}%
          </div>
        </div>
      </div>

      {/* Demographics */}
      <div className="rounded-xl px-3 py-2.5 border border-border/50 bg-secondary/30 space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">Population</div>
          <div className="text-[11px] font-bold tabular-nums">{municipality.population.toLocaleString()}</div>
        </div>
        <div className="flex items-center justify-between">
          <div className="text-[8px] text-muted-foreground uppercase tracking-widest font-semibold">Poverty Rate</div>
          <div className="text-[11px] font-bold tabular-nums">{municipality.povertyRate}%</div>
        </div>
      </div>

    </div>
  );
}

/* ─── Region summary chip (spec §1) ─── */
function SummaryChip({ label, value, color, tip }: { label: string; value: number; color: string; tip?: string }) {
  return (
    <div
      className="rounded-lg border border-border/40 bg-secondary/30 px-2 py-1.5 flex flex-col items-start"
      title={tip}
    >
      <span className="text-[8px] uppercase tracking-wider text-muted-foreground font-semibold">{label}</span>
      <span className="font-mono-num text-base font-bold tabular-nums leading-none mt-0.5" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

/* ─── Trigger composition bar (spec §3) ─── */
export function TriggerCompositionBar({ provinceId }: { provinceId: string | null }) {
  const props = getTriggerComposition(provinceId);
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <div className="w-1 h-4 rounded-full bg-primary" />
        <h3 className="text-[11px] font-bold uppercase tracking-wider">Trigger Composition</h3>
      </div>
      <div className="flex h-3 w-full rounded-md overflow-hidden border border-border/40">
        {TRIGGER_CATEGORIES.map((t, i) => (
          <div
            key={t.key}
            title={`${t.name} — ${(props[i] * 100).toFixed(0)}%`}
            style={{ width: `${props[i] * 100}%`, background: t.color }}
            className="animate-bar-fill"
          />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2">
        {TRIGGER_CATEGORIES.map((t, i) => (
          <div key={t.key} className="flex items-center gap-1.5 text-[9px]">
            <span className="w-2 h-2 rounded-sm shrink-0" style={{ background: t.color }} />
            <span className="truncate text-muted-foreground">{t.name}</span>
            <span className="ml-auto font-mono-num font-bold tabular-nums text-foreground/90">
              {(props[i] * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
