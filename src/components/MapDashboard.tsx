import { useEffect, useState, useMemo, useCallback } from "react";
import { regionsData, municipalitiesData } from "@/data/mockData";
import { FilterState, RegionData, MunicipalityData } from "@/data/types";
import { PhilippineMap } from "@/components/PhilippineMap";
import { LeftPanel, RfiiScoreCardBody, RiskDriversCardBody, ProvinceRankingCardBody } from "@/components/LeftPanel";
import {
  ShapNarrativeCardBody,
  NewsArticlesCardBody,
} from "@/components/RightAnalyticsPanel";
import { QuarterTimeSlider, buildQuarters } from "@/components/QuarterTimeSlider";
import { ForecastHistoryModal } from "@/components/ForecastHistoryModal";
import { RegionMetadataPopup } from "@/components/RegionMetadataPopup";
import { MapLegend } from "@/components/MapLegend";
import { MapControls } from "@/components/MapControls";
import { AboutModal } from "@/components/AboutModal";
import { FeedbackModal } from "@/components/FeedbackModal";
import { MapSearch } from "@/components/MapSearch";
import { getRejectedProvinceIdsForQuarter, onRejectionsChanged } from "@/lib/rejections";

function FloatingCard({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div
      className={`px-4 py-3.5 shadow-xl bg-card/85 text-card-foreground border border-border/60 ${className}`}
      style={{
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        borderRadius: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

const COL_W = 320;
const PEEK_W = 24;
const COL_TOP = 0;
const COL_BOTTOM = 64;

interface Props {
  showAboutFeedback?: boolean;
}

export function MapDashboard({ showAboutFeedback = true }: Props) {
  const [filters] = useState<FilterState>({ year: 2024, riskLevel: "all", region: "all" });
  const [selectedRegion, setSelectedRegion] = useState<RegionData | null>(null);
  const [previewRegion, setPreviewRegion] = useState<RegionData | null>(null);
  const [selectedMunicipality, setSelectedMunicipality] = useState<MunicipalityData | null>(null);
  const [metadataRegion] = useState<RegionData | null>(null);
  const [metadataOpen, setMetadataOpen] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  const quarters = useMemo(() => buildQuarters(), []);
  const initialIdx = Math.max(0, quarters.findIndex((q) => q.current));
  const [quarterId, setQuarterId] = useState<string>(quarters[initialIdx].id);
  const currentQuarter = quarters.find((q) => q.id === quarterId) ?? quarters[0];
  const quarterLabel = `${currentQuarter.label} ${currentQuarter.year} · ${currentQuarter.monthsLabel.toUpperCase()}`;

  const [rejectedIds, setRejectedIds] = useState<Set<string>>(() => getRejectedProvinceIdsForQuarter(quarterId));
  useEffect(() => {
    setRejectedIds(getRejectedProvinceIdsForQuarter(quarterId));
    return onRejectionsChanged(() => setRejectedIds(getRejectedProvinceIdsForQuarter(quarterId)));
  }, [quarterId]);

  const filteredRegions = useMemo(() => {
    return regionsData.filter((r) => {
      if (filters.riskLevel !== "all" && r.riskLevel !== filters.riskLevel) return false;
      if (filters.region !== "all" && r.id !== filters.region) return false;
      return true;
    });
  }, [filters]);

  const handleRegionClick = useCallback((r: RegionData) => {
    setSelectedRegion((prev) => (prev?.id === r.id ? null : r));
    setSelectedMunicipality(null);
  }, []);

  const handleMunicipalityClick = useCallback((m: MunicipalityData) => {
    setSelectedMunicipality((prev) => (prev?.id === m.id ? null : m));
  }, []);

  const handleHover = useCallback((r: RegionData | null) => setPreviewRegion(r), []);

  useEffect(() => {
    const onFocus = (e: Event) => {
      const detail = (e as CustomEvent).detail as { provinceId?: string; municipalityId?: string };
      if (detail.provinceId) {
        const r = regionsData.find((rr) => rr.id === detail.provinceId);
        if (r) setSelectedRegion(r);
      }
      if (detail.municipalityId) {
        const m = municipalitiesData.find((mm) => mm.id === detail.municipalityId);
        if (m) {
          const prov = regionsData.find((p) => p.id === m.provinceId);
          if (prov) setSelectedRegion(prov);
          setSelectedMunicipality(m);
        }
      }
    };
    window.addEventListener("aipheed:focus-region", onFocus as EventListener);
    return () => window.removeEventListener("aipheed:focus-region", onFocus as EventListener);
  }, []);

  const effectiveRegion = selectedRegion ?? previewRegion;
  const hasSelection = !!effectiveRegion;
  const bothCollapsed = leftCollapsed && rightCollapsed;

  return (
    <div className="h-full w-full relative overflow-hidden">
      {/* Map */}
      <div className="absolute inset-0">
        <PhilippineMap
          regions={filteredRegions}
          selectedRegion={selectedRegion}
          selectedMunicipality={selectedMunicipality}
          onHover={handleHover}
          onClick={handleRegionClick}
          onMunicipalityClick={handleMunicipalityClick}
          rejectedProvinceIds={rejectedIds}
        />
      </div>

      {/* LEFT floating column */}
      <div
        className={`absolute z-[1001] flex flex-col gap-2 transition-[left] duration-300 ${leftCollapsed ? "cursor-pointer" : "animate-slide-in-left"}`}
        style={{ top: COL_TOP, bottom: COL_BOTTOM, left: leftCollapsed ? -(COL_W - PEEK_W) : 0, width: COL_W }}
        onClick={() => leftCollapsed && setLeftCollapsed(false)}
        aria-label={leftCollapsed ? "Show left sidebar" : undefined}
      >
          {hasSelection ? (
            <div className="left-sidebar-scrollbar-left flex-1 min-h-0 overflow-y-auto thin-scrollbar">
              <FloatingCard className="p-0 overflow-hidden">
                <LeftPanel
                  selectedRegion={effectiveRegion}
                  previewRegion={previewRegion}
                  selectedMunicipality={selectedMunicipality}
                  onClose={() => { setSelectedRegion(null); setSelectedMunicipality(null); }}
                  onCloseMunicipality={() => setSelectedMunicipality(null)}
                  onRegionClick={handleRegionClick}
                  onMunicipalityClick={handleMunicipalityClick}
                  rejectedProvinceIds={rejectedIds}
                />
              </FloatingCard>
            </div>
          ) : (
            <>
              <div className="left-sidebar-scrollbar-left flex-1 min-h-0 overflow-y-auto thin-scrollbar flex flex-col gap-2">
                <FloatingCard>
                  <RfiiScoreCardBody quarterId={quarterId} quarterLabel={quarterLabel} />
                </FloatingCard>
                <FloatingCard>
                  <RiskDriversCardBody quarterId={quarterId} />
                </FloatingCard>
                <FloatingCard>
                  <ProvinceRankingCardBody onRegionClick={handleRegionClick} />
                </FloatingCard>
              </div>
              {showAboutFeedback && (
                <FloatingCard className="!py-2 shrink-0">
                  <div className="flex items-center justify-between gap-2">
                    <AboutModal />
                    <span className="text-[10px] text-muted-foreground/30">·</span>
                    <FeedbackModal />
                  </div>
                </FloatingCard>
              )}
            </>
          )}
      </div>

      {/* RIGHT floating column */}
      <div
        className={`absolute z-[1001] flex flex-col gap-2 overflow-y-auto thin-scrollbar transition-[right] duration-300 ${rightCollapsed ? "cursor-pointer" : "animate-slide-in-right"}`}
        style={{ top: COL_TOP, bottom: COL_BOTTOM, right: rightCollapsed ? -(COL_W - PEEK_W) : 0, width: COL_W }}
        onClick={() => rightCollapsed && setRightCollapsed(false)}
        aria-label={rightCollapsed ? "Show right sidebar" : undefined}
      >
          {effectiveRegion && rejectedIds.has(effectiveRegion.id) ? (
            <FloatingCard>
              <div className="py-8 text-center space-y-2">
                <p className="text-[11px] font-semibold text-muted-foreground">No forecast available</p>
                <p className="text-[10px] text-muted-foreground/60 leading-snug px-2">
                  This quarter's forecast for <span className="font-semibold">{effectiveRegion.name}</span> has been rejected by the admin.
                </p>
              </div>
            </FloatingCard>
          ) : (
            <>
              <FloatingCard>
                <ShapNarrativeCardBody
                  quarter={currentQuarter}
                  selectedRegion={effectiveRegion}
                  selectedMunicipality={selectedMunicipality}
                />
              </FloatingCard>
              <FloatingCard>
                <NewsArticlesCardBody selectedRegion={effectiveRegion} />
              </FloatingCard>
            </>
          )}
      </div>

      {/* Map search — top, beside left sidebar */}
      <div
        className="absolute z-[1001] animate-fade-in transition-[left] duration-300"
        style={{ left: (leftCollapsed ? PEEK_W : COL_W) + 12, top: 12 }}
      >
        <MapSearch />
      </div>

      <MapControls
        leftOffset={leftCollapsed ? PEEK_W : COL_W}
        rightOffset={rightCollapsed ? PEEK_W : COL_W}
        bothCollapsed={bothCollapsed}
        onToggleBoth={() => {
          const nextCollapsed = !bothCollapsed;
          setLeftCollapsed(nextCollapsed);
          setRightCollapsed(nextCollapsed);
        }}
      />

      <div
        className="absolute z-[1001] pointer-events-auto animate-slide-up transition-[left,right] duration-300"
        style={{ left: (leftCollapsed ? PEEK_W : COL_W) + 12, right: (rightCollapsed ? PEEK_W : COL_W) + 12, bottom: 12 }}
      >
        <QuarterTimeSlider
          value={quarterId}
          onChange={setQuarterId}
          quarters={quarters}
          onOpenHistory={() => setHistoryOpen(true)}
        />
      </div>

      <div className="absolute z-[1000] animate-fade-in transition-[left] duration-300" style={{ left: (leftCollapsed ? PEEK_W : COL_W) + 12, bottom: 72 }}>
        <MapLegend />
      </div>

      <div className="absolute bottom-2 right-3 z-[1000] text-[9px] text-muted-foreground/70 px-2 py-1">
        © OpenStreetMap · Protomaps
      </div>

      <RegionMetadataPopup region={metadataRegion} open={metadataOpen} onOpenChange={setMetadataOpen} />

      <ForecastHistoryModal
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        quarters={quarters}
        value={quarterId}
        onSelect={(id) => {
          setQuarterId(id);
          setHistoryOpen(false);
        }}
      />
    </div>
  );
}
