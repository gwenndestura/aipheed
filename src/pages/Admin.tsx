import { useEffect, useMemo, useState, useCallback } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import {
  MessageSquareText,
  ClipboardCheck,
  LogOut,
  Eye,
  FileText,
  CheckCircle2,
  CircleDashed,
  X,
  Map as MapIcon,
  Menu,
  Sun,
  Moon,
  Star,
} from "lucide-react";
import { regionsData } from "@/data/mockData";
import { RISK_COLORS, RISK_LABELS, FilterState, RegionData, MunicipalityData } from "@/data/types";
import { EarlyWarningBanner } from "@/components/EarlyWarningBanner";
import { PhilippineMap } from "@/components/PhilippineMap";
import { LeftPanel } from "@/components/LeftPanel";
import { MapDashboard } from "@/components/MapDashboard";
import { MapControls } from "@/components/MapControls";
import { MapLegend } from "@/components/MapLegend";
import { SidebarPeek } from "@/components/SidebarPeek";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { PSTClock } from "@/components/PSTClock";
import logoDark from "@/assets/logo-dark.png";
import logoLight from "@/assets/logo-light.png";
import logoIcon from "@/assets/logo-icon.png";

import { QuarterTimeSlider, buildQuarters } from "@/components/QuarterTimeSlider";
import { RightAnalyticsPanel } from "@/components/RightAnalyticsPanel";
import { RegionMetadataPopup } from "@/components/RegionMetadataPopup";
import { toast } from "@/hooks/use-toast";
import { loadFeedback, SUS_QUESTIONS, type FeedbackSubmission } from "@/components/FeedbackModal";
import municipalitiesGeo from "@/data/calabarzon-municipalities.json";
import { DEFAULT_SHAP, getShapForProvince } from "@/data/quarterData";
import {
  loadRejections,
  addRejection,
  removeRejection,
  isRejected,
  type RejectionEntry,
  onRejectionsChanged,
} from "@/lib/rejections";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

type Section = "map" | "review" | "feedback";
const THEME_KEY = "aipheed_theme";
const ADMIN_DOMAINS = ["calabarzon.da.gov.ph"];

function getCurrentUser(): { email: string; role: string } | null {
  try {
    const user = JSON.parse(sessionStorage.getItem("aipheed_user") ?? "null");
    return user?.role === "admin" ? user : null;
  } catch {
    return null;
  }
}

function isAuthorizedAdmin(email?: string) {
  if (!email) return false;
  const lower = email.toLowerCase();
  return ADMIN_DOMAINS.some((d) => lower.endsWith(`@${d}`));
}

interface ReviewItem {
  province: string;
  riskLevel: "low" | "moderate" | "high" | "severe";
  riskScore: number;
  status: "Staged" | "Approved" | "Rejected";
  quarter: string;
  date: string;
}

const REVIEW_YEARS = [2024, 2025, 2026] as const;
const QUARTER_END_DATES: Record<string, string> = {
  Q1: "Mar 15", Q2: "Jun 15", Q3: "Sep 15", Q4: "Dec 15",
};

const initialReviewItems: ReviewItem[] = REVIEW_YEARS.flatMap((year) =>
  (["Q1", "Q2", "Q3", "Q4"] as const).flatMap((q) =>
    regionsData.map((r, idx) => {
      // Deterministic status seed by year+quarter+province
      const seed = (year + q.charCodeAt(1) + idx) % 5;
      const status: ReviewItem["status"] =
        year === 2026 && q === "Q4" && r.name === "Quezon"
          ? "Staged"
          : year === 2026 && q === "Q2" && r.name === "Quezon"
          ? "Approved"
          : year < 2026
          ? "Approved"
          : seed < 2 ? "Staged" : seed === 4 ? "Staged" : "Approved";
      // Vary score slightly by quarter for realism
      const drift = ((q.charCodeAt(1) - 49) * 0.01) + ((2026 - year) * -0.02);
      return {
        province: r.name,
        riskLevel: r.riskLevel,
        riskScore: Math.max(0, Math.min(1, r.riskScore + drift)),
        status,
        quarter: `${year}-${q}`,
        date: `${QUARTER_END_DATES[q]}, ${year}`,
      };
    })
  )
);

// Seed localStorage with the deterministic initial rejections so the map
// reflects them immediately. Uses the same seed formula as initialReviewItems
// so the source of truth for "Rejected" is localStorage, not local state.
const REJECTIONS_SEED_KEY = "aipheed_rejections_seeded_v2";
if (!localStorage.getItem(REJECTIONS_SEED_KEY)) {
  // Clear any rejections from a previous seed version before re-seeding.
  localStorage.removeItem("aipheed_rejections");
  REVIEW_YEARS.forEach((year) => {
    (["Q1", "Q2", "Q3", "Q4"] as const).forEach((q) => {
      regionsData.forEach((r, idx) => {
        const seed = (year + q.charCodeAt(1) + idx) % 5;
        const shouldSeed =
          year >= 2026 &&
          !(year === 2026 && q === "Q4" && r.name === "Quezon") &&
          !(year === 2026 && q === "Q2" && r.name === "Quezon") &&
          seed === 4;
        if (shouldSeed) {
          addRejection({
            province: r.name,
            provinceId: r.name.toLowerCase(),
            quarter: `${year}-${q}`,
            reason: "Pre-existing rejection",
            timestamp: new Date().toISOString(),
          });
        }
      });
    });
  });
  localStorage.setItem(REJECTIONS_SEED_KEY, "1");
}

export default function Admin({ initialSection }: { initialSection?: Section } = {}) {
  const [section, setSection] = useState<Section>(initialSection ?? "map");
  const navigate = useNavigate();
  const user = getCurrentUser();

  if (!user) return <Navigate to="/login" replace />;
  if (!isAuthorizedAdmin(user.email)) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-background">
        <div className="max-w-sm bg-card border border-border/60 rounded-2xl p-6 text-center">
          <h1 className="text-base font-bold mb-2">Access denied</h1>
          <p className="text-[11px] text-muted-foreground mb-4">
            Admin access is restricted to <span className="font-bold">@calabarzon.da.gov.ph</span> emails.
          </p>
          <button
            onClick={() => navigate("/")}
            className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-[11px] font-bold"
          >
            Back to map
          </button>
        </div>
      </div>
    );
  }

  const logout = () => {
    sessionStorage.removeItem("aipheed_user");
    navigate("/login");
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-background overflow-hidden">
      <AdminNavbar
        user={user}
        section={section}
        onSection={setSection}
        onLogout={logout}
      />

      <EarlyWarningBanner onReview={() => setSection("review")} />

      <main className="flex-1 overflow-hidden relative">
        <div className={section === "map" ? "h-full w-full" : "hidden"}>
          <AdminMapView />
        </div>
        {section === "review" && <div className="h-full overflow-auto"><ReviewSection /></div>}
        {section === "feedback" && <div className="h-full overflow-auto"><FeedbackSection /></div>}
      </main>
    </div>
  );
}

/* ---------- Admin Navbar — mirrors TopNavbar styling ---------- */
function AdminNavbar({
  user,
  section,
  onSection,
  onLogout,
}: {
  user: { email: string; role: string };
  section: Section;
  onSection: (s: Section) => void;
  onLogout: () => void;
}) {
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem(THEME_KEY) as "dark" | "light") ?? "dark"
  );
  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    localStorage.setItem(THEME_KEY, theme);
    window.dispatchEvent(new CustomEvent("aipheed:theme", { detail: { theme } }));
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <header className="relative h-16 shrink-0 bg-card/95 backdrop-blur-sm border-b border-border/50 z-[1002] flex items-center px-4 gap-3">
      {/* Brand → admin map (stays within admin, no sign-out) */}
      <button
        onClick={() => onSection("map")}
        className="flex items-center mr-2 hover:opacity-90 transition-opacity shrink-0"
        aria-label="aiPHeed admin home"
      >
        <img src={logoIcon} alt="aiPHeed" className="sm:hidden h-9 w-9 object-contain" />
        <img src={logoDark} alt="aiPHeed" className="hidden sm:block dark-only h-10 w-auto object-contain" />
        <img src={logoLight} alt="aiPHeed" className="hidden sm:block light-only h-10 w-auto object-contain" />
        <span className="hidden md:inline ml-2 text-[9px] uppercase tracking-widest text-primary font-bold">Admin</span>
      </button>

      {/* Nav links — centered */}
      <nav className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1">
        <NavBtn label="Mapping" active={section === "map"} onClick={() => onSection("map")} />
        <NavBtn label="Review" active={section === "review"} onClick={() => onSection("review")} />
        <NavBtn label="Feedback" active={section === "feedback"} onClick={() => onSection("feedback")} />
      </nav>

      <div className="flex-1" />

      {/* PST clock */}
      <PSTClock />

      {/* Hamburger */}
      <Sheet>
        <SheetTrigger className="p-1.5 rounded-md hover:bg-secondary/60 transition-colors" aria-label="Menu">
          <Menu className="h-4 w-4" />
        </SheetTrigger>
        <SheetContent side="right" className="w-[280px] bg-card border-l border-border/50 z-[1100]">
          <SheetHeader>
            <SheetTitle className="text-sm">Admin Menu</SheetTitle>
          </SheetHeader>
          <div className="mt-6 space-y-1">
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground/60 font-semibold px-2 mb-1">
              Account
            </p>
            <div className="px-2 py-2 rounded-md bg-secondary/40 flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-[9px] font-bold text-primary">
                {user.email.slice(0, 2).toUpperCase()}
              </div>
              <span className="text-[10px] text-muted-foreground truncate">{user.email}</span>
            </div>
            <button
              onClick={onLogout}
              className="mt-1 w-full flex items-center gap-2 px-2 py-2 rounded-md text-[11px] hover:bg-secondary/60 text-foreground"
            >
              <LogOut className="h-3.5 w-3.5" /> Sign out
            </button>

            <div className="pt-3 mt-3 border-t border-border/40">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground/60 font-semibold px-2 mb-2">
                Appearance
              </p>
              <div className="flex items-center justify-between gap-3 px-2 py-2 rounded-md">
                <span className="flex items-center gap-3">
                  {theme === "dark" ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
                  <span className="text-[11px] font-medium">
                    {theme === "dark" ? "Dark mode" : "Light mode"}
                  </span>
                </span>
                <button
                  onClick={toggleTheme}
                  role="switch"
                  aria-checked={theme === "light"}
                  aria-label="Toggle dark / light mode"
                  className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors border ${
                    theme === "light"
                      ? "bg-primary/30 border-primary/50"
                      : "bg-secondary border-border/50"
                  }`}
                >
                  <span className="absolute left-1 text-[8px]">🌙</span>
                  <span className="absolute right-1 text-[8px]">☀️</span>
                  <span
                    className={`relative z-10 inline-block h-4 w-4 transform rounded-full bg-card shadow-md transition-transform ${
                      theme === "light" ? "translate-x-5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </header>
  );
}

/* ---------- Admin Map View — same floating-card layout as user page ---------- */
function AdminMapView() {
  return <MapDashboard showAboutFeedback={false} />;
}

function NavBtn({ label, active, onClick }: { label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`group relative px-4 py-2 text-[12px] font-semibold uppercase tracking-[0.14em] transition-colors ${
        active ? "text-primary" : "text-foreground/80 hover:text-primary"
      }`}
    >
      <span>{label}</span>
      <span
        className={`pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-1 h-[2px] rounded-full bg-primary transition-all duration-300 ${
          active ? "w-6 opacity-100" : "w-0 opacity-0 group-hover:w-6 group-hover:opacity-100"
        }`}
      />
    </button>
  );
}

function SectionHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="px-6 pt-6 pb-4 border-b border-border/40">
      <h1 className="text-lg font-bold tracking-tight">{title}</h1>
      <p className="text-[11px] text-muted-foreground mt-0.5">{subtitle}</p>
    </div>
  );
}

/* (Feedback section removed per spec — admin no longer surfaces SUS feedback) */

/* ---------- Review ---------- */
type ReviewTab = "queue" | "log";

function ReviewSection() {
  const [items, setItems] = useState<ReviewItem[]>(initialReviewItems);
  const [open, setOpen] = useState<ReviewItem | null>(null);
  const [year, setYear] = useState<number>(2026);
  const [quarter, setQuarter] = useState<"all" | "Q1" | "Q2" | "Q3" | "Q4">("all");
  const [tab, setTab] = useState<ReviewTab>("queue");

  // Sync rejected status from persisted log so a fresh load reflects rejections.
  const [rejections, setRejections] = useState<RejectionEntry[]>(() => loadRejections());
  useEffect(() => onRejectionsChanged(() => setRejections(loadRejections())), []);

  const itemsWithStatus = useMemo(
    () =>
      items.map((it) =>
        rejections.some((r) => r.province === it.province && r.quarter === it.quarter)
          ? { ...it, status: "Rejected" as const }
          : it
      ),
    [items, rejections]
  );

  const statusBadge = (s: ReviewItem["status"]) => {
    const cfg = {
      Approved: { icon: <CheckCircle2 className="h-3 w-3" />, color: "text-risk-low" },
      Staged: { icon: <CircleDashed className="h-3 w-3" />, color: "text-muted-foreground" },
      Rejected: { icon: <X className="h-3 w-3" />, color: "text-destructive" },
    }[s];
    return (
      <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold ${cfg.color}`}>
        {cfg.icon} {s}
      </span>
    );
  };

  const approve = (it: ReviewItem) => {
    // Approving clears any prior rejection
    removeRejection(it.province.toLowerCase(), it.quarter);
    setItems((prev) => prev.map((x) => (x.province === it.province && x.quarter === it.quarter ? { ...x, status: "Approved" } : x)));
    setOpen(null);
    toast({ title: "Report approved", description: `${it.province} ${it.quarter} forecast is now public-ready.` });
  };

  const unpublish = (it: ReviewItem) => {
    setItems((prev) => prev.map((x) => (x.province === it.province && x.quarter === it.quarter ? { ...x, status: "Staged" } : x)));
    setOpen(null);
    toast({ title: "Publishing undone", description: `${it.province} ${it.quarter} forecast reverted to Staged.` });
  };

  const reject = (it: ReviewItem, reason: string, notes?: string) => {
    addRejection({
      province: it.province,
      provinceId: it.province.toLowerCase(),
      quarter: it.quarter,
      reason,
      notes,
      timestamp: new Date().toISOString(),
    });
    setItems((prev) => prev.map((x) => (x.province === it.province && x.quarter === it.quarter ? { ...x, status: "Rejected" } : x)));
    setOpen(null);
    toast({
      variant: "destructive",
      title: "Forecast rejected",
      description: `Forecast for ${it.province} ${formatQuarterLabel(it.quarter)} has been rejected and will not be published.`,
    });
  };

  const filtered = itemsWithStatus.filter(
    (it) => it.quarter.startsWith(`${year}-`) && (quarter === "all" || it.quarter.endsWith(`-${quarter}`))
  );
  const sorted = [...filtered].sort((a, b) => {
    const qa = Number(a.quarter.split("-Q")[1]);
    const qb = Number(b.quarter.split("-Q")[1]);
    if (qa !== qb) return qb - qa;
    return a.province.localeCompare(b.province);
  });

  return (
    <div>
      <SectionHeader title="Review" subtitle="Approve, monitor, and audit risk reports per province" />

      <div className="px-6 pt-4">
        <div className="inline-flex rounded-lg border border-border/50 bg-card p-0.5">
          {(["queue", "log"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-[11px] font-bold rounded-md transition-colors ${
                tab === t ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t === "queue" ? "Review Queue" : "Rejection Log"}
            </button>
          ))}
        </div>
      </div>

      {tab === "queue" ? (
        <div className="p-6 pt-4">
          <div className="mb-4 flex items-center gap-3 flex-wrap">
            <label className="text-[11px] font-semibold text-muted-foreground">Year</label>
            <select
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="h-8 rounded-md border border-border/60 bg-card px-2 text-[11px] font-semibold focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {REVIEW_YEARS.slice().reverse().map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
            <label className="text-[11px] font-semibold text-muted-foreground ml-2">Quarter</label>
            <select
              value={quarter}
              onChange={(e) => setQuarter(e.target.value as typeof quarter)}
              className="h-8 rounded-md border border-border/60 bg-card px-2 text-[11px] font-semibold focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="all">All quarters</option>
              <option value="Q4">Q4</option>
              <option value="Q3">Q3</option>
              <option value="Q2">Q2</option>
              <option value="Q1">Q1</option>
            </select>
            <span className="ml-auto text-[10px] text-muted-foreground">
              {sorted.length} report{sorted.length === 1 ? "" : "s"} · {year}
              {quarter !== "all" ? ` · ${quarter}` : ""}
            </span>
          </div>
          <div className="bg-card border border-border/50 rounded-xl overflow-hidden">
            <table className="w-full text-[11px]">
              <thead className="bg-secondary/40 text-muted-foreground">
                <tr>
                  <th className="text-left font-semibold px-4 py-2.5">Province</th>
                  <th className="text-left font-semibold px-4 py-2.5">Quarter</th>
                  <th className="text-left font-semibold px-4 py-2.5">Risk Level</th>
                  <th className="text-left font-semibold px-4 py-2.5">Risk Level Score</th>
                  <th className="text-left font-semibold px-4 py-2.5">Status</th>
                  <th className="text-left font-semibold px-4 py-2.5">Action</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <tr key={`${r.province}-${r.quarter}`} className="border-t border-border/40 hover:bg-secondary/30">
                    <td className="px-4 py-2.5 font-medium">{r.province}</td>
                    <td className="px-4 py-2.5 tabular-nums text-muted-foreground">
                      <span className="font-semibold text-foreground">{r.quarter}</span>
                      <span className="block text-[9px] text-muted-foreground/70">{r.date}</span>
                    </td>
                    <td className="px-4 py-2.5">
                      <span
                        className="px-2 py-0.5 rounded-full border text-[10px] font-bold"
                        style={{
                          color: RISK_COLORS[r.riskLevel],
                          borderColor: `${RISK_COLORS[r.riskLevel]}55`,
                          background: `${RISK_COLORS[r.riskLevel]}15`,
                        }}
                      >
                        {RISK_LABELS[r.riskLevel]}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 tabular-nums font-bold" style={{ color: RISK_COLORS[r.riskLevel] }}>
                      {r.riskScore.toFixed(2)}
                    </td>
                    <td className="px-4 py-2.5">{statusBadge(r.status)}</td>
                    <td className="px-4 py-2.5">
                      <button
                        onClick={() => setOpen(r)}
                        className="text-primary hover:underline text-[11px] font-semibold flex items-center gap-1"
                      >
                        <Eye className="h-3 w-3" /> View Details
                      </button>
                    </td>
                  </tr>
                ))}
                {sorted.length === 0 && (
                  <tr><td colSpan={6} className="px-4 py-6 text-center text-muted-foreground text-[11px]">No reports for {year}{quarter !== "all" ? ` ${quarter}` : ""}.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <RejectionLog
          rejections={rejections}
          onClear={(r) => {
            removeRejection(r.provinceId, r.quarter);
            setItems((prev) =>
              prev.map((x) =>
                x.province === r.province && x.quarter === r.quarter
                  ? { ...x, status: "Staged" }
                  : x
              )
            );
          }}
        />
      )}

      {open && (
        <ReviewDetailModal
          item={open}
          onClose={() => setOpen(null)}
          onApprove={() => approve(open)}
          onUnpublish={() => unpublish(open)}
          onReject={(reason, notes) => reject(open, reason, notes)}
        />
      )}
    </div>
  );
}

function formatQuarterLabel(qid: string): string {
  const [y, q] = qid.split("-");
  return `${q} ${y}`;
}

function RejectionLog({ rejections, onClear }: { rejections: RejectionEntry[]; onClear: (r: RejectionEntry) => void }) {
  return (
    <div className="p-6 pt-4">
      <div className="bg-card border border-border/50 rounded-xl overflow-hidden">
        <table className="w-full text-[11px]">
          <thead className="bg-secondary/40 text-muted-foreground">
            <tr>
              <th className="text-left font-semibold px-4 py-2.5">Province</th>
              <th className="text-left font-semibold px-4 py-2.5">Quarter</th>
              <th className="text-left font-semibold px-4 py-2.5">Reason</th>
              <th className="text-left font-semibold px-4 py-2.5">Notes</th>
              <th className="text-left font-semibold px-4 py-2.5">Timestamp</th>
              <th className="text-left font-semibold px-4 py-2.5">Action</th>
            </tr>
          </thead>
          <tbody>
            {rejections.length === 0 && (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-muted-foreground italic">No rejections logged.</td></tr>
            )}
            {rejections.map((r) => (
              <tr key={`${r.provinceId}-${r.quarter}-${r.timestamp}`} className="border-t border-border/40 hover:bg-secondary/30">
                <td className="px-4 py-2.5 font-medium">{r.province}</td>
                <td className="px-4 py-2.5 tabular-nums">{r.quarter}</td>
                <td className="px-4 py-2.5">{r.reason}</td>
                <td className="px-4 py-2.5 text-muted-foreground">{r.notes || "—"}</td>
                <td className="px-4 py-2.5 tabular-nums text-muted-foreground">{new Date(r.timestamp).toLocaleString()}</td>
                <td className="px-4 py-2.5">
                  <button
                    onClick={() => onClear(r)}
                    className="text-primary hover:underline text-[11px] font-semibold"
                  >
                    Restore
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-muted-foreground mt-2">Audit log of rejected forecasts. Restoring removes the rejection so the report can be re-reviewed.</p>
    </div>
  );
}

function ReviewDetailModal({
  item,
  onClose,
  onApprove,
  onUnpublish,
  onReject,
}: {
  item: ReviewItem;
  onClose: () => void;
  onApprove: () => void;
  onUnpublish: () => void;
  onReject: (reason: string, notes?: string) => void;
}) {
  const region = regionsData.find((r) => r.name === item.province);
  const provinceId = item.province.toLowerCase();
  const [rejectOpen, setRejectOpen] = useState(false);

  // Real municipality / city names — full list per province from the canonical CALABARZON dataset.
  const muniData = useMemo(() => {
    const list = (municipalitiesGeo as Array<{ id: string; name: string; pid: string }>)
      .filter((m) => m.pid === provinceId);
    return list
      .map((m, i) => {
        const seed = (m.id.length * 13 + i * 7) % 100;
        return { name: m.name, score: +(0.25 + (seed / 100) * 0.55).toFixed(2) };
      })
      .sort((a, b) => b.score - a.score);
  }, [provinceId]);

  const shapData = useMemo(() => {
    const raw = getShapForProvince(provinceId);
    return raw.length ? raw : DEFAULT_SHAP;
  }, [provinceId]);

  const trend = region?.historicalTrend ?? [];

  return (
    <div className="fixed inset-0 z-[2000] bg-background/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-card border border-border/60 rounded-2xl shadow-2xl w-full max-w-3xl max-h-[88vh] overflow-hidden flex flex-col">
        <div className="px-5 py-3.5 border-b border-border/40 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold">{item.province} — Risk Review</h2>
            <p className="text-[10px] text-muted-foreground">
              <span className="font-semibold text-primary">{item.quarter}</span> · {item.date} · Risk Level {item.riskScore.toFixed(2)} · {RISK_LABELS[item.riskLevel]} · Status: {item.status}
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-md hover:bg-secondary/60"><X className="h-4 w-4" /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-2 gap-4">
          <div className="col-span-2 p-3 rounded-lg border border-border/50 bg-secondary/30 flex items-center gap-3">
            <div className="bg-primary/15 rounded-lg p-2"><FileText className="h-4 w-4 text-primary" /></div>
            <div>
              <p className="text-[12px] font-semibold">{item.province} — {item.quarter} Risk Report</p>
              <p className="text-[10px] text-muted-foreground">Quarter forecast · reviewed by DA analyst</p>
            </div>
          </div>

          <div className="col-span-2 bg-card border border-border/40 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-[11px] font-bold">Municipality / City scores</h4>
              <span className="text-[10px] text-muted-foreground">{muniData.length} LGUs · sorted by Risk Level</span>
            </div>
            <div className="max-h-64 overflow-y-auto thin-scrollbar pr-1">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                {muniData.map((m) => (
                  <div key={m.name} className="flex items-center gap-2 text-[11px] py-0.5">
                    <span className="flex-1 truncate" title={m.name}>{m.name}</span>
                    <div className="w-20 h-1.5 rounded-full bg-secondary/40 overflow-hidden">
                      <div
                        className="h-full"
                        style={{
                          width: `${Math.min(100, m.score * 100)}%`,
                          background: m.score >= 0.5 ? "hsl(var(--risk-high))" : "hsl(var(--risk-low))",
                        }}
                      />
                    </div>
                    <span className="w-10 text-right tabular-nums font-bold">{m.score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-span-2 bg-card border border-border/40 rounded-lg p-3">
            <h4 className="text-[11px] font-bold mb-2">SHAP feature contributions</h4>
            <ResponsiveContainer width="100%" height={Math.max(180, shapData.length * 28)}>
              <BarChart data={[...shapData].sort((a, b) => b.value - a.value)} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
                <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} domain={[-0.25, 0.25]} />
                <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 0, background: "hsl(var(--card))", border: "1px solid hsl(var(--border))" }} formatter={(v: number) => v.toFixed(3)} />
                <Bar dataKey="value" radius={[0, 0, 0, 0]}>
                  {shapData.map((s, i) => (
                    <Cell key={i} fill={s.value >= 0 ? "hsl(var(--risk-high))" : "hsl(var(--risk-low))"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* SHAP explanations — narrative */}
          <div className="col-span-2 bg-card border border-border/40 rounded-lg p-3">
            <h4 className="text-[11px] font-bold mb-1">SHAP explanations</h4>
            <p className="text-[10px] text-muted-foreground mb-3">
              Feature-level reasons behind this quarter's Risk Level forecast.
            </p>

            <ShapNarrativeBlock shap={shapData} />
          </div>
          <div className="col-span-2 bg-card border border-border/40 rounded-lg p-3">
            <h4 className="text-[11px] font-bold mb-2">Provincial ranking (CALABARZON)</h4>
            <div className="space-y-1.5">
              {regionsData
                .slice()
                .sort((a, b) => b.riskScore - a.riskScore)
                .map((r, i) => (
                  <div key={r.id} className={`flex items-center gap-3 px-2 py-1.5 rounded ${r.name === item.province ? "bg-primary/10 ring-1 ring-primary/40" : ""}`}>
                    <span className="text-[10px] w-5 text-muted-foreground tabular-nums">#{i + 1}</span>
                    <span className="text-[11px] flex-1">{r.name}</span>
                    <span className="text-[11px] tabular-nums font-bold" style={{ color: RISK_COLORS[r.riskLevel] }}>
                      {r.riskScore.toFixed(2)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        <div className="px-5 py-3 border-t border-border/40 flex items-center justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 rounded-md border border-border/50 text-[11px] hover:bg-secondary/60">Close</button>
          {item.status === "Rejected" ? (
            <span className="text-[11px] text-destructive font-bold px-3 py-1.5">Rejected — view in Rejection Log</span>
          ) : item.status === "Approved" ? (
            <>
              <button
                onClick={() => setRejectOpen(true)}
                className="px-3 py-1.5 rounded-md border border-destructive/60 text-destructive text-[11px] font-bold hover:bg-destructive/10"
              >
                Reject
              </button>
              <button onClick={onUnpublish} className="px-3 py-1.5 rounded-md border border-primary/50 text-primary text-[11px] font-bold hover:bg-primary/10">Undo Publishing</button>
            </>
          ) : (
            <>
              <button
                onClick={() => setRejectOpen(true)}
                className="px-3 py-1.5 rounded-md border border-destructive/60 text-destructive text-[11px] font-bold hover:bg-destructive/10"
              >
                Reject
              </button>
              <button onClick={onApprove} className="px-3 py-1.5 rounded-md bg-risk-low text-white text-[11px] font-bold hover:bg-risk-low/90">Approve & Publish</button>
            </>
          )}
        </div>
      </div>

      {rejectOpen && (
        <RejectReasonModal
          item={item}
          onCancel={() => setRejectOpen(false)}
          onConfirm={(reason, notes) => {
            setRejectOpen(false);
            onReject(reason, notes);
          }}
        />
      )}
    </div>
  );
}

const REJECTION_REASONS = [
  "Data quality issue",
  "Insufficient signal / limited articles",
  "Conflicts with field intelligence",
  "Model anomaly / outlier prediction",
  "Sensitive context — withhold publication",
  "Other",
] as const;

const SHAP_NARRATIVES: Record<string, { up: string; down: string }> = {
  "Engel Coefficient": {
    up: "Rising share of income spent on food pushed the risk score up",
    down: "Falling share of income spent on food reduced the risk score",
  },
  "FPSI Price Stress": {
    up: "Food CPI year-on-year acceleration pushed the risk score up",
    down: "Easing food CPI year-on-year reduced the risk score",
  },
  "Dependency Rate": {
    up: "Higher dependency ratio in households pushed the risk score up",
    down: "Lower dependency ratio in households reduced the risk score",
  },
  "Household Size": {
    up: "Larger average household size pushed the risk score up",
    down: "Smaller average household size reduced the risk score",
  },
  "Income Decile": {
    up: "Weaker income decile profile pushed the risk score up",
    down: "Stronger income decile profile reduced the risk score",
  },
  "Income Sources": {
    up: "Narrower income sources pushed the risk score up",
    down: "More diversified income sources reduced the risk score",
  },
};

function explainShap(feature: string, value: number): string {
  const dir = value >= 0 ? "up" : "down";
  const sign = value >= 0 ? "+" : "−";
  const abs = Math.abs(value).toFixed(2);
  const phrase =
    SHAP_NARRATIVES[feature]?.[dir] ??
    (value >= 0
      ? `${feature} pushed the risk score up`
      : `${feature} reduced the risk score`);
  return `${phrase} by ${sign}${abs}.`;
}

function ShapNarrativeBlock({ shap }: { shap: { feature: string; value: number }[] }) {
  const positives = shap.filter((s) => s.value > 0).sort((a, b) => b.value - a.value);
  const negatives = shap.filter((s) => s.value < 0).sort((a, b) => a.value - b.value);

  const Section = ({
    title,
    items,
    accent,
  }: {
    title: string;
    items: { feature: string; value: number }[];
    accent: "up" | "down";
  }) => (
    <div>
      <h5
        className="text-[11px] font-bold uppercase tracking-wider mb-2"
        style={{ color: accent === "up" ? "hsl(var(--risk-high))" : "hsl(var(--risk-low))" }}
      >
        {title}
      </h5>
      {items.length === 0 ? (
        <p className="text-[11px] text-muted-foreground italic">None for this quarter.</p>
      ) : (
        <ol className="space-y-2.5">
          {items.map((s) => (
            <li key={s.feature} className="text-[11px] leading-snug">
              <p className="font-bold text-foreground">{s.feature}</p>
              <p className="text-muted-foreground">{explainShap(s.feature, s.value)}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      <Section title="Top positive drivers" items={positives} accent="up" />
      <Section title="Top suppressors" items={negatives} accent="down" />
    </div>
  );
}

function RejectReasonModal({
  item,
  onCancel,
  onConfirm,
}: {
  item: ReviewItem;
  onCancel: () => void;
  onConfirm: (reason: string, notes?: string) => void;
}) {
  const [reason, setReason] = useState<string>("");
  const [other, setOther] = useState("");
  const [notes, setNotes] = useState("");

  const finalReason = reason === "Other" ? other.trim() : reason;
  const valid = finalReason.length > 0;

  return (
    <div className="fixed inset-0 z-[2100] bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-card border border-destructive/40 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
        <div className="px-5 py-3.5 border-b border-border/40 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-destructive">Reject forecast</h2>
            <p className="text-[10px] text-muted-foreground">
              {item.province} · {item.quarter} · This action is permanent until restored from the Rejection Log.
            </p>
          </div>
          <button onClick={onCancel} className="p-1.5 rounded-md hover:bg-secondary/60"><X className="h-4 w-4" /></button>
        </div>

        <div className="p-5 space-y-3">
          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground block mb-1.5">
              Rejection reason <span className="text-destructive">*</span>
            </label>
            <select
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full h-9 rounded-md border border-border/60 bg-background px-2 text-[12px] focus:outline-none focus:ring-2 focus:ring-destructive"
            >
              <option value="">Select a reason…</option>
              {REJECTION_REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            {reason === "Other" && (
              <input
                value={other}
                onChange={(e) => setOther(e.target.value)}
                placeholder="Specify reason…"
                className="mt-2 w-full h-9 rounded-md border border-border/60 bg-background px-2 text-[12px] focus:outline-none focus:ring-2 focus:ring-destructive"
              />
            )}
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground block mb-1.5">
              Notes <span className="text-muted-foreground/60 normal-case font-normal">(optional)</span>
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              placeholder="Additional context for audit log…"
              className="w-full rounded-md border border-border/60 bg-background px-2 py-1.5 text-[12px] focus:outline-none focus:ring-2 focus:ring-destructive resize-none"
            />
          </div>
        </div>

        <div className="px-5 py-3 border-t border-border/40 flex items-center justify-end gap-2">
          <button onClick={onCancel} className="px-3 py-1.5 rounded-md border border-border/50 text-[11px] hover:bg-secondary/60">Cancel</button>
          <button
            onClick={() => valid && onConfirm(finalReason, notes.trim() || undefined)}
            disabled={!valid}
            className="px-3 py-1.5 rounded-md bg-destructive text-destructive-foreground text-[11px] font-bold hover:bg-destructive/90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Confirm rejection
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Feedback (SUS submissions list) ---------- */
function FeedbackSection() {
  const [items, setItems] = useState<FeedbackSubmission[]>(() => loadFeedback());
  const [open, setOpen] = useState<FeedbackSubmission | null>(null);

  const avg = items.length
    ? Math.round(items.reduce((a, f) => a + f.score, 0) / items.length)
    : 0;

  return (
    <div>
      <SectionHeader title="Feedback" subtitle="System Usability Scale (SUS) submissions from public users" />
      <div className="p-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Stat label="Submissions" value={items.length.toString()} />
          <Stat label="Average SUS" value={`${avg}/100`} accent />
          <Stat label="Latest" value={items[0]?.date ?? "—"} />
        </div>

        <div className="bg-card border border-border/50 rounded-xl overflow-hidden">
          <table className="w-full text-[11px]">
            <thead className="bg-secondary/40 text-muted-foreground">
              <tr>
                <th className="text-left font-semibold px-4 py-2.5">Date</th>
                <th className="text-left font-semibold px-4 py-2.5">Name</th>
                <th className="text-left font-semibold px-4 py-2.5">Agency</th>
                <th className="text-left font-semibold px-4 py-2.5">Score</th>
                <th className="text-left font-semibold px-4 py-2.5">Action</th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground italic">
                    No feedback submissions yet.
                  </td>
                </tr>
              )}
              {items.map((f) => (
                <tr key={f.id} className="border-t border-border/40 hover:bg-secondary/30">
                  <td className="px-4 py-2.5 tabular-nums">{f.date}</td>
                  <td className="px-4 py-2.5 font-medium">{f.demographics.fullName || "Anonymous"}</td>
                  <td className="px-4 py-2.5 text-muted-foreground">{f.demographics.agency || "—"}</td>
                  <td className="px-4 py-2.5">
                    <span
                      className="inline-flex items-center gap-1 font-bold tabular-nums"
                      style={{ color: f.score >= 68 ? "hsl(var(--risk-low))" : "hsl(var(--risk-high))" }}
                    >
                      <Star className="h-3 w-3" /> {f.score}/100
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <button
                      onClick={() => setOpen(f)}
                      className="text-primary hover:underline text-[11px] font-semibold flex items-center gap-1"
                    >
                      <Eye className="h-3 w-3" /> View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {open && <FeedbackDetailModal item={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-card border border-border/50 rounded-xl px-4 py-3">
      <p className="text-[9px] uppercase tracking-widest text-muted-foreground/70 font-semibold">{label}</p>
      <p className={`font-mono-num text-2xl font-bold mt-1 tabular-nums ${accent ? "text-primary" : ""}`}>{value}</p>
    </div>
  );
}

function FeedbackDetailModal({ item, onClose }: { item: FeedbackSubmission; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-[2000] bg-background/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-card border border-border/60 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[88vh] overflow-hidden flex flex-col">
        <div className="px-5 py-3.5 border-b border-border/40 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold">SUS Feedback · {item.date}</h2>
            <p className="text-[10px] text-muted-foreground">
              {item.demographics.fullName || "Anonymous"} · {item.demographics.agency || "—"} · Score{" "}
              <span className="font-bold text-foreground">{item.score}/100</span>
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-md hover:bg-secondary/60"><X className="h-4 w-4" /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <KV k="Email" v={item.demographics.email || "—"} />
            <KV k="Designation" v={item.demographics.designation || "—"} />
            <KV k="Age" v={item.demographics.age || "—"} />
            <KV k="Sex" v={item.demographics.sex || "—"} />
            <KV k="Client type" v={item.demographics.clientType || "—"} />
            <KV k="Province" v={item.demographics.province || "—"} />
            <KV k="Municipality" v={item.demographics.municipality || "—"} />
          </div>

          <div>
            <h4 className="text-[11px] font-bold uppercase tracking-wider mb-2">Likert responses</h4>
            <ol className="space-y-1.5">
              {SUS_QUESTIONS.map((q, i) => (
                <li key={i} className="text-[11px] flex items-start gap-2">
                  <span className="text-muted-foreground tabular-nums w-5">{i + 1}.</span>
                  <span className="flex-1">{q}</span>
                  <span className="font-bold tabular-nums w-6 text-right">{item.answers[i] ?? "—"}</span>
                </li>
              ))}
            </ol>
          </div>

          {(item.liked || item.improvements) && (
            <div className="grid grid-cols-1 gap-2">
              {item.liked && (
                <div className="bg-secondary/30 border border-border/40 rounded-md p-3">
                  <p className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground mb-1">Liked most</p>
                  <p className="text-[11px] leading-snug">{item.liked}</p>
                </div>
              )}
              {item.improvements && (
                <div className="bg-secondary/30 border border-border/40 rounded-md p-3">
                  <p className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground mb-1">Suggestions</p>
                  <p className="text-[11px] leading-snug">{item.improvements}</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-border/40 flex items-center justify-end">
          <button onClick={onClose} className="px-3 py-1.5 rounded-md border border-border/50 text-[11px] hover:bg-secondary/60">Close</button>
        </div>
      </div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-secondary/30 border border-border/40 rounded-md px-2.5 py-1.5">
      <p className="text-[9px] uppercase tracking-wider text-muted-foreground/70">{k}</p>
      <p className="text-[11px] font-medium truncate">{v}</p>
    </div>
  );
}
