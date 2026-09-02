import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ChevronDown } from "lucide-react";
import { TopNavbar } from "@/components/TopNavbar";
import logoDark from "@/assets/logo-dark.png";
import heroBg from "@/assets/hero-bg.jpg";

const THEME_KEY = "aipheed_theme";

const AMBER = "#F5A623";
const SERIF = "Georgia, 'Times New Roman', Times, serif";
const TILE_BG = "rgba(245,166,35,0.12)";
const TILE_BORDER = "rgba(245,166,35,0.35)";

export default function Landing() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  useEffect(() => {
    const t = (localStorage.getItem(THEME_KEY) as "light" | "dark") ?? "dark";
    document.documentElement.classList.toggle("light", t === "light");
    setTheme(t);
    const obs = new MutationObserver(() => {
      setTheme(document.documentElement.classList.contains("light") ? "light" : "dark");
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);

  // Intersection observer fade-in
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>("[data-reveal]");
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("opacity-100", "translate-y-0");
            e.target.classList.remove("opacity-0", "translate-y-4");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  const palette = theme === "light"
    ? {
        "--lp-bg": "#F7F7F8",
        "--lp-bg2": "#FFFFFF",
        "--lp-card": "#FFFFFF",
        "--lp-fg": "#0D0F14",
        "--lp-muted": "#4B5563",
        "--lp-muted2": "#6B7280",
        "--lp-divider": "#D1D5DB",
        "--lp-footer": "#EEF0F3",
        "--lp-overlay-1": "rgba(255,255,255,0.35)",
        "--lp-overlay-2": "rgba(255,255,255,0.65)",
        "--lp-border-strong": "rgba(0,0,0,0.35)",
        "--lp-soft-bg": "rgba(0,0,0,0.03)",
        "--lp-soft-border": "rgba(0,0,0,0.06)",
        "--lp-hover": "rgba(0,0,0,0.04)",
      } as React.CSSProperties
    : {
        "--lp-bg": "#0D0F14",
        "--lp-bg2": "#111318",
        "--lp-card": "#1A1D25",
        "--lp-fg": "#FFFFFF",
        "--lp-muted": "#9CA3AF",
        "--lp-muted2": "#6B7280",
        "--lp-divider": "#3F4451",
        "--lp-footer": "#08090D",
        "--lp-overlay-1": "rgba(13,15,20,0.35)",
        "--lp-overlay-2": "rgba(13,15,20,0.65)",
        "--lp-border-strong": "rgba(255,255,255,0.6)",
        "--lp-soft-bg": "rgba(255,255,255,0.02)",
        "--lp-soft-border": "rgba(255,255,255,0.04)",
        "--lp-hover": "rgba(255,255,255,0.05)",
      } as React.CSSProperties;

  return (
    <div
      className="min-h-screen w-full text-[var(--lp-fg)]"
      style={{ background: "var(--lp-bg)", fontFamily: "Inter, ui-sans-serif, system-ui", ...palette }}
    >
      <div className="sticky top-0 z-[1002]">
        <TopNavbar />
      </div>

      <Hero />
      <StatBand />
      <About />
      <Capabilities />
      <RiskDrivers />
      <Methodology />
      <Disclaimer />
      <FAQ />
      <Footer />
    </div>
  );
}

/* ---------------- SHARED ---------------- */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <p
        className="text-[11px] font-semibold uppercase"
        style={{ color: AMBER, letterSpacing: "0.18em" }}
      >
        {children}
      </p>
      <span className="mt-2 block h-[3px] w-9" style={{ background: AMBER }} />
    </div>
  );
}

function sectionHeading(extra = "") {
  return `text-[var(--lp-fg)] font-semibold leading-tight ${extra}`;
}

/* Fires once when the element scrolls into view */
function useInView<T extends HTMLElement>(threshold = 0.3) {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setInView(true);
          io.disconnect();
        }
      },
      { threshold }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return { ref, inView };
}

/* Counts numeric strings up from zero; passes text through unchanged */
function CountUp({
  value,
  start,
  className,
  style,
}: {
  value: string;
  start: boolean;
  className?: string;
  style?: React.CSSProperties;
}) {
  const numeric = /^[\d,]+$/.test(value);
  const target = numeric ? parseInt(value.replace(/,/g, ""), 10) : 0;
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!start || !numeric) return;
    let raf = 0;
    const duration = 1200;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setN(Math.round(target * eased));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [start, numeric, target]);
  return (
    <span className={className} style={style}>
      {numeric ? n.toLocaleString() : value}
    </span>
  );
}

/* ---------------- HERO ---------------- */
function Hero() {
  return (
    <section
      className="relative overflow-hidden px-6 sm:px-10 pt-20 pb-24 text-center"
      style={{ backgroundColor: "var(--lp-bg)" }}
    >
      {/* Background image */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `url(${heroBg})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          opacity: 0.32,
        }}
      />
      {/* Yellow + red ambient light over the image */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 30% 20%, rgba(245,166,35,0.30), transparent 55%), radial-gradient(ellipse at 75% 80%, rgba(239,68,68,0.25), transparent 55%), linear-gradient(to bottom, var(--lp-overlay-1), var(--lp-overlay-2))",
        }}
      />
      <div className="relative mx-auto max-w-4xl" data-reveal style={{ transition: "all .5s ease" }}>
        <h1
          className="text-[var(--lp-fg)] font-semibold"
          style={{ fontFamily: SERIF, fontSize: "clamp(2.1rem, 4.6vw, 3.4rem)", lineHeight: 1.15 }}
        >
          Food Insecurity Forecasting for CALABARZON
        </h1>

        <p
          className="mt-6 font-bold uppercase"
          style={{ fontSize: "clamp(1.05rem,2vw,1.4rem)", letterSpacing: "0.18em" }}
        >
          <span style={{ color: "hsl(var(--risk-low))" }}>Mapped.</span>{" "}
          <span style={{ color: "hsl(var(--risk-moderate))" }}>Explained.</span>{" "}
          <span style={{ color: "hsl(var(--risk-high))" }}>Forecasted.</span>
        </p>

        <p
          className="mt-6 mx-auto text-[16px] leading-relaxed"
          style={{ color: "var(--lp-muted)", maxWidth: 640 }}
        >
          aiPHeed is an interactive geospatial forecasting system for the Department of
          Agriculture Region IV-A. It combines natural language processing and ensemble machine
          learning to identify food stress before it peaks.
        </p>

        <div className="mt-9 flex flex-wrap justify-center gap-3">
          <Link
            to="/dashboard"
            className="group inline-flex items-center gap-2 px-6 py-3 font-semibold text-[14px] transition-colors"
            style={{ background: AMBER, color: "#0D0F14" }}
          >
            Explore the dashboard
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
          <a
            href="#methodology"
            className="group inline-flex items-center gap-2 px-6 py-3 font-semibold text-[14px] border transition-colors hover:bg-[var(--lp-hover)]"
            style={{ borderColor: "var(--lp-fg)", color: "var(--lp-fg)" }}
          >
            Read the methodology
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </a>
        </div>

        {/* Risk scale legend */}
        <div className="mt-14 mx-auto max-w-2xl">
          <div
            className="h-2 w-full"
            style={{
              background: "linear-gradient(to right, #EAB308, #EF4444)",
              border: "1px solid var(--lp-divider)",
            }}
          />
          <div
            className="mt-2 flex justify-between text-[10px] font-semibold"
            style={{ color: "var(--lp-muted)", letterSpacing: "0.16em" }}
          >
            <span>LOWER RISK</span>
            <span>HIGHER RISK</span>
          </div>
        </div>
      </div>

      <p className="absolute bottom-4 right-6 text-[10px]" style={{ color: "var(--lp-muted2)" }}>
        Photo by Atom Araullo on{" "}
        <a
          href="https://www.gmanetwork.com/news/specials/content/193/the-hunger-pandemic/"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-[var(--lp-fg)] transition-colors"
        >
          GMA News
        </a>
      </p>
    </section>
  );
}

/* ---------------- STAT BAND ---------------- */
function StatBand() {
  const stats = [
    { v: "5", l: "Provinces" },
    { v: "142", l: "Cities and municipalities" },
    { v: "17,791", l: "News articles analysed" },
    { v: "Quarterly", l: "Forecast cadence" },
  ];
  const { ref, inView } = useInView<HTMLDivElement>(0.4);
  return (
    <section
      className="px-6 sm:px-10"
      style={{
        background: "var(--lp-bg)",
        borderTop: "1px solid var(--lp-divider)",
        borderBottom: "1px solid var(--lp-divider)",
      }}
    >
      <div ref={ref} className="mx-auto max-w-6xl grid grid-cols-2 sm:grid-cols-4 py-10">
        {stats.map((s, i) => (
          <div
            key={i}
            className="px-4 sm:px-8 py-3 text-center sm:text-left sm:border-l sm:first:border-l-0"
            style={{
              borderColor: "var(--lp-divider)",
              opacity: inView ? 1 : 0,
              transform: inView ? "translateY(0)" : "translateY(8px)",
              transition: `opacity .5s ease ${i * 90}ms, transform .5s ease ${i * 90}ms`,
            }}
          >
            <CountUp
              value={s.v}
              start={inView}
              className="block font-semibold leading-none text-[var(--lp-fg)]"
              style={{ fontFamily: SERIF, fontSize: "clamp(1.35rem,2.2vw,1.8rem)" }}
            />
            <p
              className="mt-2 text-[10px] font-semibold uppercase"
              style={{ color: "var(--lp-muted)", letterSpacing: "0.14em" }}
            >
              {s.l}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ---------------- ABOUT ---------------- */
function About() {
  return (
    <section id="about" className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg)" }}>
      <div
        className="mx-auto max-w-6xl grid lg:grid-cols-2 gap-12 items-start opacity-0 translate-y-4"
        data-reveal
        style={{ transition: "all .5s ease" }}
      >
        <div>
          <SectionLabel>Overview</SectionLabel>
          <h2 className={sectionHeading("mb-6")} style={{ fontFamily: SERIF, fontSize: "clamp(1.7rem,3vw,2.3rem)" }}>
            Continuous forecasts between national survey cycles
          </h2>
          <p className="text-[15px] leading-relaxed mb-4" style={{ color: "var(--lp-muted)" }}>
            aiPHeed was developed as a thesis for the Bachelor of Science in Computer Science with
            Intelligent Systems Track at De La Salle University - Dasmariñas. Researchers: Destura,
            Princess Gwenn A.; Esico, Christina M.; and Melindo, Angel Vhea P.
          </p>
          <p className="text-[15px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
            Some government agencies rely on nutrition surveys released every two to three years.
            aiPHeed supplements this cycle by delivering continuously updated, quarterly
            province-level food insecurity risk forecasts, disaggregated to all 142 cities and
            municipalities of Region IV-A, through a publicly accessible geospatial dashboard.
          </p>
        </div>

        <SummaryCard />
      </div>
    </section>
  );
}

function SummaryCard() {
  return (
    <div style={{ background: "var(--lp-card)", border: "1px solid var(--lp-divider)" }}>
      <div
        className="flex items-center justify-between px-6 py-3"
        style={{ borderBottom: "1px solid var(--lp-divider)" }}
      >
        <p className="text-[10px] font-semibold uppercase" style={{ color: "var(--lp-muted)", letterSpacing: "0.16em" }}>
          Q2 2026 · April to June
        </p>
        <div className="flex gap-2">
          <span
            className="text-[9px] font-bold uppercase tracking-wider px-2 py-1"
            style={{ background: "rgba(249,115,22,0.14)", color: "#C2410C", border: "1px solid rgba(249,115,22,0.4)" }}
          >
            High risk
          </span>
          <span
            className="text-[9px] font-bold uppercase tracking-wider px-2 py-1 inline-flex items-center gap-1"
            style={{ background: "rgba(239,68,68,0.14)", color: "#B91C1C", border: "1px solid rgba(239,68,68,0.4)" }}
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: "#B91C1C" }} /> Active alert
          </span>
        </div>
      </div>

      <div className="p-6">
        <h3 className="text-[var(--lp-fg)] text-base font-semibold mb-6" style={{ fontFamily: SERIF }}>
          CALABARZON regional summary
        </h3>

        <div className="mb-6">
          <p className="text-[10px] uppercase tracking-widest mb-2" style={{ color: "var(--lp-muted2)" }}>
            Regional Food Insecurity Index
          </p>
          <p className="font-semibold leading-none" style={{ fontFamily: SERIF, fontSize: "3.25rem", color: AMBER }}>
            0.46
            <span className="text-lg ml-2" style={{ color: "var(--lp-muted2)" }}>/ 1.00</span>
          </p>
        </div>

        <div className="flex items-center gap-2 mb-3">
          <span className="h-2 w-2 rounded-full" style={{ background: "#EAB308" }} />
          <span className="text-[12px] font-semibold uppercase tracking-wider text-[var(--lp-fg)]">
            Moderate risk
          </span>
        </div>

        <div
          className="relative h-2 w-full"
          style={{ background: "linear-gradient(to right, #EAB308, #F97316, #EF4444)", border: "1px solid var(--lp-divider)" }}
        >
          <div className="absolute -top-1 h-4 w-[2px]" style={{ left: "46%", background: "var(--lp-fg)" }} />
        </div>
        <div className="mt-2 flex justify-between text-[9px] tracking-widest" style={{ color: "var(--lp-muted2)" }}>
          <span>0.00</span>
          <span>1.00</span>
        </div>
      </div>
    </div>
  );
}

/* ---------------- CAPABILITIES ---------------- */
function Capabilities() {
  const items = [
    { title: "Geospatial risk mapping", body: "Province and municipality choropleth on an interactive map, classified from Low to Severe." },
    { title: "Zero-shot Filipino-English NLP", body: "XLM-RoBERTa scores bilingual news articles against 10 food insecurity hypotheses with no labelled training data required." },
    { title: "Bias-corrected FSSI", body: "The Food Stress Sentiment Index upweights under-covered rural provinces to correct for capital-city media concentration." },
    { title: "LightGBM with Optuna tuning", body: "Ensemble machine learning optimised via Bayesian hyperparameter search across 100 trials, benchmarked against six comparison models." },
    { title: "SHAP feature attribution", body: "Every forecast is separated into named driver contributions for transparent, interpretable decision support." },
    { title: "Municipal disaggregation", body: "Province forecasts distributed to all 142 LGUs using PSA poverty incidence (60 percent) and population density (40 percent) as vulnerability weights." },
  ];

  return (
    <section id="capabilities" className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .5s ease" }}>
        <SectionLabel>Capabilities</SectionLabel>
        <h2 className={sectionHeading("mb-10 max-w-3xl")} style={{ fontFamily: SERIF, fontSize: "clamp(1.7rem,3vw,2.3rem)" }}>
          What the system does
        </h2>

        <div
          className="grid gap-px md:grid-cols-2 lg:grid-cols-3"
          style={{ background: "var(--lp-divider)", border: "1px solid var(--lp-divider)" }}
        >
          {items.map((it, i) => (
            <div key={i} className="p-6" style={{ background: "var(--lp-card)" }}>
              <span
                className="block text-[22px] font-semibold mb-3 pb-3"
                style={{ fontFamily: SERIF, color: AMBER, borderBottom: "1px solid var(--lp-divider)" }}
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <h3 className="text-[var(--lp-fg)] font-semibold text-[15px] mb-2">{it.title}</h3>
              <p className="text-[13px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>{it.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------------- RISK DRIVERS ---------------- */
function RiskDrivers() {
  const drivers = [
    { label: "Market and prices", value: 0.21, body: "Rice retail price and Food CPI volatility raise the index across all provinces." },
    { label: "Climate stress", value: 0.17, body: "PAGASA typhoon count, rainfall anomaly, and ENSO phase shift the seasonal baseline." },
    { label: "Employment", value: 0.12, body: "Unemployment and underemployment reduce household purchasing power." },
    { label: "OFW remittance", value: -0.09, body: "BSP remittance inflows reduce risk in dependent provinces such as Cavite and Batangas." },
    { label: "Fish kill", value: 0.08, body: "Algal blooms in Taal and Laguna Lake disrupt aquaculture supply chains." },
  ];
  const scale = 0.25; // full-bar magnitude

  return (
    <section className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .5s ease" }}>
        <SectionLabel>Risk drivers</SectionLabel>
        <h2 className={sectionHeading("mb-8")} style={{ fontFamily: SERIF, fontSize: "clamp(1.6rem,2.8vw,2.1rem)" }}>
          What drives food insecurity risk in CALABARZON
        </h2>

        <div style={{ background: "var(--lp-card)", border: "1px solid var(--lp-divider)" }}>
          <div
            className="flex items-center justify-between px-5 py-3"
            style={{ borderBottom: "1px solid var(--lp-divider)" }}
          >
            <p className="text-[10px] font-semibold uppercase" style={{ color: "var(--lp-muted)", letterSpacing: "0.16em" }}>
              Top risk drivers · Q2 2026
            </p>
            <p className="text-[10px] font-semibold uppercase" style={{ color: "var(--lp-muted2)", letterSpacing: "0.1em" }}>
              Contribution to index
            </p>
          </div>

          {drivers.map((d, i) => {
            const isNeg = d.value < 0;
            const pct = Math.min(Math.abs(d.value) / scale, 1) * 100;
            const display = `${isNeg ? "−" : "+"}${Math.abs(d.value).toFixed(2)}`;
            return (
              <div
                key={i}
                className="grid grid-cols-1 sm:grid-cols-[1.4fr_1fr_auto] items-center gap-4 px-5 py-4"
                style={{ borderBottom: i < drivers.length - 1 ? "1px solid var(--lp-divider)" : "none" }}
              >
                <div className="min-w-0">
                  <p className="text-[var(--lp-fg)] font-semibold text-[13px]">{d.label}</p>
                  <p className="text-[12px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>{d.body}</p>
                </div>

                {/* Diverging magnitude bar */}
                <div
                  className="relative flex h-6 w-full items-stretch"
                  style={{ background: "var(--lp-soft-bg)", border: "1px solid var(--lp-divider)" }}
                >
                  <div className="flex-1 flex justify-end">
                    {isNeg && <div style={{ width: `${pct}%`, background: "rgba(21,128,61,0.55)" }} />}
                  </div>
                  <div style={{ width: 1, background: "var(--lp-divider)" }} />
                  <div className="flex-1">
                    {!isNeg && <div className="h-full" style={{ width: `${pct}%`, background: "rgba(185,28,28,0.55)" }} />}
                  </div>
                </div>

                <span
                  className="justify-self-start sm:justify-self-end font-mono text-[12px] font-bold px-2.5 py-1"
                  style={{
                    background: isNeg ? "rgba(21,128,61,0.10)" : "rgba(185,28,28,0.10)",
                    color: isNeg ? "#15803D" : "#B91C1C",
                    border: `1px solid ${isNeg ? "rgba(21,128,61,0.35)" : "rgba(185,28,28,0.35)"}`,
                  }}
                >
                  {display}
                </span>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-[11px]" style={{ color: "var(--lp-muted2)" }}>
          Red bars raise forecast risk; green bars lower it. Values are SHAP contributions to the regional index.
        </p>
      </div>
    </section>
  );
}

/* ---------------- METHODOLOGY ---------------- */
function Methodology() {
  const sources = [
    { tag: "Text", body: "Filipino-English news corpus · RSS, Google News, GDELT · 17,791 articles · 2020 to 2025" },
    { tag: "Statistics", body: "Government statistics · PSA, BSP, DOE, PhilRice, SWS" },
    { tag: "Climate", body: "Climate data · PAGASA (typhoon count, rainfall anomaly, ENSO phase, drought alert)" },
    { tag: "Prices", body: "Commodity and price data · Food CPI, rice retail price, unemployment rate" },
    { tag: "Anchors", body: "Survey anchors · DA-FNRI ENNS 2021 and 2023" },
  ];

  const steps = [
    { t: "Data collection", d: "Multi-source ingestion via RSS and APIs" },
    { t: "NLP processing", d: "XLM-RoBERTa zero-shot classification" },
    { t: "Feature engineering", d: "FSSI, lag features, bias correction" },
    { t: "Model training", d: "LightGBM tuned with Optuna, 100 trials" },
    { t: "Spatial disaggregation", d: "Vulnerability-weighted LGU split" },
    { t: "Dashboard output", d: "Quarterly choropleth and SHAP panels" },
  ];

  const metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Maximum lead time"];

  return (
    <section id="methodology" className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .5s ease" }}>
        <SectionLabel>Methodology</SectionLabel>
        <h2 className={sectionHeading("mb-12")} style={{ fontFamily: SERIF, fontSize: "clamp(1.7rem,3vw,2.3rem)" }}>
          How the forecasting pipeline works
        </h2>

        <h3 className="text-[var(--lp-fg)] font-semibold text-[13px] mb-4 uppercase" style={{ letterSpacing: "0.12em" }}>
          Data sources
        </h3>
        <div className="mb-16" style={{ border: "1px solid var(--lp-divider)" }}>
          {sources.map((s, i) => (
            <div
              key={i}
              className="flex items-center gap-4 px-4 py-3"
              style={{
                background: "var(--lp-card)",
                borderBottom: i < sources.length - 1 ? "1px solid var(--lp-divider)" : "none",
              }}
            >
              <span
                className="text-[10px] font-bold uppercase tracking-wider px-2 py-1 w-24 text-center shrink-0"
                style={{ background: TILE_BG, color: AMBER, border: `1px solid ${TILE_BORDER}` }}
              >
                {s.tag}
              </span>
              <p className="text-[13px]" style={{ color: "var(--lp-muted)" }}>{s.body}</p>
            </div>
          ))}
        </div>

        <h3 className="text-[var(--lp-fg)] font-semibold text-[13px] mb-4 uppercase" style={{ letterSpacing: "0.12em" }}>
          Six-stage pipeline
        </h3>
        <div
          className="grid gap-px sm:grid-cols-2 lg:grid-cols-3 mb-16"
          style={{ background: "var(--lp-divider)", border: "1px solid var(--lp-divider)" }}
        >
          {steps.map((s, i) => (
            <div key={i} className="p-5" style={{ background: "var(--lp-card)" }}>
              <div className="flex items-center gap-3 mb-2">
                <span
                  className="inline-flex h-8 w-8 items-center justify-center text-[13px] font-bold"
                  style={{ background: AMBER, color: "#0D0F14", fontFamily: SERIF }}
                >
                  {i + 1}
                </span>
                <p className="text-[var(--lp-fg)] font-semibold text-[13px]">{s.t}</p>
              </div>
              <p className="text-[12px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>{s.d}</p>
            </div>
          ))}
        </div>

        <h3 className="text-[var(--lp-fg)] font-semibold text-[13px] mb-4 uppercase" style={{ letterSpacing: "0.12em" }}>
          Evaluation metrics
        </h3>
        <div className="flex flex-wrap gap-2">
          {metrics.map((m, i) => (
            <span
              key={i}
              className="text-[11px] font-semibold uppercase tracking-wider px-3 py-1.5"
              style={{ background: "var(--lp-card)", color: "var(--lp-fg)", borderLeft: `2px solid ${AMBER}`, borderTop: "1px solid var(--lp-divider)", borderRight: "1px solid var(--lp-divider)", borderBottom: "1px solid var(--lp-divider)" }}
            >
              {m}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------------- DISCLAIMER ---------------- */
function Disclaimer() {
  const points = [
    "Relies entirely on public data. It does not include real-time field reports, satellite imagery, or granular agricultural production figures.",
    "Forecast lead times are empirically determined. Forecasts are only published when model performance meets acceptable thresholds.",
    "The NLP pipeline was not trained on labelled local data due to the absence of a Filipino-English food insecurity text dataset. This is a resource gap in the field, not a design decision.",
    "aiPHeed outputs are indicative risk rankings only. They must not replace official government assessment procedures or serve as official food insecurity classifications.",
  ];
  return (
    <section id="disclaimer" className="px-6 sm:px-10 py-20" style={{ background: "var(--lp-bg)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .5s ease" }}>
        <div
          className="p-8"
          style={{ background: "var(--lp-card)", border: "1px solid var(--lp-divider)", borderLeft: "4px solid #C2410C" }}
        >
          <p className="text-[11px] font-bold uppercase mb-4" style={{ color: "#C2410C", letterSpacing: "0.16em" }}>
            Data disclaimer and limitations
          </p>
          <ul className="space-y-3">
            {points.map((p, i) => (
              <li key={i} className="flex gap-3 text-[14px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
                <span className="mt-2 h-1.5 w-1.5 shrink-0" style={{ background: "#C2410C" }} />
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

/* ---------------- FAQ ---------------- */
function FAQ() {
  const items = [
    {
      q: "Why does aiPHeed forecast province-level risk instead of household-level?",
      a: "Household-level forecasts require granular survey microdata that is not publicly released at quarterly cadence. aiPHeed forecasts at the province level, the finest reliable signal in public data, and disaggregates to municipalities using PSA vulnerability weights.",
    },
    {
      q: "What does \"zero-shot\" NLP mean?",
      a: "Zero-shot classification means the model (XLM-RoBERTa) scores text against natural-language hypotheses without ever being fine-tuned on labelled food insecurity examples. This is necessary because no Filipino-English food insecurity dataset exists.",
    },
    {
      q: "Are municipal forecasts independent predictions?",
      a: "No. Municipal scores are deterministically derived from province forecasts using a vulnerability index combining PSA poverty incidence (60 percent) and population density (40 percent). They inherit province uncertainty.",
    },
    {
      q: "Who is the intended user of aiPHeed?",
      a: "DA Region IV-A field offices, LGU planning units, academic researchers, and journalists tracking food security signals. It is a decision-support layer, not a replacement for official assessment.",
    },
    {
      q: "Can the system be adapted for other Philippine regions?",
      a: "Yes. The pipeline is region-agnostic. Only the news corpus filters, PSA province codes, and vulnerability weights need to be re-fit for a new region.",
    },
  ];
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section id="faq" className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .5s ease" }}>
        <SectionLabel>Reference</SectionLabel>
        <h2 className={sectionHeading("mb-10")} style={{ fontFamily: SERIF, fontSize: "clamp(1.7rem,3vw,2.3rem)" }}>
          Frequently asked questions
        </h2>
        <div style={{ border: "1px solid var(--lp-divider)" }}>
          {items.map((it, i) => {
            const isOpen = open === i;
            return (
              <div
                key={i}
                style={{
                  borderBottom: i < items.length - 1 ? "1px solid var(--lp-divider)" : "none",
                  background: "var(--lp-card)",
                }}
              >
                <button
                  onClick={() => setOpen(isOpen ? null : i)}
                  className="w-full flex items-center gap-4 p-5 text-left hover:bg-[var(--lp-hover)] transition-colors"
                >
                  <span
                    className="text-[13px] font-bold shrink-0"
                    style={{ fontFamily: SERIF, color: isOpen ? AMBER : "var(--lp-muted2)" }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="flex-1 text-[var(--lp-fg)] font-semibold text-[14px]">{it.q}</span>
                  <ChevronDown
                    className="h-5 w-5 shrink-0 transition-transform"
                    style={{ color: isOpen ? AMBER : "var(--lp-muted)", transform: isOpen ? "rotate(180deg)" : "rotate(0)" }}
                  />
                </button>
                {isOpen && (
                  <div className="pl-14 pr-5 pb-5 text-[13px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
                    {it.a}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ---------------- FOOTER ---------------- */
function Footer() {
  const team = [
    { name: "Destura", url: "https://www.linkedin.com/in/princess-gwenn-destura" },
    { name: "Melindo", url: "https://www.linkedin.com/in/angelvheamelindo/" },
    { name: "Esico", url: "https://www.linkedin.com/in/christina-esico/" },
  ];
  return (
    <footer
      className="relative px-6 sm:px-10 pt-16 pb-6"
      style={{ background: "#08090D", borderTop: "1px solid var(--lp-divider)" }}
    >
      <div className="mx-auto max-w-7xl">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 sm:gap-6 lg:gap-10 mb-12">
          {/* Brand + description + bottom links */}
          <div className="col-span-1">
            <div className="flex items-center gap-3 mb-5">
              <img src={logoDark} alt="aiPHeed" className="h-10 sm:h-12 w-auto object-contain" />
            </div>
            <p className="text-[14px] leading-relaxed mb-6" style={{ color: "var(--lp-muted)" }}>
              aiPHeed forecasts food insecurity risk in CALABARZON on a quarterly basis, combining
              Filipino-English natural language processing and ensemble machine learning to identify
              food stress before it peaks.
            </p>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">Home</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">About</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#capabilities" className="hover:text-[var(--lp-fg)] transition-colors">Capabilities</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#faq" className="hover:text-[var(--lp-fg)] transition-colors">FAQ</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="mailto:aipheed.research@dlsud.edu.ph" className="hover:text-[var(--lp-fg)] transition-colors">Contact</a>
            </div>
          </div>

          {/* Product */}
          <div className="lg:pl-6 col-span-1">
            <h4 className="text-[var(--lp-fg)] font-semibold text-[15px] mb-5">Product</h4>
            <ul className="space-y-3 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <li><Link to="/dashboard" className="hover:text-[var(--lp-fg)] transition-colors">Mapping</Link></li>
              <li><Link to="/data" className="hover:text-[var(--lp-fg)] transition-colors">Data</Link></li>
              <li><Link to="/visualization" className="hover:text-[var(--lp-fg)] transition-colors">Visualization</Link></li>
              <li><a href="#methodology" className="hover:text-[var(--lp-fg)] transition-colors">Methodology</a></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-[var(--lp-fg)] font-semibold text-[15px] mb-5">Company</h4>
            <ul className="space-y-3 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <li><a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">About</a></li>
              <li><a href="#disclaimer" className="hover:text-[var(--lp-fg)] transition-colors">Disclaimer</a></li>
              <li><a href="mailto:aipheed.research@dlsud.edu.ph" className="hover:text-[var(--lp-fg)] transition-colors">Contact</a></li>
            </ul>
          </div>
        </div>

        <div
          className="pt-6 flex flex-row items-center justify-between gap-x-6 gap-y-2 text-[12px] flex-wrap"
          style={{ borderTop: "1px solid rgba(255,255,255,0.06)", color: "var(--lp-muted2)" }}
        >
          <p>© 2026 aiPHeed. All rights reserved.</p>
          <p className="flex items-center gap-1.5 flex-wrap">
            <span>Developed by</span>
            {team.map((m, i) => (
              <span key={i} className="flex items-center gap-1.5">
                <a
                  href={m.url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold hover:text-[var(--lp-fg)] transition-colors"
                  style={{ color: "var(--lp-muted)" }}
                >
                  {m.name}
                </a>
                {i < team.length - 1 && <span style={{ color: "var(--lp-divider)" }}>·</span>}
              </span>
            ))}
          </p>
        </div>
      </div>
    </footer>
  );
}
