import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  MapPin,
  Languages,
  Scale,
  Sparkles,
  Network,
  Layers,
  ChevronDown,
  TrendingUp,
  CloudRain,
  Briefcase,
  Send,
  Fish,
} from "lucide-react";
import { TopNavbar } from "@/components/TopNavbar";
import desturaImg from "@/assets/researchers/destura.png";
import esicoImg from "@/assets/researchers/esico.jpg";
import melindoImg from "@/assets/researchers/melindo.jpg";
import logoIcon from "@/assets/logo-light.png";
import logoDark from "@/assets/logo-dark.png";
import heroBg from "@/assets/hero-bg.jpg";

const THEME_KEY = "aipheed_theme";

const AMBER = "#F5A623";

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


/* ---------------- HERO ---------------- */
function Hero() {
  return (
    <section
      className="relative overflow-hidden px-6 sm:px-10 pt-20 pb-28 text-center"
      style={{ backgroundColor: "var(--lp-bg)" }}
    >
      {/* Background image (low opacity) */}
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
      {/* Gradient overlay (yellow + red) on top of the image */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 30% 20%, rgba(245,166,35,0.30), transparent 55%), radial-gradient(ellipse at 75% 80%, rgba(239,68,68,0.25), transparent 55%), linear-gradient(to bottom, var(--lp-overlay-1), var(--lp-overlay-2))",
        }}
      />
      <div className="relative mx-auto max-w-6xl" data-reveal style={{ transition: "all .8s ease" }}>

        <h1
          className="font-extrabold leading-[1.02] text-[var(--lp-fg)]"
          style={{ fontSize: "clamp(2.6rem, 7.2vw, 4.5rem)", letterSpacing: "-0.02em" }}
        >
          CALABARZON FOOD <br className="hidden sm:block" />
          <span style={{ color: AMBER }}>INSECURITY</span>
        </h1>

        <p
          className="mt-8 font-bold uppercase"
          style={{ fontSize: "clamp(1.1rem,2.1vw,1.5rem)", letterSpacing: "0.18em" }}
        >
          <span style={{ color: "hsl(var(--risk-low))" }}>Mapped.</span>{" "}
          <span style={{ color: "hsl(var(--risk-moderate))" }}>Explained.</span>{" "}
          <span style={{ color: "hsl(var(--risk-high))" }}>Forecasted.</span>
        </p>

        <p
          className="mt-8 mx-auto text-[17px] leading-relaxed"
          style={{ color: "var(--lp-muted)", maxWidth: 620 }}
        >
          aiPHeed is an interactive geospatial food insecurity forecasting system for DA
          Region IV-A, combining NLP and ensemble machine learning to detect food stress before it
          peaks.
        </p>

        <div className="mt-10 flex flex-wrap justify-center gap-4">
          <Link
            to="/dashboard"
            className="group inline-flex items-center gap-2 rounded-full px-7 py-3.5 font-semibold text-[14px] transition-all"
            style={{ background: AMBER, color: "#0D0F14", boxShadow: "0 8px 30px rgba(245,166,35,0.35)" }}
          >
            Explore the Dashboard
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
          <a
            href="#about"
            className="group inline-flex items-center gap-2 rounded-full px-7 py-3.5 font-semibold text-[14px] border transition-all hover:bg-[var(--lp-hover)]"
            style={{ borderColor: "var(--lp-border-strong)", color: "#fff" }}
          >
            Read the Research
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </a>
        </div>

        {/* Risk gradient bar */}
        <div className="mt-16 mx-auto max-w-2xl">
          <div
            className="h-2 w-full rounded-full"
            style={{
              background: "linear-gradient(to right, #EAB308, #EF4444)",
            }}
          />
          <div
            className="mt-2 flex justify-between text-[10px] font-semibold tracking-[0.2em]"
            style={{ color: "var(--lp-muted)" }}
          >
            <span>LOW</span>
            <span>HIGH</span>
          </div>
        </div>
      </div>

      <p
        className="absolute bottom-4 right-6 text-[10px]"
        style={{ color: "var(--lp-muted2)" }}
      >
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

/* ---------------- ABOUT ---------------- */
function About() {
  return (
    <section
      id="about"
      className="px-6 sm:px-10 py-24"
      style={{ background: "var(--lp-bg2)" }}
    >
      <div
        className="mx-auto max-w-6xl grid lg:grid-cols-2 gap-12 items-start opacity-0 translate-y-4"
        data-reveal
        style={{ transition: "all .8s ease" }}
      >
        <div>
          <p
            className="text-[11px] font-semibold uppercase mb-4"
            style={{ color: AMBER, letterSpacing: "0.28em" }}
          >
            About the System
          </p>
          <h2 className="text-[var(--lp-fg)] font-bold text-[clamp(1.8rem,3.5vw,2.6rem)] leading-tight mb-6">
            Built for the gap between surveys.
          </h2>
          <p className="text-[15px] leading-relaxed mb-4" style={{ color: "var(--lp-muted)" }}>
            aiPHeed was developed as a thesis for the Bachelor of Science in Computer Science with
            Intelligence Systems Track at De La Salle University - Dasmariñas. Researchers:
            Destura, Princess Gwenn A.; Esico, Christina M.; and Melindo, Angel Vhea P.
          </p>
          <p className="text-[15px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
            Some government agencies rely on surveys released every two to three years. aiPHeed
            bridges this gap by delivering continuously updated, quarterly province-level food
            insecurity risk forecasts, disaggregated to all 142 cities and municipalities of Region
            IV-A, through a publicly accessible geospatial dashboard.
          </p>
        </div>

        <SummaryCard />
      </div>
    </section>
  );
}

function SummaryCard() {
  return (
    <div
      className="rounded-xl p-6"
      style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.2)" }}
    >
      <div className="flex items-center justify-between mb-4">
        <p
          className="text-[10px] font-semibold uppercase"
          style={{ color: AMBER, letterSpacing: "0.22em" }}
        >
          Q2 2026 · Apr–Jun
        </p>
        <div className="flex gap-2">
          <span
            className="text-[9px] font-bold uppercase tracking-wider px-2 py-1 rounded-full"
            style={{ background: "rgba(249,115,22,0.18)", color: "#F97316", border: "1px solid rgba(249,115,22,0.4)" }}
          >
            High · Risk
          </span>
          <span
            className="text-[9px] font-bold uppercase tracking-wider px-2 py-1 rounded-full inline-flex items-center gap-1"
            style={{ background: "rgba(239,68,68,0.18)", color: "#EF4444", border: "1px solid rgba(239,68,68,0.4)" }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-[#EF4444]" /> Active Alert
          </span>
        </div>
      </div>

      <h3 className="text-[var(--lp-fg)] text-lg font-bold mb-6">CALABARZON Summary</h3>

      <div className="mb-6">
        <p className="text-[10px] uppercase tracking-widest mb-2" style={{ color: "var(--lp-muted2)" }}>
          Regional Food Insecurity Index
        </p>
        <p className="font-bold leading-none" style={{ fontSize: "3.5rem", color: AMBER }}>
          0.46
          <span className="text-xl ml-2" style={{ color: "var(--lp-muted2)" }}>/ 1.00</span>
        </p>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <span className="h-2 w-2 rounded-full bg-[#EAB308]" />
        <span className="text-[12px] font-bold uppercase tracking-wider text-[var(--lp-fg)]">
          Moderate · Risk
        </span>
      </div>

      <div className="relative h-2 w-full rounded-full" style={{ background: "linear-gradient(to right, #EAB308, #F97316, #EF4444)" }}>
        <div
          className="absolute -top-1 h-4 w-1 rounded-sm bg-white shadow-lg"
          style={{ left: "46%" }}
        />
      </div>
      <div className="mt-2 flex justify-between text-[9px] tracking-widest" style={{ color: "var(--lp-muted2)" }}>
        <span>0.00</span>
        <span>1.00</span>
      </div>
    </div>
  );
}

/* ---------------- CAPABILITIES ---------------- */
function Capabilities() {
  const items = [
    { icon: MapPin, title: "Geospatial Risk Mapping", body: "Province and municipality choropleth on an interactive Leaflet map, color-coded Low → Moderate → High → Severe." },
    { icon: Languages, title: "Zero-Shot Filipino-English NLP", body: "XLM-RoBERTa scores bilingual news articles against 10 food insecurity hypotheses with no labeled training data required." },
    { icon: Scale, title: "Bias-Corrected FSSI", body: "The Food Stress Sentiment Index upweights under-covered rural provinces to correct for capital-city media concentration." },
    { icon: Sparkles, title: "LightGBM + Optuna Tuning", body: "Ensemble ML optimized via Bayesian hyperparameter search across 100 trials, benchmarked against six comparison models." },
    { icon: Network, title: "SHAP Feature Attribution", body: "Every forecast is broken into named driver contributions (e.g., 'Engel Coefficient +0.19') for transparent, interpretable decision support." },
    { icon: Layers, title: "Municipal Disaggregation", body: "Province forecasts distributed to all 142 LGUs using PSA poverty incidence (60%) and population density (40%) as vulnerability weights." },
  ];

  return (
    <section className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .8s ease" }}>
        <p className="text-[11px] font-semibold uppercase mb-4" style={{ color: AMBER, letterSpacing: "0.28em" }}>
          System Capabilities
        </p>
        <h2 className="text-[var(--lp-fg)] font-bold text-[clamp(1.8rem,3.5vw,2.6rem)] leading-tight mb-12 max-w-3xl">
          Six things aiPHeed does that no other Philippine system does.
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {items.map((it, i) => (
            <div
              key={i}
              className="rounded-xl p-6 transition-all hover:-translate-y-1"
              style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.2)" }}
            >
              <h3 className="text-[var(--lp-fg)] font-bold text-[16px] mb-2">{it.title}</h3>
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
    { icon: TrendingUp, label: "Market / Prices", value: "+0.21", body: "Rice retail price + Food CPI volatility lift the index across all provinces." },
    { icon: CloudRain, label: "Climate Stress", value: "+0.17", body: "PAGASA typhoon count, rainfall anomaly, and ENSO phase shift the seasonal baseline." },
    { icon: Briefcase, label: "Employment", value: "+0.12", body: "Unemployment rate + underemployment compress household purchasing power." },
    { icon: Send, label: "OFW Remittance", value: "−0.09", body: "BSP remittance inflows reduce risk in dependent provinces (Cavite, Batangas)." },
    { icon: Fish, label: "Fish Kill", value: "+0.08", body: "Algal blooms in Taal & Laguna Lake disrupt aquaculture supply chains." },
  ];

  return (
    <section className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .8s ease" }}>
        <p className="text-[11px] font-semibold uppercase mb-4" style={{ color: AMBER, letterSpacing: "0.28em" }}>
          Risk Drivers
        </p>
        <h2 className="font-bold text-[clamp(1.6rem,3vw,2.2rem)] leading-tight mb-10" style={{ color: AMBER }}>
          WHAT DRIVES FOOD INSECURITY RISK IN CALABARZON
        </h2>

        <div className="rounded-xl p-6" style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.2)" }}>
          <p className="text-[10px] font-semibold uppercase mb-5 tracking-[0.22em]" style={{ color: AMBER }}>
            ● Top Risk Drivers · Q2 2026
          </p>
          <div className="space-y-3">
            {drivers.map((d, i) => {
              const isNeg = d.value.startsWith("−");
              return (
                <div
                  key={i}
                  className="flex items-center gap-4 p-3 rounded-lg"
                  style={{ background: "var(--lp-soft-bg)", border: "1px solid var(--lp-soft-border)" }}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-[var(--lp-fg)] font-semibold text-[13px]">{d.label}</p>
                    <p className="text-[11px]" style={{ color: "var(--lp-muted)" }}>{d.body}</p>
                  </div>
                  <span
                    className="font-mono text-[12px] font-bold px-2.5 py-1 rounded-md"
                    style={{
                      background: isNeg ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.12)",
                      color: isNeg ? "#22C55E" : "#EF4444",
                      border: `1px solid ${isNeg ? "rgba(34,197,94,0.35)" : "rgba(239,68,68,0.35)"}`,
                    }}
                  >
                    {d.value}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------------- METHODOLOGY ---------------- */
function Methodology() {
  const sources = [
    "Filipino-English News Corpus · RSS, Google News, GDELT · 17,791 articles · 2020–2025",
    "Government Statistics · PSA, BSP, DOE, PhilRice, SWS",
    "Climate Data · PAGASA (typhoon count, rainfall anomaly, ENSO phase, drought alert)",
    "Commodity & Price Data · Food CPI, rice retail price, unemployment rate",
    "Survey Anchors · DA-FNRI ENNS 2021 & 2023",
  ];

  const steps = [
    { n: "①", t: "Data Collection", d: "Multi-source ingestion via RSS & APIs" },
    { n: "②", t: "NLP Processing", d: "XLM-RoBERTa zero-shot classification" },
    { n: "③", t: "Feature Engineering", d: "FSSI, lag features, bias correction" },
    { n: "④", t: "ML Training", d: "LightGBM tuned with Optuna 100 trials" },
    { n: "⑤", t: "Spatial Disaggregation", d: "Vulnerability-weighted LGU split" },
    { n: "⑥", t: "Dashboard Output", d: "Quarterly choropleth + SHAP panels" },
  ];

  const metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Max Lead Time"];

  return (
    <section className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-6xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .8s ease" }}>
        <p className="text-[11px] font-semibold uppercase mb-4" style={{ color: AMBER, letterSpacing: "0.28em" }}>
          Methodology
        </p>
        <h2 className="text-[var(--lp-fg)] font-bold text-[clamp(1.8rem,3.5vw,2.6rem)] leading-tight mb-12">
          How the pipeline works.
        </h2>

        <h3 className="text-[var(--lp-fg)] font-bold text-[15px] mb-4 uppercase tracking-wider">Data Sources</h3>
        <div className="grid sm:grid-cols-2 gap-3 mb-14">
          {sources.map((s, i) => (
            <div
              key={i}
              className="flex items-start gap-3 p-3 rounded-lg"
              style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.15)" }}
            >
              <span className="mt-1.5 h-2 w-2 rounded-full shrink-0" style={{ background: AMBER }} />
              <p className="text-[13px]" style={{ color: "var(--lp-muted)" }}>{s}</p>
            </div>
          ))}
        </div>

        <h3 className="text-[var(--lp-fg)] font-bold text-[15px] mb-4 uppercase tracking-wider">6-Stage Pipeline</h3>
        <div className="flex flex-wrap items-stretch gap-2 mb-14">
          {steps.map((s, i) => (
            <div key={i} className="flex items-stretch gap-2 flex-1 min-w-[180px]">
              <div
                className="rounded-lg p-4 flex-1"
                style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.2)" }}
              >
                <p className="font-bold text-2xl mb-1" style={{ color: AMBER }}>{s.n}</p>
                <p className="text-[var(--lp-fg)] font-semibold text-[13px] mb-1">{s.t}</p>
                <p className="text-[11px]" style={{ color: "var(--lp-muted)" }}>{s.d}</p>
              </div>
              {i < steps.length - 1 && (
                <div className="hidden xl:flex items-center" style={{ color: AMBER }}>
                  <ArrowRight className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
        </div>

        <h3 className="text-[var(--lp-fg)] font-bold text-[15px] mb-4 uppercase tracking-wider">Evaluation Metrics</h3>
        <div className="flex flex-wrap gap-2">
          {metrics.map((m, i) => (
            <span
              key={i}
              className="text-[11px] font-semibold uppercase tracking-wider px-3 py-1.5 rounded-full"
              style={{ background: "rgba(245,166,35,0.1)", color: AMBER, border: "1px solid rgba(245,166,35,0.3)" }}
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
    "Relies entirely on public data. Does not include real-time field reports, satellite imagery, or granular agricultural production figures.",
    "Forecast lead times are empirically determined; forecasts are only published when model performance meets acceptable thresholds.",
    "NLP pipeline was not trained on labeled local data due to the absence of a Filipino-English food insecurity text dataset — a resource gap in the field, not a design decision.",
    "aiPHeed outputs are indicative risk rankings only and must not replace official government assessment procedures or serve as official food insecurity classifications.",
  ];
  return (
    <section className="px-6 sm:px-10 py-20" style={{ background: "var(--lp-bg)" }}>
      <div className="mx-auto max-w-5xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .8s ease" }}>
        <div
          className="rounded-xl p-7"
          style={{
            background: "var(--lp-card)",
            borderLeft: "4px solid #F97316",
            border: "1px solid rgba(249,115,22,0.3)",
            borderLeftWidth: 4,
          }}
        >
          <p className="text-[11px] font-bold uppercase mb-4 tracking-[0.22em]" style={{ color: "#F97316" }}>
            Data Disclaimer & Limitations
          </p>
          <ul className="space-y-3">
            {points.map((p, i) => (
              <li key={i} className="flex gap-3 text-[14px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
                <span style={{ color: "#F97316" }}>•</span>
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
      a: "Household-level forecasts require granular survey microdata that is not publicly released at quarterly cadence. aiPHeed forecasts at the province level (the finest reliable signal in public data) and disaggregates to municipalities using PSA vulnerability weights.",
    },
    {
      q: "What does \"zero-shot\" NLP mean?",
      a: "Zero-shot classification means the model (XLM-RoBERTa) scores text against natural-language hypotheses without ever being fine-tuned on labeled food insecurity examples — essential because no Filipino-English food insecurity dataset exists.",
    },
    {
      q: "Are municipal forecasts independent predictions?",
      a: "No. Municipal scores are deterministically derived from province forecasts using a vulnerability index combining PSA poverty incidence (60%) and population density (40%). They inherit province uncertainty.",
    },
    {
      q: "Who is the intended user of aiPHeed?",
      a: "DA REGION IV-A field offices, LGU planning units, academic researchers, and journalists tracking food security signals. It is a decision-support layer, not a replacement for official assessment.",
    },
    {
      q: "Can the system be adapted for other Philippine regions?",
      a: "Yes. The pipeline is region-agnostic — only the news corpus filters, PSA province codes, and vulnerability weights need to be re-fit for a new region.",
    },
  ];
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section className="px-6 sm:px-10 py-24" style={{ background: "var(--lp-bg2)" }}>
      <div className="mx-auto max-w-4xl opacity-0 translate-y-4" data-reveal style={{ transition: "all .8s ease" }}>
        <h2 className="text-[var(--lp-fg)] font-bold text-[clamp(1.6rem,3vw,2.2rem)] mb-10 uppercase tracking-wider">
          Frequently Asked Questions
        </h2>
        <div className="space-y-3">
          {items.map((it, i) => {
            const isOpen = open === i;
            return (
              <div
                key={i}
                className="rounded-xl overflow-hidden"
                style={{ background: "var(--lp-card)", border: "1px solid rgba(245,166,35,0.2)" }}
              >
                <button
                  onClick={() => setOpen(isOpen ? null : i)}
                  className="w-full flex items-center justify-between gap-4 p-5 text-left hover:bg-[var(--lp-hover)] transition-colors"
                >
                  <span className="text-[var(--lp-fg)] font-semibold text-[14px]">{it.q}</span>
                  <ChevronDown
                    className="h-5 w-5 shrink-0 transition-transform"
                    style={{ color: AMBER, transform: isOpen ? "rotate(180deg)" : "rotate(0)" }}
                  />
                </button>
                {isOpen && (
                  <div className="px-5 pb-5 text-[13px] leading-relaxed" style={{ color: "var(--lp-muted)" }}>
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
    { name: "Destura", img: desturaImg, url: "https://www.linkedin.com/in/princess-gwenn-destura" },
    { name: "Melindo", img: melindoImg, url: "https://www.linkedin.com/in/angelvheamelindo/" },
    { name: "Esico", img: esicoImg, url: "https://www.linkedin.com/in/christina-esico/" },
  ];
  return (
    <footer
      className="relative px-6 sm:px-10 pt-16 pb-6 overflow-hidden"
      style={{
        background:
          "radial-gradient(ellipse at 85% 30%, rgba(245,166,35,0.08), transparent 55%), radial-gradient(ellipse at 10% 80%, rgba(239,68,68,0.06), transparent 55%), #08090D",
        borderTop: "1px solid rgba(245,166,35,0.15)",
      }}
    >
      <div className="mx-auto max-w-7xl">
        <div className="grid grid-cols-4 gap-4 sm:gap-6 lg:gap-10 mb-12">
          {/* Brand + description + bottom links */}
          <div className="col-span-1">
            <div className="flex items-center gap-3 mb-5">
              <img src={logoDark} alt="aiPHeed" className="h-10 sm:h-12 w-auto object-contain" />
            </div>
            <p className="text-[14px] leading-relaxed mb-6" style={{ color: "var(--lp-muted)" }}>
              aiPHeed forecasts food insecurity in CALABARZON quarter-by-quarter — combining
              Filipino-English NLP and ensemble ML to detect food stress before it peaks.
            </p>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">Home</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">About</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#capabilities" className="hover:text-[var(--lp-fg)] transition-colors">Features</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="#faq" className="hover:text-[var(--lp-fg)] transition-colors">FAQ</a>
              <span style={{ color: "var(--lp-divider)" }}>•</span>
              <a href="mailto:aipheed.research@dlsud.edu.ph" className="hover:text-[var(--lp-fg)] transition-colors">Contact</a>
            </div>
          </div>

          {/* Product */}
          <div className="lg:pl-6 col-span-1">
            <h4 className="text-[var(--lp-fg)] font-bold text-[15px] mb-5">Product</h4>
            <ul className="space-y-3 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <li><Link to="/dashboard" className="hover:text-[var(--lp-fg)] transition-colors">Mapping</Link></li>
              <li><Link to="/data" className="hover:text-[var(--lp-fg)] transition-colors">Data</Link></li>
              <li><Link to="/visualization" className="hover:text-[var(--lp-fg)] transition-colors">Visualization</Link></li>
              <li><a href="#methodology" className="hover:text-[var(--lp-fg)] transition-colors">Methodology</a></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-[var(--lp-fg)] font-bold text-[15px] mb-5">Company</h4>
            <ul className="space-y-3 text-[14px]" style={{ color: "var(--lp-muted)" }}>
              <li><a href="#about" className="hover:text-[var(--lp-fg)] transition-colors">About</a></li>
              <li><a href="#disclaimer" className="hover:text-[var(--lp-fg)] transition-colors">Disclaimer</a></li>
              <li><a href="mailto:aipheed.research@dlsud.edu.ph" className="hover:text-[var(--lp-fg)] transition-colors">Contact</a></li>
            </ul>
          </div>

          {/* Meet the Developers */}
          <div className="col-span-1">
            <h4 className="text-[var(--lp-fg)] font-bold text-[15px] mb-5 lg:text-right">Meet the Developers</h4>
            <div className="flex flex-wrap lg:justify-end gap-3 sm:gap-5">
              {team.map((m, i) => (
                <a
                  key={i}
                  href={m.url}
                  target="_blank"
                  rel="noreferrer"
                  className="group flex flex-col items-center text-center transition-transform hover:-translate-y-1"
                >
                  <div
                    className="h-16 w-16 rounded-full overflow-hidden mb-2 transition-all"
                    style={{ border: "2px solid rgba(245,166,35,0.4)" }}
                  >
                    <img src={m.img} alt={m.name} className="h-full w-full object-cover" />
                  </div>
                  <p className="text-[var(--lp-fg)] text-[12px] font-semibold group-hover:text-[#F5A623] transition-colors">
                    {m.name}
                  </p>
                </a>
              ))}
            </div>
          </div>
        </div>

        <div
          className="pt-6 flex flex-row items-center justify-between gap-3 text-[12px] flex-wrap"
          style={{ borderTop: "1px solid rgba(255,255,255,0.06)", color: "var(--lp-muted2)" }}
        >
          <p>© 2026 aiPHeed. All rights reserved.</p>
          <p className="flex items-center gap-1.5">
            <span className="text-[var(--lp-fg)] font-semibold">aiPHeed</span>
          </p>
        </div>
      </div>
    </footer>
  );
}
