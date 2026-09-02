import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from "@/components/ui/dialog";
import { Info } from "lucide-react";
import logoIcon from "@/assets/logo-icon.png";

export function AboutModal() {
  return (
    <Dialog>
      <DialogTrigger className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors">
        <Info className="h-3 w-3" />
        About & Methodology
      </DialogTrigger>
      <DialogContent className="z-[1200] max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2.5">
            <div className="rounded-lg p-1 bg-transparent">
              <img src={logoIcon} alt="aiPHeed" className="h-7 w-7 object-contain" />
            </div>
            <DialogTitle className="text-sm">About aiPHeed — DA REGION IV-A</DialogTitle>
          </div>
          <DialogDescription className="text-[11px] leading-relaxed pt-2">
            aiPHeed is an interactive geospatial food-insecurity forecasting system for DA Region IV-A.
            It uses zero-shot multilingual NLP and ensemble machine learning to publish quarterly,
            3-month-ahead risk forecasts at province and municipal level. The system is intended to
            support anticipatory action — it does not replace official food-insecurity classifications.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 mt-3">
          <Section title="Forecast vs Active Alert">
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Each forecast displays a <b className="text-foreground">risk probability</b> and a{" "}
              <b className="text-foreground">HIGH / LOW</b> label using a <b>0.50</b> internal display cutoff.
              The <b className="text-foreground">active alert</b> system is gated separately at{" "}
              <b>0.60</b>, a stricter threshold designed for anticipatory action.
            </p>
          </Section>

          <Section title="Data sources">
            <ul className="text-[11px] leading-relaxed text-muted-foreground list-disc pl-4 space-y-0.5">
              <li>Philippine Statistics Authority (PSA) — CPI, FIES, LFS</li>
              <li>SWS Hunger Surveys (retrospective verification)</li>
              <li>Geocoded news corpus — Inquirer, PhilStar, BusinessWorld, ABS-CBN</li>
              <li>PAGASA climate / typhoon advisories</li>
              <li>BSP OFW remittance series · BFAR fish-kill bulletins</li>
            </ul>
            <p className="text-[10px] text-muted-foreground/70 mt-1.5">
              Publication cadence: quarterly forecasts, 3-month lead time.
            </p>
          </Section>

          <Section title="LIMITED_SIGNAL flag">
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Provinces with fewer than <b>5 geocoded articles</b> in a quarter are flagged{" "}
              <b className="text-foreground">LIMITED_SIGNAL</b>. The badge propagates to all 142 LGUs
              under that province (Algorithm Rule 5).
            </p>
          </Section>

          <Section title="Disclaimer">
            <p className="text-[11px] leading-relaxed italic text-foreground/90">
              "Results obtained serve only as potential risk indicators and are not to be construed
              as official food insecurity classifications or policy recommendations."
            </p>
          </Section>

          <Section title="Contact">
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              For verified social protection assistance, contact{" "}
              <b className="text-foreground">DA Region IV-A (CALABARZON)</b> · BPI Compound, Marawoy,
              Lipa City, Batangas · (043) 774-2005 · ord@calabarzon.da.gov.ph
            </p>
          </Section>

          <div className="grid grid-cols-2 gap-2 pt-1">
            <Row k="Model" v="LightGBM + SMOTE" />
            <Row k="Performance" v="F1 0.82 · AUC 0.87" />
            <Row k="Coverage" v="5 provinces · 142 LGUs" />
            <Row k="Refresh" v="Quarterly · 3-mo lead" />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="text-[10px] font-bold uppercase tracking-wider text-primary/90 mb-1">{title}</h4>
      {children}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between text-[11px] border border-border/40 rounded-md px-2 py-1.5 bg-secondary/30">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-semibold text-right">{v}</span>
    </div>
  );
}
