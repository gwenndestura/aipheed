import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Home, Database, BarChart3, Menu, LogIn, Sun, Moon, X, Mail, Lock, ArrowRight, ShieldCheck } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { DisplayPreferencesModal } from "./DisplayPreferencesModal";
import { toast } from "@/hooks/use-toast";
import { PSTClock } from "./PSTClock";
import logoDark from "@/assets/logo-dark.png";
import logoLight from "@/assets/logo-light.png";
import logoIcon from "@/assets/logo-icon.png";

interface Props {
  active?: "home" | "data" | "viz" | "none";
}

const THEME_KEY = "aipheed_theme";

export function TopNavbar({ active = "none" }: Props) {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (localStorage.getItem(THEME_KEY) as "dark" | "light") ?? "dark";
  });
  const [loginOpen, setLoginOpen] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    localStorage.setItem(THEME_KEY, theme);
    window.dispatchEvent(new CustomEvent("aipheed:theme", { detail: { theme } }));
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <header className="relative h-16 shrink-0 bg-card/95 backdrop-blur-sm border-b border-border/50 z-[1002] flex items-center px-4 gap-3">
      {/* Brand → landing */}
      <Link to="/" className="flex items-center mr-2 hover:opacity-90 transition-opacity shrink-0" aria-label="aiPHeed home">
        <img src={logoIcon} alt="aiPHeed" className="sm:hidden h-9 w-9 object-contain" />
        <img src={logoDark} alt="aiPHeed — Food Insecurity Forecasting" className="hidden sm:block dark-only h-10 w-auto object-contain" />
        <img src={logoLight} alt="aiPHeed — Food Insecurity Forecasting" className="hidden sm:block light-only h-10 w-auto object-contain" />
      </Link>

      {/* Nav links — centered */}
      <nav className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1">
        <NavBtn label="Mapping" active={active === "home"} to="/dashboard" />
        <NavBtn label="Data" active={active === "data"} to="/data" />
        <NavBtn label="Visualization" active={active === "viz"} to="/visualization" />
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
            <SheetTitle className="text-sm">Menu</SheetTitle>
          </SheetHeader>
          <div className="mt-6 space-y-1">
            <p className="text-[9px] uppercase tracking-widest text-muted-foreground/60 font-semibold px-2 mb-1">
              Account
            </p>
            <button onClick={() => setLoginOpen(true)} className="w-full">
              <MenuRow icon={LogIn} label="DA Admin Login" hint="Restricted administrator access" />
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

            <div className="pt-3 mt-3 border-t border-border/40">
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground/60 font-semibold px-2 mb-2">
                Settings
              </p>
              <DisplayPreferencesModal />
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <AdminLoginModal open={loginOpen} onOpenChange={setLoginOpen} />
    </header>
  );
}

function AdminLoginModal({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      const lower = email.trim().toLowerCase();
      if (!lower.endsWith("@calabarzon.da.gov.ph")) {
        setLoading(false);
        toast({
          title: "DA admin access only",
          description: "Public users do not sign in. Use the public map without an account.",
          variant: "destructive",
        });
        return;
      }
      if (lower !== "admin@calabarzon.da.gov.ph" || password.trim() !== "admin2026") {
        setLoading(false);
        toast({
          title: "Invalid credentials",
          description: "Try admin@calabarzon.da.gov.ph / admin2026 (demo).",
          variant: "destructive",
        });
        return;
      }
      sessionStorage.setItem("aipheed_user", JSON.stringify({ email: lower, role: "admin" }));
      toast({ title: "Welcome, DA Administrator", description: "Routing to admin dashboard…" });
      onOpenChange(false);
      navigate("/admin");
    }, 600);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="z-[1300] max-w-sm p-7 bg-card/95 backdrop-blur-md border border-border/50 rounded-2xl shadow-2xl shadow-primary/10">
        <DialogHeader>
          <div className="flex flex-col items-center text-center">
            <img src={logoIcon} alt="aiPHeed" className="h-20 w-20 object-contain mb-3 drop-shadow-[0_0_24px_rgba(232,69,60,0.35)]" />
            <DialogTitle className="text-lg font-bold tracking-tight">DA Admin Sign In</DialogTitle>
            <DialogDescription className="text-[11px] text-muted-foreground mt-1">
              Authorized access for feedback and review workflows
            </DialogDescription>
          </div>
        </DialogHeader>

        <form onSubmit={submit} className="space-y-3 mt-2">
          <div>
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Email</label>
            <div className="relative mt-1">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="off"
                className="w-full h-10 pl-9 pr-3 text-[12px] bg-secondary/50 border border-border/40 rounded-lg focus:outline-none focus:border-primary/50 focus:bg-secondary/80 placeholder:text-muted-foreground/50"
              />
            </div>
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Password</label>
            <div className="relative mt-1">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
                className="w-full h-10 pl-9 pr-3 text-[12px] bg-secondary/50 border border-border/40 rounded-lg focus:outline-none focus:border-primary/50 focus:bg-secondary/80 placeholder:text-muted-foreground/50"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full h-10 mt-2 flex items-center justify-center gap-2 rounded-lg bg-primary text-primary-foreground text-[12px] font-bold hover:bg-primary/90 transition-colors shadow-md shadow-primary/20 disabled:opacity-60"
          >
            {loading ? "Checking access…" : "Sign in to Admin"}
            {!loading && <ArrowRight className="h-3.5 w-3.5" />}
          </button>
        </form>

        <div className="mt-5 p-2.5 rounded-lg bg-secondary/40 border border-border/40 text-[10px] text-muted-foreground flex items-start gap-2">
          <ShieldCheck className="h-3.5 w-3.5 text-primary shrink-0 mt-0.5" />
          <span>
            Admin access is restricted to <span className="font-bold text-foreground">@calabarzon.da.gov.ph</span> emails.
            Demo credentials: <span className="font-bold text-foreground">admin@calabarzon.da.gov.ph</span> /{" "}
            <span className="font-bold text-foreground">admin2026</span>.
          </span>
        </div>

        <p className="text-center text-[10px] text-muted-foreground mt-4">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="hover:text-foreground underline underline-offset-2"
          >
            Back to public map
          </button>
        </p>
      </DialogContent>
    </Dialog>
  );
}

function NavBtn({ label, active, to }: { label: string; active?: boolean; to: string }) {
  return (
    <Link
      to={to}
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
    </Link>
  );
}

function MenuRow({ icon: Icon, label, hint }: { icon: typeof Home; label: string; hint?: string }) {
  return (
    <span className="w-full flex items-center gap-3 px-2 py-2 rounded-md hover:bg-secondary/60 transition-colors text-left">
      <Icon className="h-3.5 w-3.5 text-muted-foreground" />
      <span className="flex-1 min-w-0">
        <span className="block text-[11px] font-medium">{label}</span>
        {hint && <span className="block text-[9px] text-muted-foreground/70 truncate">{hint}</span>}
      </span>
    </span>
  );
}
