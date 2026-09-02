import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Mail, Lock, ArrowRight, ShieldCheck } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import logoIcon from "@/assets/logo-icon.png";

const ADMIN_DOMAINS = ["calabarzon.da.gov.ph"];

function isDaAdminEmail(value: string) {
  const email = value.trim().toLowerCase();
  return ADMIN_DOMAINS.some((domain) => email.endsWith(`@${domain}`));
}

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      if (!isDaAdminEmail(email)) {
        setLoading(false);
        sessionStorage.removeItem("aipheed_user");
        toast({
          title: "DA admin access only",
          description: "Public users do not sign in. Use the public map without an account.",
          variant: "destructive",
        });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      sessionStorage.setItem(
        "aipheed_user",
        JSON.stringify({ email: normalizedEmail, role: "admin" })
      );
      toast({
        title: "Welcome, DA Administrator",
        description: "Routing to admin dashboard…",
      });
      navigate("/admin");
    }, 600);
  };

  return (
    <div className="min-h-screen w-full bg-background flex items-center justify-center p-6 relative overflow-hidden">
      {/* Decorative gradient orbs */}
      <div className="absolute -top-32 -left-32 h-96 w-96 rounded-full bg-primary/15 blur-3xl" />
      <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full bg-risk-severe/10 blur-3xl" />

      <div className="relative w-full max-w-sm bg-card/95 backdrop-blur-md border border-border/50 rounded-2xl shadow-2xl shadow-primary/10 p-7">
        <div className="flex flex-col items-center text-center mb-6">
          <img src={logoIcon} alt="aiPHeed" className="h-20 w-20 object-contain mb-3 drop-shadow-[0_0_24px_rgba(232,69,60,0.35)]" />
          <h1 className="text-lg font-bold tracking-tight">DA Admin Sign In</h1>
          <p className="text-[11px] text-muted-foreground mt-1">
            Authorized access for feedback and review workflows
          </p>
        </div>

        <form onSubmit={submit} className="space-y-3">
          <div>
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">
              Email
            </label>
            <div className="relative mt-1">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder=""
                autoComplete="off"
                className="w-full h-10 pl-9 pr-3 text-[12px] bg-secondary/50 border border-border/40 rounded-lg focus:outline-none focus:border-primary/50 focus:bg-secondary/80 placeholder:text-muted-foreground/50"
              />
            </div>
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">
              Password
            </label>
            <div className="relative mt-1">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder=""
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
          </span>
        </div>

        <p className="text-center text-[10px] text-muted-foreground mt-4">
          <Link to="/" className="hover:text-foreground underline underline-offset-2">
            Back to public map
          </Link>
        </p>
      </div>
    </div>
  );
}
