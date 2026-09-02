import { nationalTrendData } from "@/data/mockData";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, Line, ComposedChart } from "recharts";

export function NationalTrendChart() {
  return (
    <div className="bg-card border rounded-lg p-4">
      <div className="flex items-center justify-between mb-1">
        <div>
          <h3 className="text-sm font-bold">National Risk Level & FPSI Trend</h3>
          <p className="text-xs text-muted-foreground">12-month rolling national average</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "hsl(35, 90%, 55%)" }} />
            Risk Level
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "hsl(200, 80%, 50%)" }} />
            FPSI
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={nationalTrendData} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="rfiiGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(35, 90%, 55%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(35, 90%, 55%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="month"
            tick={{ fontSize: 10, fill: "hsl(215, 15%, 55%)" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "hsl(215, 15%, 55%)" }}
            tickLine={false}
            axisLine={false}
            width={35}
            domain={[0, 0.7]}
          />
          <Tooltip
            contentStyle={{
              fontSize: 12,
              borderRadius: 0,
              backgroundColor: "hsl(222, 25%, 12%)",
              border: "1px solid hsl(222, 20%, 18%)",
              color: "hsl(210, 20%, 92%)",
            }}
            formatter={(v: number, name: string) => [v.toFixed(3), name === "rfii" ? "Risk Level" : "FPSI"]}
          />
          <Area
            type="monotone"
            dataKey="rfii"
            stroke="hsl(35, 90%, 55%)"
            strokeWidth={2}
            fill="url(#rfiiGrad)"
          />
          <Line
            type="monotone"
            dataKey="fpsi"
            stroke="hsl(200, 80%, 50%)"
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
