import { ArrowRight } from "lucide-react";

const STAGES = [
  { label: "Camera", detail: "RTSP / USB frame" },
  { label: "Waterline", detail: "vision/shoreline.py" },
  { label: "Expected", detail: "baseline/expected_shoreline.py" },
  { label: "Residual + score", detail: "anomaly/*" },
  { label: "Dashboard / alert", detail: "WebSocket push" },
];

/** Visual summary of the end-to-end data flow (documented in docs/ARCHITECTURE.md). */
export function DataFlow() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {STAGES.map((stage, index) => (
        <div key={stage.label} className="flex items-center gap-2">
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <p className="text-xs font-semibold text-slate-800">{stage.label}</p>
            <p className="text-[11px] text-slate-500">{stage.detail}</p>
          </div>
          {index < STAGES.length - 1 && (
            <ArrowRight className="h-4 w-4 text-slate-400" aria-hidden />
          )}
        </div>
      ))}
    </div>
  );
}
