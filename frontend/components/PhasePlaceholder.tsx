import { Construction } from "lucide-react";

import { Card } from "@/components/Card";

export function PhasePlaceholder({
  module,
  phase,
  description,
  planned,
}: {
  module: string;
  phase: number;
  description: string;
  planned: string[];
}) {
  return (
    <Card
      title={`${module} module`}
      subtitle={`Scheduled for Phase ${phase}`}
      className="border-dashed"
    >
      <div className="flex items-start gap-3">
        <Construction className="mt-0.5 h-5 w-5 text-amber-500" aria-hidden />
        <div className="space-y-3 text-sm text-slate-600">
          <p>{description}</p>
          <div>
            <p className="font-medium text-slate-800">Planned capabilities</p>
            <ul className="mt-1 list-inside list-disc space-y-0.5">
              {planned.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </Card>
  );
}
