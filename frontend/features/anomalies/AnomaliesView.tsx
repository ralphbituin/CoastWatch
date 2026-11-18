import { PageHeader } from "@/components/PageHeader";
import { PhasePlaceholder } from "@/components/PhasePlaceholder";

export function AnomaliesView() {
  return (
    <div>
      <PageHeader
        title="Anomalies"
        description="Residuals, retreat velocity, scoring and system states."
      />
      <PhasePlaceholder
        module="Anomalies"
        phase={6}
        description="Compute shoreline residuals and retreat velocity, score deviation from the learned baseline, and report explainable detection events with conservative system states."
        planned={[
          "Z-score / rolling threshold / CUSUM; Isolation Forest & One-Class SVM later",
          "Explainable anomaly score (never a validated tsunami probability)",
          "States: NORMAL, ANOMALY_DETECTED, DATA_UNRELIABLE, OFFICIAL_TSUNAMI_ADVISORY",
          "Event persistence and detection explanations",
        ]}
      />
    </div>
  );
}
