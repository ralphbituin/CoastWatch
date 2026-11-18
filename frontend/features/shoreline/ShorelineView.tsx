import { PageHeader } from "@/components/PageHeader";
import { PhasePlaceholder } from "@/components/PhasePlaceholder";

export function ShorelineView() {
  return (
    <div>
      <PageHeader
        title="Shoreline"
        description="Waterline extraction, calibration and tide-aware baseline."
      />
      <PhasePlaceholder
        module="Shoreline"
        phase={4}
        description="Segment water vs. land, extract the waterline ('where is the waterline now'), calibrate to meters, and learn the expected shoreline ('where should the waterline be given the tide')."
        planned={[
          "U-Net / DeepLabV3+ segmentation from MODEL_PATH artifacts",
          "Contour-based waterline extraction with temporal smoothing",
          "Homography calibration and camera-movement detection",
          "Tide-aware baseline with prediction intervals",
        ]}
      />
    </div>
  );
}
