import { PageHeader } from "@/components/PageHeader";
import { PhasePlaceholder } from "@/components/PhasePlaceholder";

export function CamerasView() {
  return (
    <div>
      <PageHeader
        title="Cameras"
        description="RTSP and USB camera acquisition, health and calibration."
      />
      <PhasePlaceholder
        module="Cameras"
        phase={3}
        description="Connect RTSP and USB cameras, monitor stream health, capture frames, and detect black/frozen/blurry or obstructed video."
        planned={[
          "RTSP provider with automatic reconnection",
          "USB provider via OpenCV VideoCapture",
          "Frame quality checks and camera health monitoring",
          "Per-camera calibration profile and monitoring region",
        ]}
      />
    </div>
  );
}
