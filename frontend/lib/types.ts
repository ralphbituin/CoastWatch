/** Shared API types mirroring the backend Pydantic schemas. */

export type HealthStatus = "ok" | "degraded" | "error";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface DatabaseHealthResponse {
  status: string;
  database: string;
  reachable: boolean;
  latency_ms: number | null;
  server_version: string | null;
  timescaledb_version: string | null;
  postgis_version: string | null;
  error: string | null;
}

export interface ModuleStatusPayload {
  module: string;
  phase: number;
  implemented: boolean;
  message: string;
  planned_endpoints: string[];
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

/** System states as defined by the anomaly engine contract. */
export type SystemState =
  | "NORMAL"
  | "ANOMALY_DETECTED"
  | "DATA_UNRELIABLE"
  | "OFFICIAL_TSUNAMI_ADVISORY";

// ---------------------------------------------------------------------------
// Tide module (mirrors backend Pydantic schemas)
// ---------------------------------------------------------------------------

export interface TideEvent {
  timestamp: string;
  height: number;
  type: "high" | "low";
  unit: string;
  datum: string | null;
}

export interface TideStation {
  id: number;
  code: string;
  name: string;
  region: string | null;
  country_iso: string;
  country_name: string;
  latitude: number;
  longitude: number;
  timezone: string;
  data_source: string;
  vertical_datum: string | null;
  measurement_unit: string;
  prediction_start: string | null;
  prediction_end: string | null;
  monitoring_station_id: number | null;
  dataset_count: number;
}

export interface TidePoint {
  timestamp: string;
  height: number;
  unit: string;
  datum: string | null;
  quality_status: string;
}

export interface TidePhaseResponse {
  station_code: string;
  station_name: string;
  requested_timestamp: string;
  local_timestamp: string;
  server_time: string;
  predicted_height_m: number;
  unit: string;
  vertical_datum: string | null;
  direction: string;
  quality: string;
  quality_note: string;
  source: string | null;
  dataset_id: number | null;
  timezone: string;
  interpolation_method: string;
  previous_event: TideEvent | null;
  next_event: TideEvent | null;
}

export interface TideChartResponse {
  station_code: string;
  start: string;
  end: string;
  method: string;
  count: number;
  coverage_start: string | null;
  coverage_end: string | null;
  sufficient_for_7d: boolean;
  points: TidePoint[];
  events: TideEvent[];
}

export interface ImportPreviewResponse {
  station_code: string | null;
  timezone: string;
  file_hash: string | null;
  columns_detected: string[];
  column_mapping: Record<string, unknown>;
  total_records: number;
  valid_records: number;
  rejected_records: number;
  rejection_summary: Record<string, unknown>;
  rejection_samples: Array<Record<string, unknown>>;
  coverage_start: string | null;
  coverage_end: string | null;
  message: string;
}

export interface ImportCommitResponse extends ImportPreviewResponse {
  dataset_id: number | null;
  import_id: number;
  status: "committed" | "partial" | "rejected" | "duplicate";
  imported_records: number;
}

export interface DatasetImportRead {
  id: number;
  tide_station_id: number;
  station_code: string;
  dataset_id: number | null;
  source_filename: string | null;
  file_hash: string | null;
  total_records: number;
  imported_records: number;
  rejected_records: number;
  status: string;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Stations module
// ---------------------------------------------------------------------------

export interface MonitoringStation {
  id: number;
  code: string;
  name: string;
  region: string | null;
  country_iso: string;
  country_name: string;
  latitude: number;
  longitude: number;
  timezone: string;
  vertical_datum: string | null;
  description: string | null;
  tide_station_count: number;
  created_at: string;
}

export interface Country {
  id: number;
  iso_code: string;
  name: string;
  default_timezone: string;
}

// ---------------------------------------------------------------------------
// Cameras module (mirrors backend Pydantic schemas)
// ---------------------------------------------------------------------------

export type CameraType = "rtsp" | "usb" | "upload";

export interface Camera {
  id: number;
  code: string;
  name: string;
  monitoring_station_id: number;
  station_code: string | null;
  station_name: string | null;
  camera_type: CameraType;
  rtsp_url_redacted: string | null;
  device_index: number | null;
  resolution_width: number | null;
  resolution_height: number | null;
  fps: number | null;
  latitude: number | null;
  longitude: number | null;
  orientation_deg: number | null;
  monitoring_region: Record<string, unknown> | null;
  enabled: boolean;
  active_calibration_id: number | null;
  created_at: string;
}

export interface CameraStatus {
  camera_id: number;
  code: string;
  enabled: boolean;
  connected: boolean;
  healthy: boolean;
  message: string;
  last_frame_at: string | null;
  frames_captured: number;
  checked_at: string;
}

export interface CameraQuality {
  score: number;
  is_black: boolean;
  is_frozen: boolean;
  is_blurry: boolean;
  is_obstructed: boolean;
  is_timestamp_inconsistent: boolean;
  flags: string[];
}

export interface CameraTestConnectionResult {
  camera_id: number;
  ok: boolean;
  message: string;
  latency_ms: number | null;
  frame_width: number | null;
  frame_height: number | null;
  captured_at: string | null;
  quality: CameraQuality | null;
}

export interface CameraCalibration {
  id: number;
  camera_id: number;
  version: number;
  reference_points: Record<string, unknown> | null;
  homography_matrix: unknown[][] | null;
  meters_per_pixel: number | null;
  is_active: boolean;
  valid_from: string | null;
  valid_to: string | null;
  notes: string | null;
  created_at: string;
}
