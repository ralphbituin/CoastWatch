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
// Tide station registry (prediction/import types were removed with the feature)
// ---------------------------------------------------------------------------

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

export type CameraType = "rtsp" | "usb" | "upload" | "video" | "youtube_live";

export interface CameraDevice {
  index: number | null;
  name: string;
  source: "local" | "host_agent";
  camera_type: "usb" | "rtsp";
  backend: string;
  available: boolean;
  is_virtual: boolean;
  width: number | null;
  height: number | null;
  fps: number | null;
  device_index: number | null;
  stream_url: string | null;
  detail: string | null;
}

export interface CameraDeviceList {
  devices: CameraDevice[];
  host_agent_configured: boolean;
  host_agent_available: boolean;
  warnings: string[];
}

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
  video_path: string | null;
  youtube_url_redacted: string | null;
  stream_quality: string | null;
  stream_format: string | null;
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
  is_live?: boolean | null;
  stream_state?: string | null;
  stream_session_id?: string | null;
  stream_recording_id?: string | null;
  frames_persisted?: number | null;
  frames_rejected?: number | null;
  reconnect_count?: number | null;
  last_error?: string | null;
  last_rolling_classification?: string | null;
  last_event_at?: string | null;
  resolved_quality?: string | null;
  resolved_format?: string | null;
  experimental_notice?: string;
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
  is_live?: boolean | null;
  resolved_format?: string | null;
  resolved_quality?: string | null;
  source_url_redacted?: string | null;
}

export interface CameraRoiPolygon {
  coordinate_space: "normalized";
  points: [number, number][];
  frame_width: number | null;
  frame_height: number | null;
}

export interface CameraCalibration {
  id: number;
  camera_id: number;
  version: number;
  reference_points: Record<string, unknown> | null;
  homography_matrix: unknown[][] | null;
  meters_per_pixel: number | null;
  roi_polygon: CameraRoiPolygon | null;
  is_active: boolean;
  valid_from: string | null;
  valid_to: string | null;
  notes: string | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Shorelines module (mirrors backend Pydantic schemas)
// ---------------------------------------------------------------------------

export type ShorelineSystemState =
  | "NORMAL"
  | "OBSERVED"
  | "ANOMALY_DETECTED"
  | "DATA_UNRELIABLE"
  | "OFFICIAL_TSUNAMI_ADVISORY";

export interface ShorelineObservationSummary {
  camera_id: number;
  camera_code: string | null;
  observed_at: string;
  monitoring_station_id: number;
  station_code: string | null;
  calibration_id: number | null;
  calibrated_position_m: number | null;
  segmentation_confidence: number | null;
  frame_quality_score: number | null;
  frame_quality_flags: string[] | null;
  created_at: string;
  recording_id?: string | null;
  frame_index?: number | null;
  extraction_status?: string | null;
  extraction_reason?: string | null;
  shoreline_position_px?: number | null;
  signed_position_px?: number | null;
  position_axis?: string | null;
  shoreline_axis?: string | null;
  measured_coverage?: number | null;
  segment_count?: number | null;
  inference_time_ms?: number | null;
  capture_timestamp_source?: string | null;
  capture_timestamp_confidence?: number | null;
  elapsed_seconds?: number | null;
  stream_session_id?: string | null;
  ingest_latency_ms?: number | null;
}

export interface ShorelineObservationDetail extends ShorelineObservationSummary {
  shoreline_geometry: [number, number][] | null;
  /** Ordered segments; absent for historical observations. */
  shoreline_segments?: [number, number][][] | null;
  shoreline_axis?: "horizontal" | "vertical" | null;
  interpolated_ranges?: [number, number][] | null;
  measured_coverage?: number | null;
  source_frame_path: string | null;
  position_sign?: string | null;
  segment_measurements?: Array<Record<string, unknown>> | null;
  class_fractions?: Record<string, number> | null;
  capture_timestamp_source?: string | null;
  capture_timestamp_confidence?: number | null;
}

// ---------------------------------------------------------------------------
// Phase 5/6 prototype: baselines, anomalies, archive videos
// ---------------------------------------------------------------------------

export interface ShorelineBaseline {
  id: number;
  monitoring_station_id: number;
  camera_id: number | null;
  tide_station_id: number | null;
  recording_id: string | null;
  name: string;
  model_type: string;
  status: string | null;
  provisional: boolean | null;
  reason: string | null;
  sample_count: number;
  is_active: boolean;
  position_axis: string | null;
  position_sign: string | null;
  position_unit: string | null;
  detector_mode?: string | null;
  r_squared: number | null;
  model_params: Record<string, unknown> | null;
  prediction_interval: Record<string, unknown> | null;
  valid_from: string | null;
  valid_to: string | null;
  notes: string | null;
  created_at: string;
}

export interface ShorelineBaselineStatus {
  available: boolean;
  status: string;
  reason: string | null;
  detector_mode?: string | null;
  parameters?: Record<string, unknown> | null;
  experimental_notice: string;
  baselines: ShorelineBaseline[];
}

export type AnomalyClassification =
  | "NORMAL"
  | "UNUSUAL_SHORELINE_POSITION"
  | "RAPID_SHORELINE_RETREAT"
  | "PERSISTENT_DRAWDOWN"
  | "POSSIBLE_DRAWDOWN"
  | "DATA_UNRELIABLE"
  | "EXPLORATORY_MOVEMENT_ONLY"
  | "EXPLORATORY_ROLLING_BASELINE"
  | "PENDING_PERSISTENCE";

export interface AnomalyEvent {
  id: number;
  detected_at: string;
  monitoring_station_id: number;
  camera_id: number | null;
  recording_id: string | null;
  frame_index: number | null;
  state: string;
  classification: AnomalyClassification | null;
  anomaly_score: number | null;
  baseline_status: string | null;
  tide_adjusted: boolean | null;
  tide_data_available: boolean | null;
  tide_source: string | null;
  tide_height_m: number | null;
  observed_position_px: number | null;
  expected_position_px: number | null;
  residual_px: number | null;
  robust_z: number | null;
  movement_rate_px_s: number | null;
  persistence_count: number | null;
  quality_status: string | null;
  reason: string | null;
  explanation: string | null;
  contributing_factors: Record<string, unknown> | null;
  acknowledged: boolean;
  created_at: string;
  detector_mode?: string | null;
  resolved_detector_mode?: string | null;
  mode_reason?: string | null;
  onset_frame_index?: number | null;
  rolling_baseline_px?: number | null;
  rolling_scale_px?: number | null;
  scale_floored?: boolean | null;
  cumulative_drawdown_px?: number | null;
  baseline_frozen?: boolean | null;
  event_max_classification?: string | null;
  onset_time?: string | null;
  detection_time?: string | null;
  detection_latency_s?: number | null;
  event_window_seconds?: number | null;
  cooldown_state?: string | null;
  config_hash?: string | null;
}

export interface AnomalyEventList {
  total: number;
  items: AnomalyEvent[];
  experimental_notice: string;
}

export interface AnomalySummary {
  total: number;
  by_classification: Record<string, number>;
  by_state: Record<string, number>;
  by_baseline_status: Record<string, number>;
  experimental_notice: string;
}

export interface VideoRecording {
  id: number;
  recording_id: string;
  camera_id: number | null;
  monitoring_station_id: number | null;
  file_path: string;
  file_size_bytes: number | null;
  duration_seconds: number | null;
  fps: number | null;
  frame_count: number | null;
  width: number | null;
  height: number | null;
  video_codec: string | null;
  audio_codec: string | null;
  timezone: string | null;
  timestamp_source: string | null;
  timestamp_confidence: number | null;
  processing_status: string | null;
  processed_frames: number | null;
  sampled_frames: number | null;
  metadata_json: Record<string, unknown> | null;
  notes: string | null;
  created_at: string;
}

export interface ShorelineObserveResult {
  station_id: number;
  camera_id: number;
  observed_at: string;
  actual_shoreline_m: number | null;
  expected_shoreline_m: number | null;
  residual_m: number | null;
  retreat_velocity_m_s: number | null;
  anomaly_score: number | null;
  state: ShorelineSystemState;
  details: Record<string, unknown>;
}

export interface ShorelineModuleStatus {
  module: string;
  phase: number;
  segmentation_model_available: boolean;
  implemented_endpoints: string[];
  notes: string[];
  architecture?: string | null;
  encoder?: string | null;
  model_version?: string | null;
  class_names?: string[] | null;
  class_ids?: Record<string, number> | null;
  water_classes?: number[] | null;
  ignore_index?: number | null;
  checkpoint_id?: string | null;
  checkpoint_sha256?: string | null;
  providers?: string[];
  precision?: Record<string, unknown>;
  limitations?: string[];
}

export interface ShorelineCalibration {
  id: number;
  camera_id: number;
  camera_code: string | null;
  version: number;
  meters_per_pixel: number | null;
  is_active: boolean;
  valid_from: string | null;
  valid_to: string | null;
  notes: string | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Phase 7: live stream sessions and the Predictor + Anomaly Test Lab
// ---------------------------------------------------------------------------

export interface StreamSession {
  id: number;
  session_id: string;
  recording_id: string;
  camera_id: number;
  camera_code: string | null;
  started_at: string;
  ended_at: string | null;
  status: string;
  source_url_redacted: string | null;
  sample_interval_seconds: number | null;
  last_frame_at: string | null;
  frames_captured: number;
  frames_persisted: number;
  frames_rejected: number;
  reconnect_count: number;
  last_error: string | null;
  last_rolling_classification: string | null;
  last_event_at: string | null;
  detector_config: Record<string, unknown> | null;
  metadata_json: Record<string, unknown> | null;
  created_at: string;
  experimental_notice?: string;
}

export interface CameraStreamStatus {
  camera_id: number;
  state: string;
  session_id: string | null;
  recording_id: string | null;
  started_at: string | null;
  last_frame_at: string | null;
  frames_captured: number;
  frames_persisted: number;
  frames_rejected: number;
  consecutive_failures: number;
  reconnect_count: number;
  last_error: string | null;
  last_rolling_classification: string | null;
  last_event_at: string | null;
  sample_interval_seconds: number;
  is_live: boolean | null;
  experimental_notice?: string;
}

export interface LabRun {
  run_id: string;
  kind: string;
  status: string;
  created_at: string;
  updated_at: string;
  timezone: string;
  source: Record<string, unknown>;
  progress: Record<string, unknown>;
  row_count: number;
  event_count: number;
  errors: string[];
  warnings: string[];
  artifacts: string[];
  persist: boolean;
  camera_id: number | null;
  recording_id: string | null;
  experimental_notice: string;
}

export interface LabRow {
  frame_index?: number | null;
  observed_at?: string | null;
  observed_at_local?: string | null;
  elapsed_seconds?: number | null;
  timestamp_source?: string | null;
  timestamp_confidence?: number | null;
  capture_timestamp_source?: string | null;
  capture_timestamp_confidence?: number | null;
  shoreline_position_px?: number | null;
  signed_position_px?: number | null;
  position_axis?: string | null;
  segment_count?: number | null;
  segment_measurements?: Array<Record<string, unknown>> | null;
  segmentation_confidence?: number | null;
  class_fractions?: Record<string, number> | null;
  inference_time_ms?: number | null;
  horizon_y_px?: number | null;
  extraction_status?: string | null;
  extraction_reason?: string | null;
  rolling_baseline_px?: number | null;
  rolling_scale_px?: number | null;
  residual_px?: number | null;
  robust_z?: number | null;
  cumulative_drawdown_px?: number | null;
  movement_rate_px_s?: number | null;
  classification?: string | null;
  state?: string | null;
  quality_status?: string | null;
  in_event?: boolean | null;
  event_max_classification?: string | null;
  onset_time?: string | null;
  detection_time?: string | null;
  persistence_count?: number | null;
  camera_motion_gate?: string | null;
  segment_evidence?: Array<Record<string, unknown>> | null;
  reason?: string | null;
  [key: string]: unknown;
}

export interface LabRowsResponse {
  run_id: string;
  total: number;
  limit: number;
  offset: number;
  items: LabRow[];
  experimental_notice: string;
}
