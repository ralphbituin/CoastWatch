"""Application configuration.

All settings are read from environment variables (and an optional ``.env``
file at the repository root). No secrets are hard-coded here — this module
only declares the *names* and sane non-secret defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the CoastWatch backend."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    project_name: str = "CoastWatch"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    api_v1_prefix: str = Field(default="/api/v1")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    # "json" for structured machine-readable logs, "console" for humans.
    log_format: str = Field(default="console")

    # --- CORS (comma separated, e.g. "http://localhost:3000,http://127.0.0.1:3000") ---
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")

    # --- Database ---
    postgres_user: str = Field(default="coastwatch")
    postgres_password: str = Field(default="coastwatch")
    postgres_db: str = Field(default="coastwatch")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    # If set, takes precedence over the individual POSTGRES_* variables.
    database_url: str | None = Field(default=None)

    # --- Domain defaults ---
    # IANA time zone used as the default for Philippine monitoring locations.
    default_timezone: str = Field(default="Asia/Manila")
    # Directory that holds trained model artifacts (Phase 4+). Never commit weights.
    model_path: str = Field(default=str(REPO_ROOT / "ml" / "models"))
    # Root directory for operator-provided datasets (CSV tide tables, videos).
    data_dir: str = Field(default=str(REPO_ROOT / "data"))

    # --- Phase 7 live capture / test lab (in-process, bounded) ---
    # Autostart is opt-in so a developer/test process never opens a real stream
    # or a camera device just by importing the app.
    live_capture_autostart: bool = Field(default=False)
    live_capture_interval_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    # Lower bound for serving the worker's most recent frame/mask/shoreline.
    # The effective value is max(this, 2.5 * live_capture_interval_seconds).
    preview_cache_max_age_seconds: float = Field(default=60.0, ge=0.0)
    # ONNX Runtime intra-op threads. 0 leaves ORT's default; the CPU sweep in
    # the latency report picked the documented value.
    onnx_intra_op_threads: int = Field(default=0, ge=0)
    # Process-wide CPU inference cap. Multiple camera workers share the CPU;
    # one run at a time keeps ONNX from oversubscribing every core.
    onnx_max_concurrent_inferences: int = Field(default=1, ge=1, le=8)
    lab_max_concurrent_runs: int = Field(default=2, ge=1, le=8)
    lab_run_ttl_seconds: float = Field(default=6 * 3600.0, ge=60.0)
    lab_max_images: int = Field(default=50, ge=1, le=500)
    lab_max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    lab_max_total_upload_bytes: int = Field(default=40 * 1024 * 1024, ge=1024)
    lab_max_video_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1024)
    lab_max_video_samples: int = Field(default=120, ge=1, le=2000)
    lab_max_stream_samples: int = Field(default=60, ge=1, le=1000)
    lab_max_stream_seconds: float = Field(default=600.0, ge=5.0, le=86400.0)

    # --- Tide-aware baseline gates (retained, but no predictor feeds them) ---
    # |slope| at a timestamp below this (m per hour) is treated as slack water.
    tide_slack_slope_tolerance_m_per_hour: float = Field(default=0.01)
    # Minimum prominence (m) for a local extremum to count as high/low water.
    tide_extrema_prominence_m: float = Field(default=0.05)

    # --- Class-aware shoreline extraction (Ocean/Beach boundary) ---
    # Area thresholds are fractions of the frame, so a 640x360 preview and a
    # 1920x1080 frame behave consistently. Explicit *_px values passed to the
    # extractor act as a floor on top of these ratios.
    shoreline_min_ocean_component_ratio: float = Field(default=0.005)
    shoreline_min_beach_component_ratio: float = Field(default=0.0005)
    # Morphological closing of the Ocean mask only (never the stored classes).
    # Kernel width = ratio * min(frame height, width), clamped and made odd.
    # 0.03 closes a ~11 px gap at 640x360 and ~32 px at 1920x1080.
    shoreline_close_kernel_ratio: float = Field(default=0.03)
    shoreline_close_kernel_min_px: int = Field(default=3)
    shoreline_close_kernel_max_px: int = Field(default=61)
    # Median filter along the shoreline, as a fraction of the short frame side.
    # 0.008 is 3 px at 640x360 and 9 px at 1920x1080: removes single-pixel
    # spikes while preserving wave-scale variation. Set to 0 to disable.
    shoreline_median_kernel_ratio: float = Field(default=0.008)
    # Split a line when consecutive points are farther apart than this fraction
    # of the scanline length (columns for land_below/above, rows otherwise).
    # 0.03 is ~11 px at 360 rows and ~58 px at 1920 columns; the minimum is 8 px.
    shoreline_max_gap_ratio: float = Field(default=0.03)
    # Drop segments shorter than this fraction of the scanline length.
    shoreline_min_segment_ratio: float = Field(default=0.015)
    # Ignore this fraction of the short frame side at every frame edge.
    shoreline_margin_ratio: float = Field(default=0.005)
    # How many pixels on the land side must be Beach for a contact to count.
    shoreline_beach_check_depth_ratio: float = Field(default=0.02)
    # "auto" infers the land side from the class centroids. Pin one of
    # land_below / land_above / land_left / land_right for a fixed camera whose
    # geometry the heuristic gets wrong.
    shoreline_orientation: str = Field(default="auto")
    # Ocean component selection when the area filter keeps more than one large
    # component. "coastline" keeps the component with the longest shared border
    # with the dominant Beach plus any other open (Sky/border) seaward
    # component, discarding land-side false positives (blue deck furniture,
    # pools, rock shadows, on-screen overlays). "all" restores the pre-fix
    # area-only behaviour; single-component frames are unaffected either way.
    shoreline_ocean_component_selection: str = Field(default="coastline")
    # Secondary shoreline branches (after component selection and gap
    # splitting) must have open water at least this fraction of the short frame
    # side, measured perpendicular to the local shoreline tangent. The anchor
    # branch is always kept. Default 0.06 comes from the follow-up audit: the
    # surviving landward strip in reference_frame_2 had 5.6% local depth, while
    # the smallest real branch in the six original fixtures had 13%.
    # Set to 0 to disable this additional branch filter.
    shoreline_min_branch_depth_ratio: float = Field(default=0.06)
    # One continuous line / one axis contract. When true, the final shoreline
    # is exactly one bridged trace (or unavailable); set false to preserve the
    # previous multi-segment output for rollback/comparison.
    shoreline_single_line_mode: bool = Field(default=True)
    # Minimum fraction of the chosen axis that must be directly measured (not
    # linearly bridged). Default 0.20 comes from the single-line audit: the
    # smallest audited real coastline covered 0.23 of its axis, while sparse
    # candidates below 0.20 were not a usable single interpretation.
    shoreline_min_measured_coverage_ratio: float = Field(default=0.20)

    @field_validator("model_path", "data_dir", mode="after")
    @classmethod
    def _resolve_against_repo_root(cls, value: str) -> str:
        """Resolve relative paths from the repository root.

        ``.env`` ships ``MODEL_PATH=./ml/models``. Resolving against the process
        working directory made the artifact undiscoverable when the API is
        started from ``backend/`` instead of the repository root. Absolute values
        (e.g. the Docker mount ``/app/models``) are left untouched.
        """
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy URL for the (synchronous) PostgreSQL + PostGIS database."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


settings = get_settings()
