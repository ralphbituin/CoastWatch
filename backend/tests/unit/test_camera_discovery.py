"""Unit tests for camera device discovery (no hardware required)."""

from __future__ import annotations

from app.services.camera import discovery


def test_parse_ffmpeg_dshow_new_layout() -> None:
    output = """
[dshow @ 0000] "Lihappe8 Webcam L69SN 1080P" (video)
[dshow @ 0000]   Alternative name "@device_pnp_\\\\?\\usb#vid_0c46"
[dshow @ 0000] "OBS Virtual Camera" (video)
[dshow @ 0000] "Microphone (Realtek(R) Audio)" (audio)
"""

    assert discovery._parse_ffmpeg_dshow_devices(output) == [
        "Lihappe8 Webcam L69SN 1080P",
        "OBS Virtual Camera",
    ]


def test_parse_ffmpeg_dshow_legacy_sections() -> None:
    output = """
[dshow @ 0000] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0000]  "Integrated Camera"
[dshow @ 0000]     Alternative name "@device_pnp_..."
[dshow @ 0000] DirectShow audio devices
[dshow @ 0000]  "Microphone"
"""

    assert discovery._parse_ffmpeg_dshow_devices(output) == ["Integrated Camera"]


def test_discover_prefers_local_over_host_agent_duplicate(monkeypatch) -> None:
    local = [
        discovery.DiscoveredCamera(
            index=0,
            name="Lihappe8 Webcam L69SN 1080P",
            source="local",
            camera_type="usb",
            backend="directshow",
            device_index=0,
        )
    ]
    agent = [
        discovery.DiscoveredCamera(
            index=0,
            name="Lihappe8 Webcam L69SN 1080P",
            source="host_agent",
            camera_type="rtsp",
            backend="dshow->mjpeg",
            stream_url="http://host.docker.internal:8765/stream/0.mjpg",
        )
    ]

    class _Settings:
        host_camera_agent_url = "http://host.docker.internal:8765"
        host_camera_agent_timeout_seconds = 1.0

    monkeypatch.setattr(discovery, "_local_cameras", lambda probe: local)
    monkeypatch.setattr(
        discovery, "_host_agent_cameras", lambda base_url, timeout, probe=False: list(agent)
    )
    monkeypatch.setattr(discovery, "get_settings", lambda: _Settings())

    result = discovery.discover_cameras(include_host_agent=True)

    assert result.host_agent_available is True
    assert [(device.source, device.name) for device in result.devices] == [
        ("local", "Lihappe8 Webcam L69SN 1080P")
    ]


def test_devices_route_serializes_discovery(monkeypatch) -> None:
    from app.api.routes import cameras as camera_routes

    result = discovery.DiscoveryResult(
        devices=[
            discovery.DiscoveredCamera(
                index=0,
                name="Lihappe8 Webcam L69SN 1080P",
                source="local",
                camera_type="usb",
                backend="directshow",
                device_index=0,
            )
        ],
        warnings=[],
        host_agent_configured=True,
        host_agent_available=False,
    )
    monkeypatch.setattr(discovery, "discover_cameras", lambda probe: result)

    response = camera_routes.list_camera_devices(probe=False)

    assert response.devices[0].name == "Lihappe8 Webcam L69SN 1080P"
    assert response.devices[0].camera_type == "usb"
    assert response.host_agent_configured is True


def test_discover_keeps_host_agent_when_no_local_device(monkeypatch) -> None:
    agent = [
        discovery.DiscoveredCamera(
            index=0,
            name="OBS Virtual Camera",
            source="host_agent",
            camera_type="rtsp",
            backend="dshow->mjpeg",
            stream_url="http://host.docker.internal:8765/stream/0.mjpg",
        )
    ]

    class _Settings:
        host_camera_agent_url = "http://host.docker.internal:8765"
        host_camera_agent_timeout_seconds = 1.0

    monkeypatch.setattr(discovery, "_local_cameras", lambda probe: [])
    monkeypatch.setattr(
        discovery, "_host_agent_cameras", lambda base_url, timeout, probe=False: list(agent)
    )
    monkeypatch.setattr(discovery, "get_settings", lambda: _Settings())

    result = discovery.discover_cameras(include_host_agent=True)

    assert [(device.source, device.camera_type) for device in result.devices] == [
        ("host_agent", "rtsp")
    ]
    assert result.devices[0].stream_url == "http://host.docker.internal:8765/stream/0.mjpg"
