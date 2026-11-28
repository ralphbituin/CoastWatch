# USB cameras

How CoastWatch discovers USB webcams and how to make them visible to the
backend when the backend runs in Docker.

## 1. What "detect cameras" returns

`GET /api/v1/cameras/devices` (also the **Detect cameras** button on the
Cameras page) lists, without registering anything:

| Field | Meaning |
|---|---|
| `name` | Device name from the OS/driver, e.g. `Lihappe8 Webcam L69SN 1080P`, `OBS Virtual Camera` |
| `source` | `local` (visible to the backend process) or `host_agent` (Windows bridge) |
| `camera_type` | `usb` for local device indices; `rtsp` for host-agent MJPEG streams |
| `device_index` | OpenCV/V4L2/DirectShow index for local devices; `null` for host-agent streams |
| `stream_url` | MJPEG URL for host-agent devices; `null` for local USB |
| `available` | Result of `probe=true`; false means the device could not be opened |
| `is_virtual` | Name looks like a virtual camera (OBS, loopback) |
| `width`/`height`/`fps` | Probed properties, when available |
| `backend` | `v4l2`, `directshow`, `avfoundation`, or `dshow->mjpeg` |
| `detail` | Explanation when a device is unavailable |

`probe=false` (the API default) is fast and does not open hardware.
`probe=true` (what the UI uses) opens each device briefly, so it is slower and
can fail while another application is using the camera.

## 2. The Windows/Docker constraint

Docker Desktop on Windows runs **Linux** containers. A Linux container cannot
see a Windows USB webcam: there is no `/dev/video0` to pass through. Docker
Desktop's `--device` flag does not bridge DirectShow devices.

CoastWatch therefore supports two working setups.

### A. Windows + Docker: host camera agent (recommended)

A small agent runs natively on Windows, enumerates DirectShow cameras, and
re-serves them as MJPEG. The backend (in Docker) reads them as network streams.

```powershell
# One command: start the agent, then the Docker stack. The agent is stopped
# when you Ctrl+C out of Compose.
powershell -ExecutionPolicy Bypass -File scripts\start-with-cameras.ps1
```

Or run the two pieces separately:

```powershell
# 1. Start the agent on the Windows host (leave it running)
powershell -ExecutionPolicy Bypass -File scripts\host-camera-agent.ps1

# 2. In another terminal, start the stack
docker compose up --build
```

The agent prints the cameras it found:

```text
CoastWatch host camera agent on http://0.0.0.0:8765
Detected cameras:
  [0] Lihappe8 Webcam L69SN 1080P
  [1] OBS Virtual Camera
```

`docker-compose.yml` already sets:

```yaml
HOST_CAMERA_AGENT_URL: http://host.docker.internal:8765
```

and adds `host.docker.internal:host-gateway` so the name resolves on Linux
Docker Engine too.

In the UI: **Cameras → Detect cameras**. Host-agent entries show
`host agent` and **Use stream**. Selecting one fills the stream URL with
`http://host.docker.internal:8765/stream/<index>.mjpg` and registers the camera
as a network (`rtsp`) camera. The backend fetches the stream; the browser never
connects to the agent directly.

Useful agent endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and device count |
| `GET /devices` | Names (add `?probe=1` for resolution/FPS/availability) |
| `GET /stream/<index>.mjpg` | MJPEG stream (`?width=1280&height=720&fps=15&quality=80`) |
| `GET /snapshot/<index>.jpg` | One JPEG, useful for a quick check |

Test it directly:

```powershell
curl http://localhost:8765/devices
```

**OBS Virtual Camera:** start OBS and click **Start Virtual Camera** first.
The agent can only enumerate it while it exists. If OBS is not running, it will
not appear.

### B. Linux host: pass V4L2 devices into the container

On a Linux host the camera is a normal `/dev/videoN` node:

```bash
ls -l /dev/video*
docker compose -f docker-compose.yml -f docker-compose.usb.yml up --build
```

Edit `docker-compose.usb.yml` to list the nodes you actually have. Compose
fails if a listed device is missing, so remove unused lines. The image runs as
root, so no extra group is normally needed; `group_add` is included for hosts
that run the container as a non-root user.

### C. Native backend on Windows (no Docker for the backend)

Running the backend directly on Windows gives it direct DirectShow access:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Keep PostgreSQL in Docker. The **Detect cameras** button then lists local
DirectShow devices with `source: local` and `camera_type: usb`; selecting one
fills the device index.

## 3. Registering a detected camera

1. Open **Cameras**.
2. Choose the monitoring station, code and name.
3. Click **Detect cameras**.
4. Click a device:
   - local USB → transport becomes `usb`, device index is filled;
   - host agent → transport becomes `rtsp` (network stream) and the MJPEG URL
     is filled.
5. Optionally set resolution/FPS, then **Register camera**.
6. Use **Test connection** to verify a frame can be read.

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Detect returns nothing, warning mentions `host camera agent unavailable` | Agent is not running, or port 8765 is blocked. Start the agent; check `curl http://localhost:8765/health` on the host. If Windows Firewall blocks Docker's network, allow the port for Private/Domain profiles (admin PowerShell): `New-NetFirewallRule -DisplayName "CoastWatch host camera agent" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow -Profile Private,Domain` |
| Host agent lists the camera but `probe` says unavailable | Another app owns it (OBS, Teams, browser, Zoom). Close it or stop the OBS virtual camera. |
| `OBS Virtual Camera` missing | OBS is not running or virtual camera is not started. |
| `device_index 0` fails but a name is listed | V4L2/DirectShow exposes multiple nodes; click the detected entry (the UI uses the probed index) or try the next index. |
| Camera works natively but not in Docker | Expected on Windows without the host agent. Use setup A or C. |
| Docker Desktop fails to start the backend after adding USB devices | You used `docker-compose.usb.yml` on Windows. Do not apply that file there; use the host agent. |
| Stream opens then stops | A second client tried to use the same physical camera. The agent serves one client per device at a time. |

## 5. Security notes

The host camera agent has **no authentication** and serves live video. Bind it
to `127.0.0.1` when the backend runs natively on the same machine. For Docker
Desktop it must listen on an address the container can reach (`0.0.0.0`);
restrict TCP 8765 with Windows Firewall to local/Docker networks and do not
expose it to the public internet. The backend's stream URL is redacted in API
responses like any other camera URL.
