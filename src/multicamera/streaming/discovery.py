"""Network discovery for MJPEG streaming services.

Two complementary discovery strategies:

1. **mDNS / Zeroconf** (primary) — listens for services advertising standard
   types like ``_http._tcp``, ``_mjpeg._tcp``, or ``_rtsp._tcp`` on the local
   network.  This is passive, fast, and battery-friendly.

2. **Subnet HTTP probe** (fallback) — actively scans common MJPEG ports on the
   local /24 subnet.  Useful when cameras don't advertise via mDNS.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import aiohttp
from PySide6.QtCore import QThread, Signal
from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

MDNS_SERVICE_TYPES = [
    "_http._tcp.local.",
    "_mjpeg._tcp.local.",
    "_rtsp._tcp.local.",
]

COMMON_MJPEG_PORTS = [8080, 8081, 8082, 8083, 80, 81, 554, 9000, 9001]
MJPEG_PATHS = ["/video", "/stream", "/mjpeg", "/cam", "/", "/video_feed"]
PROBE_TIMEOUT = 1.5  # seconds per connection attempt


# ── Data model ───────────────────────────────────────────────────

@dataclass
class DiscoveredService:
    """A discovered MJPEG streaming endpoint."""

    host: str
    port: int
    path: str = "/"
    name: str = ""
    url: str = ""
    source: str = ""  # "mdns" | "probe"
    stream_type: str = "unknown"  # "rgb" | "ir" | "control" | "unknown"
    eye: str = "unknown"  # "left" | "right" | "unknown"

    def __post_init__(self):
        if not self.url:
            self.url = f"http://{self.host}:{self.port}{self.path}"
        if not self.name:
            self.name = f"{self.host}:{self.port}"
        guessed_type, guessed_eye = infer_stream_role(
            name=self.name,
            url=self.url,
            path=self.path,
            port=self.port,
        )
        if self.stream_type == "unknown":
            self.stream_type = guessed_type
        if self.eye == "unknown":
            self.eye = guessed_eye

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_camera_stream(self) -> bool:
        return self.stream_type in {"rgb", "ir"} and self.eye in {"left", "right"}


# ── mDNS / Zeroconf listener ────────────────────────────────────

class MdnsListener:
    """Collects mDNS service announcements via ``zeroconf``."""

    def __init__(self):
        self._services: Dict[str, DiscoveredService] = {}
        self._zc: Zeroconf | None = None
        self._browsers: list[ServiceBrowser] = []
        self._on_change: Callable[[], None] | None = None

    @property
    def services(self) -> List[DiscoveredService]:
        return list(self._services.values())

    def start(self, on_change: Callable[[], None] | None = None):
        self._on_change = on_change
        self._zc = Zeroconf()
        for stype in MDNS_SERVICE_TYPES:
            browser = ServiceBrowser(self._zc, stype, handlers=[self._handler])
            self._browsers.append(browser)
        logger.info("mDNS listener started for %s", MDNS_SERVICE_TYPES)

    def stop(self):
        if self._zc is not None:
            self._zc.close()
            self._zc = None
        self._browsers.clear()
        logger.info("mDNS listener stopped")

    def _handler(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ):
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info is None:
                return
            self._on_service_found(info, name)
        elif state_change == ServiceStateChange.Removed:
            key_prefix = name.split(".")[0]
            removed = [k for k in self._services if k.startswith(key_prefix)]
            for k in removed:
                del self._services[k]
            if removed and self._on_change:
                self._on_change()

    def _on_service_found(self, info: ServiceInfo, full_name: str):
        addresses = info.parsed_scoped_addresses()
        server = _normalise_mdns_host(info.server)
        if not addresses and not server:
            return

        # Prefer the advertised .local hostname for stable UI/config URLs.
        # Fall back to IP address when the service does not publish a server name.
        host = server or addresses[0]
        port = info.port
        props = _decode_properties(info.properties)
        path = props.get("path", "/")
        friendly = _service_instance_name(full_name)
        stream_type, eye = infer_stream_role(
            name=full_name,
            path=path,
            port=port,
            properties=props,
        )

        svc = DiscoveredService(
            host=host,
            port=port,
            path=path,
            name=friendly,
            source="mdns",
            stream_type=stream_type,
            eye=eye,
        )
        self._services[svc.key] = svc
        logger.info("mDNS discovered: %s -> %s", friendly, svc.url)
        if self._on_change:
            self._on_change()


# ── Subnet HTTP probe (fallback) ────────────────────────────────

def _normalise_mdns_host(server: str | None) -> str:
    if not server:
        return ""
    return server.rstrip(".")


def _service_instance_name(full_name: str) -> str:
    return full_name.split("._", 1)[0].rstrip(".") or full_name.rstrip(".")


def _decode_properties(properties: dict[bytes, bytes | None]) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for key, value in properties.items():
        key_s = key.decode("utf-8", errors="replace").lower()
        value_s = "" if value is None else value.decode("utf-8", errors="replace")
        decoded[key_s] = value_s
    return decoded

def infer_stream_role(
    name: str = "",
    url: str = "",
    path: str = "",
    port: int | None = None,
    properties: Optional[dict[str, str]] = None,
) -> tuple[str, str]:
    """Infer camera modality and eye from service metadata.

    Devices should ideally publish TXT keys like ``type=rgb`` and ``eye=left``.
    For compatibility, this also recognises common words in service names/paths.
    """
    props = properties or {}
    prop_type = _normalise_token(
        props.get("type")
        or props.get("stream")
        or props.get("modality")
        or props.get("camera_type")
    )
    prop_eye = _normalise_token(props.get("eye") or props.get("side") or props.get("role"))

    haystack = " ".join(
        [
            name,
            url,
            path,
            str(port or ""),
            " ".join(f"{k}={v}" for k, v in props.items()),
        ]
    ).lower()

    stream_type = "unknown"
    if prop_type in {"rgb", "color", "colour", "彩色", "可见光"}:
        stream_type = "rgb"
    elif prop_type in {"ir", "infrared", "mono", "红外", "ir_camera"}:
        stream_type = "ir"
    elif prop_type in {"console", "control", "web", "ui"}:
        stream_type = "control"
    elif any(word in haystack for word in ("console", "control", "admin", "webui")):
        stream_type = "control"
    elif any(word in haystack for word in ("rgb", "color", "colour", "彩色", "可见光")):
        stream_type = "rgb"
    elif any(word in haystack for word in ("ir", "infrared", "gray", "grey", "mono", "红外")):
        stream_type = "ir"

    eye = "unknown"
    if prop_eye in {"left", "l"}:
        eye = "left"
    elif prop_eye in {"right", "r"}:
        eye = "right"
    elif _contains_eye_word(haystack, "left") or "_l" in haystack or "-l" in haystack:
        eye = "left"
    elif _contains_eye_word(haystack, "right") or "_r" in haystack or "-r" in haystack:
        eye = "right"

    return stream_type, eye


def _normalise_token(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def _contains_eye_word(text: str, word: str) -> bool:
    aliases = {"left": ("left", "左", "左目"), "right": ("right", "右", "右目")}
    return any(alias in text for alias in aliases[word])

def _get_local_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _get_subnet_hosts(local_ip: str, mask_bits: int = 24) -> List[str]:
    parts = local_ip.split(".")
    if len(parts) != 4:
        return []
    base = struct.unpack("!I", socket.inet_aton(local_ip))[0]
    mask = (0xFFFFFFFF << (32 - mask_bits)) & 0xFFFFFFFF
    network = base & mask
    hosts = []
    count = (1 << (32 - mask_bits)) - 2
    for i in range(1, min(count + 1, 255)):
        ip = socket.inet_ntoa(struct.pack("!I", network + i))
        if ip != local_ip:
            hosts.append(ip)
    return hosts


async def _probe_mjpeg(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    path: str,
) -> Optional[DiscoveredService]:
    url = f"http://{host}:{port}{path}"
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT)
        ) as resp:
            ct = resp.headers.get("Content-Type", "").lower()
            if resp.status == 200 and (
                "multipart/x-mixed-replace" in ct
                or "image/jpeg" in ct
                or "video" in ct
            ):
                return DiscoveredService(
                    host=host, port=port, path=path, url=url, source="probe"
                )
    except Exception:
        pass
    return None


async def _probe_host(
    session: aiohttp.ClientSession,
    host: str,
    ports: List[int],
    paths: List[str],
) -> List[DiscoveredService]:
    results: List[DiscoveredService] = []
    found_ports: Set[int] = set()
    tasks = []
    for port in ports:
        for path in paths:
            tasks.append(_probe_mjpeg(session, host, port, path))
    done = await asyncio.gather(*tasks, return_exceptions=True)
    for r in done:
        if isinstance(r, DiscoveredService) and r.port not in found_ports:
            found_ports.add(r.port)
            results.append(r)
    return results


async def scan_subnet(
    progress_callback=None,
    ports: Optional[List[int]] = None,
    paths: Optional[List[str]] = None,
    mask_bits: int = 24,
    max_concurrent: int = 50,
) -> List[DiscoveredService]:
    """Scan the local /24 subnet for MJPEG services (active probe)."""
    local_ip = _get_local_ip()
    if not local_ip:
        logger.warning("Cannot determine local IP")
        return []

    if ports is None:
        ports = COMMON_MJPEG_PORTS
    if paths is None:
        paths = MJPEG_PATHS

    hosts = _get_subnet_hosts(local_ip, mask_bits)
    logger.info("Subnet probe: %d hosts on %s/%d", len(hosts), local_ip, mask_bits)

    all_results: List[DiscoveredService] = []
    sem = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _scan_one(host: str, idx: int):
            async with sem:
                results = await _probe_host(session, host, ports, paths)
                all_results.extend(results)
                if progress_callback:
                    progress_callback(idx + 1, len(hosts))

        tasks = [_scan_one(h, i) for i, h in enumerate(hosts)]
        await asyncio.gather(*tasks)

    logger.info("Subnet probe complete: found %d services", len(all_results))
    return all_results


# ── Qt worker threads ────────────────────────────────────────────

class DiscoveryWorker(QThread):
    """Runs mDNS listening + optional subnet probe in a background thread.

    Emits *service_found* in real-time as mDNS services appear, and
    *scan_progress* / *scan_finished* during the subnet probe phase.
    """

    service_found = Signal(object)       # DiscoveredService
    service_removed = Signal(str)        # service key
    scan_progress = Signal(int, int)     # current, total
    scan_finished = Signal(list)         # List[DiscoveredService]
    # legacy compat
    progress = Signal(int, int)
    finished = Signal(list)

    def __init__(self, enable_mdns: bool = True, enable_probe: bool = True, parent=None):
        super().__init__(parent)
        self._enable_mdns = enable_mdns
        self._enable_probe = enable_probe
        self._mdns = MdnsListener()
        self._all: Dict[str, DiscoveredService] = {}
        self._running = True

    def run(self):
        # Phase 1: mDNS (runs for a few seconds to collect broadcasts)
        if self._enable_mdns:
            self._mdns.start(on_change=self._on_mdns_change)
            # Give mDNS 3 seconds to collect responses
            for _ in range(30):
                if not self._running:
                    break
                self.msleep(100)
            for svc in self._mdns.services:
                self._all[svc.key] = svc

        # Phase 2: subnet probe
        if self._enable_probe and self._running:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                probe_results = loop.run_until_complete(
                    scan_subnet(
                        progress_callback=self._on_probe_progress,
                    )
                )
                for svc in probe_results:
                    if svc.key not in self._all:
                        self._all[svc.key] = svc
            except Exception as e:
                logger.error("Subnet probe failed: %s", e)
            finally:
                loop.close()

        # keep mDNS listener alive until explicitly stopped
        results = list(self._all.values())
        self.scan_finished.emit(results)
        self.finished.emit(results)

    def stop(self):
        self._running = False
        self._mdns.stop()
        self.wait(3000)

    def _on_mdns_change(self):
        for svc in self._mdns.services:
            if svc.key not in self._all:
                self._all[svc.key] = svc
                self.service_found.emit(svc)

    def _on_probe_progress(self, current: int, total: int):
        self.scan_progress.emit(current, total)
        self.progress.emit(current, total)


class MdnsWatcher(QThread):
    """Long-running mDNS watcher — keeps listening after initial discovery.

    Use this for continuous monitoring of service availability.
    """

    service_added = Signal(object)    # DiscoveredService
    service_removed = Signal(str)     # service key
    services_changed = Signal(list)   # full current list

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mdns = MdnsListener()
        self._running = True
        self._prev_keys: Set[str] = set()

    def run(self):
        self._mdns.start(on_change=self._check_diff)
        while self._running:
            self.msleep(500)
        self._mdns.stop()

    def stop(self):
        self._running = False
        self.wait(3000)

    def _check_diff(self):
        current = {s.key: s for s in self._mdns.services}
        current_keys = set(current.keys())

        for key in current_keys - self._prev_keys:
            self.service_added.emit(current[key])
        for key in self._prev_keys - current_keys:
            self.service_removed.emit(key)

        self._prev_keys = current_keys
        self.services_changed.emit(list(current.values()))
