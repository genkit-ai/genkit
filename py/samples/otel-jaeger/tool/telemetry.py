# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S310,S404,S603,S607

"""Docker-free local telemetry stack for the OTel sample.

Downloads the Jaeger and otelcol-contrib release binaries into a local
cache (skipping the download if they already exist), writes a collector
config, then spawns both processes:

  app --OTLP:4318--> otelcol-contrib --OTLP:14317--> jaeger (UI :16686)
                            \\--debug--> collector.log (metrics + logs)

Env overrides for locked-down networks:
  JAEGER_BIN / OTEL_COLLECTOR_BIN         use an existing binary, skip download
  JAEGER_VERSION / OTEL_COLLECTOR_VERSION pin a release tag instead of latest
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

JAEGER_UI_PORT = 16686
COLLECTOR_OTLP_GRPC_PORT = 4317
COLLECTOR_OTLP_HTTP_PORT = 4318
JAEGER_OTLP_GRPC_PORT = 14317
JAEGER_OTLP_HTTP_PORT = 14318

USER_AGENT = 'genkit-py-telemetry'


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    otel_dir = project_root / '.otel'
    bin_dir = otel_dir / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)

    plat, arch = _platform_arch()
    print(f'Platform: {plat}/{arch}')

    otelcol_path = _ensure_binary(
        executable_name='otelcol-contrib',
        repo='open-telemetry/opentelemetry-collector-releases',
        binary_name_in_archive='otelcol-contrib',
        bin_env_var='OTEL_COLLECTOR_BIN',
        version_env_var='OTEL_COLLECTOR_VERSION',
        is_jaeger=False,
        plat=plat,
        arch=arch,
        bin_dir=bin_dir,
    )
    jaeger_path = _ensure_binary(
        executable_name='jaeger',
        repo='jaegertracing/jaeger',
        binary_name_in_archive='jaeger',
        bin_env_var='JAEGER_BIN',
        version_env_var='JAEGER_VERSION',
        is_jaeger=True,
        plat=plat,
        arch=arch,
        bin_dir=bin_dir,
    )

    _pkill(otelcol_path)
    _pkill(jaeger_path)

    config_file = otel_dir / 'collector.yaml'
    config_file.write_text(_collector_config(), encoding='utf-8')
    print(f'Wrote collector config: {config_file}')

    jaeger_log = otel_dir / 'jaeger.log'
    collector_log = otel_dir / 'collector.log'
    processes: list[subprocess.Popen[bytes]] = []

    def shutdown(*, code: int = 0) -> None:
        print('\nShutting down...')
        for proc in processes:
            proc.send_signal(signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise SystemExit(code)

    signal.signal(signal.SIGINT, lambda *_: shutdown())
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda *_: shutdown())

    print(f'Starting Jaeger... logs: {jaeger_log}')
    jaeger = _spawn_logged(
        jaeger_path,
        [
            f'--set=receivers.otlp.protocols.grpc.endpoint=127.0.0.1:{JAEGER_OTLP_GRPC_PORT}',
            f'--set=receivers.otlp.protocols.http.endpoint=127.0.0.1:{JAEGER_OTLP_HTTP_PORT}',
        ],
        jaeger_log,
    )
    processes.append(jaeger)
    if not _wait_until_ready(jaeger, JAEGER_UI_PORT, 'Jaeger', jaeger_log):
        shutdown(code=1)
    print('Jaeger is up.')

    print(f'Starting otelcol-contrib... logs: {collector_log}')
    collector = _spawn_logged(otelcol_path, ['--config', str(config_file)], collector_log)
    processes.append(collector)
    if not _wait_until_ready(collector, COLLECTOR_OTLP_HTTP_PORT, 'Collector', collector_log):
        shutdown(code=1)
    print('Collector is up.')

    print(
        f"""
Local telemetry environment is running.

  Jaeger UI:  http://localhost:{JAEGER_UI_PORT}
  OTLP in:    http://localhost:{COLLECTOR_OTLP_HTTP_PORT} (http)  |  localhost:{COLLECTOR_OTLP_GRPC_PORT} (grpc)
  Metrics:    tail -f {collector_log}

Run the sample in another terminal:
  export GEMINI_API_KEY=...
  python src/main.py

Press Ctrl+C to stop."""
    )

    while True:
        for proc in processes:
            if proc.poll() is not None:
                shutdown()
        time.sleep(0.5)


def _platform_arch() -> tuple[str, str]:
    system = platform.system().lower()
    plat = {'darwin': 'darwin', 'windows': 'windows'}.get(system, 'linux')
    machine = platform.machine().lower()
    if machine in {'x86_64', 'x64', 'amd64'}:
        arch = 'amd64'
    elif machine in {'arm64', 'aarch64'}:
        arch = 'arm64'
    else:
        arch = machine
    return plat, arch


def _collector_config() -> str:
    return f"""
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:{COLLECTOR_OTLP_GRPC_PORT}"
      http:
        endpoint: "0.0.0.0:{COLLECTOR_OTLP_HTTP_PORT}"
processors:
  batch:
    timeout: 1s
exporters:
  otlp:
    endpoint: "127.0.0.1:{JAEGER_OTLP_GRPC_PORT}"
    tls:
      insecure: true
  debug:
    verbosity: detailed
service:
  telemetry:
    logs:
      level: "info"
    metrics:
      level: "none"
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
"""


def _ensure_binary(
    *,
    executable_name: str,
    repo: str,
    binary_name_in_archive: str,
    bin_env_var: str,
    version_env_var: str,
    is_jaeger: bool,
    plat: str,
    arch: str,
    bin_dir: Path,
) -> str:
    override = os.environ.get(bin_env_var)
    if override:
        print(f'Using {executable_name} from {bin_env_var}={override}')
        return override

    target = bin_dir / executable_name
    if target.exists():
        print(f'{executable_name} already cached: {target}')
        return str(target)

    print(f'{executable_name} not found; resolving release from {repo}...')
    ext = 'zip' if plat == 'windows' else 'tar.gz'
    pinned = os.environ.get(version_env_var)
    download_url, asset_name = (
        _resolve_jaeger_asset(repo, plat, arch, pinned)
        if is_jaeger
        else _resolve_otelcol_asset(repo, plat, arch, ext, pinned)
    )

    with tempfile.TemporaryDirectory(prefix='genkit-otel-') as tmp:
        archive_path = Path(tmp) / asset_name
        print(f'Downloading {asset_name}...')
        _download(download_url, archive_path)
        print('Extracting...')
        _extract(archive_path, Path(tmp))
        bin_name = f'{binary_name_in_archive}.exe' if plat == 'windows' else binary_name_in_archive
        found = _find_file(Path(tmp), bin_name)
        if found is None:
            raise RuntimeError(f'Binary "{bin_name}" not found in {asset_name}')
        shutil.copy2(found, target)
        target.chmod(target.stat().st_mode | 0o111)
    print(f'Installed {executable_name}: {target}')
    return str(target)


def _resolve_otelcol_asset(repo: str, plat: str, arch: str, ext: str, pinned: str | None) -> tuple[str, str]:
    release = (
        _get_json(f'https://api.github.com/repos/{repo}/releases/tags/{pinned}')
        if pinned
        else _get_json(f'https://api.github.com/repos/{repo}/releases/latest')
    )
    tag = str(release['tag_name'])
    version = tag[1:] if tag.startswith('v') else tag
    asset_name = f'otelcol-contrib_{version}_{plat}_{arch}.{ext}'
    for asset in release['assets']:
        if asset['name'] == asset_name:
            return str(asset['browser_download_url']), asset_name
    raise RuntimeError(f'Asset not found in {tag}: {asset_name}')


def _resolve_jaeger_asset(repo: str, plat: str, arch: str, pinned: str | None) -> tuple[str, str]:
    suffix = f'-{plat}-{arch}.zip' if plat == 'windows' else f'-{plat}-{arch}.tar.gz'
    if pinned:
        releases = [_get_json(f'https://api.github.com/repos/{repo}/releases/tags/{pinned}')]
    else:
        releases = [
            r
            for r in _get_json_list(f'https://api.github.com/repos/{repo}/releases')
            if not r.get('prerelease') and str(r.get('tag_name', '')).startswith('v')
        ]
        releases.sort(key=lambda r: _semver_tuple(str(r['tag_name'])), reverse=True)
    for release in releases:
        for asset in release.get('assets', []):
            name = str(asset['name'])
            if name.startswith('jaeger-2.') and name.endswith(suffix):
                return str(asset['browser_download_url']), name
    raise RuntimeError(f'No Jaeger v2 asset for {plat}/{arch}')


def _semver_tuple(tag: str) -> tuple[int, ...]:
    parts = tag.lstrip('v').split('.')
    return tuple(int(p) if p.isdigit() else 0 for p in parts)


def _get_json(url: str) -> dict[str, Any]:
    data = json.loads(_http_text(url))
    if not isinstance(data, dict):
        raise RuntimeError(f'Expected object from {url}')
    return data


def _get_json_list(url: str) -> list[dict[str, Any]]:
    data = json.loads(_http_text(url))
    if not isinstance(data, list):
        raise RuntimeError(f'Expected list from {url}')
    return [item for item in data if isinstance(item, dict)]


def _http_text(url: str) -> str:
    req = Request(url, headers={'User-Agent': USER_AGENT})
    with urlopen(req) as resp:  # noqa: S310
        return resp.read().decode('utf-8')


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={'User-Agent': USER_AGENT})
    with urlopen(req) as resp, dest.open('wb') as out:  # noqa: S310
        shutil.copyfileobj(resp, out)


def _extract(archive_path: Path, dest_dir: Path) -> None:
    if archive_path.suffix == '.zip' or str(archive_path).endswith('.zip'):
        subprocess.run(['tar', '-xf', str(archive_path), '-C', str(dest_dir)], check=True)
    else:
        subprocess.run(['tar', '-xzf', str(archive_path), '-C', str(dest_dir)], check=True)


def _find_file(directory: Path, name: str) -> Path | None:
    for path in directory.rglob(name):
        if path.is_file():
            return path
    return None


def _pkill(pattern: str) -> None:
    try:
        subprocess.run(['pkill', '-f', pattern], check=False)
    except FileNotFoundError:
        return


def _spawn_logged(executable: str, args: list[str], log_file: Path) -> subprocess.Popen[bytes]:
    log = log_file.open('ab')
    return subprocess.Popen([executable, *args], stdout=log, stderr=log)


def _wait_until_ready(proc: subprocess.Popen[bytes], port: int, name: str, log_file: Path) -> bool:
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            _dump_log(name, port, log_file)
            return False
        if _port_open(port):
            return True
        time.sleep(0.5)
    _dump_log(name, port, log_file)
    return False


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(('localhost', port), timeout=1):
            return True
    except OSError:
        return False


def _dump_log(name: str, port: int, log_file: Path) -> None:
    print(f'{name} failed to start (port {port}).', file=sys.stderr)
    if log_file.exists():
        text = log_file.read_text(encoding='utf-8', errors='replace')
        print(text[-2000:], file=sys.stderr)


if __name__ == '__main__':
    main()
