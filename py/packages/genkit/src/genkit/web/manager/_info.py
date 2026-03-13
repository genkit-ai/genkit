# Copyright 2025 Google LLC
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

"""Utility functions to get server information."""

import functools
import importlib.metadata
import os
import platform
import socket
import sys
import time
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

import pathlib

from ._server import ServerConfig

# TODO(#4354): OpenTelemetry integration


def get_health_info(config: ServerConfig) -> dict[str, Any]:
    """Get health information.

    Returns:
        A dictionary containing health information.
    """
    uptime_seconds = round(time.time() - config.start_time, 2) if config.start_time else None

    d = {
        'status': 'ok',
        'timestamp': time.time(),
        'uptime_seconds': uptime_seconds,
    }
    return d


@functools.lru_cache(maxsize=16)
def _get_system_info() -> dict[str, Any]:
    """Get system information (memoized).

    Returns:
        A dictionary containing system information.
    """
    return {
        'cpu_count': os.cpu_count(),
        'hostname': platform.node(),
        'os': platform.system(),
        'platform': platform.platform(),
        'python_implementation': platform.python_implementation(),
        'python_version': platform.python_version(),
    }


@functools.lru_cache(maxsize=16)
def _get_process_info() -> dict[str, Any]:
    """Get process information (memoized).

    Returns:
        A dictionary containing process information.
    """
    info = {
        'cwd': pathlib.Path.cwd(),
        'pid': os.getpid(),
        'argv': sys.argv,
        'executable': sys.executable,
        'start_time': time.time(),
    }

    if psutil is not None:
        try:
            process = psutil.Process()
            info.update({
                'memory_rss': process.memory_info().rss,  # in bytes
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'threads_count': process.num_threads(),
                'open_files_count': len(process.open_files()),
                'create_time': process.create_time(),
                'username': process.username(),
                'command_line': process.cmdline(),
            })
        except (psutil.Error, AttributeError):
            # Handle potential psutil errors gracefully
            pass

    return info


@functools.lru_cache(maxsize=16)
def _get_memory_info() -> dict[str, Any]:
    """Get memory information (memoized).

    Returns:
        A dictionary containing memory information.
    """
    if psutil is None:
        return {'available': False}

    try:
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()

        return {
            'available': True,
            'virtual': {
                'total': virtual_memory.total,
                'available': virtual_memory.available,
                'percent': virtual_memory.percent,
                'used': virtual_memory.used,
                'free': virtual_memory.free,
            },
            'swap': {
                'total': swap_memory.total,
                'used': swap_memory.used,
                'free': swap_memory.free,
                'percent': swap_memory.percent,
            },
        }
    except (psutil.Error, AttributeError):
        return {'available': False}


@functools.lru_cache(maxsize=16)
def _get_disk_info() -> dict[str, Any]:
    """Get disk information (memoized).

    Returns:
        A dictionary containing disk information.
    """
    if psutil is None:
        return {'available': False}

    try:
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters(perdisk=False)

        info = {
            'available': True,
            'usage': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent,
            },
        }

        if disk_io:
            info['io'] = {
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
            }

        return info
    except (psutil.Error, AttributeError):
        return {'available': False}


@functools.lru_cache(maxsize=16)
def _get_network_info() -> dict[str, Any]:
    """Get network information (memoized).

    Returns:
        A dictionary containing network information.
    """
    info: dict[str, Any] = {
        'hostname': socket.gethostname(),
    }

    # Try to get local IP address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # This doesn't actually establish a connection
            s.connect(('8.8.8.8', 80))
            ip_address = s.getsockname()[0]
            info['ip_address'] = ip_address
    except (OSError, IndexError):
        pass

    if psutil is not None:
        try:
            net_io = psutil.net_io_counters()
            info['stats'] = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout,
            }

            # Get network interfaces
            interfaces = {}
            for name, addrs in psutil.net_if_addrs().items():
                addresses = []
                for addr in addrs:
                    addr_info = {
                        'family': str(addr.family),
                        'address': addr.address,
                    }
                    if addr.netmask:
                        addr_info['netmask'] = addr.netmask
                    if hasattr(addr, 'broadcast') and addr.broadcast:
                        addr_info['broadcast'] = addr.broadcast
                    addresses.append(addr_info)

                if addresses:
                    interfaces[name] = addresses

            if interfaces:
                info['interfaces'] = interfaces
        except (psutil.Error, AttributeError):
            pass

    return info


@functools.lru_cache(maxsize=16)
def _get_deps_info() -> dict[str, Any]:
    """Get Python dependencies information (memoized).

    Returns:
        A dictionary containing Python dependencies information.
    """
    try:
        # Get all installed packages
        packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}

        # Get actively used packages (those that are imported)
        modules = set(sys.modules.keys())
        active_packages = {}
        for pkg_name, version in packages.items():
            # Check if this package name (or package_name.something) is in sys.modules
            if pkg_name in modules or any(m.startswith(f'{pkg_name}.') for m in modules):
                active_packages[pkg_name] = version

        # Filter web-related packages
        web_related_keywords = [
            'http',
            'web',
            'api',
            'server',
            'rest',
            'json',
            'async',
            'wsgi',
            'asgi',
        ]
        web_packages = {}
        for pkg_name, version in packages.items():
            pkg_lower = pkg_name.lower()
            if any(keyword in pkg_lower for keyword in web_related_keywords):
                web_packages[pkg_name] = version

        # Get dependencies for this specific module
        module_deps = {}
        try:
            current_module = __name__.split('.')[0]  # Get the top-level package name
            if current_module in packages:
                module_deps[current_module] = packages[current_module]

                # Try to find dependencies of this module if possible
                try:
                    # Try multiple approaches to find dependencies
                    for pkg_name in packages:
                        pkg_meta = importlib.metadata.metadata(pkg_name)
                        requires = pkg_meta.get_all('Requires-Dist') or []
                        for req in requires:
                            if current_module in req.lower():
                                module_deps[pkg_name] = packages[pkg_name]
                except Exception:  # noqa: S110 - best-effort dependency discovery
                    pass
        except Exception:  # noqa: S110 - best-effort module info gathering
            pass

        return {
            'available': True,
            'active': active_packages,
            'web_related': web_packages,
            'module_deps': {k: v.strip() for k, v in module_deps.items()},
            'total_count': len(packages),
            # Uncomment if you want all packages (can be large)
            # 'all': packages,
        }
    except Exception:
        return {'available': False}


@functools.lru_cache(maxsize=16)
def _get_env_info(env_prefix: str | None = None) -> dict[str, str]:
    """Get environment variables (memoized).

    Args:
        env_prefix: Optional prefix to filter environment variables.
            If provided, only variables starting with this prefix will be included.

    Returns:
        A dictionary containing environment variables.
    """
    env_vars = {}
    # List of sensitive environment variable name patterns to exclude
    sensitive_patterns = [
        'TOKEN',
        'SECRET',
        'PASSWORD',
        'KEY',
        'AUTH',
        'CREDENTIAL',
        'CERT',
        'PWD',
        'PASS',
        'API_KEY',
        'APIKEY',
        'ACCESS_KEY',
        'PRIVATE_KEY',
    ]
    for key, value in os.environ.items():
        # Skip if we're filtering by prefix and this doesn't match
        if env_prefix and not key.startswith(env_prefix):
            continue
        # Skip sensitive environment variables
        if any(pattern in key.upper() for pattern in sensitive_patterns):
            continue
        env_vars[key] = value
    return env_vars


def get_server_info(config: ServerConfig) -> dict[str, Any]:
    """Get server information.

    Args:
        config: The server configuration.

    Returns:
        A dictionary containing server information.
    """
    has_prefix = hasattr(config, 'env_prefix')
    env_prefix = config.env_prefix if has_prefix else None  # type: ignore[attr-defined]

    # Get feature flags if available
    feature_flags: dict[str, Any] = getattr(config, 'feature_flags', {})

    return {
        'process': _get_process_info(),
        'service': {
            'name': config.name,
            'port': config.port,
            'version': config.version,
        },
        'system': _get_system_info(),
        'environment': _get_env_info(env_prefix=env_prefix),
        'memory': _get_memory_info(),
        'disk': _get_disk_info(),
        'network': _get_network_info(),
        'dependencies': _get_deps_info(),
        'runtime': {
            'python_path': sys.executable,
            'python_version': sys.version,
            'sys_path': sys.path,
        },
        'feature_flags': feature_flags,
    }
