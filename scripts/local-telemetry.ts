/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Docker-free local telemetry stack for the OpenTelemetry samples (any
// language: JS, Go, Python, Dart). Downloads the Jaeger and otelcol-contrib
// release binaries into a local cache (skipping the download if they already
// exist), writes a collector config, then spawns both processes:
//
//   app --OTLP:4317--> otelcol-contrib --OTLP:14317--> jaeger (UI :16686)
//                            \--debug--> collector.log (metrics + logs)
//
// Jaeger v2's binary is itself an OTel collector, so it ingests OTLP directly;
// the separate otelcol-contrib lets us also debug-log metrics/logs.
//
// Run from the repo root:
//   npx tsx scripts/local-telemetry.ts
//
// Env overrides for locked-down networks:
//   JAEGER_BIN / OTEL_COLLECTOR_BIN         use an existing binary, skip download
//   JAEGER_VERSION / OTEL_COLLECTOR_VERSION pin a release tag instead of latest
//   OTEL_CACHE_DIR                          where to cache binaries + config
//                                           (default: <repo>/.otel)

import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { Socket } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const JAEGER_UI_PORT = 16686;

// The collector owns the app-facing OTLP ports.
const COLLECTOR_OTLP_GRPC_PORT = 4317;
const COLLECTOR_OTLP_HTTP_PORT = 4318;

// Jaeger v2 is itself an OTel collector whose OTLP receiver binds both gRPC and
// HTTP by default, so both must be moved off 4317/4318 to avoid colliding with
// the collector above.
const JAEGER_OTLP_GRPC_PORT = 14317;
const JAEGER_OTLP_HTTP_PORT = 14318;

const cacheDir = process.env.OTEL_CACHE_DIR
  ? process.env.OTEL_CACHE_DIR
  : join(process.cwd(), '.otel');
const binDir = join(cacheDir, 'bin');

interface PlatformArch {
  platform: string;
  arch: string;
}

async function main(): Promise<void> {
  mkdirSync(binDir, { recursive: true });

  const { platform, arch } = platformArch();
  console.log(`Platform: ${platform}/${arch}`);

  const otelcolPath = await ensureBinary({
    executableName: 'otelcol-contrib',
    repo: 'open-telemetry/opentelemetry-collector-releases',
    binaryNameInArchive: 'otelcol-contrib',
    binEnvVar: 'OTEL_COLLECTOR_BIN',
    versionEnvVar: 'OTEL_COLLECTOR_VERSION',
    isJaeger: false,
    platform,
    arch,
  });

  const jaegerPath = await ensureBinary({
    executableName: 'jaeger',
    repo: 'jaegertracing/jaeger',
    binaryNameInArchive: 'jaeger',
    binEnvVar: 'JAEGER_BIN',
    versionEnvVar: 'JAEGER_VERSION',
    isJaeger: true,
    platform,
    arch,
  });

  // Clean up any stale processes from a previous run. Match on the cached
  // binary paths so we only touch instances this script started.
  pkill(otelcolPath);
  pkill(jaegerPath);

  const configPath = join(cacheDir, 'collector.yaml');
  writeFileSync(configPath, collectorConfig());
  console.log(`Wrote collector config: ${configPath}`);

  const jaegerLog = join(cacheDir, 'jaeger.log');
  const collectorLog = join(cacheDir, 'collector.log');

  const processes: ChildProcess[] = [];
  let shuttingDown = false;
  const shutdown = (code = 0): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log('\nShutting down...');
    for (const p of processes) p.kill('SIGTERM');
    process.exit(code);
  };
  process.on('SIGINT', () => shutdown());
  process.on('SIGTERM', () => shutdown());

  // Start Jaeger. Its OTLP receiver is moved to 14317/14318 so it does not
  // squat on the app-facing 4317/4318 the collector owns.
  console.log(`Starting jaeger... logs: ${jaegerLog}`);
  const jaeger = spawnLogged(
    jaegerPath,
    [
      '--set',
      `receivers.otlp.protocols.grpc.endpoint=0.0.0.0:${JAEGER_OTLP_GRPC_PORT}`,
      '--set',
      `receivers.otlp.protocols.http.endpoint=0.0.0.0:${JAEGER_OTLP_HTTP_PORT}`,
    ],
    jaegerLog
  );
  processes.push(jaeger);
  if (!(await waitUntilReady(jaeger, JAEGER_UI_PORT, 'Jaeger', jaegerLog))) {
    shutdown(1);
    return;
  }
  console.log('Jaeger is up.');

  // Start the collector (receives from the app on 4317/4318).
  console.log(`Starting otelcol-contrib... logs: ${collectorLog}`);
  const collector = spawnLogged(
    otelcolPath,
    ['--config', configPath],
    collectorLog
  );
  processes.push(collector);
  if (
    !(await waitUntilReady(
      collector,
      COLLECTOR_OTLP_HTTP_PORT,
      'Collector',
      collectorLog
    ))
  ) {
    shutdown(1);
    return;
  }
  console.log('Collector is up.');

  console.log(`
Local telemetry environment is running.

  Jaeger UI:  http://localhost:${JAEGER_UI_PORT}
  OTLP in:    http://localhost:${COLLECTOR_OTLP_HTTP_PORT} (http)  |  localhost:${COLLECTOR_OTLP_GRPC_PORT} (grpc)
  Metrics:    tail -f ${collectorLog}

Run a sample in another terminal, pointing it at the collector, e.g.:
  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:${COLLECTOR_OTLP_HTTP_PORT}

Press Ctrl+C to stop.`);

  // Keep the process alive until a child exits or the user interrupts.
  await Promise.race(processes.map((p) => onExit(p)));
  shutdown();
}

function platformArch(): PlatformArch {
  const platform =
    process.platform === 'darwin'
      ? 'darwin'
      : process.platform === 'win32'
        ? 'windows'
        : 'linux';
  const arch =
    process.arch === 'x64'
      ? 'amd64'
      : process.arch === 'arm64'
        ? 'arm64'
        : process.arch;
  return { platform, arch };
}

function collectorConfig(): string {
  return `receivers:
  otlp:
    protocols:
      grpc:
        # 0.0.0.0 so both IPv4 and IPv6 loopback resolve (the app may dial
        # localhost as either).
        endpoint: "0.0.0.0:${COLLECTOR_OTLP_GRPC_PORT}"
      http:
        endpoint: "0.0.0.0:${COLLECTOR_OTLP_HTTP_PORT}"
processors:
  batch:
    timeout: 1s
exporters:
  otlp:
    endpoint: "127.0.0.1:${JAEGER_OTLP_GRPC_PORT}"
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
`;
}

interface EnsureBinaryArgs {
  executableName: string;
  repo: string;
  binaryNameInArchive: string;
  binEnvVar: string;
  versionEnvVar: string;
  isJaeger: boolean;
  platform: string;
  arch: string;
}

/**
 * Ensures a binary exists in `binDir`, downloading + extracting from the latest
 * (or `versionEnvVar`-pinned) GitHub release when missing.
 */
async function ensureBinary(args: EnsureBinaryArgs): Promise<string> {
  const override = process.env[args.binEnvVar];
  if (override) {
    console.log(
      `Using ${args.executableName} from ${args.binEnvVar}=${override}`
    );
    return override;
  }

  const target = join(binDir, args.executableName);
  if (existsSync(target)) {
    console.log(`${args.executableName} already cached: ${target}`);
    return target;
  }

  console.log(
    `${args.executableName} not found; resolving release from ${args.repo}...`
  );
  const ext = args.platform === 'windows' ? 'zip' : 'tar.gz';
  const pinned = process.env[args.versionEnvVar];

  const { downloadUrl, assetName } = args.isJaeger
    ? await resolveJaegerAsset(args.repo, args.platform, args.arch, pinned)
    : await resolveOtelcolAsset(
        args.repo,
        args.platform,
        args.arch,
        ext,
        pinned
      );

  const tmp = mkdtempSync(join(tmpdir(), 'genkit-otel-'));
  try {
    const archivePath = join(tmp, assetName);
    console.log(`Downloading ${assetName}...`);
    await download(downloadUrl, archivePath);

    console.log('Extracting...');
    extract(archivePath, tmp);

    const binName =
      args.platform === 'windows'
        ? `${args.binaryNameInArchive}.exe`
        : args.binaryNameInArchive;
    const found = findFile(tmp, binName);
    if (!found) {
      throw new Error(`Binary "${binName}" not found in ${assetName}`);
    }
    copyFileSync(found, target);
    if (args.platform !== 'windows') chmodSync(target, 0o755);
    console.log(`Installed ${args.executableName}: ${target}`);
    return target;
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

interface ResolvedAsset {
  downloadUrl: string;
  assetName: string;
}

/** otelcol assets are exact-named: `otelcol-contrib_<ver>_<plat>_<arch>.<ext>`. */
async function resolveOtelcolAsset(
  repo: string,
  platform: string,
  arch: string,
  ext: string,
  pinned?: string
): Promise<ResolvedAsset> {
  const release = await getJson(
    pinned
      ? `https://api.github.com/repos/${repo}/releases/tags/${pinned}`
      : `https://api.github.com/repos/${repo}/releases/latest`
  );
  const tag = release.tag_name as string;
  const version = tag.startsWith('v') ? tag.substring(1) : tag;
  const assetName = `otelcol-contrib_${version}_${platform}_${arch}.${ext}`;
  const assets = (release.assets ?? []) as Array<Record<string, unknown>>;
  const asset = assets.find((a) => a.name === assetName);
  if (!asset) {
    throw new Error(`No otelcol-contrib asset "${assetName}" in ${tag}`);
  }
  return {
    downloadUrl: asset.browser_download_url as string,
    assetName,
  };
}

/** Jaeger v2 assets are named `jaeger-2.<...>-<plat>-<arch>.<ext>`. */
async function resolveJaegerAsset(
  repo: string,
  platform: string,
  arch: string,
  pinned?: string
): Promise<ResolvedAsset> {
  const release = await getJson(
    pinned
      ? `https://api.github.com/repos/${repo}/releases/tags/${pinned}`
      : `https://api.github.com/repos/${repo}/releases/latest`
  );
  const suffix = `-${platform}-${arch}.${platform === 'windows' ? 'zip' : 'tar.gz'}`;
  const assets = (release.assets ?? []) as Array<Record<string, unknown>>;
  const asset = assets.find((a) => {
    const name = a.name as string;
    return name.startsWith('jaeger-2.') && name.endsWith(suffix);
  });
  if (!asset) {
    throw new Error(`No Jaeger v2 asset for ${platform}/${arch}`);
  }
  return {
    downloadUrl: asset.browser_download_url as string,
    assetName: asset.name as string,
  };
}

async function getJson(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'genkit-local-telemetry' },
  });
  if (!res.ok) {
    throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as Record<string, unknown>;
}

async function download(url: string, dest: string): Promise<void> {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'genkit-local-telemetry' },
  });
  if (!res.ok || !res.body) {
    throw new Error(`Download failed for ${url}: ${res.status}`);
  }
  const sink = createWriteStream(dest);
  await new Promise<void>((resolve, reject) => {
    sink.on('error', reject);
    sink.on('finish', resolve);
    // Node's fetch body is a web ReadableStream; pump it into the file sink.
    const reader = res.body!.getReader();
    const pump = (): void => {
      reader
        .read()
        .then(({ done, value }) => {
          if (done) {
            sink.end();
            return;
          }
          sink.write(Buffer.from(value));
          pump();
        })
        .catch(reject);
    };
    pump();
  });
}

function extract(archivePath: string, destDir: string): void {
  const result = archivePath.endsWith('.zip')
    ? spawnSync('unzip', ['-o', archivePath, '-d', destDir])
    : spawnSync('tar', ['-xzf', archivePath, '-C', destDir]);
  if (result.status !== 0) {
    throw new Error(`Extraction failed: ${result.stderr}`);
  }
}

function findFile(dir: string, name: string): string | undefined {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findFile(full, name);
      if (found) return found;
    } else if (entry.name === name) {
      return full;
    }
  }
  return undefined;
}

function pkill(pattern: string): void {
  try {
    spawnSync('pkill', ['-f', pattern]);
  } catch {
    // pkill may be missing or match nothing; ignore.
  }
}

function spawnLogged(
  executable: string,
  args: string[],
  logFile: string
): ChildProcess {
  const sink = createWriteStream(logFile, { flags: 'a' });
  const child = spawn(executable, args);
  child.stdout?.pipe(sink);
  child.stderr?.pipe(sink);
  return child;
}

function onExit(child: ChildProcess): Promise<number> {
  return new Promise((resolve) =>
    child.on('exit', (code) => resolve(code ?? 0))
  );
}

/**
 * Waits until `child` is serving on `port`, or fails fast if the process exits
 * first (e.g. a port-bind error). On failure, dumps the log tail.
 */
async function waitUntilReady(
  child: ChildProcess,
  port: number,
  name: string,
  logFile: string
): Promise<boolean> {
  // Race the port coming up against the process dying. A dead child resolves
  // first here, avoiding a false positive from another process on the port.
  const exited = onExit(child).then(() => 'exited' as const);
  const ready = waitForPort(port).then((ok) => (ok ? 'ready' : 'timeout'));
  const result = await Promise.race([exited, ready]);

  if (result === 'ready') return true;

  console.error(`${name} failed to start (port ${port}).`);
  if (existsSync(logFile)) {
    const log = readFileSync(logFile, 'utf-8');
    console.error(log.length > 2000 ? log.slice(-2000) : log);
  }
  return false;
}

async function waitForPort(
  port: number,
  timeoutSeconds = 30
): Promise<boolean> {
  const deadline = Date.now() + timeoutSeconds * 1000;
  while (Date.now() < deadline) {
    const ok = await new Promise<boolean>((resolve) => {
      const socket = new Socket();
      socket.setTimeout(1000);
      socket.once('connect', () => {
        socket.destroy();
        resolve(true);
      });
      socket.once('timeout', () => {
        socket.destroy();
        resolve(false);
      });
      socket.once('error', () => resolve(false));
      socket.connect(port, 'localhost');
    });
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
