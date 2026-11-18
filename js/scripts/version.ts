#!/usr/bin/env npx tsx
// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Automated version bumping for Genkit JS packages using conventional commits.
 *
 * Analyzes git history per package to determine the appropriate semver bump,
 * propagates bumps to dependents, generates changelogs, and creates git
 * commits/tags.
 *
 * Usage:
 *   npx tsx js/scripts/version.ts [options]
 *
 * Options:
 *   --rc <tag>     Create RC release with given pre-release tag (e.g. beta, rc)
 *   --graduate     Graduate RC versions to stable releases
 *   --dry-run      Preview changes without applying them
 *   --no-commit    Skip creating a git commit
 *   --no-tags      Skip creating git tags
 *   -h, --help     Print usage information
 */

import { execSync } from 'child_process';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'fs';
import { join, relative, resolve } from 'path';
import { parseArgs } from 'util';

// ---------------------------------------------------------------------------
// Types & enums
// ---------------------------------------------------------------------------

enum BumpType {
  none = 0,
  patch = 1,
  minor = 2,
  major = 3,
}

interface SemVer {
  major: number;
  minor: number;
  patch: number;
  prerelease: string | null;
}

interface ConventionalCommit {
  type: string;
  scope: string | null;
  isBreaking: boolean;
  message: string;
}

interface Package {
  name: string;
  path: string; // absolute path to the package directory
  relPath: string; // path relative to repo root (e.g. "js/core")
  version: SemVer;
  private: boolean;
  dependencies: string[];
  devDependencies: string[];
  peerDependencies: string[];
  optionalDependencies: string[];
}

// ---------------------------------------------------------------------------
// SemVer helpers (no external deps)
// ---------------------------------------------------------------------------

function parseSemVer(version: string): SemVer {
  const match = version.match(/^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$/);
  if (!match) throw new Error(`Invalid semver: ${version}`);
  return {
    major: parseInt(match[1]),
    minor: parseInt(match[2]),
    patch: parseInt(match[3]),
    prerelease: match[4] ?? null,
  };
}

function formatSemVer(v: SemVer): string {
  const base = `${v.major}.${v.minor}.${v.patch}`;
  return v.prerelease ? `${base}-${v.prerelease}` : base;
}

function isPreRelease(v: SemVer): boolean {
  return v.prerelease !== null;
}

function bumpSemVer(v: SemVer, bump: BumpType): SemVer {
  if (bump === BumpType.none) return v;
  if (v.major === 0) {
    // 0.x.y: major → minor, minor/patch → patch
    if (bump === BumpType.major)
      return { major: 0, minor: v.minor + 1, patch: 0, prerelease: null };
    return { major: 0, minor: v.minor, patch: v.patch + 1, prerelease: null };
  }
  switch (bump) {
    case BumpType.major:
      return { major: v.major + 1, minor: 0, patch: 0, prerelease: null };
    case BumpType.minor:
      return { major: v.major, minor: v.minor + 1, patch: 0, prerelease: null };
    case BumpType.patch:
      return {
        major: v.major,
        minor: v.minor,
        patch: v.patch + 1,
        prerelease: null,
      };
    default:
      return v;
  }
}

// ---------------------------------------------------------------------------
// Conventional commit parsing
// ---------------------------------------------------------------------------

function parseConventionalCommit(message: string): ConventionalCommit {
  const regex = /^(\w+)(?:\((.*?)\))?(!?):\s*(.*)/;
  const match = regex.exec(message);
  if (match) {
    return {
      type: match[1],
      scope: match[2] ?? null,
      isBreaking: match[3] === '!' || message.includes('BREAKING CHANGE'),
      message: match[4],
    };
  }
  return {
    type: 'chore',
    scope: null,
    isBreaking: message.includes('BREAKING CHANGE'),
    message,
  };
}

function commitBumpType(commit: ConventionalCommit): BumpType {
  if (commit.isBreaking) return BumpType.major;
  if (commit.type === 'feat') return BumpType.minor;
  if (
    commit.type === 'fix' ||
    commit.type === 'perf' ||
    commit.type === 'refactor'
  )
    return BumpType.patch;
  return BumpType.none;
}

// ---------------------------------------------------------------------------
// Git helpers
// ---------------------------------------------------------------------------

function exec(cmd: string, options?: { cwd?: string }): string {
  try {
    return execSync(cmd, {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
      ...options,
    }).trim();
  } catch {
    return '';
  }
}

function tagExists(tagName: string): boolean {
  const result = exec(`git tag -l "${tagName}"`);
  return result === tagName;
}

function getLatestTag(packageName: string): string | null {
  // Try the standard tag format: <name>@<version>
  const result = exec(
    `git describe --tags --match "${packageName}@*" --abbrev=0`
  );
  return result || null;
}

function getCommitsSince(tag: string | null, path: string): string[] {
  const range = tag ? `${tag}..HEAD` : 'HEAD';
  const result = exec(`git log ${range} --format=%s -- "${path}"`);
  if (!result) return [];
  return result.split('\n');
}

// ---------------------------------------------------------------------------
// Workspace discovery
// ---------------------------------------------------------------------------

function discoverPackages(jsRoot: string): Package[] {
  const repoRoot = resolve(jsRoot, '..');
  const packages: Package[] = [];

  // Use pnpm to list workspace packages
  const pnpmOutput = exec('pnpm ls -r --depth -1 --json', { cwd: jsRoot });
  if (!pnpmOutput) {
    // Fallback: manually find package.json files in known locations
    return discoverPackagesManual(jsRoot, repoRoot);
  }

  try {
    const parsed = JSON.parse(pnpmOutput);
    for (const entry of parsed) {
      const pkgPath = entry.path;
      if (!pkgPath) continue;
      const pkgJsonPath = join(pkgPath, 'package.json');
      if (!existsSync(pkgJsonPath)) continue;
      const pkg = loadPackage(pkgJsonPath, repoRoot);
      if (pkg) packages.push(pkg);
    }
  } catch {
    return discoverPackagesManual(jsRoot, repoRoot);
  }

  return packages;
}

function discoverPackagesManual(jsRoot: string, repoRoot: string): Package[] {
  const packages: Package[] = [];
  const dirs = [
    join(jsRoot, 'core'),
    join(jsRoot, 'ai'),
    join(jsRoot, 'genkit'),
  ];

  // Add plugins
  const pluginsDir = join(jsRoot, 'plugins');
  if (existsSync(pluginsDir)) {
    const pluginEntries = readdirSync(pluginsDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => join(pluginsDir, entry.name));
    dirs.push(...pluginEntries);
  }

  for (const dir of dirs) {
    const pkgJsonPath = join(dir, 'package.json');
    if (!existsSync(pkgJsonPath)) continue;
    const pkg = loadPackage(pkgJsonPath, repoRoot);
    if (pkg) packages.push(pkg);
  }

  return packages;
}

function loadPackage(pkgJsonPath: string, repoRoot: string): Package | null {
  const raw = JSON.parse(readFileSync(pkgJsonPath, 'utf-8'));
  const pkgDir = resolve(pkgJsonPath, '..');
  const relPath = relative(repoRoot, pkgDir).replace(/\\/g, '/');

  return {
    name: raw.name,
    path: pkgDir,
    relPath,
    version: parseSemVer(raw.version ?? '0.0.0'),
    private: raw.private === true,
    dependencies: Object.keys(raw.dependencies ?? {}),
    devDependencies: Object.keys(raw.devDependencies ?? {}),
    peerDependencies: Object.keys(raw.peerDependencies ?? {}),
    optionalDependencies: Object.keys(raw.optionalDependencies ?? {}),
  };
}

function allDeps(pkg: Package): string[] {
  return [
    ...pkg.dependencies,
    ...pkg.devDependencies,
    ...pkg.peerDependencies,
    ...pkg.optionalDependencies,
  ];
}

/** Returns true if the package is publishable (not private, not a testapp/doc-snippet). */
function isPublishable(pkg: Package): boolean {
  if (pkg.private) return false;
  // Only include packages from js/core, js/ai, js/genkit, js/plugins/*
  // Exclude testapps, doc-snippets, third_party, etc.
  if (
    pkg.relPath.includes('testapps/') ||
    pkg.relPath.includes('doc-snippets') ||
    pkg.relPath.includes('third_party/')
  ) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Version planning (mirrors Dart VersionPlanner)
// ---------------------------------------------------------------------------

function planBumps(
  packages: Package[],
  rcTag: string | null,
  graduate: boolean
): Map<string, SemVer> {
  const publishable = packages.filter(isPublishable);
  const proposedBumps = new Map<string, SemVer>();

  // Phase 1: Determine direct bumps from commit history
  for (const pkg of publishable) {
    const currentTagName = `${pkg.name}@${formatSemVer(pkg.version)}`;
    const currentTagFound = tagExists(currentTagName);
    const latestTag = getLatestTag(pkg.name);
    const commits = getCommitsSince(latestTag, pkg.path);

    // Skip if current version is already tagged and no new commits
    if (currentTagFound && commits.length === 0) continue;
    if (commits.length === 0 && !graduate && !rcTag) continue;

    let maxBump = BumpType.none;
    for (const msg of commits) {
      const commit = parseConventionalCommit(msg);
      const bump = commitBumpType(commit);
      if (bump > maxBump) maxBump = bump;
    }

    if (maxBump === BumpType.none && !graduate && !rcTag) continue;

    const current = pkg.version;
    let next: SemVer;

    if (graduate) {
      // Graduate: strip prerelease
      if (!isPreRelease(current)) continue;
      next = {
        major: current.major,
        minor: current.minor,
        patch: current.patch,
        prerelease: null,
      };
    } else if (rcTag) {
      // RC release
      if (
        isPreRelease(current) &&
        current.prerelease?.startsWith(rcTag + '.')
      ) {
        // Already on this RC tag — bump the RC number
        const parts = current.prerelease.split('.');
        const rcNum = parts.length > 1 ? parseInt(parts[1]) || 0 : 0;
        next = {
          major: current.major,
          minor: current.minor,
          patch: current.patch,
          prerelease: `${rcTag}.${rcNum + 1}`,
        };
      } else {
        // New RC: bump base version first, then add RC suffix
        const base = bumpSemVer(
          current,
          maxBump === BumpType.none ? BumpType.patch : maxBump
        );
        next = { ...base, prerelease: `${rcTag}.1` };
      }
    } else {
      // Normal release
      if (maxBump === BumpType.none) continue;
      next = bumpSemVer(current, maxBump);
    }

    proposedBumps.set(pkg.name, next);
  }

  // Phase 2: Propagate bumps to downstream dependents
  let changed = true;
  while (changed) {
    changed = false;
    for (const pkg of publishable) {
      if (proposedBumps.has(pkg.name)) continue;

      const deps = allDeps(pkg);
      const hasBumpedDep = deps.some((d) => proposedBumps.has(d));
      if (!hasBumpedDep) continue;

      let next: SemVer;
      if (graduate && isPreRelease(pkg.version)) {
        next = {
          major: pkg.version.major,
          minor: pkg.version.minor,
          patch: pkg.version.patch,
          prerelease: null,
        };
      } else {
        next = bumpSemVer(pkg.version, BumpType.patch);
        if (rcTag) {
          if (
            isPreRelease(pkg.version) &&
            pkg.version.prerelease?.startsWith(rcTag + '.')
          ) {
            const parts = pkg.version.prerelease.split('.');
            const rcNum = parts.length > 1 ? parseInt(parts[1]) || 0 : 0;
            next = {
              major: pkg.version.major,
              minor: pkg.version.minor,
              patch: pkg.version.patch,
              prerelease: `${rcTag}.${rcNum + 1}`,
            };
          } else {
            next = { ...next, prerelease: `${rcTag}.1` };
          }
        }
      }

      proposedBumps.set(pkg.name, next);
      changed = true;
    }
  }

  return proposedBumps;
}

// ---------------------------------------------------------------------------
// Changelog generation
// ---------------------------------------------------------------------------

function filterReverts(messages: string[]): string[] {
  const toSkip = new Array(messages.length).fill(false);

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.startsWith('Revert "') && msg.endsWith('"')) {
      const original = msg.slice(8, -1);
      for (let j = 0; j < messages.length; j++) {
        if (!toSkip[j] && messages[j] === original) {
          toSkip[i] = true;
          toSkip[j] = true;
          break;
        }
      }
    }
  }

  return messages.filter((_, i) => !toSkip[i]);
}

function buildChangelogEntry(
  version: SemVer,
  commitMessages: string[]
): string {
  const filtered = filterReverts(commitMessages);
  const versionStr = formatSemVer(version);

  if (filtered.length === 0) {
    return `## ${versionStr}\n\n - Updated internal dependencies.\n`;
  }

  const breaking: string[] = [];
  const feats: string[] = [];
  const fixes: string[] = [];
  const others: string[] = [];

  for (const msg of filtered) {
    const commit = parseConventionalCommit(msg);
    if (commit.type === 'chore' && !commit.isBreaking) continue;

    if (commit.isBreaking) breaking.push(commit.message);
    else if (commit.type === 'feat') feats.push(commit.message);
    else if (commit.type === 'fix') fixes.push(commit.message);
    else others.push(commit.message);
  }

  const lines: string[] = [`## ${versionStr}\n`];

  if (breaking.length) {
    lines.push('### Breaking Changes\n');
    for (const m of breaking) lines.push(` - ${m}`);
    lines.push('');
  }
  if (feats.length) {
    lines.push('### Features\n');
    for (const m of feats) lines.push(` - ${m}`);
    lines.push('');
  }
  if (fixes.length) {
    lines.push('### Fixes\n');
    for (const m of fixes) lines.push(` - ${m}`);
    lines.push('');
  }
  if (others.length) {
    lines.push('### Other Changes\n');
    for (const m of others) lines.push(` - ${m}`);
    lines.push('');
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Version applying
// ---------------------------------------------------------------------------

function applyBumps(
  packages: Package[],
  bumps: Map<string, SemVer>
): Set<string> {
  const modified = new Set<string>();

  for (const [pkgName, newVersion] of bumps) {
    const pkg = packages.find((p) => p.name === pkgName);
    if (!pkg) continue;

    const versionStr = formatSemVer(newVersion);
    console.log(`Bumping ${pkgName} to ${versionStr}...`);

    // Use npm version to update package.json (no git tag/commit)
    execSync(`npm version "${versionStr}" --no-git-tag-version`, {
      cwd: pkg.path,
      stdio: 'pipe',
    });
    modified.add(pkgName);

    // Generate changelog
    const latestTag = getLatestTag(pkg.name);
    const commits = getCommitsSince(latestTag, pkg.path);
    const entry = buildChangelogEntry(newVersion, commits);

    const changelogPath = join(pkg.path, 'CHANGELOG.md');
    if (existsSync(changelogPath)) {
      const existing = readFileSync(changelogPath, 'utf-8');
      if (existing.includes(`## ${versionStr}`)) {
        console.log(
          `  Changelog for ${versionStr} already exists in ${pkgName}. Skipping.`
        );
      } else {
        writeFileSync(changelogPath, `${entry}\n${existing}`);
      }
    } else {
      writeFileSync(changelogPath, entry);
    }
  }

  return modified;
}

// ---------------------------------------------------------------------------
// Git commit & tag
// ---------------------------------------------------------------------------

function createCommitAndTags(
  packages: Package[],
  bumps: Map<string, SemVer>,
  modified: Set<string>,
  doCommit: boolean,
  doTags: boolean
): void {
  if (!doCommit) {
    console.log('\nSkipping git commit due to --no-commit flag.');
    return;
  }

  console.log('\nStaging files...');
  for (const pkgName of modified) {
    const pkg = packages.find((p) => p.name === pkgName);
    if (!pkg) continue;
    exec(`git add "${join(pkg.path, 'package.json')}"`);
    if (bumps.has(pkgName)) {
      exec(`git add "${join(pkg.path, 'CHANGELOG.md')}"`);
    }
  }

  console.log('Creating git commit...');
  const commitResult = exec('git commit -m "chore(release): publish packages"');
  if (!commitResult) {
    console.log('Warning: Failed to create git commit (nothing to commit?).');
    return;
  }

  if (!doTags) {
    console.log('\nSkipping git tags due to --no-tags flag.');
    return;
  }

  console.log('\nCreating git tags...');
  for (const [pkgName, newVersion] of bumps) {
    const versionStr = formatSemVer(newVersion);
    const tag = `${pkgName}@${versionStr}`;

    if (tagExists(tag)) {
      console.log(`  Tag ${tag} already exists. Skipping.`);
      continue;
    }

    exec(`git tag -a "${tag}" -m "Release ${tag}"`);
    console.log(`  Created annotated tag ${tag}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function printUsage() {
  console.log(
    `
Usage: npx tsx js/scripts/version.ts [options]

Automatically bump JS package versions based on conventional commits.

Options:
  --rc <tag>     Create an RC release with the given pre-release tag (e.g. beta, rc)
  --graduate     Graduate RC versions to stable releases
  --dry-run      Preview changes without applying them
  --no-commit    Skip creating a git commit (default: commit)
  --no-tags      Skip creating git tags (default: tags)
  -h, --help     Print this usage information
`.trim()
  );
}

function main() {
  const { values } = parseArgs({
    options: {
      rc: { type: 'string' },
      graduate: { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
      commit: { type: 'boolean', default: true },
      tags: { type: 'boolean', default: true },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: false,
  });

  if (values.help) {
    printUsage();
    process.exit(0);
  }

  const rcTag = values.rc ?? null;
  const graduate = values.graduate!;
  const dryRun = values['dry-run']!;
  const doCommit = values.commit!;
  const doTags = values.tags!;

  // Find js root (script is at js/scripts/version.ts)
  const jsRoot = resolve(__dirname);
  // Actually, __dirname would be js/scripts. Go up one level.
  const jsDir = resolve(jsRoot, '..');

  // If we're being called from repo root, detect js/ directory
  const resolvedJsDir = existsSync(join(jsDir, 'pnpm-workspace.yaml'))
    ? jsDir
    : existsSync(join(process.cwd(), 'js', 'pnpm-workspace.yaml'))
      ? join(process.cwd(), 'js')
      : (() => {
          console.error(
            'Could not find js/pnpm-workspace.yaml. Run from the repo root or js/ directory.'
          );
          process.exit(1);
        })();

  console.log(`Discovering packages from ${resolvedJsDir}...`);
  const packages = discoverPackages(resolvedJsDir);
  const publishable = packages.filter(isPublishable);
  console.log(
    `Found ${packages.length} packages (${publishable.length} publishable).`
  );

  // Plan bumps
  const bumps = planBumps(packages, rcTag, graduate);

  if (bumps.size === 0) {
    console.log('No changes found. Nothing to bump.');
    return;
  }

  // Print proposed bumps
  console.log('\nProposed bumps:');
  for (const [name, newVersion] of bumps) {
    const pkg = packages.find((p) => p.name === name)!;
    console.log(
      `  ${name}: ${formatSemVer(pkg.version)} → ${formatSemVer(newVersion)}`
    );
  }

  if (dryRun) {
    console.log('\n--- Changelog Previews ---');
    for (const [pkgName, newVersion] of bumps) {
      const pkg = packages.find((p) => p.name === pkgName)!;
      const latestTag = getLatestTag(pkg.name);
      const commits = getCommitsSince(latestTag, pkg.path);
      const entry = buildChangelogEntry(newVersion, commits);
      console.log(`\nPackage: ${pkgName}`);
      console.log(entry.trimEnd());
      console.log('--------------------------');
    }
    console.log('\nDry run complete. No files were changed.');
    return;
  }

  // Apply bumps
  const modified = applyBumps(packages, bumps);

  // Commit and tag
  createCommitAndTags(packages, bumps, modified, doCommit, doTags);

  console.log('\nDone!');
}

main();
