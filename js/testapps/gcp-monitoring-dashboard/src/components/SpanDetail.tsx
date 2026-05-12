import { useState } from 'react';
import {
  CheckCircle2,
  Clock,
  Copy,
  Check,
  XCircle,
  AlertCircle,
  Tag,
  Cpu,
  FileText,
} from 'lucide-react';
import type { SpanData } from '../types';

function StatusBadge({ status }: { status: string }) {
  if (status === 'success')
    return (
      <span className="metric-badge bg-accent-green/20 text-accent-green">
        <CheckCircle2 className="h-3 w-3" /> Success
      </span>
    );
  if (status === 'error')
    return (
      <span className="metric-badge bg-accent-red/20 text-accent-red">
        <XCircle className="h-3 w-3" /> Error
      </span>
    );
  return (
    <span className="metric-badge bg-text-tertiary/20 text-text-secondary">
      <AlertCircle className="h-3 w-3" /> Unknown
    </span>
  );
}

function TypeBadge({ subtype }: { subtype: string }) {
  if (!subtype) return null;
  return (
    <span className={`type-badge type-badge-${subtype}`}>{subtype}</span>
  );
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms === 0) return '<1ms';
  return `${Math.round(ms)}ms`;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="rounded p-1 text-text-tertiary hover:bg-surface-3 hover:text-text-secondary"
      title="Copy to clipboard"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-accent-green" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

/** Simple JSON syntax highlighter using CSS classes */
function JsonDisplay({ value }: { value: string }) {
  if (!value || value === '<redacted>') {
    return (
      <div className="rounded bg-surface-2 px-3 py-2 text-xs italic text-text-tertiary">
        {value || '(empty)'}
      </div>
    );
  }

  // Try to parse and pretty-print JSON
  let formatted = value;
  try {
    const parsed = JSON.parse(value);
    formatted = JSON.stringify(parsed, null, 2);
  } catch {
    // Not JSON, use raw value
  }

  // Simple syntax highlighting
  const highlighted = formatted
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // strings
    .replace(
      /"([^"\\]*(\\.[^"\\]*)*)"/g,
      (match, content) => {
        // Check if it's a key (followed by colon)
        return `<span class="json-string">${match}</span>`;
      }
    )
    // numbers
    .replace(
      /\b(-?\d+\.?\d*([eE][+-]?\d+)?)\b/g,
      '<span class="json-number">$1</span>'
    )
    // booleans and null
    .replace(
      /\b(true|false|null)\b/g,
      '<span class="json-keyword">$1</span>'
    );

  return (
    <pre
      className="overflow-x-auto rounded bg-surface-2 px-3 py-2 font-mono text-xs leading-relaxed text-text-secondary"
      dangerouslySetInnerHTML={{ __html: highlighted }}
    />
  );
}

/** Filter labels to show only interesting genkit-specific ones */
function getDisplayLabels(labels: Record<string, string>): [string, string][] {
  const skipPrefixes = ['g.co/', 'genkit/metadata/context'];
  const genkitPrefix = 'genkit/';

  return Object.entries(labels)
    .filter(([key]) => {
      if (skipPrefixes.some((p) => key.startsWith(p))) return false;
      return true;
    })
    .map(([key, value]) => {
      // Strip genkit/ prefix for cleaner display
      const displayKey = key.startsWith(genkitPrefix)
        ? key.slice(genkitPrefix.length)
        : key;
      return [displayKey, value] as [string, string];
    })
    .sort(([a], [b]) => a.localeCompare(b));
}

export function SpanDetail({
  span,
  traceId,
}: {
  span: SpanData;
  traceId: string;
}) {
  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* Header */}
      <div className="border-b border-border/60 px-4 py-3.5">
        <div className="mb-2 flex items-center gap-2">
          <h3 className="truncate text-base font-semibold text-text-primary">
            {span.name}
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={span.status} />
          <TypeBadge subtype={span.subtype} />
          {span.modelName && (
            <span className="metric-badge bg-accent-orange/20 text-accent-orange">
              <Cpu className="h-3 w-3" /> {span.modelName}
            </span>
          )}
        </div>
      </div>

      {/* Metadata */}
      <div className="border-b border-border/60 px-4 py-3.5">
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-text-tertiary">Trace ID</span>
            <div className="mt-0.5 flex items-center gap-1">
              <code className="truncate font-mono text-text-secondary">
                {traceId}
              </code>
              <CopyButton text={traceId} />
            </div>
          </div>
          <div>
            <span className="text-text-tertiary">Span ID</span>
            <div className="mt-0.5 flex items-center gap-1">
              <code className="truncate font-mono text-text-secondary">
                {span.spanId}
              </code>
              <CopyButton text={span.spanId} />
            </div>
          </div>
          <div>
            <span className="text-text-tertiary">Duration</span>
            <div className="mt-0.5 flex items-center gap-1 text-text-primary">
              <Clock className="h-3 w-3 text-text-tertiary" />
              {formatDuration(span.durationMs)}
            </div>
          </div>
          <div>
            <span className="text-text-tertiary">Start Time</span>
            <div className="mt-0.5 text-text-secondary">
              {formatTime(span.startTime)}
            </div>
          </div>
          {span.path && (
            <div className="col-span-2">
              <span className="text-text-tertiary">Path</span>
              <div className="mt-0.5">
                <code className="break-all font-mono text-[11px] text-text-secondary">
                  {span.path}
                </code>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="border-b border-border/60 px-4 py-3.5">
        <div className="mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
            <FileText className="h-3 w-3" /> Input
          </span>
          {span.input && span.input !== '<redacted>' && (
            <CopyButton text={span.input} />
          )}
        </div>
        <JsonDisplay value={span.input} />
      </div>

      {/* Output */}
      <div className="border-b border-border/60 px-4 py-3.5">
        <div className="mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
            <FileText className="h-3 w-3" /> Output
          </span>
          {span.output && span.output !== '<redacted>' && (
            <CopyButton text={span.output} />
          )}
        </div>
        <JsonDisplay value={span.output} />
      </div>

      {/* Labels */}
      {Object.keys(span.labels).length > 0 && (
        <div className="px-4 py-3.5">
          <span className="mb-2 flex items-center gap-1.5 text-xs font-medium text-text-secondary">
            <Tag className="h-3 w-3" /> Labels
          </span>
          <div className="mt-2 space-y-1">
            {getDisplayLabels(span.labels).map(([key, value]) => (
              <div
                key={key}
                className="flex items-baseline gap-2 text-xs"
              >
                <span className="flex-shrink-0 font-mono text-accent-purple">
                  {key}
                </span>
                <span className="truncate text-text-secondary">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
