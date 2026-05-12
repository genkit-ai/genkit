import { useState } from 'react';
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  XCircle,
  AlertCircle,
} from 'lucide-react';
import type { SpanData } from '../types';

function StatusIcon({ status }: { status: string }) {
  if (status === 'success')
    return <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0 text-accent-green" />;
  if (status === 'error')
    return <XCircle className="h-3.5 w-3.5 flex-shrink-0 text-accent-red" />;
  return <AlertCircle className="h-3.5 w-3.5 flex-shrink-0 text-text-tertiary" />;
}

function TypeBadge({ subtype }: { subtype: string }) {
  if (!subtype) return null;
  const cls = `type-badge type-badge-${subtype}`;
  return <span className={cls}>{subtype}</span>;
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms === 0) return '<1ms';
  return `${Math.round(ms)}ms`;
}

/** Compute the width percentage of this span relative to the root duration. */
function barStyle(
  span: SpanData,
  rootStart: number,
  rootDuration: number
): { left: string; width: string } {
  if (rootDuration === 0)
    return { left: '0%', width: '100%' };

  const spanStart = new Date(span.startTime).getTime();
  const left = ((spanStart - rootStart) / rootDuration) * 100;
  const width = Math.max((span.durationMs / rootDuration) * 100, 0.5); // min 0.5% for visibility

  return {
    left: `${Math.max(0, left).toFixed(1)}%`,
    width: `${Math.min(width, 100 - Math.max(0, left)).toFixed(1)}%`,
  };
}

function SpanNode({
  span,
  depth,
  selectedSpanId,
  onSelect,
  rootStart,
  rootDuration,
}: {
  span: SpanData;
  depth: number;
  selectedSpanId: string | null;
  onSelect: (span: SpanData) => void;
  rootStart: number;
  rootDuration: number;
}) {
  const [expanded, setExpanded] = useState(depth < 3); // auto-expand first 3 levels
  const hasChildren = span.children && span.children.length > 0;
  const isSelected = span.spanId === selectedSpanId;
  const bar = barStyle(span, rootStart, rootDuration);

  const barColor =
    span.status === 'error'
      ? 'bg-accent-red/60'
      : span.subtype === 'model'
        ? 'bg-accent-orange/60'
        : span.subtype === 'flow'
          ? 'bg-accent-blue/60'
          : span.subtype === 'tool'
            ? 'bg-accent-green/60'
            : 'bg-accent-purple/40';

  return (
    <div>
      <div
        className={`group flex cursor-pointer items-center border-l-2 py-1 transition-colors ${
          isSelected
            ? 'border-accent-blue bg-accent-blue/8'
            : 'border-transparent hover:bg-surface-2/60'
        }`}
        style={{ paddingLeft: `${depth * 20 + 10}px`, paddingRight: '10px' }}
        onClick={() => onSelect(span)}
      >
        {/* Expand/collapse toggle */}
        <button
          className="mr-1 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded text-text-tertiary hover:text-text-secondary"
          onClick={(e) => {
            e.stopPropagation();
            if (hasChildren) setExpanded(!expanded);
          }}
        >
          {hasChildren ? (
            expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )
          ) : (
            <span className="h-3.5 w-3.5" /> // spacer
          )}
        </button>

        {/* Status icon */}
        <StatusIcon status={span.status} />

        {/* Span name */}
        <span className="ml-2 mr-2 truncate text-sm text-text-primary">
          {span.name}
        </span>

        {/* Type badge */}
        <TypeBadge subtype={span.subtype} />

        {/* Spacer */}
        <div className="flex-1" />

        {/* Timing bar + duration */}
        <div className="mr-2 hidden w-32 md:block">
          <div className="relative h-3 w-full rounded-sm bg-surface-3">
            <div
              className={`absolute top-0 h-full rounded-sm ${barColor}`}
              style={{ left: bar.left, width: bar.width }}
            />
          </div>
        </div>

        <span className="w-16 flex-shrink-0 text-right text-xs text-text-secondary">
          {formatDuration(span.durationMs)}
        </span>
      </div>

      {/* Children */}
      {hasChildren && expanded && (
        <div>
          {span.children.map((child) => (
            <SpanNode
              key={child.spanId}
              span={child}
              depth={depth + 1}
              selectedSpanId={selectedSpanId}
              onSelect={onSelect}
              rootStart={rootStart}
              rootDuration={rootDuration}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function SpanTree({
  rootSpan,
  selectedSpanId,
  onSelectSpan,
}: {
  rootSpan: SpanData;
  selectedSpanId: string | null;
  onSelectSpan: (span: SpanData) => void;
}) {
  const rootStart = new Date(rootSpan.startTime).getTime();
  const rootDuration = rootSpan.durationMs;

  return (
    <div className="divide-y divide-border/30">
      <SpanNode
        span={rootSpan}
        depth={0}
        selectedSpanId={selectedSpanId}
        onSelect={onSelectSpan}
        rootStart={rootStart}
        rootDuration={rootDuration}
      />
    </div>
  );
}
