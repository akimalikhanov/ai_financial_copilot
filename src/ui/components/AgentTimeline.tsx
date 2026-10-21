import React from 'react';
import { ChevronDown } from 'lucide-react';
import type { ActivityEvent, ActivityStatus, OutcomeDetail } from '../services/api';

// The backend words every row and the outcome; this file only folds events and renders them.

type RowStatus = ActivityStatus | 'running';

interface Child {
  id: string;
  kind: 'search' | 'note' | 'company';
  label: string;
  status: RowStatus;
  note?: string;
  // Shown on hover under the label: where a search's new excerpts came from, or a note's claim.
  detail?: string;
}

interface Row {
  id: string;
  label: string;
  runningLabel?: string;
  status: RowStatus;
  note?: string;
  hidden: boolean;
  startedAt: number; // seconds, as on the wire
  endedAt?: number;
  children: Child[];
}

export interface Timeline {
  stages: Row[];
  passes: number;
  outcome?: OutcomeDetail & { ts: number };
  done: boolean;
}

export const emptyTimeline = (): Timeline => ({ stages: [], passes: 0, done: false });

const str = (v: unknown): string | undefined => (typeof v === 'string' && v ? v : undefined);
const statusOf = (d: Record<string, unknown> | undefined): ActivityStatus =>
  (d?.status as ActivityStatus | undefined) ?? 'ok';

interface CompanyDetail { name: string; years: number[]; found: boolean }

const companyRow = (c: CompanyDetail): Child => ({
  id: c.name,
  kind: 'company',
  label: c.name,
  status: c.found ? 'ok' : 'warn',
  note: c.found ? c.years.join(', ') || undefined : 'not in your documents',
});

// Applies one activity event. Used live (SSE) and when replaying a persisted
// `trace.activity` log: same fold, two sources.
export function applyActivityEvent(t: Timeline, ev: ActivityEvent): Timeline {
  const d = ev.detail;
  const mapStage = (id: string, fn: (r: Row) => Row) =>
    ({ ...t, stages: t.stages.map(r => (r.id === id ? fn(r) : r)) });
  // Searches and notes belong to the stage that started last.
  const addChild = (child: Child) => {
    const last = t.stages[t.stages.length - 1];
    if (!last || last.children.some(c => c.id === child.id)) return t;
    return mapStage(last.id, r => ({ ...r, children: [...r.children, child] }));
  };
  switch (ev.kind) {
    case 'stage_started':
      if (t.stages.some(r => r.id === ev.id)) return t;
      return {
        ...t,
        stages: [...t.stages, {
          id: ev.id,
          label: ev.label ?? '',
          runningLabel: str(d?.running_label),
          status: 'running',
          hidden: d?.hidden === true,
          startedAt: ev.ts,
          children: [],
        }],
      };
    case 'stage_ended':
      return mapStage(ev.id, r => ({
        ...r,
        status: statusOf(d),
        note: str(d?.note),
        hidden: typeof d?.hidden === 'boolean' ? d.hidden : r.hidden,
        endedAt: ev.ts,
        children: Array.isArray(d?.companies)
          ? [...(d.companies as CompanyDetail[]).map(companyRow), ...r.children]
          : r.children,
      }));
    case 'tool_call_started':
      return addChild({ id: ev.id, kind: 'search', label: ev.label ?? '', status: 'running' });
    case 'note':
      return addChild({
        id: ev.id, kind: 'note', label: ev.label ?? '', status: statusOf(d), note: str(d?.note), detail: str(d?.claim),
      });
    case 'tool_call_ended':
      return {
        ...t,
        stages: t.stages.map(r => ({
          ...r,
          children: r.children.map(c =>
            c.id === ev.id ? { ...c, status: statusOf(d), note: str(d?.note), detail: str(d?.sources) } : c
          ),
        })),
      };
    case 'round_started':
      return { ...t, passes: t.passes + 1 };
    case 'outcome':
      return { ...t, outcome: { ...(d as unknown as OutcomeDetail), ts: ev.ts } };
    default:
      return t;
  }
}

// Stream ended or failed: rows still open did not finish, so they read as skipped with no
// duration instead of looking successful.
export function closeTimeline(t: Timeline): Timeline {
  const skip = <R extends { status: RowStatus }>(r: R): R => (r.status === 'running' ? { ...r, status: 'skipped' } : r);
  return { ...t, done: true, stages: t.stages.map(r => ({ ...skip(r), children: r.children.map(skip) })) };
}

// Rehydrates timelines for messages loaded from history, so they survive a reload.
export function timelinesFromMessages(
  messages: { id: string; trace?: { activity?: ActivityEvent[] } | null }[]
): Record<string, Timeline> {
  const result: Record<string, Timeline> = {};
  for (const m of messages) {
    if (m.trace?.activity?.length) {
      result[m.id] = closeTimeline(m.trace.activity.reduce(applyActivityEvent, emptyTimeline()));
    }
  }
  return result;
}

// `3.8s`, `41s`, `1m 12s`; nothing below 0.1s.
function fmtDuration(s: number): string | null {
  if (s < 0.1) return null;
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  const r = Math.round(s);
  return `${Math.floor(r / 60)}m ${r % 60}s`;
}


const STATUS_ICON: Record<RowStatus, { glyph: string; color: string; label: string }> = {
  running: { glyph: '◌', color: 'var(--accent)', label: 'running' },
  ok: { glyph: '✓', color: 'var(--success)', label: 'done' },
  skipped: { glyph: '–', color: 'var(--text-faint)', label: 'skipped' },
  warn: { glyph: '⚠', color: 'var(--warning)', label: 'warning' },
  fail: { glyph: '✕', color: 'var(--danger)', label: 'failed' },
};

function StatusIcon({ status }: { status: RowStatus }) {
  const s = STATUS_ICON[status];
  return (
    <span
      role="img"
      aria-label={s.label}
      className={`inline-block w-3.5 text-center flex-shrink-0 ${status === 'running' ? 'animate-pulse' : ''}`}
      style={{ color: s.color }}
    >
      {s.glyph}
    </span>
  );
}

function ActivityRow({ row, now }: { row: Row; now: number }) {
  const dur = row.endedAt !== undefined
    ? fmtDuration(row.endedAt - row.startedAt)
    : row.status === 'running' ? fmtDuration(now - row.startedAt) : null;
  return (
    <div className="tc-row-enter">
      <div className="flex items-baseline gap-2 py-0.5 text-xs">
        <StatusIcon status={row.status} />
        <span className="min-w-0 truncate text-[var(--text)]">
          {row.status === 'running' ? row.runningLabel ?? row.label : row.label}
          {row.note && <span className="text-[var(--text-faint)]"> · {row.note}</span>}
        </span>
        {dur && <span className="ml-auto pl-3 flex-shrink-0 text-[var(--text-faint)] tabular-nums">{dur}</span>}
      </div>
      {row.children.length > 0 && (
        <div className="ml-5 flex flex-col">
          {row.children.map(c => (
            <div
              key={c.id}
              className="tc-row-enter flex items-baseline gap-2 py-0.5 text-[11px]"
              title={c.detail ? `${c.label}\n${c.detail}` : c.label}
            >
              <StatusIcon status={c.status} />
              <span className={`min-w-0 truncate text-[var(--text-muted)] ${c.status === 'running' ? 'tc-shimmer' : ''}`}>
                {c.label}
              </span>
              {c.note && <span className="ml-auto pl-3 flex-shrink-0 text-[var(--text-faint)]">{c.note}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function OutcomeLines({ lines }: { lines: string[] }) {
  return (
    <div className="flex flex-col py-0.5 text-[11px] text-[var(--text-muted)]">
      {lines.map((line, i) => (
        <div key={i} className={`flex items-baseline gap-2 ${i > 0 ? 'ml-5' : ''}`}>
          {i === 0 && <span aria-hidden className="inline-block w-3.5 text-center flex-shrink-0">■</span>}
          <span>{line}</span>
        </div>
      ))}
    </div>
  );
}

export function AgentTimeline({ timeline }: { timeline: Timeline }) {
  const [expanded, setExpanded] = React.useState(false);
  const [now, setNow] = React.useState(() => Date.now() / 1000);
  const { stages, outcome, done } = timeline;

  React.useEffect(() => {
    if (done) return;
    const id = setInterval(() => setNow(Date.now() / 1000), 100);
    return () => clearInterval(id);
  }, [done]);

  if (outcome?.hidden) return null;

  const visible = stages.filter(r => !r.hidden);
  const first = stages[0]?.startedAt;
  // No outcome when done means a message from before outcomes existed.
  const end = outcome?.ts ?? (done ? Math.max(...stages.map(r => r.endedAt ?? r.startedAt)) : now);
  const total = first !== undefined ? fmtDuration(end - first) : null;

  let pillStatus: RowStatus;
  let pillText: string;
  if (outcome) {
    pillStatus = outcome.status;
    pillText = outcome.text;
  } else if (done) {
    pillStatus = 'ok';
    pillText = 'Done';
  } else {
    // The last visible stage keeps the pill while hidden stages run.
    const current = visible[visible.length - 1];
    const searches = current?.children.filter(c => c.kind === 'search').length ?? 0;
    pillStatus = 'running';
    pillText = [
      current ? current.runningLabel ?? current.label : 'Preparing…',
      searches > 0 && timeline.passes > 1 ? `pass ${timeline.passes}` : null,
      searches > 0 ? `${searches} search${searches === 1 ? '' : 'es'}` : null,
    ].filter(Boolean).join(' · ');
  }

  const lines = outcome?.lines ?? [];
  // Lines close the stage that did the searching; the last such stage, or the end.
  const linesAfter = [...visible].reverse().find(r => r.children.some(c => c.kind !== 'company')) ?? visible[visible.length - 1];
  const expandable = !outcome || lines.length > 0
    || visible.some(r => r.children.length > 0 || r.status === 'warn' || r.status === 'fail');

  const pillBody = (
    <>
      <StatusIcon status={pillStatus} />
      <span className="font-medium text-[var(--text)] text-left">{pillText}</span>
      {total && <span className="text-[var(--text-faint)] flex-shrink-0">· {total}</span>}
    </>
  );
  const pillClass = 'flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--surface-1)] text-xs';

  return (
    <div className="pl-4 py-1">
      {expandable ? (
        <button
          type="button"
          onClick={() => setExpanded(e => !e)}
          aria-expanded={expanded}
          className={`${pillClass} hover:bg-[var(--surface-2)]`}
          style={{ transition: 'background 0.2s' }}
        >
          {pillBody}
          <ChevronDown
            size={11}
            aria-hidden
            className="text-[var(--text-faint)] ml-0.5 flex-shrink-0"
            style={{ transition: 'transform 0.25s ease', transform: expanded ? 'rotate(180deg)' : 'none' }}
          />
        </button>
      ) : (
        <div className={`${pillClass} w-fit`}>{pillBody}</div>
      )}

      {expandable && (
        <div
          className="grid"
          style={{
            gridTemplateRows: expanded ? '1fr' : '0fr',
            opacity: expanded ? 1 : 0,
            transition: 'grid-template-rows 0.3s ease, opacity 0.25s ease',
          }}
        >
          <div className="overflow-hidden">
            <div className="mt-2 ml-1 flex flex-col max-w-[var(--content-max)]">
              {visible.map(row => (
                <React.Fragment key={row.id}>
                  <ActivityRow row={row} now={now} />
                  {row === linesAfter && lines.length > 0 && <OutcomeLines lines={lines} />}
                </React.Fragment>
              ))}
              {!linesAfter && lines.length > 0 && <OutcomeLines lines={lines} />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
