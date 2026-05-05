import React, { useRef, useEffect, useState } from 'react';
import { Filter, Layers, CheckCircle, Plus, ChevronDown } from 'lucide-react';
import { Scope } from '../types';
import { Badge, Button } from './ui';

interface ScopeBarProps {
  scope: Scope;
  docCount: number;
  onFilterChange: (filters: Partial<Scope['filters']>) => void;
  onModeChange: (mode: Scope['mode']) => void;
  onAddFiles?: () => void;
  filterOptions?: { companies: string[]; years: number[] };
}

const MODES = [
  { id: 'allDocs' as Scope['mode'], label: 'All Docs', icon: Layers },
  { id: 'filteredByMetadata' as Scope['mode'], label: 'Filtered', icon: Filter },
  { id: 'selectedDocs' as Scope['mode'], label: 'Selected', icon: CheckCircle },
];

export const ScopeBar: React.FC<ScopeBarProps> = ({ scope, docCount, onModeChange, onFilterChange, onAddFiles, filterOptions }) => {
  const activeIdx = MODES.findIndex(m => m.id === scope.mode);
  const containerRef = useRef<HTMLDivElement>(null);
  const btnRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({});

  useEffect(() => {
    const btn = btnRefs.current[activeIdx];
    const container = containerRef.current;
    if (!btn || !container) return;
    const containerRect = container.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    setIndicatorStyle({
      left: btnRect.left - containerRect.left,
      width: btnRect.width,
    });
  }, [activeIdx, scope.mode]);

  return (
    <div className="flex items-center gap-3 px-4 h-14 border-b border-[var(--border)] bg-[var(--surface-1)] flex-wrap shrink-0">
      {/* Sliding pill group */}
      <div
        ref={containerRef}
        className="relative flex bg-[var(--surface-2)] border border-[var(--border)] rounded-lg p-1"
      >
        {/* Sliding indicator */}
        <div
          className="absolute top-1 bottom-1 bg-[var(--surface-3)] rounded-md transition-all duration-200 ease-out shadow-sm"
          style={indicatorStyle}
        />
        {MODES.map((mode, idx) => (
          <button
            key={mode.id}
            ref={el => { btnRefs.current[idx] = el; }}
            onClick={() => onModeChange(mode.id)}
            className={`
              relative z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors duration-150
              ${scope.mode === mode.id ? 'text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'}
            `}
          >
            <mode.icon size={12} />
            {mode.label}
          </button>
        ))}
      </div>

      {/* Filters (only when filteredByMetadata) */}
      {scope.mode === 'filteredByMetadata' && (
        <div className="flex flex-wrap gap-2 animate-fade-in">
          <div className="relative">
            <select
              className="appearance-none bg-[var(--input-bg)] text-xs border border-[var(--input-border)] rounded-full
                px-3 py-1.5 pr-7 text-[var(--text)] focus:border-[var(--input-border-focus)] focus:ring-1
                focus:ring-[var(--focus-ring)] outline-none cursor-pointer transition-colors
                hover:border-[var(--border-strong)]"
              value={scope.filters.company?.[0] ?? ''}
              onChange={e => onFilterChange({ company: e.target.value ? [e.target.value] : undefined })}
            >
              <option value="">All Companies</option>
              {(filterOptions?.companies ?? []).map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <ChevronDown size={10} className="absolute right-2 top-[7px] text-[var(--text-faint)] pointer-events-none" />
          </div>

          <div className="relative">
            <select
              className="appearance-none bg-[var(--input-bg)] text-xs border border-[var(--input-border)] rounded-full
                px-3 py-1.5 pr-7 text-[var(--text)] focus:border-[var(--input-border-focus)] focus:ring-1
                focus:ring-[var(--focus-ring)] outline-none cursor-pointer transition-colors
                hover:border-[var(--border-strong)]"
              value={scope.filters.year?.[0] ?? ''}
              onChange={e => onFilterChange({ year: e.target.value ? [parseInt(e.target.value)] : undefined })}
            >
              <option value="">All Years</option>
              {(filterOptions?.years ?? []).map(y => <option key={y} value={y}>{y}</option>)}
            </select>
            <ChevronDown size={10} className="absolute right-2 top-[7px] text-[var(--text-faint)] pointer-events-none" />
          </div>
        </div>
      )}

      {/* Selected docs actions */}
      {scope.mode === 'selectedDocs' && (
        <div className="flex items-center gap-2 animate-fade-in">
          <Button size="sm" variant="secondary" onClick={onAddFiles} className="h-7 gap-1 text-xs border-dashed">
            <Plus size={12} /> Add Files
          </Button>
          <span className="text-xs text-[var(--text-muted)]">
            {scope.docIds.length === 0 ? 'No files selected.' : `${scope.docIds.length} file(s) in scope.`}
          </span>
        </div>
      )}

      {/* Doc count badge — pushed to the right */}
      <div className="ml-auto flex-shrink-0">
        <Badge variant="accent" className="h-7 px-3 font-mono text-[10px]">
          {docCount} DOCS
        </Badge>
      </div>
    </div>
  );
};
