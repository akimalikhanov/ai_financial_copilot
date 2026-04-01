import React from 'react';
import { Filter, Layers, CheckCircle, Plus } from 'lucide-react';
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

export const ScopeBar: React.FC<ScopeBarProps> = ({ scope, docCount, onModeChange, onFilterChange, onAddFiles, filterOptions }) => {
  return (
    <div className="flex flex-col gap-4 p-4 border-b border-[var(--border)] bg-[var(--surface-1)]">
      {/* Mode Switcher */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-[var(--text-faint)] uppercase tracking-wider mr-2">Scope</span>
        <div className="flex bg-[var(--surface-2)] border border-[var(--border)] rounded-lg p-1">
          {[
            { id: 'allDocs', label: 'All Docs', icon: Layers },
            { id: 'filteredByMetadata', label: 'Filtered', icon: Filter },
            { id: 'selectedDocs', label: 'Selected', icon: CheckCircle }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id as Scope['mode'])}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all
                ${scope.mode === mode.id
                  ? 'bg-[var(--surface-3)] text-[var(--text)] shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'}
              `}
            >
              <mode.icon size={12} />
              {mode.label}
            </button>
          ))}
        </div>

        {/* Active Doc Count Badge */}
        <div className="ml-auto flex items-center gap-2">
             <Badge variant="accent" className="h-7 px-3 font-mono">
                {docCount} DOCS ACTIVE
             </Badge>
        </div>
      </div>

      {/* Filters (Only visible if 'filteredByMetadata') */}
      {scope.mode === 'filteredByMetadata' && (
        <div className="flex flex-wrap gap-2 animate-fade-in">
          <select
            className="appearance-none bg-[var(--input-bg)] text-xs border border-[var(--input-border)] rounded-md px-3 py-1.5 text-[var(--text)] focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)] outline-none cursor-pointer"
            value={scope.filters.company?.[0] ?? ''}
            onChange={(e) => onFilterChange({ company: e.target.value ? [e.target.value] : undefined })}
          >
            <option value="">All Companies</option>
            {(filterOptions?.companies ?? []).map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <select
            className="appearance-none bg-[var(--input-bg)] text-xs border border-[var(--input-border)] rounded-md px-3 py-1.5 text-[var(--text)] focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)] outline-none cursor-pointer"
            value={scope.filters.year?.[0] ?? ''}
            onChange={(e) => onFilterChange({ year: e.target.value ? [parseInt(e.target.value)] : undefined })}
          >
            <option value="">All Years</option>
            {(filterOptions?.years ?? []).map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
      )}

      {/* Selected Docs Actions */}
      {scope.mode === 'selectedDocs' && (
        <div className="flex items-center gap-2 animate-fade-in">
            <Button
              size="sm"
              variant="secondary"
              onClick={onAddFiles}
              className="h-7 gap-1 text-xs border-dashed"
            >
                <Plus size={12} /> Add Files
            </Button>
            <span className="text-xs text-[var(--text-muted)] ml-2">
                {scope.docIds.length === 0
                    ? "No documents selected. Add files to begin."
                    : `${scope.docIds.length} file(s) in scope.`}
            </span>
        </div>
      )}
    </div>
  );
};
