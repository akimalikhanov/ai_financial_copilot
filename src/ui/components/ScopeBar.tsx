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
}

export const ScopeBar: React.FC<ScopeBarProps> = ({ scope, docCount, onModeChange, onFilterChange, onAddFiles }) => {
  return (
    <div className="flex flex-col gap-4 p-4 border-b border-zinc-800 bg-zinc-900/30">
      {/* Mode Switcher */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-mono text-zinc-500 uppercase tracking-widest mr-2">Scope</span>
        <div className="flex bg-zinc-900 border border-zinc-800 rounded-sm p-0.5">
          {[
            { id: 'allDocs', label: 'All Docs', icon: Layers },
            { id: 'filteredByMetadata', label: 'Filtered', icon: Filter },
            { id: 'selectedDocs', label: 'Selected', icon: CheckCircle }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id as Scope['mode'])}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-sm text-xs font-medium transition-all
                ${scope.mode === mode.id 
                  ? 'bg-zinc-800 text-zinc-100 shadow-sm' 
                  : 'text-zinc-500 hover:text-zinc-300'}
              `}
            >
              <mode.icon size={12} />
              {mode.label}
            </button>
          ))}
        </div>
        
        {/* Active Doc Count Badge */}
        <div className="ml-auto flex items-center gap-2">
             <Badge variant="outline" className="h-7 px-3 bg-zinc-950 font-mono text-accent-400 border-accent-900/30">
                {docCount} DOCS ACTIVE
             </Badge>
        </div>
      </div>

      {/* Filters (Only visible if 'filteredByMetadata') */}
      {scope.mode === 'filteredByMetadata' && (
        <div className="flex flex-wrap gap-2 animate-fade-in">
          <select 
            className="bg-zinc-900 text-xs border border-zinc-700 rounded-sm px-2 py-1 text-zinc-300 focus:border-accent-500 outline-none"
            onChange={(e) => onFilterChange({ company: e.target.value ? [e.target.value] : undefined })}
          >
            <option value="">All Companies</option>
            <option value="NVIDIA Corp">NVIDIA Corp</option>
            <option value="Tesla Inc">Tesla Inc</option>
            <option value="Apple Inc">Apple Inc</option>
          </select>
          
          <select 
             className="bg-zinc-900 text-xs border border-zinc-700 rounded-sm px-2 py-1 text-zinc-300 focus:border-accent-500 outline-none"
             onChange={(e) => onFilterChange({ year: e.target.value ? [parseInt(e.target.value)] : undefined })}
          >
            <option value="">All Years</option>
            <option value="2023">2023</option>
            <option value="2022">2022</option>
          </select>

           <select 
             className="bg-zinc-900 text-xs border border-zinc-700 rounded-sm px-2 py-1 text-zinc-300 focus:border-accent-500 outline-none"
             onChange={(e) => onFilterChange({ type: e.target.value ? [e.target.value] : undefined })}
          >
            <option value="">All Types</option>
            <option value="Annual Report">Annual Report</option>
            <option value="10-K">10-K</option>
          </select>
        </div>
      )}

      {/* Selected Docs Actions */}
      {scope.mode === 'selectedDocs' && (
        <div className="flex items-center gap-2 animate-fade-in">
            <Button size="sm" onClick={onAddFiles} className="h-7 gap-1 text-xs border-dashed border-zinc-700 bg-zinc-900/50 text-zinc-400 hover:text-accent-400 hover:border-accent-500/50">
                <Plus size={12} /> Add Files
            </Button>
            <span className="text-xs text-zinc-500 ml-2">
                {scope.docIds.length === 0 
                    ? "No documents selected. Add files to begin." 
                    : `${scope.docIds.length} file(s) in scope.`}
            </span>
        </div>
      )}
    </div>
  );
};