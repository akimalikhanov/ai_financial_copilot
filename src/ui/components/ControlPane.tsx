import React, { useState } from 'react';
import {
  Settings2, ChevronRight, ChevronLeft, Thermometer,
  Hash, Brain, MessageSquareText, Zap, DollarSign,
  Clock, BarChart3, Sparkles, Info, RotateCcw, ChevronDown, ChevronUp
} from 'lucide-react';
import { Button, Badge } from './ui';

// ============================================
// TYPES
// ============================================

export interface ModelParams {
  temperature: number;
  maxTokens: number;
  reasoningEffort: 'low' | 'medium' | 'high' | null;
  verbosity: 'low' | 'medium' | 'high' | null;
}

export interface RequestStats {
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  totalTokens: number;
  cost: number;
  latencyMs: number;
  ttftMs: number | null;
  tps: number | null;
  model: string;
  timestamp: number;
}

export interface ModelCapabilities {
  supportsTemperature: boolean;
  supportsReasoningEffort: boolean;
  supportsVerbosity: boolean;
}

interface ControlPaneProps {
  isOpen: boolean;
  onToggle: () => void;
  params: ModelParams;
  onParamsChange: (params: Partial<ModelParams>) => void;
  capabilities: ModelCapabilities;
  stats: RequestStats | null;
  statsHistory: RequestStats[];
}

// ============================================
// SLIDER COMPONENT
// ============================================

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  icon: React.ReactNode;
  disabled?: boolean;
  formatValue?: (value: number) => string;
  hint?: string;
}

const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  icon,
  disabled = false,
  formatValue = (v) => v.toString(),
  hint,
}) => {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`space-y-2 ${disabled ? 'opacity-40' : ''}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[var(--icon)]">{icon}</span>
          <span className="text-xs font-medium text-[var(--text)]">{label}</span>
          {hint && (
            <div className="group relative">
              <Info size={12} className="text-[var(--text-faint)] cursor-help" />
              <div className="absolute left-0 top-full mt-2 hidden group-hover:block z-50">
                <div className="bg-[var(--surface-3)] text-[var(--text)] text-[10px] px-2 py-1 rounded-md shadow-lg whitespace-pre-line border border-[var(--border)] max-w-[200px]">
                  {hint.split(/<br\s*\/?>/i).map((part, i, arr) => (
                    <React.Fragment key={i}>
                      {part.trim()}
                      {i < arr.length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
        <span className="text-xs font-mono text-[var(--accent)] bg-[var(--accent-subtle)] px-2 py-0.5 rounded">
          {formatValue(value)}
        </span>
      </div>
      <div className="relative h-2 group">
        <div className="absolute inset-0 bg-[var(--surface-3)] rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] rounded-full transition-all duration-150"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-md border-2 border-[var(--accent)] transition-all duration-150 group-hover:scale-110"
          style={{ left: `calc(${percentage}% - 8px)` }}
        />
      </div>
    </div>
  );
};

// ============================================
// SEGMENTED CONTROL COMPONENT
// ============================================

interface SegmentedControlProps {
  label: string;
  value: string | null;
  options: { value: string; label: string }[];
  onChange: (value: string | null) => void;
  icon: React.ReactNode;
  disabled?: boolean;
  hint?: string;
}

const SegmentedControl: React.FC<SegmentedControlProps> = ({
  label,
  value,
  options,
  onChange,
  icon,
  disabled = false,
  hint,
}) => {
  return (
    <div className={`space-y-2 ${disabled ? 'opacity-40' : ''}`}>
      <div className="flex items-center gap-2">
        <span className="text-[var(--icon)]">{icon}</span>
        <span className="text-xs font-medium text-[var(--text)]">{label}</span>
        {hint && (
          <div className="group relative">
            <Info size={12} className="text-[var(--text-faint)] cursor-help" />
            <div className="absolute left-0 top-full mt-2 hidden group-hover:block z-50">
              <div className="bg-[var(--surface-3)] text-[var(--text)] text-[10px] px-2 py-1 rounded-md shadow-lg whitespace-pre-line border border-[var(--border)] max-w-[200px]">
                {hint.split(/<br\s*\/?>/i).map((part, i, arr) => (
                  <React.Fragment key={i}>
                    {part.trim()}
                    {i < arr.length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="flex bg-[var(--surface-2)] rounded-lg p-1 border border-[var(--border)]">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(value === opt.value ? null : opt.value)}
            disabled={disabled}
            className={`
              flex-1 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider rounded-md transition-all duration-150
              ${value === opt.value
                ? 'bg-[var(--accent)] text-white shadow-sm'
                : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--surface-3)]'
              }
              disabled:cursor-not-allowed
            `}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
};

// ============================================
// COLLAPSIBLE SECTION COMPONENT
// ============================================

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon,
  defaultOpen = true,
  children,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-[var(--surface-2)] rounded-lg border border-[var(--border)] overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-[var(--surface-3)] transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-[var(--icon)]">{icon}</span>
          <span className="text-xs font-semibold text-[var(--text)] uppercase tracking-wider">{title}</span>
        </div>
        {isOpen ? (
          <ChevronUp size={14} className="text-[var(--text-muted)]" />
        ) : (
          <ChevronDown size={14} className="text-[var(--text-muted)]" />
        )}
      </button>
      {isOpen && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );
};

// ============================================
// STAT CARD COMPONENT
// ============================================

interface StatCardProps {
  label: string;
  value: string;
  subValue?: string;
  icon: React.ReactNode;
  variant?: 'default' | 'accent' | 'warning';
}

const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subValue,
  icon,
  variant = 'default',
}) => {
  const variants = {
    default: 'text-[var(--text)]',
    accent: 'text-[var(--accent)]',
    warning: 'text-[var(--warning)]',
  };

  return (
    <div className="bg-[var(--surface-2)] rounded-lg p-3 border border-[var(--border)] hover:border-[var(--border-subtle)] transition-colors">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[var(--text-faint)]">{icon}</span>
        <span className="text-[10px] uppercase tracking-wider text-[var(--text-faint)] font-medium">{label}</span>
      </div>
      <div className={`text-lg font-semibold font-mono ${variants[variant]}`}>
        {value}
      </div>
      {subValue && (
        <div className="text-[10px] text-[var(--text-faint)] font-mono mt-0.5">
          {subValue}
        </div>
      )}
    </div>
  );
};

// ============================================
// MAIN CONTROL PANE COMPONENT
// ============================================

export const ControlPane: React.FC<ControlPaneProps> = ({
  isOpen,
  onToggle,
  params,
  onParamsChange,
  capabilities,
  stats,
  statsHistory,
}) => {
  const [activeTab, setActiveTab] = useState<'params' | 'stats'>('params');

  // Format cost as currency
  const formatCost = (cost: number) => {
    if (cost < 0.01) return `$${cost.toFixed(6)}`;
    if (cost < 1) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
  };

  // Format latency
  const formatLatency = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format TTFT
  const formatTTFT = (ms: number | null) => {
    if (ms === null || ms === undefined) return 'N/A';
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format TPS
  const formatTPS = (tps: number | null) => {
    if (tps === null || tps === undefined) return 'N/A';
    return `${tps.toFixed(1)}`;
  };

  // Calculate session totals
  const sessionTotals = statsHistory.reduce(
    (acc, stat) => ({
      tokens: acc.tokens + stat.totalTokens,
      cost: acc.cost + stat.cost,
      requests: acc.requests + 1,
    }),
    { tokens: 0, cost: 0, requests: 0 }
  );

  // Reset params to defaults
  const handleReset = () => {
    onParamsChange({
      temperature: 0.2,
      maxTokens: 2000,
      reasoningEffort: null,
      verbosity: null,
    });
  };

  return (
    <>
      {/* Toggle Button (visible when closed) */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed right-0 top-1/2 -translate-y-1/2 z-40 bg-[var(--surface-2)] border border-[var(--border)] border-r-0 rounded-l-lg p-2 hover:bg-[var(--surface-3)] transition-colors shadow-lg group"
          title="Open Control Pane"
        >
          <Settings2 size={18} className="text-[var(--icon)] group-hover:text-[var(--accent)] transition-colors" />
        </button>
      )}

      {/* Control Pane Panel */}
      <div
        className={`
          fixed right-0 top-14 bottom-0 z-30 w-96
          bg-[var(--surface-1)] border-l border-[var(--border)]
          transform transition-transform duration-300 ease-out
          flex flex-col
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        {/* Header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-[var(--border)] shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-[var(--accent)]" />
            <span className="font-semibold text-sm text-[var(--text)]">Control Pane</span>
          </div>
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md hover:bg-[var(--surface-2)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
          >
            <ChevronRight size={16} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex border-b border-[var(--border)] shrink-0">
          <button
            onClick={() => setActiveTab('params')}
            className={`
              flex-1 px-4 py-3 text-xs font-semibold uppercase tracking-wider transition-colors relative
              ${activeTab === 'params'
                ? 'text-[var(--accent)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }
            `}
          >
            <span className="flex items-center justify-center gap-2">
              <Settings2 size={14} />
              Parameters
            </span>
            {activeTab === 'params' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--accent)]" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('stats')}
            className={`
              flex-1 px-4 py-3 text-xs font-semibold uppercase tracking-wider transition-colors relative
              ${activeTab === 'stats'
                ? 'text-[var(--accent)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }
            `}
          >
            <span className="flex items-center justify-center gap-2">
              <BarChart3 size={14} />
              Stats
            </span>
            {activeTab === 'stats' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--accent)]" />
            )}
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'params' ? (
            <div className="space-y-6">
              {/* Temperature */}
              <Slider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.1}
                onChange={(v) => onParamsChange({ temperature: v })}
                icon={<Thermometer size={14} />}
                disabled={!capabilities.supportsTemperature}
                formatValue={(v) => v.toFixed(1)}
                hint="Controls randomness.
                <br />
                Lower = more focused, higher = more creative"
              />

              {/* Max Tokens */}
              <Slider
                label="Max Tokens"
                value={params.maxTokens}
                min={100}
                max={16000}
                step={100}
                onChange={(v) => onParamsChange({ maxTokens: v })}
                icon={<Hash size={14} />}
                formatValue={(v) => v.toLocaleString()}
                hint="Maximum length of the response"
              />

              {/* Reasoning Effort */}
              <SegmentedControl
                label="Reasoning Effort"
                value={params.reasoningEffort}
                options={[
                  { value: 'low', label: 'Low' },
                  { value: 'medium', label: 'Med' },
                  { value: 'high', label: 'High' },
                ]}
                onChange={(v) => onParamsChange({ reasoningEffort: v as 'low' | 'medium' | 'high' | null })}
                icon={<Brain size={14} />}
                disabled={!capabilities.supportsReasoningEffort}
                hint="How much the model should 'think' before responding.
                <br />
                Low = less thinking, high = more thinking"
              />

              {/* Verbosity */}
              <SegmentedControl
                label="Verbosity"
                value={params.verbosity}
                options={[
                  { value: 'low', label: 'Concise' },
                  { value: 'medium', label: 'Normal' },
                  { value: 'high', label: 'Detailed' },
                ]}
                onChange={(v) => onParamsChange({ verbosity: v as 'low' | 'medium' | 'high' | null })}
                icon={<MessageSquareText size={14} />}
                disabled={!capabilities.supportsVerbosity}
                hint="How detailed the response should be"
              />

              {/* Reset Button */}
              <button
                onClick={handleReset}
                className="w-full flex items-center justify-center gap-2 py-2 text-xs text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] rounded-lg transition-colors border border-dashed border-[var(--border)]"
              >
                <RotateCcw size={12} />
                Reset to Defaults
              </button>

              {/* Capability Notice */}
              {(!capabilities.supportsTemperature || !capabilities.supportsReasoningEffort || !capabilities.supportsVerbosity) && (
                <div className="bg-[var(--warning-bg)] border border-[var(--warning)] rounded-lg p-3 text-xs text-[var(--warning)]">
                  <div className="flex items-start gap-2">
                    <Info size={14} className="shrink-0 mt-0.5" />
                    <span>Some parameters are disabled because the selected model doesn't support them.</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {/* Last Request Stats */}
              {stats ? (
                <>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Last Request</span>
                    <Badge variant="accent" className="text-[10px]">{stats.model}</Badge>
                  </div>

                  <div className="space-y-3">
                    {/* Tokens Usage Section */}
                    <CollapsibleSection
                      title="Tokens Usage"
                      icon={<Zap size={14} />}
                      defaultOpen={true}
                    >
                      <div className="space-y-3">
                        <div className="grid grid-cols-3 gap-2">
                          <StatCard
                            label="Input"
                            value={stats.inputTokens.toLocaleString()}
                            icon={<Zap size={12} />}
                          />
                          <StatCard
                            label="Output"
                            value={stats.outputTokens.toLocaleString()}
                            icon={<Zap size={12} />}
                          />
                          <StatCard
                            label="Reasoning"
                            value={stats.reasoningTokens > 0 ? stats.reasoningTokens.toLocaleString() : '—'}
                            icon={<Brain size={12} />}
                            variant={stats.reasoningTokens > 0 ? 'default' : 'default'}
                          />
                        </div>

                        {/* Token Distribution Bar */}
                        <div className="bg-[var(--surface-1)] rounded-lg p-3 border border-[var(--border)]">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] uppercase tracking-wider text-[var(--text-faint)]">Distribution</span>
                            <span className="text-[10px] font-mono text-[var(--text-muted)]">{stats.totalTokens.toLocaleString()} total</span>
                          </div>
                          <div className="h-2.5 bg-[var(--surface-3)] rounded-full overflow-hidden flex">
                            {stats.inputTokens > 0 && (
                              <div
                                className="bg-[var(--accent)] transition-all duration-300"
                                style={{ width: `${(stats.inputTokens / stats.totalTokens) * 100}%` }}
                                title={`Input: ${stats.inputTokens}`}
                              />
                            )}
                            {stats.outputTokens > 0 && (
                              <div
                                className="bg-[var(--warning)] transition-all duration-300"
                                style={{ width: `${(stats.outputTokens / stats.totalTokens) * 100}%` }}
                                title={`Output: ${stats.outputTokens}`}
                              />
                            )}
                            {stats.reasoningTokens > 0 && (
                              <div
                                className="transition-all duration-300"
                                style={{
                                  width: `${(stats.reasoningTokens / stats.totalTokens) * 100}%`,
                                  backgroundColor: 'rgba(139, 92, 246, 0.7)' // Purple for reasoning
                                }}
                                title={`Reasoning: ${stats.reasoningTokens}`}
                              />
                            )}
                          </div>
                          <div className="flex items-center justify-between mt-2 text-[10px]">
                            <div className="flex items-center gap-1">
                              <div className="w-2 h-2 rounded-full bg-[var(--accent)]" />
                              <span className="text-[var(--text-faint)]">Input</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-2 h-2 rounded-full bg-[var(--warning)]" />
                              <span className="text-[var(--text-faint)]">Output</span>
                            </div>
                            {stats.reasoningTokens > 0 && (
                              <div className="flex items-center gap-1">
                                <div
                                  className="w-2 h-2 rounded-full"
                                  style={{ backgroundColor: 'rgba(139, 92, 246, 0.7)' }}
                                />
                                <span className="text-[var(--text-faint)]">Reasoning</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </CollapsibleSection>

                    {/* Time Stats Section */}
                    <CollapsibleSection
                      title="Time Stats"
                      icon={<Clock size={14} />}
                      defaultOpen={true}
                    >
                      <div className="grid grid-cols-3 gap-2">
                        <StatCard
                          label="Latency"
                          value={formatLatency(stats.latencyMs)}
                          variant={stats.latencyMs > 5000 ? 'warning' : 'default'}
                          icon={<Clock size={12} />}
                        />
                        <StatCard
                          label="TTFT"
                          value={formatTTFT(stats.ttftMs)}
                          icon={<Clock size={12} />}
                        />
                        <StatCard
                          label="TPS"
                          value={formatTPS(stats.tps)}
                          icon={<Zap size={12} />}
                        />
                      </div>
                    </CollapsibleSection>

                    {/* Cost Section */}
                    <CollapsibleSection
                      title="Cost"
                      icon={<DollarSign size={14} />}
                      defaultOpen={true}
                    >
                      <div>
                        <StatCard
                          label="Total Cost"
                          value={formatCost(stats.cost)}
                          variant="accent"
                          icon={<DollarSign size={12} />}
                        />
                      </div>
                    </CollapsibleSection>
                  </div>
                </>
              ) : (
                <div className="text-center py-8">
                  <div className="w-12 h-12 mx-auto bg-[var(--surface-2)] rounded-xl flex items-center justify-center mb-3 border border-[var(--border)]">
                    <BarChart3 size={20} className="text-[var(--text-faint)]" />
                  </div>
                  <p className="text-sm text-[var(--text-muted)]">No requests yet</p>
                  <p className="text-xs text-[var(--text-faint)] mt-1">Stats will appear after your first message</p>
                </div>
              )}

              {/* Session Totals */}
              {statsHistory.length > 0 && (
                <>
                  <div className="h-px bg-[var(--border)] my-4" />

                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Session Totals</span>
                    <span className="text-[10px] text-[var(--text-faint)]">{sessionTotals.requests} requests</span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <StatCard
                      label="Total Tokens"
                      value={sessionTotals.tokens.toLocaleString()}
                      icon={<Zap size={12} />}
                    />
                    <StatCard
                      label="Total Cost"
                      value={formatCost(sessionTotals.cost)}
                      variant="accent"
                      icon={<DollarSign size={12} />}
                    />
                  </div>
                </>
              )}

              {/* Recent History */}
              {statsHistory.length > 1 && (
                <>
                  <div className="h-px bg-[var(--border)] my-4" />

                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">Recent Requests</span>
                  </div>

                  <div className="space-y-2 max-h-80 overflow-y-auto">
                    {statsHistory.slice(-5).reverse().map((stat, idx) => {
                      const requestNumber = statsHistory.length - idx;
                      return (
                        <div
                          key={stat.timestamp}
                          className="flex items-center justify-between py-2.5 px-3 bg-[var(--surface-2)] rounded-lg border border-[var(--border)] text-xs hover:bg-[var(--surface-3)] transition-colors"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-[var(--text-faint)] font-mono shrink-0">#{requestNumber}</span>
                            <span className="text-[var(--text)] truncate">{stat.totalTokens.toLocaleString()} tokens</span>
                          </div>
                          <div className="flex items-center gap-3 shrink-0">
                            <span className="text-[var(--accent)] font-mono">{formatCost(stat.cost)}</span>
                            <span className="text-[var(--text-faint)] font-mono">{formatLatency(stat.latencyMs)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-[var(--border)] bg-[var(--surface-1)] shrink-0">
          <div className="flex items-center justify-between text-[10px] text-[var(--text-faint)]">
            <span>Press <kbd className="px-1.5 py-0.5 bg-[var(--surface-2)] rounded border border-[var(--border)] font-mono">⌘ K</kbd> for shortcuts</span>
            <span>v2.4.0</span>
          </div>
        </div>
      </div>
    </>
  );
};

export default ControlPane;
