import React, { useState } from 'react';
import { Button } from './ui';
import type { ClarificationCandidate, ClarificationEntity, ClarificationPick, ScopeClarification } from '../types';

interface ClarificationCardProps {
  card: ScopeClarification;
  companies: string[];
  disabled: boolean;
  // Companies in the scope bar's company filter; 0 when the bar isn't filtering by company.
  filteredCompanies: number;
  onReply: (picks: ClarificationPick[]) => void;
  onNarrowScope: () => void;
  onAskAgain: () => void;
  onUpload: () => void;
}

const candidateLabel = (c: ClarificationCandidate) =>
  c.years.length ? `${c.company} · ${c.years.join(', ')}` : c.company;

const CompanySelect: React.FC<{ companies: string[]; disabled: boolean; onPick: (company: string) => void }> = ({ companies, disabled, onPick }) => (
  <select
    className="text-xs rounded-full border border-[var(--input-border)] bg-[var(--input-bg)] text-[var(--text)] px-3 py-1.5 outline-none"
    value=""
    disabled={disabled || companies.length === 0}
    onChange={e => e.target.value && onPick(e.target.value)}
  >
    <option value="">Another company…</option>
    {companies.map(c => <option key={c} value={c}>{c}</option>)}
  </select>
);

export const ClarificationCard: React.FC<ClarificationCardProps> = ({ card, companies, disabled, filteredCompanies, onReply, onNarrowScope, onAskAgain, onUpload }) => {
  const [picks, setPicks] = useState<Record<string, ClarificationPick>>({});

  if (card.outcome === 'too_broad') {
    if (card.named_companies) return null;  // the streamed text already asks to split the question
    // The user picks companies in the scope bar; this card only gets them there and re-asks.
    const ready = filteredCompanies >= 1 && filteredCompanies <= card.max_companies;
    return (
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" disabled={disabled} onClick={onNarrowScope}>Narrow scope</Button>
        <Button size="sm" disabled={disabled || !ready} onClick={onAskAgain}>Ask again</Button>
        <span className="text-xs text-[var(--text-faint)]">
          {filteredCompanies > card.max_companies
            ? `${filteredCompanies} companies picked, up to ${card.max_companies}`
            : `${filteredCompanies} of up to ${card.max_companies} companies picked in the scope bar`}
        </span>
      </div>
    );
  }

  const choose = (pick: ClarificationPick) => {
    const next = { ...picks, [pick.raw_span]: pick };
    setPicks(next);
    if (Object.keys(next).length === card.unresolved.length) onReply(Object.values(next));
  };
  const isPicked = (span: string, match: Partial<ClarificationPick>) => {
    const p = picks[span];
    return !!p && (match.company === undefined || p.company === match.company)
      && (match.include_outside_scope === undefined || p.include_outside_scope === match.include_outside_scope);
  };
  const off = disabled || Object.keys(picks).length === card.unresolved.length;

  const renderEntity = (u: ClarificationEntity) => {
    if (u.outcome === 'outside_scope') {
      const name = u.candidates[0]?.company ?? u.raw_span;
      return (
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant={isPicked(u.raw_span, { include_outside_scope: true }) ? 'primary' : 'secondary'} disabled={off}
            onClick={() => choose({ raw_span: u.raw_span, include_outside_scope: true })}>Include {name}</Button>
          <Button size="sm" variant={isPicked(u.raw_span, { include_outside_scope: false }) ? 'primary' : 'secondary'} disabled={off}
            onClick={() => choose({ raw_span: u.raw_span, include_outside_scope: false })}>Keep selection</Button>
        </div>
      );
    }
    return (
      <div className="flex flex-wrap items-center gap-2">
        {u.candidates.map(c => (
          <Button key={c.company} size="sm" variant={isPicked(u.raw_span, { company: c.company }) ? 'primary' : 'secondary'} disabled={off}
            onClick={() => choose({ raw_span: u.raw_span, company: c.company })}>{candidateLabel(c)}</Button>
        ))}
        <CompanySelect companies={companies} disabled={off} onPick={company => choose({ raw_span: u.raw_span, company })} />
        {u.outcome === 'none' && (
          <Button size="sm" variant="ghost" disabled={disabled} onClick={onUpload}>Upload a document</Button>
        )}
      </div>
    );
  };

  return (
    <div className="mt-3 space-y-3">
      {card.resolved.length > 0 && (
        <div className="text-xs text-[var(--text-faint)]">Already matched: {card.resolved.join(', ')}</div>
      )}
      {card.unresolved.map(u => (
        <div key={u.raw_span} className="space-y-1.5">
          <div className="text-xs font-medium text-[var(--text-muted)]">“{u.raw_span}”</div>
          {renderEntity(u)}
        </div>
      ))}
    </div>
  );
};
