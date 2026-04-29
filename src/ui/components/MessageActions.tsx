import React, { useState, useRef, useEffect } from 'react';
import { Copy, Check, ThumbsUp, ThumbsDown, X } from 'lucide-react';
import { submitMessageFeedback, deleteMessageFeedback } from '../services/api';
import { MessageFeedback } from '../types';

interface MessageActionsProps {
  messageId: string;
  content: string;
  feedback?: MessageFeedback | null;
  onFeedbackChange: (feedback: MessageFeedback | null) => void;
}

export const MessageActions: React.FC<MessageActionsProps> = ({
  messageId,
  content,
  feedback,
  onFeedbackChange,
}) => {
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [showCommentFor, setShowCommentFor] = useState<'down' | null>(null);
  const [commentDraft, setCommentDraft] = useState('');
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const downBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!showCommentFor) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (popoverRef.current?.contains(target)) return;
      if (downBtnRef.current?.contains(target)) return;
      setShowCommentFor(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowCommentFor(null);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [showCommentFor]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  const persistRating = async (rating: 'up' | 'down', comment: string | null) => {
    const prev = feedback ?? null;
    setBusy(true);
    try {
      onFeedbackChange({ rating, comment });
      const res = await submitMessageFeedback(messageId, rating, comment);
      onFeedbackChange({ rating: res.rating, comment: res.comment ?? null });
    } catch {
      onFeedbackChange(prev);
    } finally {
      setBusy(false);
    }
  };

  const handleUp = async () => {
    if (busy) return;
    if (feedback?.rating === 'up') {
      const prev = feedback;
      setBusy(true);
      onFeedbackChange(null);
      try {
        await deleteMessageFeedback(messageId);
      } catch {
        onFeedbackChange(prev);
      } finally {
        setBusy(false);
      }
      return;
    }
    setShowCommentFor(null);
    await persistRating('up', null);
  };

  const handleDown = async () => {
    if (busy) return;
    if (feedback?.rating === 'down') {
      // Re-clicking down: toggle popover open to edit/clear
      setCommentDraft(feedback.comment ?? '');
      setShowCommentFor((s) => (s === 'down' ? null : 'down'));
      return;
    }
    setCommentDraft('');
    setShowCommentFor('down');
    await persistRating('down', null);
  };

  const submitComment = async () => {
    const trimmed = commentDraft.trim();
    setShowCommentFor(null);
    await persistRating('down', trimmed.length > 0 ? trimmed : null);
  };

  const removeFeedback = async () => {
    if (busy) return;
    const prev = feedback;
    setBusy(true);
    onFeedbackChange(null);
    setShowCommentFor(null);
    try {
      await deleteMessageFeedback(messageId);
    } catch {
      onFeedbackChange(prev ?? null);
    } finally {
      setBusy(false);
    }
  };

  const btnBase =
    'inline-flex items-center justify-center w-6 h-6 rounded transition-colors text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]';
  const btnActive = 'text-[var(--accent)] bg-[var(--surface-2)] hover:bg-[var(--surface-2)] hover:text-[var(--accent)]';

  return (
    <div className="relative flex items-center gap-1 pl-2.5 pt-1">
      <button
        type="button"
        onClick={handleCopy}
        className={btnBase}
        title="Copy"
        aria-label="Copy message"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <button
        type="button"
        onClick={handleUp}
        disabled={busy}
        className={`${btnBase} ${feedback?.rating === 'up' ? btnActive : ''}`}
        title="Good response"
        aria-label="Thumbs up"
      >
        <ThumbsUp size={14} fill={feedback?.rating === 'up' ? 'currentColor' : 'none'} />
      </button>
      <button
        ref={downBtnRef}
        type="button"
        onClick={handleDown}
        disabled={busy}
        className={`${btnBase} ${feedback?.rating === 'down' ? btnActive : ''}`}
        title="Bad response"
        aria-label="Thumbs down"
      >
        <ThumbsDown size={14} fill={feedback?.rating === 'down' ? 'currentColor' : 'none'} />
      </button>

      {showCommentFor === 'down' && (
        <div
          ref={popoverRef}
          className="absolute top-full left-4 mt-1 z-20 w-72 bg-[var(--surface-1)] border border-[var(--border)] rounded-lg shadow-[var(--shadow-md)] p-3"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-[var(--text)]">What went wrong? (optional)</span>
            <button
              type="button"
              onClick={() => setShowCommentFor(null)}
              className="text-[var(--text-faint)] hover:text-[var(--text)]"
              aria-label="Close"
            >
              <X size={14} />
            </button>
          </div>
          <textarea
            value={commentDraft}
            onChange={(e) => setCommentDraft(e.target.value)}
            placeholder="Tell us more…"
            className="w-full text-xs bg-[var(--input-bg)] border border-[var(--input-border)] rounded-md px-2 py-1.5 text-[var(--text)] placeholder:text-[var(--placeholder)] focus:outline-none focus:border-[var(--accent)] resize-none"
            rows={3}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                submitComment();
              }
            }}
          />
          <div className="flex items-center justify-between mt-2 gap-2">
            {feedback?.rating === 'down' ? (
              <button
                type="button"
                onClick={removeFeedback}
                className="text-[11px] text-[var(--text-faint)] hover:text-[var(--text)]"
              >
                Remove rating
              </button>
            ) : <span />}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowCommentFor(null)}
                className="text-xs px-2 py-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submitComment}
                disabled={busy}
                className="text-xs px-2.5 py-1 rounded-md bg-[var(--accent)] text-white hover:opacity-90 disabled:opacity-50"
              >
                Submit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
