import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  MessageSquare, BookOpen, Plus, Send, Search,
  LogOut, User, FileText, MoreHorizontal, Trash2, ArrowRight, Layers, AlertTriangle, ChevronDown, Bot, Sun, Moon, Settings2, Pencil, Loader2
} from 'lucide-react';
import { Document, Chat, Message, Scope, ViewMode, MobileTab, Citation, ReferenceItem } from './types';
import { CitedText } from './components/CitedText';
import {
  chatEnqueue,
  chatStreamSubscribe,
  fetchChatStats,
  fetchModels,
  getMe,
  listDocuments,
  createConversation,
  fetchMessages,
  updateConversation,
  fetchConversations,
  deleteConversation,
  ApiError,
  ModelInfo,
  type RequestStatsItem,
  type UserInfo,
  type DocumentListItemResponse,
} from './services/api';
import { useAuth } from './context/AuthContext';
import { LoginPage } from './components/LoginPage';
import { Button, Input, Badge, Card, ChatBubble, Toggle } from './components/ui';
import { ScopeBar } from './components/ScopeBar';
import { EvidencePanel } from './components/EvidencePanel';
import { UploadModal } from './components/UploadModal';
import { DocPickerModal } from './components/DocPickerModal';
import { ControlPane, ModelParams, RequestStats, ModelCapabilities } from './components/ControlPane';

// Fallback models (used while loading or on error)
const FALLBACK_MODELS: ModelInfo[] = [
  { id: 'gpt-4o-mini', name: 'GPT-4o-mini' },
];

// --- Helpers ---
const generateId = () => Math.random().toString(36).slice(2, 11);
const mapDocStatus = (s: string): Document['status'] =>
  s === 'ready' ? 'Ready' : s === 'failed' ? 'Error' : 'Processing';

const toYear = (v: unknown): number => {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v);
  if (typeof v === 'string' && v.trim().length > 0) {
    const n = Number(v);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return 0;
};

/** Map a backend MessageResponse (with metadata) to a frontend Message, restoring citations. */
const toUiMessage = (msg: { id: string; role: string; content: string; created_at: string; metadata?: Record<string, unknown> }): Message => {
  const meta = msg.metadata || {};
  const spans = meta.citation_spans as Array<{ start: number; end: number; ref_ids: string[]; display_labels: string[] }> | undefined;
  const refs = meta.references as Array<Record<string, unknown>> | undefined;
  return {
    id: msg.id,
    role: msg.role as 'user' | 'assistant',
    content: msg.content,
    timestamp: new Date(msg.created_at).getTime(),
    citations: meta.citations as Citation[] | undefined,
    citationSpans: spans?.map((s) => ({ start: s.start, end: s.end, refIds: s.ref_ids, displayLabels: s.display_labels })),
    references: refs?.map((r: Record<string, unknown>) => ({
      refId: r.ref_id as string,
      displayLabel: r.display_label as string,
      chunkId: r.chunk_id as string,
      documentId: r.document_id as string,
      documentName: r.document_name as string,
      filename: (r.filename as string | null),
      pageNumbers: r.page_numbers as number[],
      headingPath: r.heading_path as string[],
      snippet: (r.snippet as string | null),
    })),
  };
};

const toUiDoc = (d: DocumentListItemResponse): Document => ({
  id: d.id,
  title: d.extracted_title ?? d.original_filename,
  company: (d.metadata?.company as string | undefined) ?? '',
  year: toYear(d.metadata?.year),
  type: (d.metadata?.type as string | undefined) ?? '',
  pages: d.page_count ?? 0,
  status: mapDocStatus(d.status),
  tags: [],
});

function EvidenceList({
  references,
  onRefClick,
}: {
  references: ReferenceItem[];
  onRefClick: (ref: ReferenceItem) => void;
}) {
  const sorted = [...references].sort((a, b) =>
    a.displayLabel.localeCompare(b.displayLabel, undefined, { numeric: true })
  );
  const [open, setOpen] = useState(false);

  const toggle = () => setOpen((o) => !o);

  return (
    <div className="mt-3">
      {/* Section header — click to toggle */}
      <button
        onClick={toggle}
        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer
          hover:bg-[var(--surface-1)] transition-colors group mb-1"
      >
        <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-faint)] group-hover:text-[var(--accent)] transition-colors flex-shrink-0">
          Evidence
        </span>
        {/* Pill preview strip — visible only when collapsed */}
        {!open && (() => {
          const PILL_LIMIT = 2;
          const visible = sorted.slice(0, PILL_LIMIT);
          const overflow = sorted.length - PILL_LIMIT;
          return (
            <div className="flex items-center gap-1 flex-1 overflow-hidden">
              {visible.map((ref) => (
                <span
                  key={ref.refId}
                  className="inline-flex items-center justify-center h-4 min-w-[1.25rem] px-1 rounded
                    text-[9px] font-mono font-bold bg-[var(--accent-subtle)] text-[var(--accent)]
                    border border-[var(--accent)] border-opacity-30 leading-none flex-shrink-0"
                >
                  {ref.displayLabel}
                </span>
              ))}
              {overflow > 0 && (
                <span className="text-[9px] text-[var(--text-faint)] font-mono flex-shrink-0">
                  +{overflow} more
                </span>
              )}
            </div>
          );
        })()}
        {open && <div className="flex-1 h-px bg-[var(--border)]" />}
        <ChevronDown
          size={12}
          className={`text-[var(--text-faint)] group-hover:text-[var(--accent)] transition-all flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Reference cards — grid 0fr→1fr for smooth roll-out */}
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
        <div className="space-y-1 pb-0.5">
          {sorted.map((ref) => {
            const headingPath =
              ref.headingPath && ref.headingPath.length > 0 ? ref.headingPath.join(' › ') : null;
            const pages =
              ref.pageNumbers && ref.pageNumbers.length > 0
                ? `p.\u00a0${ref.pageNumbers.join(', ')}`
                : null;
            const snippet =
              ref.snippet
                ? ref.snippet.length > 140
                  ? ref.snippet.slice(0, 137) + '…'
                  : ref.snippet
                : null;

            return (
              <button
                key={ref.refId}
                onClick={() => onRefClick(ref)}
                className="w-full flex items-stretch gap-3 bg-[var(--surface-1)] border border-[var(--border)]
                  hover:border-[var(--accent)] hover:bg-[var(--surface-2)] rounded-xl px-3 py-2.5
                  text-left transition-all group"
              >
                {/* Label badge */}
                <div className="flex-shrink-0 flex items-start pt-0.5">
                  <span
                    className="inline-flex items-center justify-center h-5 min-w-[1.5rem] px-1 rounded
                      text-[10px] font-mono font-bold bg-[var(--accent-subtle)] text-[var(--accent)]
                      border border-[var(--accent)] border-opacity-30
                      group-hover:bg-[var(--accent)] group-hover:text-white group-hover:border-opacity-100
                      transition-all leading-none"
                  >
                    {ref.displayLabel}
                  </span>
                </div>

                {/* Content */}
                <div className="min-w-0 flex-1 space-y-0.5">
                  <div className="text-xs font-semibold text-[var(--text)] truncate leading-snug">
                    {ref.documentName || 'Unknown document'}
                  </div>
                  {headingPath && (
                    <div className="text-[10px] text-[var(--accent)] truncate leading-snug opacity-80">
                      {headingPath}
                    </div>
                  )}
                  {pages && (
                    <div className="text-[10px] text-[var(--text-faint)] font-mono leading-snug">
                      {pages}
                    </div>
                  )}
                  {snippet && (
                    <p className="text-[11px] text-[var(--text-faint)] italic leading-snug mt-1 line-clamp-2">
                      "{snippet}"
                    </p>
                  )}
                </div>
              </button>
            );
          })}
        </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { authChecked, isAuthenticated, setAccessToken, logout } = useAuth();

  // --- Global State ---
  const [view, setView] = useState<ViewMode>('ASK');
  const [mobileTab, setMobileTab] = useState<MobileTab>('CONVERSATION');
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Data
  const [docs, setDocs] = useState<Document[]>([]);
  const [chats, setChats] = useState<Chat[]>([]);

  // Messages (fetched from backend, keyed by conversationId)
  const [messagesByConversation, setMessagesByConversation] = useState<Record<string, Message[]>>({});
  const [messagesLoading, setMessagesLoading] = useState<Record<string, boolean>>({});
  const [messagesHasMore, setMessagesHasMore] = useState<Record<string, boolean>>({});
  const [messagesMinSeq, setMessagesMinSeq] = useState<Record<string, number>>({});

  // Models (loaded from API)
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Active Context
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState(FALLBACK_MODELS[0].id);
  const [scope, setScope] = useState<Scope>({
    mode: 'allDocs',
    docIds: [],
    filters: {}
  });

  // Evidence Panel State
  const [isEvidenceOpen, setIsEvidenceOpen] = useState(false);
  const [openPdfTabs, setOpenPdfTabs] = useState<{ doc: Document, page: number }[]>([]);
  const [activePdfDocId, setActivePdfDocId] = useState<string | null>(null);
  const [activeHighlight, setActiveHighlight] = useState<Citation['bboxHint'] | undefined>(undefined);

  // UI State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isAwaitingResponse, setIsAwaitingResponse] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isDocPickerOpen, setIsDocPickerOpen] = useState(false);

  // Control Pane State
  const [isControlPaneOpen, setIsControlPaneOpen] = useState(false);
  const [modelParams, setModelParams] = useState<ModelParams>({
    temperature: 0.2,
    maxTokens: 2000,
    reasoningEffort: null,
    verbosity: null,
  });
  const [lastRequestStats, setLastRequestStats] = useState<RequestStats | null>(null);
  const [statsHistory, setStatsHistory] = useState<RequestStats[]>([]);

  // Current user (fetched when authenticated)
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);

  // Delete Confirmation State
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  // Rename State
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // --- Theme Toggle ---
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.documentElement.classList.toggle('light', !isDarkMode);
  }, [isDarkMode]);

  // --- Derived State ---
  const activeChat = chats.find(c => c.id === activeChatId);
  const activeMessages = activeChat?.conversationId
    ? messagesByConversation[activeChat.conversationId] || []
    : [];
  const activeDocsCount = React.useMemo(() => {
    if (scope.mode === 'allDocs') return docs.length;
    if (scope.mode === 'filteredByMetadata') {
       return docs.filter(d => {
         const f = scope.filters;
         const matchCompany = !f.company?.length || f.company.includes(d.company);
         const matchYear = !f.year?.length || f.year.includes(d.year);
         const matchType = !f.type?.length || f.type.includes(d.type);
         return matchCompany && matchYear && matchType;
       }).length;
    }
    return scope.docIds.length;
  }, [docs, scope]);

  // --- Effects ---

  // Load current user when authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      setCurrentUser(null);
      return;
    }
    let cancelled = false;
    getMe()
      .then((user) => {
        if (!cancelled) setCurrentUser(user);
      })
      .catch((err) => console.error('Failed to load current user:', err));
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  // Load documents when authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      setDocs([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await listDocuments();
        if (!cancelled) setDocs(res.documents.map(toUiDoc));
      } catch (err) {
        console.error('Failed to load documents:', err);
      }
    })();
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  const refreshDocs = useCallback(async () => {
    const res = await listDocuments();
    setDocs(res.documents.map(toUiDoc));
  }, []);

  // Load conversations when authenticated (7.6)
  useEffect(() => {
    if (!isAuthenticated) return;
    let cancelled = false;
    fetchConversations()
      .then((res) => {
        if (cancelled) return;
        const mapped: Chat[] = res.conversations.map((c) => ({
          id: c.id,
          title: c.title ?? 'Untitled',
          createdAt: new Date(c.created_at).getTime(),
          conversationId: c.id,
        }));
        setChats(mapped);
      })
      .catch((err) => console.error('Failed to load conversations:', err));
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  // Fetch available models when authenticated (same as docs/conversations)
  useEffect(() => {
    if (!isAuthenticated) {
      setModels(FALLBACK_MODELS);
      setModelsLoading(false);
      return;
    }
    let cancelled = false;
    setModelsLoading(true);
    (async () => {
      try {
        const fetched = await fetchModels();
        if (!cancelled && fetched.length > 0) {
          setModels(fetched);
          // Set default model to first in list if current isn't valid
          setActiveModel((prev) => {
            const isValid = fetched.some((m) => m.id === prev);
            return isValid ? prev : fetched[0].id;
          });
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
        // Keep fallback models on error
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  // Fetch messages when conversation is selected
  useEffect(() => {
    if (!activeChat?.conversationId) return;

    const conversationId = activeChat.conversationId;

    // Skip if already have messages (avoid re-fetch on tab reload when switching back)
    if (messagesByConversation[conversationId]) return;

    let cancelled = false;
    setMessagesLoading((prev) => ({ ...prev, [conversationId]: true }));

    (async () => {
      try {
        const response = await fetchMessages(conversationId, { limit: 50 });
        if (!cancelled) {
          // Backend returns oldest first (ascending by seq); use as-is for display
          const uiMessages: Message[] = response.messages.map(toUiMessage);
          setMessagesByConversation((prev) => ({
            ...prev,
            [conversationId]: uiMessages,
          }));
          setMessagesHasMore((prev) => ({
            ...prev,
            [conversationId]: response.has_more,
          }));
          if (response.messages.length > 0) {
            const minSeq = Math.min(...response.messages.map((m) => m.seq));
            setMessagesMinSeq((prev) => ({
              ...prev,
              [conversationId]: minSeq,
            }));
          }
        }
      } catch (error) {
        console.error('Failed to fetch messages:', error);
        if (!cancelled) {
          setMessagesByConversation((prev) => ({
            ...prev,
            [conversationId]: [],
          }));
        }
      } finally {
        if (!cancelled) {
          setMessagesLoading((prev) => {
            const next = { ...prev };
            delete next[conversationId];
            return next;
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeChat?.conversationId, messagesByConversation]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeMessages, isTyping]);

  // Load older messages on scroll
  useEffect(() => {
    if (!activeChat?.conversationId || !messagesContainerRef.current) return;
    const conversationId = activeChat.conversationId;

    const container = messagesContainerRef.current;
    let scrollTimeout: ReturnType<typeof setTimeout> | null = null;

    const handleScroll = () => {
      // Debounce scroll events
      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }
      scrollTimeout = setTimeout(() => {
        // Load more when scrolled to top (within 100px)
        if (container.scrollTop < 100 && messagesHasMore[conversationId] && !messagesLoading[conversationId]) {
          const minSeq = messagesMinSeq[conversationId];
          if (minSeq !== undefined && minSeq > 0) {
            setMessagesLoading(prev => ({ ...prev, [conversationId]: true }));
            fetchMessages(conversationId, { limit: 50, before_seq: minSeq })
              .then(response => {
                // Backend returns oldest first; use as-is and prepend to existing
                const uiMessages: Message[] = response.messages.map(toUiMessage);
                setMessagesByConversation(prev => {
                  const existing = prev[conversationId] || [];
                  // Prepend older messages (they come before existing messages chronologically)
                  const allMessages = [...uiMessages, ...existing];
                  // Remove duplicates by id
                  const uniqueMessages = Array.from(
                    new Map(allMessages.map(m => [m.id, m])).values()
                  );
                  // Sort by timestamp (ascending)
                  uniqueMessages.sort((a, b) => a.timestamp - b.timestamp);
                  return {
                    ...prev,
                    [conversationId]: uniqueMessages,
                  };
                });
                setMessagesHasMore(prev => ({
                  ...prev,
                  [conversationId]: response.has_more,
                }));
                if (response.messages.length > 0) {
                  const newMinSeq = Math.min(...response.messages.map(m => m.seq));
                  setMessagesMinSeq(prev => ({
                    ...prev,
                    [conversationId]: Math.min(prev[conversationId] ?? Infinity, newMinSeq),
                  }));
                }
              })
              .catch(error => {
                console.error('Failed to load older messages:', error);
              })
              .finally(() => {
                setMessagesLoading(prev => {
                  const next = { ...prev };
                  delete next[conversationId];
                  return next;
                });
              });
          }
        }
      }, 100);
    };

    container.addEventListener('scroll', handleScroll);
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }
    };
  }, [activeChat?.conversationId, messagesHasMore, messagesLoading, messagesMinSeq]);

  // --- Streaming Mode Toggle ---

  // --- Model Capabilities (derived from active model) ---
  const modelCapabilities: ModelCapabilities = React.useMemo(() => {
    // GPT-5 series and some newer models support reasoning_effort and verbosity
    const supportsAdvanced = activeModel.startsWith('gpt-5') || activeModel.includes('o1') || activeModel.includes('o3');
    return {
      supportsTemperature: !supportsAdvanced, // GPT-5 models don't support temperature
      supportsReasoningEffort: supportsAdvanced,
      supportsVerbosity: supportsAdvanced,
    };
  }, [activeModel]);

  // --- Load stats from DB (conversation-scoped) ---
  const fetchStats = useCallback(async (conversationId: string | undefined) => {
    if (!conversationId) {
      setLastRequestStats(null);
      setStatsHistory([]);
      return;
    }
    try {
      const { requests } = await fetchChatStats(conversationId, 50);
      const mapItem = (r: RequestStatsItem): RequestStats => {
        const input = r.input_tokens ?? 0;
        const output = r.output_tokens ?? 0;
        const reasoning = r.reasoning_tokens ?? 0;
        const total = r.total_tokens ?? input + output + reasoning;
        return {
          inputTokens: input,
          outputTokens: output,
          reasoningTokens: reasoning,
          totalTokens: total,
          cost: r.cost_usd ?? 0,
          latencyMs: r.latency_ms ?? 0,
          ttftMs: r.ttft_ms ?? null,
          tps: r.tps ?? null,
          model: r.model,
          timestamp: new Date(r.created_at).getTime(),
        };
      };
      const history = requests.map(mapItem);
      setLastRequestStats(history[0] ?? null);
      setStatsHistory(history);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  }, []);

  // Fetch stats when pane opens or conversation changes; clear when no active chat
  useEffect(() => {
    if (!isControlPaneOpen || !isAuthenticated) return;
    if (!activeChat?.conversationId) {
      setLastRequestStats(null);
      setStatsHistory([]);
      return;
    }
    fetchStats(activeChat.conversationId);
  }, [isControlPaneOpen, isAuthenticated, activeChatId, activeChat?.conversationId, fetchStats]);

  // --- Handlers ---

  /** Update a message in the active conversation */
  const updateMessage = useCallback(
    (conversationId: string, msgId: string, updater: (msg: Message) => Message) => {
      setMessagesByConversation((prev) => {
        const messages = prev[conversationId] || [];
        return {
          ...prev,
          [conversationId]: messages.map((m) => (m.id === msgId ? updater(m) : m)),
        };
      });
    },
    []
  );

  /** Append a message to the active conversation */
  const appendMessage = useCallback(
    (conversationId: string, msg: Message) => {
      setMessagesByConversation((prev) => ({
        ...prev,
        [conversationId]: [...(prev[conversationId] || []), msg],
      }));
    },
    []
  );

  const stopTypingForText = useCallback((text?: string) => {
    if (!text || text.trim().length === 0) return;
    setIsTyping(false);
  }, []);

  const handleSendMessage = async (text: string = inputMessage) => {
    if (!text.trim() || !activeChatId || !activeChat?.conversationId) return;

    const conversationId = activeChat.conversationId;
    setInputMessage('');
    setIsTyping(true);
    setIsAwaitingResponse(true);

    const clientMsgId = crypto.randomUUID();
    const clientRequestId = crypto.randomUUID();

    const extraParams: Record<string, unknown> = {};
    if (modelCapabilities.supportsReasoningEffort && modelParams.reasoningEffort) {
      extraParams.reasoning_effort = modelParams.reasoningEffort;
    }
    if (modelCapabilities.supportsVerbosity && modelParams.verbosity) {
      extraParams.verbosity = modelParams.verbosity;
    }

    try {
      const enqueueRes = await chatEnqueue({
        conversation_id: conversationId,
        content: text,
        client_msg_id: clientMsgId,
        client_request_id: clientRequestId,
        model: activeModel,
        params: {
          temperature: modelParams.temperature,
          max_tokens: modelParams.maxTokens,
          ...extraParams,
        },
      });

      // Add user message to local state (from single persist+enqueue response)
      const userMsg: Message = {
        id: enqueueRes.user_message_id,
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };
      appendMessage(conversationId, userMsg);
      const placeholderId = enqueueRes.assistant_message_id;
      appendMessage(conversationId, {
        id: placeholderId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      });
      await chatStreamSubscribe(
        enqueueRes.request_id,
        // onDelta
        (chunk) => {
          updateMessage(conversationId, placeholderId, (m) => ({
            ...m,
            content: m.content + chunk.text,
          }));
          stopTypingForText(chunk.text);
        },
        // onCitationSpan
        (span) => {
          updateMessage(conversationId, placeholderId, (m) => ({
            ...m,
            citationSpans: [...(m.citationSpans || []), {
              start: span.start,
              end: span.end,
              refIds: span.ref_ids,
              displayLabels: span.display_labels,
            }],
          }));
        },
        // onReferences
        (refs) => {
          updateMessage(conversationId, placeholderId, (m) => ({
            ...m,
            references: refs.items.map((r) => ({
              refId: r.ref_id,
              displayLabel: r.display_label,
              chunkId: r.chunk_id,
              documentId: r.document_id,
              documentName: r.document_name,
              filename: r.filename,
              pageNumbers: r.page_numbers,
              headingPath: r.heading_path,
              snippet: r.snippet,
            })),
          }));
        },
        // onFinal
        (chunk) => {
          if (chunk.text) {
            updateMessage(conversationId, placeholderId, (m) => ({
              ...m,
              content: m.content + chunk.text,
            }));
          }
          if (chunk.stats) fetchStats(conversationId);
          setIsTyping(false);
          setIsAwaitingResponse(false);
        },
        // onError
        (error) => {
          updateMessage(conversationId, placeholderId, (m) => ({
            ...m,
            content: m.content || `Error: ${error.message}`,
          }));
          setIsTyping(false);
          setIsAwaitingResponse(false);
        }
      );
    } catch (err) {
      const error = err as ApiError;
      // Add user message (persistence failed, use temp id)
      appendMessage(conversationId, {
        id: `temp-${clientMsgId}`,
        role: 'user',
        content: text,
        timestamp: Date.now(),
      });
      appendMessage(conversationId, {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: Date.now(),
      });
      setIsTyping(false);
      setIsAwaitingResponse(false);
    }
  };

  const handleCitationClick = (citation: Citation) => {
    const doc = docs.find(d => d.id === citation.docId);
    if (!doc) return;

    // Open Evidence Panel
    setIsEvidenceOpen(true);
    setMobileTab('EVIDENCE');

    // Add tab if not exists
    setOpenPdfTabs(prev => {
        const exists = prev.find(p => p.doc.id === doc.id);
        if (exists) return prev.map(p => p.doc.id === doc.id ? { ...p, page: citation.page } : p);
        return [...prev, { doc, page: citation.page }];
    });

    setActivePdfDocId(doc.id);
    setActiveHighlight(citation.bboxHint);
  };

  const handleReferenceClick = (ref: ReferenceItem) => {
    const doc = docs.find(d => d.id === ref.documentId);
    if (!doc) return;

    setIsEvidenceOpen(true);
    setMobileTab('EVIDENCE');

    const page = ref.pageNumbers[0] || 1;
    setOpenPdfTabs(prev => {
      const exists = prev.find(p => p.doc.id === doc.id);
      if (exists) return prev.map(p => p.doc.id === doc.id ? { ...p, page } : p);
      return [...prev, { doc, page }];
    });

    setActivePdfDocId(doc.id);
    setActiveHighlight(undefined);
  };

  const handleNewChat = async () => {
    try {
      const response = await createConversation({
        title: 'New Analysis',
        settings: {},
      });
      const newChat: Chat = {
        id: Date.now().toString(),
        title: 'New Analysis',
        createdAt: Date.now(),
        conversationId: response.conversation_id,
      };
      setChats([newChat, ...chats]);
      setActiveChatId(newChat.id);
      // Initialize empty messages for this conversation
      setMessagesByConversation(prev => ({
        ...prev,
        [response.conversation_id]: [],
      }));
      if (window.innerWidth < 768) setIsSidebarOpen(false);
    } catch (error) {
      console.error('Failed to create conversation:', error);
      // Show error to user
      alert('Failed to create conversation. Please try again.');
    }
  };

  const handleDeleteRequest = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setChatToDelete(id);
  };

  const confirmDeleteChat = async () => {
    if (!chatToDelete) return;
    const chat = chats.find((c) => c.id === chatToDelete);
    if (!chat?.conversationId) {
      setChatToDelete(null);
      return;
    }
    try {
      await deleteConversation(chat.conversationId);
      setChats((prev) => prev.filter((c) => c.id !== chatToDelete));
      if (activeChatId === chatToDelete) setActiveChatId(null);
      setMessagesByConversation((prev) => {
        const next = { ...prev };
        delete next[chat.conversationId];
        return next;
      });
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      alert('Failed to delete conversation. Please try again.');
    } finally {
      setChatToDelete(null);
    }
  };

  const handleStartRename = (e: React.MouseEvent, chat: Chat) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditingTitle(chat.title);
  };

  const handleCancelRename = () => {
    setEditingChatId(null);
    setEditingTitle('');
  };

  const handleSaveRename = async (chat: Chat) => {
    if (!chat.conversationId || !editingTitle.trim()) {
      handleCancelRename();
      return;
    }

    const newTitle = editingTitle.trim();
    // Skip API call if title hasn't changed
    if (newTitle === chat.title) {
      handleCancelRename();
      return;
    }

    try {
      await updateConversation(chat.conversationId, { title: newTitle });
      setChats(prev => prev.map(c =>
        c.id === chat.id ? { ...c, title: newTitle } : c
      ));
      setEditingChatId(null);
      setEditingTitle('');
    } catch (error) {
      console.error('Failed to rename conversation:', error);
      alert('Failed to rename conversation. Please try again.');
      // Reset to original title on error
      setEditingTitle(chat.title);
    }
  };

  // --- Renderers ---

  const renderSidebar = () => (
    <div className={`
      fixed inset-y-0 left-0 z-30 w-64 bg-[var(--surface-1)] border-r border-[var(--border)] transform transition-transform duration-200 ease-in-out flex flex-col
      md:relative md:translate-x-0
      ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
    `}>
      {/* Logo */}
      <div className="h-14 flex items-center px-4 border-b border-[var(--border)] shrink-0">
         <div className="h-8 w-8 bg-[var(--accent)] rounded-lg flex items-center justify-center mr-3">
            <span className="font-bold text-white font-mono text-sm">AI</span>
         </div>
         <span className="font-semibold text-[var(--text)] tracking-tight">Financial Copilot</span>
      </div>

      {/* New Chat & Search */}
      <div className="p-4 space-y-3 shrink-0">
        <Button className="w-full justify-start gap-2" onClick={handleNewChat}>
          <Plus size={16} /> New Chat
        </Button>
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-[var(--text-faint)]" size={14} />
          <Input placeholder="Search chats..." className="pl-9" />
        </div>
      </div>

      {/* Chat List */}
      <div className="px-4 py-2 text-xs font-medium text-[var(--text-faint)] uppercase tracking-wider shrink-0">Recent Chats</div>
      <div className="flex-1 overflow-y-auto px-2 min-h-0">
        {chats.map(chat => (
          <div
            key={chat.id}
            onClick={() => { if(editingChatId !== chat.id) { setActiveChatId(chat.id); if(window.innerWidth < 768) setIsSidebarOpen(false); } }}
            className={`
              group flex items-center justify-between p-3 rounded-lg mb-1 cursor-pointer transition-colors
              ${activeChatId === chat.id
                ? 'bg-[var(--surface-3)] text-[var(--text)] border border-[var(--border)]'
                : 'text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] border border-transparent'}
            `}
          >
            {editingChatId === chat.id ? (
              <input
                type="text"
                value={editingTitle}
                onChange={(e) => setEditingTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleSaveRename(chat);
                  } else if (e.key === 'Escape') {
                    handleCancelRename();
                  }
                }}
                onBlur={() => handleSaveRename(chat)}
                autoFocus
                className="flex-1 bg-[var(--input-bg)] border border-[var(--input-border)] rounded px-2 py-1 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--input-border-focus)]"
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <div className="truncate text-sm font-medium pr-2 flex-1">
                  {chat.title}
                  <div className="text-[10px] text-[var(--text-faint)] font-mono mt-0.5">
                      {new Date(chat.createdAt).toLocaleDateString()}
                  </div>
              </div>
            )}
            <div className="flex items-center gap-1">
              <button
                  onClick={(e) => handleStartRename(e, chat)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-[var(--accent)] transition-opacity"
              >
                  <Pencil size={12} />
              </button>
              <button
                  onClick={(e) => handleDeleteRequest(e, chat.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-[var(--danger)] transition-opacity"
              >
                  <Trash2 size={12} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* User Profile */}
      <div className="p-4 border-t border-[var(--border)] mt-auto shrink-0 bg-[var(--surface-1)]">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-[var(--surface-2)] flex items-center justify-center border border-[var(--border)]">
            <User size={14} className="text-[var(--text-muted)]" />
          </div>
          <div className="flex-1">
             <div className="text-sm font-medium text-[var(--text)] truncate">
              {currentUser?.display_name ?? currentUser?.email ?? 'User'}
            </div>
            <div className="text-xs text-[var(--text-faint)] truncate">
              {currentUser?.display_name ? currentUser.email : (currentUser ? 'Pro Plan' : null)}
            </div>
          </div>
          <button
            onClick={() => logout()}
            className="p-1.5 text-[var(--icon)] hover:text-[var(--accent)] hover:bg-[var(--surface-2)] rounded-md transition-colors"
            title="Log out"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </div>
  );

  const renderChatArea = () => {
    // 1. No active chat selected
    if (!activeChatId) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <div className="w-16 h-16 bg-[var(--surface-2)] rounded-2xl flex items-center justify-center mb-6 border border-[var(--border)]">
                    <Bot className="text-[var(--text-faint)]" size={32} />
                </div>
                <h2 className="text-xl font-semibold text-[var(--text)] mb-2">AI Financial Copilot</h2>
                <p className="text-[var(--text-muted)] max-w-md mb-8">
                   Select a conversation from the sidebar or start a new chat.
                </p>
                <Button onClick={handleNewChat}>
                    <Plus size={16} className="mr-2" /> Start New Chat
                </Button>
            </div>
        );
    }

    // 2. Chat selected
    const isChatEmpty = activeMessages.length === 0;

    return (
      <div className="flex-1 flex flex-col min-w-0 h-full">
        {/* Scope Bar */}
        <ScopeBar
            scope={scope}
            docCount={activeDocsCount}
            onModeChange={(m) => setScope(s => ({ ...s, mode: m }))}
            onFilterChange={(f) => setScope(s => ({ ...s, filters: { ...s.filters, ...f } }))}
            onAddFiles={() => setIsDocPickerOpen(true)}
        />

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto min-h-0 p-4 md:p-8 pb-32 md:pb-8">
          {isChatEmpty ? (
            <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
                 <div className="w-16 h-16 bg-[var(--surface-2)] rounded-2xl flex items-center justify-center mb-6 border border-[var(--border)]">
                    <Bot className="text-[var(--text-faint)]" size={32} />
                </div>
                <h2 className="text-xl font-semibold text-[var(--text)] mb-2">AI Financial Copilot</h2>
                <p className="text-[var(--text-muted)] max-w-md mb-8 leading-relaxed">
                   I can help you analyze financial documents, extract key metrics, and compare company performance.
                </p>
            </div>
          ) : (
            <div className="space-y-6">
              {messagesHasMore[activeChat.conversationId] && (
                <div className="text-center text-sm text-[var(--text-faint)] py-2">
                  {messagesLoading[activeChat.conversationId] ? 'Loading older messages...' : 'Scroll up to load more'}
                </div>
              )}
              {activeMessages.map((msg) => {
                // Skip rendering assistant messages with empty content (placeholder while typing)
                if (msg.role === 'assistant' && !msg.content) return null;

                return (
                  <div key={msg.id} className={`flex gap-4 max-w-3xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''}`}>
                    {msg.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-lg bg-[var(--accent-subtle)] border border-[var(--accent)] border-opacity-20 flex items-center justify-center flex-shrink-0 mt-1">
                        <Bot size={18} className="text-[var(--accent)]" />
                      </div>
                    )}

                    <div className={`space-y-2 ${msg.role === 'user' ? 'flex flex-col items-end ml-[15%]' : 'flex-1'}`}>
                      <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-[var(--bubble-user-bg)] text-[var(--text)] rounded-br-md max-w-[85%]'
                          : 'bg-[var(--bubble-assistant-bg)] text-[var(--text)] rounded-bl-md'
                      }`} style={{ width: 'fit-content', maxWidth: '100%' }}>
                          {msg.citationSpans && msg.citationSpans.length > 0 ? (
                            <CitedText
                              text={msg.content}
                              spans={msg.citationSpans}
                              references={msg.references || []}
                              onCitationClick={handleReferenceClick}
                            />
                          ) : (
                            <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                          )}
                      </div>

                      {/* Evidence section (new structured references) */}
                      {msg.references && msg.references.length > 0 && (
                        <EvidenceList
                          references={msg.references}
                          onRefClick={handleReferenceClick}
                        />
                      )}
                      {/* Legacy citations grid (fallback for old messages) */}
                      {!msg.references?.length && msg.citations && msg.citations.length > 0 && (
                          <div className={`flex flex-wrap gap-2 mt-2 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                              {msg.citations.map((c, i) => {
                                  const doc = docs.find(d => d.id === c.docId);
                                  return (
                                      <button
                                          key={i}
                                          onClick={() => handleCitationClick(c)}
                                          className="flex items-center gap-2 bg-[var(--surface-1)] border border-[var(--border)] hover:border-[var(--accent)] hover:bg-[var(--surface-2)] px-3 py-2 rounded-lg text-left transition-all group max-w-xs"
                                      >
                                          <div className="h-8 w-8 bg-[var(--bg)] flex items-center justify-center rounded-md text-[var(--text-faint)] group-hover:text-[var(--accent)] border border-[var(--border)]">
                                              <span className="font-mono text-xs font-bold">{i+1}</span>
                                          </div>
                                          <div className="min-w-0">
                                              <div className="text-xs font-medium text-[var(--text)] truncate">{doc?.company}</div>
                                              <div className="text-[10px] text-[var(--text-faint)] font-mono truncate">Page {c.page} • {doc?.type}</div>
                                          </div>
                                      </button>
                                  );
                              })}
                          </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {isTyping && (
                <div className="flex gap-4 max-w-3xl mx-auto">
                    <div className="w-8 h-8 rounded-lg bg-[var(--accent-subtle)] border border-[var(--accent)] border-opacity-20 flex items-center justify-center flex-shrink-0">
                        <Bot size={18} className="text-[var(--accent)] animate-pulse" />
                    </div>
                    <div className="flex items-center gap-1 h-8">
                        <span className="w-2 h-2 bg-[var(--text-faint)] rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                        <span className="w-2 h-2 bg-[var(--text-faint)] rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                        <span className="w-2 h-2 bg-[var(--text-faint)] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="p-4 bg-[var(--bg)]">
          <div className="max-w-3xl mx-auto relative">
             <div className="absolute -top-8 left-0 flex items-center gap-2">
                <div className="relative">
                  <select
                    value={activeModel}
                    onChange={(e) => setActiveModel(e.target.value)}
                    className="appearance-none bg-[var(--surface-2)] border border-[var(--border)] text-xs font-medium text-[var(--text)] rounded-md px-3 py-1.5 pr-8 outline-none focus:border-[var(--accent)] hover:bg-[var(--surface-3)] cursor-pointer transition-colors"
                  >
                    {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                  <ChevronDown className="absolute right-2 top-1.5 text-[var(--text-faint)] pointer-events-none" size={12} />
                </div>
             </div>
             <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                placeholder="Ask me anything..."
                className="w-full bg-[var(--input-bg)] border border-[var(--input-border)] rounded-xl p-4 pr-14 text-sm text-[var(--text)] placeholder:text-[var(--placeholder)] focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)] min-h-[96px] resize-none font-sans shadow-sm transition-colors"
             />
             <Button
                size="icon"
                className={`absolute right-3 bottom-3 transition-all ${
                  inputMessage.trim()
                    ? 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]'
                    : 'bg-[var(--surface-3)] text-[var(--text-muted)]'
                }`}
                onClick={() => handleSendMessage()}
                disabled={!inputMessage.trim() || isAwaitingResponse}
             >
                <Send size={16} />
             </Button>
          </div>
          <div className="text-center mt-2 text-[10px] text-[var(--text-faint)]">
            AI can make mistakes. Verify with citations.
          </div>
        </div>
      </div>
    );
  };

  const renderLibrary = () => (
    <div className="flex-1 overflow-auto bg-[var(--bg)] p-6 md:p-10 animate-fade-in pb-20 md:pb-10">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
            <div>
                <h1 className="text-2xl font-semibold text-[var(--text)] mb-2">Document Library</h1>
                <p className="text-[var(--text-muted)] text-sm">Manage and analyze your financial repository.</p>
            </div>
            <Button onClick={() => setIsUploadOpen(true)} className="gap-2">
                <Plus size={16} /> Upload PDFs
            </Button>
        </div>

        {/* Filters */}
        {(() => {
          const companies = Array.from(new Set(docs.map(d => d.company).filter(Boolean))).sort();
          const years = Array.from(new Set(docs.map(d => d.year).filter(y => y > 0))).sort((a, b) => Number(b) - Number(a));
          return (
        <div className="flex gap-4 mb-6">
            <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-2.5 text-[var(--text-faint)]" size={14} />
                <Input placeholder="Search documents..." className="pl-9" />
            </div>
            <select className="appearance-none bg-[var(--input-bg)] border border-[var(--input-border)] rounded-md px-3 pr-8 text-sm text-[var(--text)] focus:border-[var(--input-border-focus)] outline-none cursor-pointer">
                <option>All Companies</option>
                {companies.map(c => <option key={c}>{c}</option>)}
            </select>
             <select className="appearance-none bg-[var(--input-bg)] border border-[var(--input-border)] rounded-md px-3 pr-8 text-sm text-[var(--text)] focus:border-[var(--input-border-focus)] outline-none cursor-pointer">
                <option>All Years</option>
                {years.map(y => <option key={y}>{y}</option>)}
            </select>
        </div>
          );
        })()}

        {/* Table */}
        <div className="border border-[var(--border)] rounded-lg overflow-hidden bg-[var(--surface-1)]">
            <table className="w-full text-left text-sm">
                <thead className="bg-[var(--surface-2)] border-b border-[var(--border)] text-[var(--text-muted)] text-xs uppercase tracking-wider">
                    <tr>
                        <th className="px-6 py-4 font-medium">Status</th>
                        <th className="px-6 py-4 font-medium">Document Name</th>
                        <th className="px-6 py-4 font-medium">Company</th>
                        <th className="px-6 py-4 font-medium">Year</th>
                        <th className="px-6 py-4 font-medium">Type</th>
                        <th className="px-6 py-4 font-medium">Pages</th>
                        <th className="px-6 py-4 font-medium text-right">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                    {docs.map(doc => (
                        <tr key={doc.id} className="hover:bg-[var(--surface-2)] transition-colors group cursor-pointer" onClick={() => {
                            // Open in side panel
                            setView('ASK');
                            handleCitationClick({ docId: doc.id, page: 1, excerpt: '' });
                            handleNewChat();
                            setScope({ mode: 'thisDoc', docIds: [doc.id], filters: {} });
                        }}>
                            <td className="px-6 py-4">
                                <Badge variant={doc.status === 'Ready' ? 'success' : 'warning'}>{doc.status}</Badge>
                            </td>
                            <td className="px-6 py-4 font-medium text-[var(--text)] flex items-center gap-2">
                                <FileText size={16} className="text-[var(--text-faint)]" />
                                {doc.title}
                            </td>
                            <td className="px-6 py-4 text-[var(--text-muted)]">{doc.company}</td>
                            <td className="px-6 py-4 font-mono text-[var(--text-faint)]">{doc.year}</td>
                            <td className="px-6 py-4 text-[var(--text-muted)]">{doc.type}</td>
                            <td className="px-6 py-4 font-mono text-[var(--text-faint)]">{doc.pages}</td>
                            <td className="px-6 py-4 text-right">
                                <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100">
                                    <MoreHorizontal size={16} />
                                </Button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
      </div>
    </div>
  );

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
        <Loader2 className="w-8 h-8 animate-spin text-[var(--accent)]" />
      </div>
    );
  }
  if (!isAuthenticated) {
    return <LoginPage onSuccess={(token) => setAccessToken(token)} />;
  }

  return (
    <div className="flex flex-col h-screen bg-[var(--bg)] text-[var(--text)] overflow-hidden font-sans">
      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUpload={() => { void refreshDocs(); }}
      />

      {/* Doc Picker Modal */}
      <DocPickerModal
        isOpen={isDocPickerOpen}
        onClose={() => setIsDocPickerOpen(false)}
        docs={docs}
        selectedIds={scope.docIds}
        onConfirm={(ids) => {
            setScope(prev => ({ ...prev, mode: 'selectedDocs', docIds: ids }));
            setIsDocPickerOpen(false);
        }}
      />

      {/* Delete Confirmation Modal */}
      {chatToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <Card className="w-full max-w-sm p-6 shadow-xl" variant="elevated">
             <div className="flex flex-col items-center text-center gap-4">
                <div className="w-12 h-12 bg-[var(--danger-bg)] rounded-full flex items-center justify-center text-[var(--danger)]">
                    <AlertTriangle size={24} />
                </div>
                <div>
                    <h3 className="text-lg font-semibold text-[var(--text)]">Delete Chat?</h3>
                    <p className="text-sm text-[var(--text-muted)] mt-1">This action cannot be undone.</p>
                </div>
                <div className="flex w-full gap-2 mt-2">
                    <Button variant="secondary" className="flex-1" onClick={() => setChatToDelete(null)}>Cancel</Button>
                    <Button variant="danger" className="flex-1" onClick={confirmDeleteChat}>Delete</Button>
                </div>
             </div>
          </Card>
        </div>
      )}

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar (Only in ASK view) */}
        {view === 'ASK' && renderSidebar()}

        {/* Content Overlay/Mask for mobile sidebar */}
        {isSidebarOpen && window.innerWidth < 768 && (
             <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={() => setIsSidebarOpen(false)} />
        )}

        {/* Center Workspace */}
        <div className="flex-1 flex flex-col min-w-0 bg-[var(--bg)] relative">

          {/* Header */}
          <header className="h-14 border-b border-[var(--border)] flex items-center justify-between px-4 bg-[var(--surface-1)] z-10">
            <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                    <Layers size={18} />
                </Button>
                {/* View Switcher */}
                <div className="flex bg-[var(--surface-2)] p-1 rounded-lg border border-[var(--border)]">
                    <button
                        onClick={() => setView('ASK')}
                        className={`px-4 py-1.5 text-xs font-semibold rounded-md transition-all ${
                          view === 'ASK'
                            ? 'bg-[var(--surface-3)] text-[var(--text)] shadow-sm'
                            : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                        }`}
                    >
                        ASK
                    </button>
                    <button
                        onClick={() => setView('LIBRARY')}
                        className={`px-4 py-1.5 text-xs font-semibold rounded-md transition-all ${
                          view === 'LIBRARY'
                            ? 'bg-[var(--surface-3)] text-[var(--text)] shadow-sm'
                            : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                        }`}
                    >
                        LIBRARY
                    </button>
                </div>
            </div>

            <div className="flex items-center gap-3">
                {/* Control Pane Toggle */}
                <button
                  onClick={() => setIsControlPaneOpen(!isControlPaneOpen)}
                  className={`p-2 rounded-lg transition-colors ${
                    isControlPaneOpen
                      ? 'text-[var(--accent)] bg-[var(--accent-subtle)]'
                      : 'text-[var(--icon)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]'
                  }`}
                  title="Control Pane"
                >
                  <Settings2 size={18} />
                </button>
                {/* Theme Toggle */}
                <button
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="p-2 rounded-lg text-[var(--icon)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
                  title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
                </button>
                <Badge variant="outline" className="h-7 px-3">v2.4.0-stable</Badge>
            </div>
          </header>

          {/* Main Views */}
          {view === 'LIBRARY' ? renderLibrary() : (
             // Ask View Container
             <div className="flex-1 flex overflow-hidden">
                {/* Mobile Tab Switch Logic */}
                <div className={`flex-1 flex flex-col ${window.innerWidth < 768 && mobileTab === 'EVIDENCE' ? 'hidden' : 'flex'}`}>
                    {renderChatArea()}
                </div>

                {/* Evidence Panel (Desktop: Side, Mobile: Full via Tab) */}
                <EvidencePanel
                    isOpen={isEvidenceOpen || (window.innerWidth < 768 && mobileTab === 'EVIDENCE')}
                    onClose={() => { setIsEvidenceOpen(false); setMobileTab('CONVERSATION'); }}
                    openDocs={openPdfTabs}
                    activeDocId={activePdfDocId}
                    onSwitchDoc={setActivePdfDocId}
                    onCloseDoc={(id) => {
                        const newTabs = openPdfTabs.filter(t => t.doc.id !== id);
                        setOpenPdfTabs(newTabs);
                        if (newTabs.length === 0) { setIsEvidenceOpen(false); setMobileTab('CONVERSATION'); }
                        else if (activePdfDocId === id) setActivePdfDocId(newTabs[0].doc.id);
                    }}
                    highlight={activeHighlight}
                />

                {/* Control Pane */}
                <ControlPane
                    isOpen={isControlPaneOpen}
                    onToggle={() => setIsControlPaneOpen(!isControlPaneOpen)}
                    params={modelParams}
                    onParamsChange={(p) => setModelParams(prev => ({ ...prev, ...p }))}
                    capabilities={modelCapabilities}
                    stats={lastRequestStats}
                    statsHistory={statsHistory}
                />
             </div>
          )}
        </div>
      </div>

      {/* Mobile Bottom Nav (Only in ASK View, Always visible) */}
      {view === 'ASK' && (
        <div className="md:hidden h-14 border-t border-[var(--border)] bg-[var(--surface-1)] flex grid grid-cols-2 fixed bottom-0 left-0 right-0 z-50 shadow-lg">
            <button
                onClick={() => { setMobileTab('CONVERSATION'); setIsEvidenceOpen(false); }}
                className={`flex flex-col items-center justify-center gap-1 transition-colors ${
                  mobileTab === 'CONVERSATION'
                    ? 'text-[var(--accent)] bg-[var(--accent-subtle)]'
                    : 'text-[var(--text-muted)]'
                }`}
            >
                <MessageSquare size={18} />
                <span className="text-[10px] font-semibold">CHAT</span>
            </button>
            <button
                onClick={() => { setMobileTab('EVIDENCE'); setIsEvidenceOpen(true); }}
                className={`flex flex-col items-center justify-center gap-1 transition-colors ${
                  mobileTab === 'EVIDENCE'
                    ? 'text-[var(--accent)] bg-[var(--accent-subtle)]'
                    : 'text-[var(--text-muted)]'
                }`}
            >
                <BookOpen size={18} />
                <span className="text-[10px] font-semibold">EVIDENCE</span>
            </button>
        </div>
      )}
    </div>
  );
}
