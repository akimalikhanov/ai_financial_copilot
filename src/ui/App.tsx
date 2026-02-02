import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  MessageSquare, BookOpen, Plus, Send, Search, 
  Settings, User, FileText, MoreHorizontal, Trash2, ArrowRight, Layers, AlertTriangle, ChevronDown, Bot
} from 'lucide-react';
import { Document, Chat, Message, Scope, ViewMode, MobileTab, Citation } from './types';
import { MOCK_DOCS, MOCK_CHATS, COMPANIES, YEARS } from './services/mockData';
import {
  chat as apiChat,
  chatStream as apiChatStream,
  ChatRequest,
  ChatMessage as ApiChatMessage,
  ApiError,
} from './services/api';
import { Button, Input, Badge, Card } from './components/ui';
import { ScopeBar } from './components/ScopeBar';
import { EvidencePanel } from './components/EvidencePanel';
import { UploadModal } from './components/UploadModal';
import { DocPickerModal } from './components/DocPickerModal';

// --- Types ---
const MODELS = [
  { id: 'gpt-4o-mini', name: 'GPT-4o-mini' },
  { id: 'gpt-4o', name: 'GPT-4o' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus' },
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
];

// --- Helpers ---
const generateId = () => Math.random().toString(36).slice(2, 11);

/** Build scope context string for system message */
const buildScopeContext = (scope: Scope, docs: Document[]): string => {
  let pool = docs;
  if (scope.mode === 'filteredByMetadata') {
    pool = docs.filter((d) => {
      const f = scope.filters;
      const matchCompany = !f.company?.length || f.company.includes(d.company);
      const matchYear = !f.year?.length || f.year.includes(d.year);
      const matchType = !f.type?.length || f.type.includes(d.type);
      return matchCompany && matchYear && matchType;
    });
  } else if (scope.mode === 'selectedDocs' || scope.mode === 'thisDoc') {
    pool = docs.filter((d) => scope.docIds.includes(d.id));
  }
  if (pool.length === 0) {
    return 'No documents are currently in scope.';
  }
  const docList = pool.map((d) => `- ${d.title} (${d.company}, ${d.year}, ${d.type})`).join('\n');
  return `The user has ${pool.length} document(s) in scope:\n${docList}`;
};

/** Convert UI messages to API messages, optionally prepending a system message */
const toApiMessages = (
  messages: Message[],
  systemPrompt?: string
): ApiChatMessage[] => {
  const apiMessages: ApiChatMessage[] = [];
  if (systemPrompt) {
    apiMessages.push({ role: 'system', content: systemPrompt });
  }
  for (const m of messages) {
    apiMessages.push({ role: m.role, content: m.content });
  }
  return apiMessages;
};

export default function App() {
  // --- Global State ---
  const [view, setView] = useState<ViewMode>('ASK');
  const [mobileTab, setMobileTab] = useState<MobileTab>('CONVERSATION');
  
  // Data
  const [docs, setDocs] = useState<Document[]>(MOCK_DOCS);
  const [chats, setChats] = useState<Chat[]>(MOCK_CHATS);
  
  // Active Context
  const [activeChatId, setActiveChatId] = useState<string | null>(MOCK_CHATS[0].id);
  const [activeModel, setActiveModel] = useState(MODELS[0].id);
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
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isDocPickerOpen, setIsDocPickerOpen] = useState(false);
  
  // Delete Confirmation State
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- Derived State ---
  const activeChat = chats.find(c => c.id === activeChatId);
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
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeChat?.messages, isTyping]);

  // --- Streaming Mode Toggle ---
  const [useStreaming, setUseStreaming] = useState(true);

  // --- Handlers ---

  /** Update a message in the active chat by id */
  const updateMessageInChat = useCallback(
    (msgId: string, updater: (msg: Message) => Message) => {
      setChats((prev) =>
        prev.map((c) =>
          c.id === activeChatId
            ? { ...c, messages: c.messages.map((m) => (m.id === msgId ? updater(m) : m)) }
            : c
        )
      );
    },
    [activeChatId]
  );

  /** Append a message to the active chat */
  const appendMessageToChat = useCallback(
    (msg: Message) => {
      setChats((prev) =>
        prev.map((c) => (c.id === activeChatId ? { ...c, messages: [...c.messages, msg] } : c))
      );
    },
    [activeChatId]
  );

  const handleSendMessage = async (text: string = inputMessage) => {
    if (!text.trim() || !activeChatId) return;

    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    // Optimistic update: add user message
    const currentChat = chats.find((c) => c.id === activeChatId);
    const messagesWithUser = currentChat ? [...currentChat.messages, userMsg] : [userMsg];
    setChats((prev) =>
      prev.map((c) => (c.id === activeChatId ? { ...c, messages: messagesWithUser } : c))
    );
    setInputMessage('');
    setIsTyping(true);

    // Build request
    const systemPrompt = buildScopeContext(scope, docs);
    const apiMessages = toApiMessages(messagesWithUser, systemPrompt);
    const request: ChatRequest = {
      messages: apiMessages,
      model: activeModel,
    };

    if (useStreaming) {
      // --- Streaming path ---
      const assistantMsgId = generateId();
      const placeholderMsg: Message = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      };
      appendMessageToChat(placeholderMsg);

      await apiChatStream(
        request,
        // onDelta
        (chunk) => {
          updateMessageInChat(assistantMsgId, (m) => ({
            ...m,
            content: m.content + chunk.text,
          }));
        },
        // onFinal
        (chunk) => {
          // Append any remaining text from final chunk
          if (chunk.text) {
            updateMessageInChat(assistantMsgId, (m) => ({
              ...m,
              content: m.content + chunk.text,
            }));
          }
          setIsTyping(false);
        },
        // onError
        (error: ApiError) => {
          updateMessageInChat(assistantMsgId, (m) => ({
            ...m,
            content: m.content || `Error: ${error.message}`,
          }));
          setIsTyping(false);
        }
      );
    } else {
      // --- Non-streaming path ---
      try {
        const response = await apiChat(request);
        const assistantMsg: Message = {
          id: generateId(),
          role: 'assistant',
          content: response.text,
          timestamp: Date.now(),
        };
        appendMessageToChat(assistantMsg);
      } catch (err) {
        const error = err as ApiError;
        const errorMsg: Message = {
          id: generateId(),
          role: 'assistant',
          content: `Error: ${error.message ?? 'Request failed'}`,
          timestamp: Date.now(),
        };
        appendMessageToChat(errorMsg);
      } finally {
        setIsTyping(false);
      }
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

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: 'New Analysis',
      createdAt: Date.now(),
      messages: []
    };
    setChats([newChat, ...chats]);
    setActiveChatId(newChat.id);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const handleDeleteRequest = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setChatToDelete(id);
  };

  const confirmDeleteChat = () => {
    if (chatToDelete) {
        setChats(prev => prev.filter(c => c.id !== chatToDelete));
        if (activeChatId === chatToDelete) setActiveChatId(null);
        setChatToDelete(null);
    }
  };

  // --- Renderers ---

  const renderSidebar = () => (
    <div className={`
      fixed inset-y-0 left-0 z-30 w-64 bg-zinc-950 border-r border-zinc-800 transform transition-transform duration-200 ease-in-out flex flex-col
      md:relative md:translate-x-0
      ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
    `}>
      <div className="h-14 flex items-center px-4 border-b border-zinc-800 shrink-0">
         <div className="h-8 w-8 bg-accent-600 rounded-sm flex items-center justify-center mr-3 shadow-[0_0_15px_-3px_rgba(2,132,199,0.5)]">
            <span className="font-bold text-white font-mono">AI</span>
         </div>
         <span className="font-bold text-zinc-100 font-mono tracking-tight">FIN.COPILOT</span>
      </div>

      <div className="p-4 space-y-4 shrink-0">
        <Button className="w-full justify-start gap-2" onClick={handleNewChat}>
          <Plus size={16} /> New Chat
        </Button>
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-zinc-500" size={14} />
          <Input placeholder="Search chats..." className="pl-9" />
        </div>
      </div>

      <div className="px-4 py-2 text-xs font-mono text-zinc-500 uppercase tracking-widest shrink-0">Recent Chats</div>
      <div className="flex-1 overflow-y-auto px-2 min-h-0">
        {chats.map(chat => (
          <div
            key={chat.id}
            onClick={() => { setActiveChatId(chat.id); if(window.innerWidth < 768) setIsSidebarOpen(false); }}
            className={`
              group flex items-center justify-between p-3 rounded-sm mb-1 cursor-pointer transition-colors
              ${activeChatId === chat.id ? 'bg-zinc-800/50 text-zinc-100 border border-zinc-700/50' : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 border border-transparent'}
            `}
          >
            <div className="truncate text-sm font-medium pr-2">
                {chat.title}
                <div className="text-[10px] opacity-60 font-mono mt-0.5">
                    {new Date(chat.createdAt).toLocaleDateString()}
                </div>
            </div>
            <button 
                onClick={(e) => handleDeleteRequest(e, chat.id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
            >
                <Trash2 size={12} />
            </button>
          </div>
        ))}
      </div>

      <div className="p-4 border-t border-zinc-800 mt-auto shrink-0 bg-zinc-950">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-zinc-800 flex items-center justify-center border border-zinc-700">
            <User size={14} className="text-zinc-400" />
          </div>
          <div className="flex-1">
             <div className="text-sm font-medium text-zinc-200">Analyst User</div>
             <div className="text-xs text-zinc-500">Pro Plan</div>
          </div>
          <Settings size={16} className="text-zinc-500 hover:text-zinc-300 cursor-pointer" />
        </div>
      </div>
    </div>
  );

  const renderChatArea = () => {
    // 1. No active chat selected
    if (!activeChatId) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <div className="w-16 h-16 bg-zinc-900 rounded-2xl flex items-center justify-center mb-6 border border-zinc-800">
                    <Bot className="text-zinc-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-zinc-100 mb-2">AI Financial Copilot</h2>
                <p className="text-zinc-500 max-w-md mb-8">
                   Select a conversation from the sidebar or start a new chat.
                </p>
                <Button onClick={handleNewChat}>
                    <Plus size={16} className="mr-2" /> Start New Chat
                </Button>
            </div>
        );
    }

    // 2. Chat selected
    const isChatEmpty = activeChat && activeChat.messages.length === 0;

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
        <div className="flex-1 overflow-y-auto p-4 md:p-8 pb-32 md:pb-8">
          {isChatEmpty ? (
            <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
                 <div className="w-16 h-16 bg-zinc-900 rounded-2xl flex items-center justify-center mb-6 border border-zinc-800">
                    <Bot className="text-zinc-600" size={32} />
                </div>
                <h2 className="text-xl font-bold text-zinc-100 mb-2">AI Financial Copilot</h2>
                <p className="text-zinc-500 max-w-md mb-8 leading-relaxed">
                   I can help you analyze financial documents, extract key metrics, and compare company performance.
                </p>
            </div>
          ) : (
            <div className="space-y-8">
              {activeChat?.messages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 max-w-3xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''}`}>
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-sm bg-accent-600/10 border border-accent-600/20 flex items-center justify-center flex-shrink-0 mt-1">
                      <Bot size={18} className="text-accent-500" />
                    </div>
                  )}
                  
                  <div className={`space-y-2 ${msg.role === 'user' ? 'text-right' : 'flex-1'}`}>
                    <div className={`inline-block p-4 rounded-lg text-sm leading-relaxed ${msg.role === 'user' ? 'bg-zinc-800 text-zinc-100 rounded-br-none' : 'text-zinc-300'}`}>
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                    
                    {/* Citations Grid */}
                    {msg.citations && msg.citations.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-2">
                            {msg.citations.map((c, i) => {
                                const doc = docs.find(d => d.id === c.docId);
                                return (
                                    <button 
                                        key={i}
                                        onClick={() => handleCitationClick(c)}
                                        className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 hover:border-accent-500/50 hover:bg-zinc-800 px-3 py-2 rounded-sm text-left transition-all group max-w-xs"
                                    >
                                        <div className="h-8 w-8 bg-zinc-950 flex items-center justify-center rounded-sm text-zinc-500 group-hover:text-accent-400 border border-zinc-800">
                                            <span className="font-mono text-xs font-bold">{i+1}</span>
                                        </div>
                                        <div className="min-w-0">
                                            <div className="text-xs font-bold text-zinc-300 truncate">{doc?.company}</div>
                                            <div className="text-[10px] text-zinc-500 font-mono truncate">Page {c.page} • {doc?.type}</div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex gap-4 max-w-3xl mx-auto">
                    <div className="w-8 h-8 rounded-sm bg-accent-600/10 border border-accent-600/20 flex items-center justify-center flex-shrink-0">
                        <Bot size={18} className="text-accent-500 animate-pulse" />
                    </div>
                    <div className="flex items-center gap-1 h-8">
                        <span className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                        <span className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                        <span className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="p-4 border-t border-zinc-800 bg-zinc-950">
          <div className="max-w-3xl mx-auto relative">
             <div className="absolute -top-8 left-0 flex items-center gap-2">
                <div className="relative">
                  <select 
                    value={activeModel}
                    onChange={(e) => setActiveModel(e.target.value)}
                    className="appearance-none bg-zinc-900 border border-zinc-800 text-xs font-medium text-zinc-300 rounded-sm px-3 py-1 pr-8 outline-none focus:border-accent-500 hover:bg-zinc-800 cursor-pointer"
                  >
                    {MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                  <ChevronDown className="absolute right-2 top-1.5 text-zinc-500 pointer-events-none" size={12} />
                </div>
             </div>
             <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                placeholder="Ask me anything..."
                className="w-full bg-zinc-900 border border-zinc-800 rounded-md p-4 pr-12 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-accent-600 focus:ring-1 focus:ring-accent-600 min-h-[60px] resize-none font-sans shadow-lg"
             />
             <Button 
                size="icon" 
                className={`absolute right-3 bottom-3 transition-all ${inputMessage.trim() ? 'bg-accent-600 text-white hover:bg-accent-500' : 'bg-zinc-800 text-zinc-500'}`}
                onClick={() => handleSendMessage()}
                disabled={!inputMessage.trim() || isTyping}
             >
                <Send size={16} />
             </Button>
          </div>
          <div className="text-center mt-2 text-[10px] text-zinc-600 font-mono">
            AI can make mistakes. Verify with citations.
          </div>
        </div>
      </div>
    );
  };

  const renderLibrary = () => (
    <div className="flex-1 overflow-auto bg-zinc-950 p-6 md:p-10 animate-fade-in pb-20 md:pb-10">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
            <div>
                <h1 className="text-2xl font-bold text-zinc-100 font-mono mb-2">DOCUMENT LIBRARY</h1>
                <p className="text-zinc-500 text-sm">Manage and analyze your financial repository.</p>
            </div>
            <Button onClick={() => setIsUploadOpen(true)} className="gap-2 bg-accent-600 hover:bg-accent-500 text-white border-0">
                <Plus size={16} /> Upload PDFs
            </Button>
        </div>

        {/* Filters */}
        <div className="flex gap-4 mb-6">
            <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-2.5 text-zinc-500" size={14} />
                <Input placeholder="Search documents..." className="pl-9 bg-zinc-900" />
            </div>
            <select className="bg-zinc-900 border border-zinc-800 rounded-sm px-3 text-sm text-zinc-300">
                <option>All Companies</option>
                {COMPANIES.map(c => <option key={c}>{c}</option>)}
            </select>
             <select className="bg-zinc-900 border border-zinc-800 rounded-sm px-3 text-sm text-zinc-300">
                <option>All Years</option>
                {YEARS.map(y => <option key={y}>{y}</option>)}
            </select>
        </div>

        {/* Table */}
        <div className="border border-zinc-800 rounded-sm overflow-hidden bg-zinc-900/20">
            <table className="w-full text-left text-sm">
                <thead className="bg-zinc-900 border-b border-zinc-800 text-zinc-500 font-mono text-xs uppercase tracking-wider">
                    <tr>
                        <th className="px-6 py-4">Status</th>
                        <th className="px-6 py-4">Document Name</th>
                        <th className="px-6 py-4">Company</th>
                        <th className="px-6 py-4">Year</th>
                        <th className="px-6 py-4">Type</th>
                        <th className="px-6 py-4">Pages</th>
                        <th className="px-6 py-4 text-right">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                    {docs.map(doc => (
                        <tr key={doc.id} className="hover:bg-zinc-900/50 transition-colors group cursor-pointer" onClick={() => {
                            // Open in side panel
                            setView('ASK');
                            handleCitationClick({ docId: doc.id, page: 1, excerpt: '' });
                            handleNewChat();
                            setScope({ mode: 'thisDoc', docIds: [doc.id], filters: {} });
                        }}>
                            <td className="px-6 py-4">
                                <Badge variant={doc.status === 'Ready' ? 'success' : 'warning'}>{doc.status}</Badge>
                            </td>
                            <td className="px-6 py-4 font-medium text-zinc-200 flex items-center gap-2">
                                <FileText size={16} className="text-zinc-500" />
                                {doc.title}
                            </td>
                            <td className="px-6 py-4 text-zinc-400">{doc.company}</td>
                            <td className="px-6 py-4 font-mono text-zinc-500">{doc.year}</td>
                            <td className="px-6 py-4 text-zinc-400">{doc.type}</td>
                            <td className="px-6 py-4 font-mono text-zinc-500">{doc.pages}</td>
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

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 overflow-hidden font-sans">
      {/* Upload Modal */}
      <UploadModal 
        isOpen={isUploadOpen} 
        onClose={() => setIsUploadOpen(false)} 
        onUpload={(meta) => setDocs(prev => [{ id: Date.now().toString(), pages: 0, status: 'Processing', tags: [], ...meta }, ...prev])}
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/80 backdrop-blur-sm p-4">
          <Card className="w-full max-w-sm bg-zinc-900 border-zinc-800 p-6 shadow-2xl">
             <div className="flex flex-col items-center text-center gap-4">
                <div className="w-12 h-12 bg-red-900/20 rounded-full flex items-center justify-center text-red-500">
                    <AlertTriangle size={24} />
                </div>
                <div>
                    <h3 className="text-lg font-bold text-zinc-100">Delete Chat?</h3>
                    <p className="text-sm text-zinc-500 mt-1">This action cannot be undone.</p>
                </div>
                <div className="flex w-full gap-2 mt-2">
                    <Button variant="ghost" className="flex-1" onClick={() => setChatToDelete(null)}>Cancel</Button>
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
        <div className="flex-1 flex flex-col min-w-0 bg-zinc-950 relative">
          
          {/* Header */}
          <header className="h-14 border-b border-zinc-800 flex items-center justify-between px-4 bg-zinc-950/80 backdrop-blur-md z-10">
            <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                    <Layers size={18} />
                </Button>
                {/* View Switcher */}
                <div className="flex bg-zinc-900 p-0.5 rounded-sm border border-zinc-800">
                    <button 
                        onClick={() => setView('ASK')} 
                        className={`px-4 py-1.5 text-xs font-bold rounded-sm transition-all ${view === 'ASK' ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                        ASK
                    </button>
                    <button 
                        onClick={() => setView('LIBRARY')} 
                        className={`px-4 py-1.5 text-xs font-bold rounded-sm transition-all ${view === 'LIBRARY' ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                        LIBRARY
                    </button>
                </div>
            </div>
            
            <div className="hidden md:flex items-center gap-2">
                <Badge variant="outline" className="h-7 px-3 border-zinc-800 text-zinc-500">v2.4.0-stable</Badge>
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
             </div>
          )}
        </div>
      </div>

      {/* Mobile Bottom Nav (Only in ASK View, Always visible) */}
      {view === 'ASK' && (
        <div className="md:hidden h-14 border-t border-zinc-800 bg-zinc-950 flex grid grid-cols-2 fixed bottom-0 left-0 right-0 z-50 shadow-lg">
            <button 
                onClick={() => { setMobileTab('CONVERSATION'); setIsEvidenceOpen(false); }}
                className={`flex flex-col items-center justify-center gap-1 ${mobileTab === 'CONVERSATION' ? 'text-accent-500 bg-zinc-900/50' : 'text-zinc-500'}`}
            >
                <MessageSquare size={18} />
                <span className="text-[10px] font-bold">CHAT</span>
            </button>
            <button 
                onClick={() => { setMobileTab('EVIDENCE'); setIsEvidenceOpen(true); }}
                className={`flex flex-col items-center justify-center gap-1 ${mobileTab === 'EVIDENCE' ? 'text-accent-500 bg-zinc-900/50' : 'text-zinc-500'}`}
            >
                <BookOpen size={18} />
                <span className="text-[10px] font-bold">EVIDENCE</span>
            </button>
        </div>
      )}
    </div>
  );
}