export interface Document {
  id: string;
  title: string;
  company: string;
  year: number;
  type: string;
  pages: number;
  status: 'Ready' | 'Processing' | 'Error';
  tags: string[];
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Citation {
  docId: string;
  page: number;
  excerpt: string;
  bboxHint?: BoundingBox;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: number;
}

export interface Chat {
  id: string;
  title: string;
  createdAt: number;
  conversationId: string; // Backend conversation UUID (required)
}

export interface ScopeFilters {
  company?: string[];
  year?: number[];
  type?: string[];
}

export interface Scope {
  mode: 'allDocs' | 'filteredByMetadata' | 'selectedDocs' | 'thisDoc';
  docIds: string[]; // For selectedDocs or thisDoc
  filters: ScopeFilters;
}

export type ViewMode = 'ASK' | 'LIBRARY';
export type MobileTab = 'CONVERSATION' | 'EVIDENCE';
