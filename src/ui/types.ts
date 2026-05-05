export interface Document {
  id: string;
  title: string;
  company: string;
  year: number;
  type: string;
  pages: number;
  status: 'Ready' | 'Processing' | 'Error';
  tags: string[];
  ingestionStage?: string;
  ingestionStageIndex?: number;
  ingestionStageTotal?: number;
}

export interface BoundingBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
  coord_origin: string;
  page: number;
}

export interface Citation {
  docId: string;
  page: number;
  excerpt: string;
  bboxHint?: BoundingBox;
}

export interface CitationSpan {
  start: number;
  end: number;
  refIds: string[];
  displayLabels: string[];
}

export interface ReferenceItem {
  refId: string;
  displayLabel: string;
  chunkId: string;
  documentId: string;
  documentName: string;
  filename: string | null;
  pageNumbers: number[];
  headingPath: string[];
  snippet: string | null;
  bboxHint?: BoundingBox | null;
}

export interface MessageFeedback {
  rating: 'up' | 'down';
  comment?: string | null;
}

export interface MessageMetadata {
  confidence?: 'low' | 'medium' | 'high' | 'none';
  ungrounded_claims?: boolean | null;
  route?: string | null;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  citationSpans?: CitationSpan[];
  references?: ReferenceItem[];
  timestamp: number;
  feedback?: MessageFeedback | null;
  metadata?: MessageMetadata;
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
