export type Role = 'system' | 'developer' | 'user' | 'assistant' | 'tool';

// --- Auth token wiring (in-memory only; set by AuthContext) ---
let getAccessToken: () => string | null = () => null;
let setAccessTokenFn: (token: string | null) => void = () => {};
let onRefreshFailureFn: () => void = () => {};

export function setAccessTokenGetter(fn: () => string | null) {
  getAccessToken = fn;
}

export function setAccessTokenSetter(fn: (token: string | null) => void) {
  setAccessTokenFn = fn;
}

export function setOnRefreshFailure(fn: () => void) {
  onRefreshFailureFn = fn;
}

export function getAccessTokenValue(): string | null {
  return getAccessToken();
}

export interface ChatMessage {
  role: Role;
  content: string;
  name?: string | null;
  tool_call_id?: string | null;
}

export interface LLMResponseStats {
  input_tokens?: number | null;
  cached_input_tokens?: number | null;
  output_tokens?: number | null;
  reasoning_tokens?: number | null;
  total_tokens?: number | null;
  latency_ms?: number | null;
  ttft_ms?: number | null;
  tps?: number | null;
  cost_usd?: number | null;
}

export interface LLMStreamChunk {
  text: string;
  is_final?: boolean;
  stats?: LLMResponseStats | null;
  raw?: Record<string, unknown> | null;
  assistant_message_id?: string;
  assistant_seq?: number;
  persisted?: boolean;
}

export interface ErrorResponse {
  error_type: string;
  message: string;
  internal_message?: string | null;
  user_message?: string | null;
  provider?: string | null;
  model?: string | null;
  is_retryable?: boolean | null;
  status_code?: number | null;
  error_code?: string | null;
  original_error_message?: string | null;
}

export interface ApiError {
  message: string;
  errorType?: string;
  statusCode?: number;
  raw?: unknown;
}

// --- Citation streaming event types ---

export interface CitationSpanEvent {
  start: number;
  end: number;
  ref_ids: string[];
  display_labels: string[];
}

export interface ReferencesEvent {
  items: {
    ref_id: string;
    display_label: string;
    chunk_id: string;
    document_id: string;
    document_name: string;
    filename: string | null;
    page_numbers: number[];
    heading_path: string[];
    snippet: string | null;
    bbox_hints?: {
      left: number;
      top: number;
      right: number;
      bottom: number;
      coord_origin: string;
      page: number;
    }[] | null;
  }[];
}

export interface MetadataEvent {
  confidence: 'low' | 'medium' | 'high' | 'none';
  ungrounded_claims: boolean | null;
  route: string | null;
}

export interface StageEvent {
  stage: string;
  index: number;
  total: number;
}

export interface ToolCallStartedEvent {
  entity: string;
}

export interface ToolCallCompletedEvent {
  entity: string;
  chunks_returned: number;
  new_chunks_added: number;
}

export interface ConversationTitleEvent {
  title: string;
  conversation_id: string;
}

type ApiErrorOptions = {
  statusCode?: number;
  fallbackMessage?: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

const joinUrl = (base: string, path: string) => {
  if (!base) return path;
  return `${base.replace(/\/+$/, '')}${path}`;
};

const toApiError = (payload: unknown, options: ApiErrorOptions = {}): ApiError => {
  if (payload && typeof payload === 'object') {
    const asError = payload as Partial<ErrorResponse> & { detail?: string };
    return {
      message:
        asError.user_message ??
        asError.message ??
        asError.detail ??
        options.fallbackMessage ??
        'Request failed',
      errorType: asError.error_type,
      statusCode: asError.status_code ?? options.statusCode,
      raw: payload,
    };
  }

  if (typeof payload === 'string' && payload.trim().length > 0) {
    return {
      message: payload,
      statusCode: options.statusCode,
      raw: payload,
    };
  }

  return {
    message: options.fallbackMessage ?? 'Request failed',
    statusCode: options.statusCode,
    raw: payload,
  };
};

const toApiErrorFromResponse = async (response: Response): Promise<ApiError> => {
  const fallbackMessage = `Request failed with status ${response.status}`;
  const text = await response.text();
  try {
    const payload = JSON.parse(text) as unknown;
    return toApiError(payload, { statusCode: response.status, fallbackMessage });
  } catch {
    return toApiError(text, { statusCode: response.status, fallbackMessage });
  }
};

const toApiErrorFromThrowable = (error: unknown): ApiError => {
    if (error instanceof TypeError && error.message.includes('NetworkError')) {
        return {
            message: 'Unable to connect to the server. Check your internet connection and try again.',
            errorType: 'NetworkError',
            raw: error,
        };
        }
  if (error instanceof Error) {
    return { message: error.message, errorType: error.name, raw: error };
  }
  return { message: 'Unknown error', raw: error };
};

type SseEvent = {
  event: string;
  data: string;
};

type FetchApiOptions = RequestInit & {
  skipAuthRetry?: boolean; // set true for /auth/refresh to avoid retry loop
};

async function fetchApi(url: string, options: FetchApiOptions = {}): Promise<Response> {
  const { skipAuthRetry, ...init } = options;
  const token = getAccessToken();

  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const config: RequestInit = { ...init, headers, credentials: 'include' };

  let res = await fetch(url, config);

  if (res.status === 401 && !skipAuthRetry) {
    try {
      const refreshRes = await fetch(joinUrl(API_BASE_URL, '/v1/auth/refresh'), {
        method: 'POST',
        credentials: 'include',
        headers: { Accept: 'application/json' },
      } as RequestInit);
      if (refreshRes.ok) {
        const data = (await refreshRes.json()) as { access_token: string };
        setAccessTokenFn(data.access_token);
        const retryHeaders = new Headers(init.headers);
        retryHeaders.set('Authorization', `Bearer ${data.access_token}`);
        res = await fetch(url, { ...init, headers: retryHeaders, credentials: 'include' });
      } else {
        onRefreshFailureFn();
        throw await toApiErrorFromResponse(refreshRes);
      }
    } catch (e) {
      onRefreshFailureFn();
      throw e;
    }
  }

  return res;
}

const parseSseEvent = (rawEvent: string): SseEvent | null => {
  const lines = rawEvent.split('\n');
  let event = '';
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
      continue;
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }

  if (!event && dataLines.length === 0) {
    return null;
  }

  return {
    event: event || 'message',
    data: dataLines.join('\n'),
  };
};

// --- Models API ---

export interface ModelInfo {
  id: string;
  name: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export const fetchModels = async (): Promise<ModelInfo[]> => {
  let response: Response;
  try {
    response = await fetchApi(joinUrl(API_BASE_URL, '/v1/models'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  } catch (error) {
    throw toApiErrorFromThrowable(error);
  }

  if (!response.ok) {
    throw await toApiErrorFromResponse(response);
  }

  const data = (await response.json()) as ModelsResponse;
  return data.models;
};

// --- Auth API ---

export interface UserInfo {
  id: string;
  email: string;
  display_name: string | null;
}

export interface LoginResponse {
  access_token: string;
  user: UserInfo;
}

export const login = async (email: string, password: string): Promise<LoginResponse> => {
  const res = await fetch(joinUrl(API_BASE_URL, '/v1/auth/login'), {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw await toApiErrorFromResponse(res);
  const data = (await res.json()) as { access_token: string };
  const meRes = await fetch(joinUrl(API_BASE_URL, '/v1/auth/me'), {
    method: 'GET',
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${data.access_token}`,
    },
  });
  if (!meRes.ok) throw await toApiErrorFromResponse(meRes);
  const user = (await meRes.json()) as UserInfo;
  return { access_token: data.access_token, user };
};

export const register = async (
  email: string,
  password: string,
  displayName?: string | null
): Promise<LoginResponse> => {
  const res = await fetch(joinUrl(API_BASE_URL, '/v1/auth/register'), {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ email, password, display_name: displayName ?? null }),
  });
  if (!res.ok) throw await toApiErrorFromResponse(res);
  const data = (await res.json()) as { access_token: string };
  const meRes = await fetch(joinUrl(API_BASE_URL, '/v1/auth/me'), {
    method: 'GET',
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${data.access_token}`,
    },
  });
  if (!meRes.ok) throw await toApiErrorFromResponse(meRes);
  const user = (await meRes.json()) as UserInfo;
  return { access_token: data.access_token, user };
};

export const refreshTokens = async (): Promise<{ access_token: string }> => {
  const res = await fetch(joinUrl(API_BASE_URL, '/v1/auth/refresh'), {
    method: 'POST',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw await toApiErrorFromResponse(res);
  return (await res.json()) as { access_token: string };
};

export const logout = async (): Promise<void> => {
  await fetch(joinUrl(API_BASE_URL, '/v1/auth/logout'), {
    method: 'POST',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
};

/** Fetch current user (requires valid access token). */
export const getMe = async (): Promise<UserInfo> => {
  const response = await fetchApi(joinUrl(API_BASE_URL, '/v1/auth/me'), {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) throw await toApiErrorFromResponse(response);
  return (await response.json()) as UserInfo;
};

// --- Documents API ---

export interface UploadDocumentResponse {
  id: string;
  status: string;
  original_filename: string;
  storage_key: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface DocumentListItemResponse {
  id: string;
  status: string;
  original_filename: string;
  created_at: string;
  extracted_title: string | null;
  page_count: number | null;
  metadata: Record<string, unknown>;
}

export interface ListDocumentsResponse {
  documents: DocumentListItemResponse[];
  total: number;
}

export const listDocuments = async (): Promise<ListDocumentsResponse> => {
  let response: Response;
  try {
    response = await fetchApi(joinUrl(API_BASE_URL, '/v1/documents'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  } catch (error) {
    throw toApiErrorFromThrowable(error);
  }
  if (!response.ok) throw await toApiErrorFromResponse(response);
  return (await response.json()) as ListDocumentsResponse;
};

export interface DocumentFilterOptionsResponse {
  companies: string[];
  years: number[];
}

export const fetchFilterOptions = async (): Promise<DocumentFilterOptionsResponse> => {
  const response = await fetchApi(joinUrl(API_BASE_URL, '/v1/documents/filter-options'), {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) throw await toApiErrorFromResponse(response);
  return response.json() as Promise<DocumentFilterOptionsResponse>;
};

export const uploadDocument = async (formData: FormData): Promise<UploadDocumentResponse> => {
  const response = await fetchApi(joinUrl(API_BASE_URL, '/v1/documents/upload'), {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw await toApiErrorFromResponse(response);
  return response.json() as Promise<UploadDocumentResponse>;
};

export const deleteDocument = async (documentId: string): Promise<void> => {
  const response = await fetchApi(joinUrl(API_BASE_URL, `/v1/documents/${documentId}`), {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) throw await toApiErrorFromResponse(response);
};

export const getPdfUrl = (documentId: string): string =>
  joinUrl(API_BASE_URL, `/v1/documents/${documentId}/pdf`);

export interface IngestionStageEvent {
  stage: string;
  stage_index: number;
  stage_total: number;
}

export const subscribeIngestionStream = (
  documentId: string,
  onStage: (event: IngestionStageEvent) => void,
  onDone: () => void,
  onError: (message: string) => void,
): (() => void) => {
  const url = joinUrl(API_BASE_URL, `/v1/documents/${documentId}/stream`);
  const token = getAccessTokenValue();
  const headers: Record<string, string> = { Accept: 'text/event-stream' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let cancelled = false;
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(url, { headers, signal: controller.signal });
      if (!response.ok || !response.body) {
        onError(`HTTP ${response.status}`);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let eventType = 'stage';

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            try {
              const payload = JSON.parse(line.slice(5).trim());
              if (eventType === 'stage') onStage(payload as IngestionStageEvent);
              else if (eventType === 'done') { onDone(); return; }
              else if (eventType === 'error') { onError(payload.message ?? 'Ingestion failed'); return; }
            } catch { /* ignore malformed */ }
            eventType = 'stage';
          }
        }
      }
    } catch (err) {
      if (!cancelled) onError(String(err));
    }
  })();

  return () => { cancelled = true; controller.abort(); };
};

// --- Conversations API ---

export interface CreateConversationRequest {
  title?: string | null;
  settings?: Record<string, unknown>;
}

export interface CreateConversationResponse {
  conversation_id: string;
}

export interface ConversationListItem {
  id: string;
  title: string | null;
  created_at: string;
  last_message_at: string | null;
}

export interface ListConversationsResponse {
  conversations: ConversationListItem[];
  total: number;
}

export const fetchConversations = async (
  limit = 50,
  offset = 0
): Promise<ListConversationsResponse> => {
  const params = new URLSearchParams();
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());
  const url = `/v1/conversations?${params}`;
  const res = await fetchApi(joinUrl(API_BASE_URL, url), {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw await toApiErrorFromResponse(res);
  return (await res.json()) as ListConversationsResponse;
};

export const createConversation = async (
  request: CreateConversationRequest
): Promise<CreateConversationResponse> => {
  let response: Response;
  try {
    response = await fetchApi(joinUrl(API_BASE_URL, '/v1/conversations'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(request),
    });
  } catch (error) {
    throw toApiErrorFromThrowable(error);
  }

  if (!response.ok) {
    throw await toApiErrorFromResponse(response);
  }

  return (await response.json()) as CreateConversationResponse;
};

export interface UpdateConversationRequest {
  title: string | null;
}

export interface UpdateConversationResponse {
  status: string;
}

export const updateConversation = async (
  conversationId: string,
  request: UpdateConversationRequest
): Promise<UpdateConversationResponse> => {
  let response: Response;
  try {
    response = await fetchApi(
      joinUrl(API_BASE_URL, `/v1/conversations/${conversationId}`),
      {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
  } catch (error) {
    throw toApiErrorFromThrowable(error);
  }

  if (!response.ok) {
    throw await toApiErrorFromResponse(response);
  }

  return (await response.json()) as UpdateConversationResponse;
};

export const deleteConversation = async (
  conversationId: string
): Promise<void> => {
  const response = await fetchApi(
    joinUrl(API_BASE_URL, `/v1/conversations/${conversationId}`),
    { method: 'DELETE', headers: { Accept: 'application/json' } }
  );
  if (!response.ok) throw await toApiErrorFromResponse(response);
};

// --- Messages API ---

export interface MessageFeedbackPayload {
  rating: 'up' | 'down';
  comment?: string | null;
}

export interface MessageResponse {
  id: string;
  role: Role;
  content: string;
  seq: number;
  created_at: string;
  metadata?: Record<string, unknown>;
  feedback?: MessageFeedbackPayload | null;
}

export const submitMessageFeedback = async (
  messageId: string,
  rating: 'up' | 'down',
  comment?: string | null,
): Promise<MessageFeedbackPayload> => {
  const response = await fetchApi(
    joinUrl(API_BASE_URL, `/v1/messages/${messageId}/feedback`),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ rating, comment: comment ?? null }),
    },
  );
  if (!response.ok) throw await toApiErrorFromResponse(response);
  const data = await response.json();
  return { rating: data.rating, comment: data.comment };
};

export const deleteMessageFeedback = async (messageId: string): Promise<void> => {
  const response = await fetchApi(
    joinUrl(API_BASE_URL, `/v1/messages/${messageId}/feedback`),
    { method: 'DELETE', headers: { Accept: 'application/json' } },
  );
  if (!response.ok && response.status !== 404) throw await toApiErrorFromResponse(response);
};

export interface FetchMessagesResponse {
  messages: MessageResponse[];
  has_more: boolean;
}

export interface FetchMessagesParams {
  limit?: number;
  after_seq?: number;
  before_seq?: number;
}

export const fetchMessages = async (
  conversationId: string,
  params?: FetchMessagesParams
): Promise<FetchMessagesResponse> => {
  let response: Response;
  try {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) {
      searchParams.append('limit', params.limit.toString());
    }
    if (params?.after_seq !== undefined) {
      searchParams.append('after_seq', params.after_seq.toString());
    }
    if (params?.before_seq !== undefined) {
      searchParams.append('before_seq', params.before_seq.toString());
    }
    const queryString = searchParams.toString();
    const url = `/v1/conversations/${conversationId}/messages${queryString ? `?${queryString}` : ''}`;
    response = await fetchApi(joinUrl(API_BASE_URL, url), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  } catch (error) {
    throw toApiErrorFromThrowable(error);
  }

  if (!response.ok) {
    throw await toApiErrorFromResponse(response);
  }

  return (await response.json()) as FetchMessagesResponse;
};

// --- Queued Chat API (producer: persist + enqueue, subscriber: stream) ---

export interface ChatEnqueueRequest {
  conversation_id: string;
  content: string;
  client_msg_id: string;
  client_request_id: string;
  model: string;
  params: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ChatEnqueueResponse {
  request_id: string;
  user_message_id: string;
  user_seq: number;
  assistant_message_id: string;
  assistant_seq: number;
  status: string;
}

export const chatEnqueue = async (
  request: ChatEnqueueRequest
): Promise<ChatEnqueueResponse> => {
  const response = await fetchApi(joinUrl(API_BASE_URL, '/v1/chat'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      ...request,
      metadata: request.metadata ?? {},
    }),
  });
  if (!response.ok) {
    throw await toApiErrorFromResponse(response);
  }
  return (await response.json()) as ChatEnqueueResponse;
};

const MAX_SSE_RETRIES = 2;
const SSE_RETRY_DELAY_MS = 500;

export const chatStreamSubscribe = async (
  requestId: string,
  onDelta: (chunk: LLMStreamChunk) => void,
  onCitationSpan: (span: CitationSpanEvent) => void,
  onReferences: (refs: ReferencesEvent) => void,
  onFinal: (chunk: LLMStreamChunk) => void,
  onError: (error: ApiError) => void,
  afterEventId?: string,
  onMetadata?: (meta: MetadataEvent) => void,
  onStage?: (stage: StageEvent) => void,
  onToolCallStarted?: (event: ToolCallStartedEvent) => void,
  onToolCallCompleted?: (event: ToolCallCompletedEvent) => void,
  onConversationTitle?: (event: ConversationTitleEvent) => void,
): Promise<void> => {
  for (let attempt = 0; attempt <= MAX_SSE_RETRIES; attempt++) {
    const result = await _doStreamAttempt(
      requestId, onDelta, onCitationSpan, onReferences, onFinal, onError, afterEventId, onMetadata, onStage, onToolCallStarted, onToolCallCompleted, onConversationTitle
    );
    if (result === 'done' || result === 'server-error') return;
    // 'connection-error': retry only if no content was delivered (safe to replay from 0-0)
    if (attempt < MAX_SSE_RETRIES) {
      await new Promise((r) => setTimeout(r, SSE_RETRY_DELAY_MS));
      continue;
    }
    // Exhausted retries
    onError({ message: 'Unable to connect to the server. Check your internet connection and try again.' });
  }
};

/**
 * Single SSE stream attempt. Returns:
 * - 'done': stream completed successfully (or onError called for a server-side error)
 * - 'server-error': non-retryable error (HTTP error, server-sent error event)
 * - 'connection-error': connection-level failure before any content events — safe to retry
 */
async function _doStreamAttempt(
  requestId: string,
  onDelta: (chunk: LLMStreamChunk) => void,
  onCitationSpan: (span: CitationSpanEvent) => void,
  onReferences: (refs: ReferencesEvent) => void,
  onFinal: (chunk: LLMStreamChunk) => void,
  onError: (error: ApiError) => void,
  afterEventId?: string,
  onMetadata?: (meta: MetadataEvent) => void,
  onStage?: (stage: StageEvent) => void,
  onToolCallStarted?: (event: ToolCallStartedEvent) => void,
  onToolCallCompleted?: (event: ToolCallCompletedEvent) => void,
  onConversationTitle?: (event: ConversationTitleEvent) => void,
): Promise<'done' | 'server-error' | 'connection-error'> {
  const params = new URLSearchParams({ request_id: requestId });
  if (afterEventId) {
    params.set('after_event_id', afterEventId);
  }
  const url = `${joinUrl(API_BASE_URL, '/v1/chat/stream')}?${params}`;
  let response: Response;
  try {
    response = await fetchApi(url, {
      method: 'GET',
      headers: { Accept: 'text/event-stream' },
    });
  } catch {
    return 'connection-error';
  }
  if (!response.ok) {
    onError(await toApiErrorFromResponse(response));
    return 'server-error';
  }
  if (!response.body) {
    onError({ message: 'Stream response has no body', statusCode: response.status });
    return 'server-error';
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let receivedContent = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundaryIndex = buffer.indexOf('\n\n');
      while (boundaryIndex !== -1) {
        const rawEvent = buffer.slice(0, boundaryIndex).trim();
        buffer = buffer.slice(boundaryIndex + 2);
        boundaryIndex = buffer.indexOf('\n\n');
        if (!rawEvent) continue;
        const parsed = parseSseEvent(rawEvent);
        if (!parsed) continue;
        if (parsed.event === 'error') {
          let payload: unknown = parsed.data;
          try {
            payload = JSON.parse(parsed.data);
          } catch {
            /* keep raw */
          }
          onError(toApiError(payload, { fallbackMessage: 'Streaming error', statusCode: response.status }));
          return 'server-error';
        }
        // Handle citation_span events
        if (parsed.event === 'citation_span') {
          try {
            const span = JSON.parse(parsed.data) as CitationSpanEvent;
            onCitationSpan(span);
            receivedContent = true;
          } catch { /* ignore malformed */ }
          continue;
        }
        // Handle references events
        if (parsed.event === 'references') {
          try {
            const refs = JSON.parse(parsed.data) as ReferencesEvent;
            onReferences(refs);
            receivedContent = true;
          } catch { /* ignore malformed */ }
          continue;
        }
        // Handle metadata events
        if (parsed.event === 'metadata') {
          try {
            const meta = JSON.parse(parsed.data) as MetadataEvent;
            onMetadata?.(meta);
          } catch { /* ignore malformed */ }
          continue;
        }
        // Handle stage progress events
        if (parsed.event === 'stage') {
          try {
            const s = JSON.parse(parsed.data) as StageEvent;
            onStage?.(s);
          } catch { /* ignore malformed */ }
          continue;
        }
        if (parsed.event === 'tool_call_started') {
          try { onToolCallStarted?.(JSON.parse(parsed.data) as ToolCallStartedEvent); } catch { /* ignore */ }
          continue;
        }
        if (parsed.event === 'tool_call_completed') {
          try { onToolCallCompleted?.(JSON.parse(parsed.data) as ToolCallCompletedEvent); } catch { /* ignore */ }
          continue;
        }
        if (parsed.event === 'conversation_title') {
          try { onConversationTitle?.(JSON.parse(parsed.data) as ConversationTitleEvent); } catch { /* ignore */ }
          continue;
        }
        // Skip other non-content server events
        if (!['delta', 'usage', 'message'].includes(parsed.event)) continue;
        let payload: LLMStreamChunk | null = null;
        try {
          payload = JSON.parse(parsed.data) as LLMStreamChunk;
        } catch {
          onError(toApiError(parsed.data, { fallbackMessage: 'Failed to parse stream payload', statusCode: response.status }));
          return 'server-error';
        }
        const isFinal = parsed.event === 'usage' || Boolean(payload.is_final);
        const safePayload: LLMStreamChunk = { ...payload, is_final: isFinal };
        if (isFinal) {
          onFinal(safePayload);
          return 'done';
        }
        onDelta(safePayload);
        receivedContent = true;
      }
    }
    if (buffer.trim().length > 0) {
      const parsed = parseSseEvent(buffer.trim());
      if (parsed?.data) {
        try {
          const payload = JSON.parse(parsed.data) as LLMStreamChunk;
          const isFinal = parsed.event === 'usage' || Boolean(payload.is_final);
          const safePayload = { ...payload, is_final: isFinal };
          if (isFinal) {
            onFinal(safePayload);
            return 'done';
          }
          onDelta(safePayload);
          receivedContent = true;
        } catch {
          onError(toApiError(buffer, { fallbackMessage: 'Failed to parse trailing payload', statusCode: response.status }));
          return 'server-error';
        }
      }
    }
    // Stream ended without a final event
    return receivedContent ? 'done' : 'connection-error';
  } catch {
    return receivedContent ? 'done' : 'connection-error';
  } finally {
    reader.releaseLock();
  }
}

// --- Chat Stats API ---

export interface RequestStatsItem {
  input_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  total_tokens: number | null;
  cost_usd: number | null;
  latency_ms: number | null;
  ttft_ms: number | null;
  tps: number | null;
  model: string;
  created_at: string;
  // Full pipeline aggregates (chat LLM + router combined).
  pipeline_cost_usd: number | null;
  pipeline_total_tokens: number | null;
}

export interface ChatStatsResponse {
  requests: RequestStatsItem[];
}

export const fetchChatStats = async (
  conversationId: string,
  limit = 50
): Promise<ChatStatsResponse> => {
  const res = await fetchApi(
    joinUrl(
      API_BASE_URL,
      `/v1/chat/stats?conversation_id=${conversationId}&limit=${limit}`
    ),
    { method: 'GET', headers: { Accept: 'application/json' } }
  );
  if (!res.ok) throw await toApiErrorFromResponse(res);
  return res.json();
};
