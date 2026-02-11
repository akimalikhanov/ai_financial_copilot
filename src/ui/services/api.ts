export type Role = 'system' | 'developer' | 'user' | 'assistant' | 'tool';

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
    const asError = payload as Partial<ErrorResponse>;
    return {
      message:
        asError.user_message ??
        asError.message ??
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
  try {
    const payload = await response.json();
    return toApiError(payload, {
      statusCode: response.status,
      fallbackMessage,
    });
  } catch {
    const payload = await response.text();
    return toApiError(payload, {
      statusCode: response.status,
      fallbackMessage,
    });
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
    response = await fetch(joinUrl(API_BASE_URL, '/v1/models'), {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
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

// --- Conversations API ---

export interface CreateConversationRequest {
  user_id?: string | null;
  title?: string | null;
  settings?: Record<string, unknown>;
}

export interface CreateConversationResponse {
  conversation_id: string;
}

export const createConversation = async (
  request: CreateConversationRequest
): Promise<CreateConversationResponse> => {
  let response: Response;
  try {
    response = await fetch(joinUrl(API_BASE_URL, '/v1/conversations'), {
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
    response = await fetch(
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

// --- Messages API ---

export interface MessageResponse {
  id: string;
  role: Role;
  content: string;
  seq: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

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
    response = await fetch(joinUrl(API_BASE_URL, url), {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
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
  const response = await fetch(joinUrl(API_BASE_URL, '/v1/chat'), {
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

export const chatStreamSubscribe = async (
  requestId: string,
  onDelta: (chunk: LLMStreamChunk) => void,
  onFinal: (chunk: LLMStreamChunk) => void,
  onError: (error: ApiError) => void,
  afterEventId?: string
): Promise<void> => {
  const params = new URLSearchParams({ request_id: requestId });
  if (afterEventId) {
    params.set('after_event_id', afterEventId);
  }
  const url = `${joinUrl(API_BASE_URL, '/v1/chat/stream')}?${params}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'text/event-stream' },
    });
  } catch (error) {
    onError(toApiErrorFromThrowable(error));
    return;
  }
  if (!response.ok) {
    onError(await toApiErrorFromResponse(response));
    return;
  }
  if (!response.body) {
    onError({ message: 'Stream response has no body', statusCode: response.status });
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
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
          return;
        }
        let payload: LLMStreamChunk | null = null;
        try {
          payload = JSON.parse(parsed.data) as LLMStreamChunk;
        } catch {
          onError(toApiError(parsed.data, { fallbackMessage: 'Failed to parse stream payload', statusCode: response.status }));
          return;
        }
        const isFinal = parsed.event === 'usage' || Boolean(payload.is_final);
        const safePayload: LLMStreamChunk = { ...payload, is_final: isFinal };
        if (isFinal) {
          onFinal(safePayload);
        } else {
          onDelta(safePayload);
        }
      }
    }
    if (buffer.trim().length > 0) {
      const parsed = parseSseEvent(buffer.trim());
      if (parsed?.data) {
        try {
          const payload = JSON.parse(parsed.data) as LLMStreamChunk;
          const isFinal = parsed.event === 'usage' || Boolean(payload.is_final);
          const safePayload = { ...payload, is_final: isFinal };
          if (isFinal) onFinal(safePayload);
          else onDelta(safePayload);
        } catch {
          onError(toApiError(buffer, { fallbackMessage: 'Failed to parse trailing payload', statusCode: response.status }));
        }
      }
    }
  } catch (error) {
    onError(toApiErrorFromThrowable(error));
  } finally {
    reader.releaseLock();
  }
};
