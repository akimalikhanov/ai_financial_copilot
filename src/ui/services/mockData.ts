import { Document, Chat, Message, Citation, Scope } from '../types';

export const COMPANIES = ['NVIDIA Corp', 'Tesla Inc', 'JPMorgan Chase', 'Apple Inc', 'Microsoft'];
export const DOC_TYPES = ['Annual Report', '10-K', 'Earnings Call', 'ESG Report'];
export const YEARS = [2024, 2023, 2022, 2021];

const generateId = () => Math.random().toString(36).substr(2, 9);

export const MOCK_DOCS: Document[] = [
  { id: 'd1', title: 'NVIDIA 2023 Annual Review', company: 'NVIDIA Corp', year: 2023, type: 'Annual Report', pages: 142, status: 'Ready', tags: ['AI', 'Hardware'] },
  { id: 'd2', title: 'Tesla Q3 2023 Update', company: 'Tesla Inc', year: 2023, type: 'Earnings Call', pages: 24, status: 'Ready', tags: ['EV', 'Energy'] },
  { id: 'd3', title: 'JPM 2022 Annual Report', company: 'JPMorgan Chase', year: 2022, type: 'Annual Report', pages: 312, status: 'Ready', tags: ['Banking'] },
  { id: 'd4', title: 'Apple 2023 10-K', company: 'Apple Inc', year: 2023, type: '10-K', pages: 88, status: 'Ready', tags: ['Consumer Tech'] },
  { id: 'd5', title: 'Microsoft 2023 Impact Report', company: 'Microsoft', year: 2023, type: 'ESG Report', pages: 56, status: 'Ready', tags: ['Sustainability'] },
];

export const MOCK_CHATS: Chat[] = [
  {
    id: 'c1',
    title: 'NVIDIA AI Revenue Analysis',
    createdAt: Date.now() - 1000000,
    messages: [
      { id: 'm1', role: 'user', content: 'What was the growth in Data Center revenue for NVIDIA?', timestamp: Date.now() - 1000000 },
      {
        id: 'm2',
        role: 'assistant',
        content: 'NVIDIA reported Data Center revenue of $15.01 billion for the fiscal year, up 41% from the previous year. This growth reflects strong demand for generative AI and large language models.',
        timestamp: Date.now() - 900000,
        citations: [
            { docId: 'd1', page: 34, excerpt: "Data Center revenue was $15.01 billion, up 41% from a year ago.", bboxHint: { x: 10, y: 40, w: 80, h: 15 } }
        ]
      }
    ]
  },
  {
    id: 'c2',
    title: 'Tesla Margins 2023',
    createdAt: Date.now() - 86400000,
    messages: [
      { id: 'm1', role: 'user', content: 'How did gross margins change for Tesla in Q3?', timestamp: Date.now() - 86400000 },
    ]
  }
];

// Helper to simulate an AI response with citations
export const generateMockResponse = (
    query: string,
    scope: Scope,
    allDocs: Document[]
): Promise<Message> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Filter docs based on scope
      let pool = allDocs;
      if (scope.mode === 'filteredByMetadata') {
        pool = allDocs.filter(d => {
            const f = scope.filters;
            const matchCompany = !f.company?.length || f.company.includes(d.company);
            const matchYear = !f.year?.length || f.year.includes(d.year);
            const matchType = !f.type?.length || f.type.includes(d.type);
            return matchCompany && matchYear && matchType;
        });
      } else if (scope.mode === 'selectedDocs' || scope.mode === 'thisDoc') {
        pool = allDocs.filter(d => scope.docIds.includes(d.id));
      }

      // If pool is empty, return a generic warning
      if (pool.length === 0) {
        resolve({
          id: generateId(),
          role: 'assistant',
          content: "I couldn't find any documents matching your current scope. Please adjust your filters or select documents to continue.",
          timestamp: Date.now()
        });
        return;
      }

      // Pick 1-2 random docs from pool for citations
      const citedDocs = pool.sort(() => 0.5 - Math.random()).slice(0, Math.min(2, pool.length));

      const citations: Citation[] = citedDocs.map(d => ({
        docId: d.id,
        page: Math.floor(Math.random() * d.pages) + 1,
        excerpt: `Simulated excerpt from ${d.title} regarding "${query}".`,
        bboxHint: { x: 10 + Math.random() * 40, y: 20 + Math.random() * 50, w: 80, h: 10 }
      }));

      const content = `Based on the analysis of ${pool.length} document(s), here is what I found regarding "${query}". The reports indicate a strong trend in this area. specifically, ${citedDocs[0].company} highlights this in their latest filing.`;

      resolve({
        id: generateId(),
        role: 'assistant',
        content,
        citations,
        timestamp: Date.now()
      });
    }, 2000); // 2s delay
  });
};
