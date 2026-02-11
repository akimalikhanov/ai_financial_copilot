import { Document } from '../types';

export const COMPANIES = ['NVIDIA Corp', 'Tesla Inc', 'JPMorgan Chase', 'Apple Inc', 'Microsoft'];
export const YEARS = [2024, 2023, 2022, 2021];

export const MOCK_DOCS: Document[] = [
  { id: 'd1', title: 'NVIDIA 2023 Annual Review', company: 'NVIDIA Corp', year: 2023, type: 'Annual Report', pages: 142, status: 'Ready', tags: ['AI', 'Hardware'] },
  { id: 'd2', title: 'Tesla Q3 2023 Update', company: 'Tesla Inc', year: 2023, type: 'Earnings Call', pages: 24, status: 'Ready', tags: ['EV', 'Energy'] },
  { id: 'd3', title: 'JPM 2022 Annual Report', company: 'JPMorgan Chase', year: 2022, type: 'Annual Report', pages: 312, status: 'Ready', tags: ['Banking'] },
  { id: 'd4', title: 'Apple 2023 10-K', company: 'Apple Inc', year: 2023, type: '10-K', pages: 88, status: 'Ready', tags: ['Consumer Tech'] },
  { id: 'd5', title: 'Microsoft 2023 Impact Report', company: 'Microsoft', year: 2023, type: 'ESG Report', pages: 56, status: 'Ready', tags: ['Sustainability'] },
];
