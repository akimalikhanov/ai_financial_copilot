/**
 * Theme Configuration - ChatGPT-inspired Design System
 *
 * Single source of truth for all design tokens.
 * Inspired by ChatGPT.com's clean, neutral aesthetic with OpenAI's green accent.
 *
 * To customize: modify the values below and everything updates automatically.
 */

export const theme = {
  dark: {
    // Backgrounds - ChatGPT style: darker sidebar, lighter main content
    bg: '#212121',              // Main chat area - lighter gray
    'surface-1': '#171717',     // Sidebar - darker
    'surface-2': '#2F2F2F',     // Hover states, cards
    'surface-3': '#3D3D3D',     // Active states, highlights

    // Borders - Very subtle 1px dividers
    border: '#3D3D3D',
    'border-subtle': '#2F2F2F',

    // Text - High contrast primary, muted secondary
    text: '#ECECEC',
    'text-muted': '#B4B4B4',
    'text-faint': '#8E8E8E',
    icon: '#B4B4B4',

    // Accent - OpenAI green
    accent: '#10A37F',
    'accent-hover': '#12B08A',
    'accent-pressed': '#0E8F6F',
    'accent-subtle': 'rgba(16, 163, 127, 0.12)',
    link: '#10A37F',
    'focus-ring': 'rgba(16, 163, 127, 0.35)',

    // Semantic colors
    success: '#22C55E',
    'success-bg': 'rgba(34, 197, 94, 0.12)',
    warning: '#F59E0B',
    'warning-bg': 'rgba(245, 158, 11, 0.12)',
    danger: '#EF4444',
    'danger-bg': 'rgba(239, 68, 68, 0.12)',

    // Inputs
    'input-bg': '#2F2F2F',
    'input-border': '#3D3D3D',
    'input-border-focus': '#10A37F',
    placeholder: '#8E8E8E',

    // Code blocks
    'code-bg': '#171717',
    'code-border': '#3D3D3D',

    // Chat bubbles
    'bubble-user-bg': '#2F2F2F',
    'bubble-assistant-bg': 'transparent',
  },

  light: {
    // Backgrounds - ChatGPT style: gray sidebar, white main content
    bg: '#FFFFFF',              // Main chat area - white
    'surface-1': '#F7F7F8',     // Sidebar - light gray
    'surface-2': '#EFEFEF',     // Hover states
    'surface-3': '#E5E5E5',     // Active states

    // Borders
    border: '#E5E5E5',
    'border-subtle': '#EFEFEF',

    // Text
    text: '#0D0D0D',
    'text-muted': '#6E6E6E',
    'text-faint': '#8E8E8E',
    icon: '#6E6E6E',

    // Accent
    accent: '#10A37F',
    'accent-hover': '#0F946F',
    'accent-pressed': '#0D8562',
    'accent-subtle': 'rgba(16, 163, 127, 0.08)',
    link: '#10A37F',
    'focus-ring': 'rgba(16, 163, 127, 0.25)',

    // Semantic
    success: '#16A34A',
    'success-bg': 'rgba(22, 163, 74, 0.1)',
    warning: '#D97706',
    'warning-bg': 'rgba(217, 119, 6, 0.1)',
    danger: '#DC2626',
    'danger-bg': 'rgba(220, 38, 38, 0.1)',

    // Inputs
    'input-bg': '#FFFFFF',
    'input-border': '#E5E5E5',
    'input-border-focus': '#10A37F',
    placeholder: '#8E8E8E',

    // Code blocks
    'code-bg': '#F7F7F8',
    'code-border': '#E5E5E5',

    // Chat bubbles
    'bubble-user-bg': '#F7F7F8',
    'bubble-assistant-bg': 'transparent',
  },
} as const;

// Typography
export const typography = {
  fontFamily: {
    sans: "'Söhne', 'Helvetica Neue', Helvetica, Arial, sans-serif",
    mono: "'Söhne Mono', 'JetBrains Mono', Menlo, Monaco, monospace",
  },
  fontSize: {
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
  },
  fontWeight: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
  },
  lineHeight: {
    tight: '1.25',
    normal: '1.5',
    relaxed: '1.625',
  },
} as const;

// Spacing & Sizing
export const spacing = {
  px: '1px',
  0: '0',
  0.5: '0.125rem',
  1: '0.25rem',
  1.5: '0.375rem',
  2: '0.5rem',
  2.5: '0.625rem',
  3: '0.75rem',
  3.5: '0.875rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  8: '2rem',
  10: '2.5rem',
  12: '3rem',
  16: '4rem',
  20: '5rem',
  24: '6rem',
} as const;

// Border radius
export const radius = {
  none: '0',
  sm: '0.25rem',   // 4px
  md: '0.375rem',  // 6px
  lg: '0.5rem',    // 8px
  xl: '0.75rem',   // 12px
  '2xl': '1rem',   // 16px
  full: '9999px',
} as const;

// Shadows
export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
} as const;

// Transitions
export const transitions = {
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
} as const;

// Generate CSS variables from theme object
export function generateCSSVariables(mode: 'dark' | 'light'): string {
  const colors = theme[mode];
  return Object.entries(colors)
    .map(([key, value]) => `  --${key}: ${value};`)
    .join('\n');
}

// Type exports for TypeScript
export type ThemeMode = 'dark' | 'light';
export type ThemeColors = typeof theme.dark;
export type ColorKey = keyof ThemeColors;
