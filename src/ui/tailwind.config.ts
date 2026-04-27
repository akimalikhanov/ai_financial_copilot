import type { Config } from 'tailwindcss';

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}', './*.tsx', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      colors: {
        bg: 'var(--bg)',
        'surface-1': 'var(--surface-1)',
        'surface-2': 'var(--surface-2)',
        'surface-3': 'var(--surface-3)',
        border: 'var(--border)',
        'border-subtle': 'var(--border-subtle)',
        text: 'var(--text)',
        'text-muted': 'var(--text-muted)',
        'text-faint': 'var(--text-faint)',
        icon: 'var(--icon)',
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          pressed: 'var(--accent-pressed)',
          subtle: 'var(--accent-subtle)',
        },
        success: {
          DEFAULT: 'var(--success)',
          bg: 'var(--success-bg)',
        },
        warning: {
          DEFAULT: 'var(--warning)',
          bg: 'var(--warning-bg)',
        },
        danger: {
          DEFAULT: 'var(--danger)',
          bg: 'var(--danger-bg)',
        },
        input: {
          bg: 'var(--input-bg)',
          border: 'var(--input-border)',
          'border-focus': 'var(--input-border-focus)',
        },
        code: {
          bg: 'var(--code-bg)',
          border: 'var(--code-border)',
        },
        bubble: {
          user: 'var(--bubble-user-bg)',
          assistant: 'var(--bubble-assistant-bg)',
        },
      },
      transitionTimingFunction: {
        'out-expo':    'cubic-bezier(.16,1,.3,1)',
        'in-out-expo': 'cubic-bezier(.87,0,.13,1)',
        'spring':      'cubic-bezier(.5,1.5,.5,1)',
      },
      animation: {
        'fade-in':           'fadeIn 0.3s cubic-bezier(.16,1,.3,1) forwards',
        'slide-in-right':    'slideInRight 0.3s cubic-bezier(.16,1,.3,1) forwards',
        'slide-in-up':       'slideInUp 0.2s cubic-bezier(.16,1,.3,1) forwards',
        'typing':            'typing 1.2s cubic-bezier(.87,0,.13,1) infinite',
        'caret-blink':       'caretBlink 1.1s ease-in-out infinite',
        'stream-reveal':     'streamReveal 120ms cubic-bezier(.16,1,.3,1) forwards',
        'modal-backdrop-in': 'modalBackdropIn 180ms cubic-bezier(.16,1,.3,1) forwards',
        'modal-backdrop-out':'modalBackdropOut 160ms cubic-bezier(.16,1,.3,1) forwards',
        'modal-card-in':     'modalCardIn 220ms cubic-bezier(.16,1,.3,1) forwards',
        'modal-card-out':    'modalCardOut 160ms cubic-bezier(.16,1,.3,1) forwards',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        slideInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        typing: {
          '0%, 100%': { opacity: '0.3' },
          '50%':      { opacity: '1' },
        },
        caretBlink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
        streamReveal: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        modalBackdropIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        modalBackdropOut: {
          '0%':   { opacity: '1' },
          '100%': { opacity: '0' },
        },
        modalCardIn: {
          '0%':   { opacity: '0', transform: 'scale(0.97) translateY(8px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        modalCardOut: {
          '0%':   { opacity: '1', transform: 'scale(1) translateY(0)' },
          '100%': { opacity: '0', transform: 'scale(0.97) translateY(8px)' },
        },
      },
      borderRadius: {
        sm:    'var(--radius-sm)',
        md:    'var(--radius-md)',
        lg:    'var(--radius-lg)',
        xl:    'var(--radius-xl)',
        '2xl': 'var(--radius-2xl)',
      },
      boxShadow: {
        'focus-ring': '0 0 0 2px var(--focus-ring)',
        xs:  'var(--shadow-xs)',
        sm:  'var(--shadow-sm)',
        md:  'var(--shadow-md)',
        lg:  'var(--shadow-lg)',
      },
      maxWidth: {
        content: 'var(--content-max)',
      },
      lineHeight: {
        tight:   'var(--leading-tight)',
        normal:  'var(--leading-normal)',
        relaxed: 'var(--leading-relaxed)',
      },
    },
  },
  plugins: [],
} satisfies Config;
