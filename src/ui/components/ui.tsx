import React from 'react';
import { Loader2 } from 'lucide-react';

/**
 * Design System Components
 * 
 * All components use CSS custom properties from index.css for theming.
 * Colors automatically adapt to dark/light mode.
 */

// ============================================
// BUTTON
// ============================================

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'primary', size = 'md', loading, children, disabled, ...props }, ref) => {
    const isDisabled = disabled || loading;
    
    const baseStyle = `
      inline-flex items-center justify-center font-sans font-medium 
      transition-all duration-150 ease-out
      focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)]
      disabled:opacity-50 disabled:cursor-not-allowed
    `.trim().replace(/\s+/g, ' ');
    
    const variants: Record<string, string> = {
      primary: `
        bg-[var(--accent)] text-white 
        hover:bg-[var(--accent-hover)] 
        active:bg-[var(--accent-pressed)]
        border border-transparent
      `.trim().replace(/\s+/g, ' '),
      secondary: `
        bg-[var(--surface-2)] text-[var(--text)] 
        border border-[var(--border)]
        hover:bg-[var(--surface-3)] hover:border-[var(--border)]
      `.trim().replace(/\s+/g, ' '),
      ghost: `
        bg-transparent text-[var(--text-muted)] 
        hover:text-[var(--text)] hover:bg-[var(--surface-2)]
        border border-transparent
      `.trim().replace(/\s+/g, ' '),
      danger: `
        bg-[var(--danger-bg)] text-[var(--danger)] 
        border border-[var(--danger)]
        hover:bg-[var(--danger)] hover:text-white
      `.trim().replace(/\s+/g, ' '),
    };

    const sizes: Record<string, string> = {
      sm: 'h-8 px-3 text-xs rounded-md gap-1.5',
      md: 'h-10 px-4 text-sm rounded-md gap-2',
      lg: 'h-12 px-6 text-base rounded-lg gap-2',
      icon: 'h-9 w-9 p-0 rounded-md',
    };

    return (
      <button
        ref={ref}
        className={`${baseStyle} ${variants[variant]} ${sizes[size]} ${className}`}
        disabled={isDisabled}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  }
);
Button.displayName = 'Button';

// ============================================
// INPUT
// ============================================

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`
          flex h-10 w-full rounded-md 
          border border-[var(--input-border)] 
          bg-[var(--input-bg)] 
          px-3 py-2 text-sm text-[var(--text)] 
          placeholder:text-[var(--placeholder)]
          transition-colors duration-150
          focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)]
          disabled:cursor-not-allowed disabled:opacity-50 
          font-sans
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

// ============================================
// TEXTAREA
// ============================================

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className = '', ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={`
          flex w-full rounded-md 
          border border-[var(--input-border)] 
          bg-[var(--input-bg)] 
          px-3 py-2 text-sm text-[var(--text)] 
          placeholder:text-[var(--placeholder)]
          transition-colors duration-150
          focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)]
          disabled:cursor-not-allowed disabled:opacity-50 
          font-sans resize-none
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      />
    );
  }
);
Textarea.displayName = 'Textarea';

// ============================================
// SELECT
// ============================================

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={`
          appearance-none h-10 rounded-md 
          border border-[var(--input-border)] 
          bg-[var(--input-bg)] 
          px-3 pr-8 text-sm text-[var(--text)]
          transition-colors duration-150
          focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-1 focus:ring-[var(--focus-ring)]
          disabled:cursor-not-allowed disabled:opacity-50 
          font-sans cursor-pointer
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {children}
      </select>
    );
  }
);
Select.displayName = 'Select';

// ============================================
// BADGE
// ============================================

export interface BadgeProps {
  children?: React.ReactNode;
  variant?: 'default' | 'outline' | 'success' | 'warning' | 'danger' | 'accent';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ 
  children, 
  variant = 'default', 
  className = '' 
}) => {
  const variants: Record<string, string> = {
    default: 'bg-[var(--surface-2)] text-[var(--text-muted)] border-transparent',
    outline: 'bg-transparent border-[var(--border)] text-[var(--text-muted)]',
    success: 'bg-[var(--success-bg)] text-[var(--success)] border-[var(--success)]',
    warning: 'bg-[var(--warning-bg)] text-[var(--warning)] border-[var(--warning)]',
    danger: 'bg-[var(--danger-bg)] text-[var(--danger)] border-[var(--danger)]',
    accent: 'bg-[var(--accent-subtle)] text-[var(--accent)] border-[var(--accent)]',
  };

  return (
    <span 
      className={`
        inline-flex items-center px-2.5 py-0.5 rounded-md 
        text-xs font-medium font-sans
        border
        ${variants[variant]} 
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      {children}
    </span>
  );
};

// ============================================
// CARD
// ============================================

export interface CardProps {
  children?: React.ReactNode;
  className?: string;
  variant?: 'default' | 'elevated';
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  className = '',
  variant = 'default'
}) => {
  const variants: Record<string, string> = {
    default: 'bg-[var(--surface-1)] border border-[var(--border)]',
    elevated: 'bg-[var(--surface-2)] border border-[var(--border)] shadow-lg',
  };

  return (
    <div 
      className={`
        rounded-lg
        ${variants[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      {children}
    </div>
  );
};

// ============================================
// SEPARATOR
// ============================================

export interface SeparatorProps {
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export const Separator: React.FC<SeparatorProps> = ({ 
  className = '', 
  orientation = 'horizontal' 
}) => (
  <div 
    className={`
      ${orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]'} 
      bg-[var(--border)]
      ${className}
    `.trim().replace(/\s+/g, ' ')} 
  />
);

// ============================================
// TOGGLE / SWITCH
// ============================================

export interface ToggleProps {
  checked?: boolean;
  onChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  label?: string;
}

export const Toggle: React.FC<ToggleProps> = ({
  checked = false,
  onChange,
  disabled = false,
  className = '',
  label,
}) => {
  return (
    <label 
      className={`
        inline-flex items-center gap-2 cursor-pointer
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange?.(!checked)}
        className={`
          relative inline-flex h-6 w-11 shrink-0 
          cursor-pointer rounded-full border-2 border-transparent 
          transition-colors duration-200 ease-in-out
          focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)]
          ${checked ? 'bg-[var(--accent)]' : 'bg-[var(--surface-3)]'}
          ${disabled ? 'cursor-not-allowed' : ''}
        `.trim().replace(/\s+/g, ' ')}
      >
        <span
          className={`
            pointer-events-none inline-block h-5 w-5 
            transform rounded-full bg-white shadow-md ring-0 
            transition duration-200 ease-in-out
            ${checked ? 'translate-x-5' : 'translate-x-0'}
          `.trim().replace(/\s+/g, ' ')}
        />
      </button>
      {label && (
        <span className="text-sm text-[var(--text)]">{label}</span>
      )}
    </label>
  );
};

// ============================================
// CODE BLOCK
// ============================================

export interface CodeBlockProps {
  children: React.ReactNode;
  language?: string;
  className?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  children,
  language,
  className = '',
}) => {
  return (
    <div 
      className={`
        rounded-lg overflow-hidden
        border border-[var(--code-border)]
        bg-[var(--code-bg)]
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      {language && (
        <div className="px-4 py-2 border-b border-[var(--code-border)] text-xs text-[var(--text-faint)] font-mono">
          {language}
        </div>
      )}
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm font-mono text-[var(--text)]">
          {children}
        </code>
      </pre>
    </div>
  );
};

// ============================================
// CHAT BUBBLE
// ============================================

export interface ChatBubbleProps {
  children: React.ReactNode;
  variant: 'user' | 'assistant';
  className?: string;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({
  children,
  variant,
  className = '',
}) => {
  const variants: Record<string, string> = {
    user: 'bg-[var(--bubble-user-bg)] text-[var(--text)] rounded-2xl rounded-br-md',
    assistant: 'bg-[var(--bubble-assistant-bg)] text-[var(--text)] rounded-2xl rounded-bl-md',
  };

  return (
    <div 
      className={`
        px-4 py-3 text-sm leading-relaxed
        ${variants[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      {children}
    </div>
  );
};

// ============================================
// AVATAR
// ============================================

export interface AvatarProps {
  src?: string;
  alt?: string;
  fallback?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const Avatar: React.FC<AvatarProps> = ({
  src,
  alt = '',
  fallback,
  size = 'md',
  className = '',
}) => {
  const sizes: Record<string, string> = {
    sm: 'h-8 w-8 text-xs',
    md: 'h-10 w-10 text-sm',
    lg: 'h-12 w-12 text-base',
  };

  return (
    <div 
      className={`
        rounded-full overflow-hidden flex items-center justify-center
        bg-[var(--surface-2)] border border-[var(--border)]
        text-[var(--text-muted)] font-medium
        ${sizes[size]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      {src ? (
        <img src={src} alt={alt} className="h-full w-full object-cover" />
      ) : (
        fallback
      )}
    </div>
  );
};

// ============================================
// SKELETON LOADER
// ============================================

export interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '' }) => (
  <div 
    className={`
      animate-pulse rounded-md bg-[var(--surface-2)]
      ${className}
    `.trim().replace(/\s+/g, ' ')} 
  />
);
