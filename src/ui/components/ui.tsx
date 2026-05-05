import React from 'react';
import { Loader2, ChevronDown } from 'lucide-react';

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

    const baseStyle = [
      'inline-flex items-center justify-center font-sans font-medium',
      'transition-all duration-150 ease-[cubic-bezier(.2,.8,.2,1)]',
      'focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)]',
      'disabled:opacity-50 disabled:cursor-not-allowed',
      'active:translate-y-[1px]',
    ].join(' ');

    const variants: Record<string, string> = {
      primary: [
        'bg-gradient-to-b from-[var(--accent)] to-[var(--accent-pressed)]',
        'dark:[background-image:none] dark:bg-[var(--accent)]',
        'text-white border border-transparent',
        'shadow-xs hover:shadow-sm',
        'hover:from-[var(--accent-hover)] hover:to-[var(--accent-pressed)]',
        'dark:hover:[background-image:none] dark:hover:bg-[var(--accent-hover)]',
      ].join(' '),
      secondary: [
        'bg-[var(--surface-2)] text-[var(--text)]',
        'border border-[var(--border)]',
        'shadow-xs hover:bg-[var(--surface-3)] hover:shadow-sm',
      ].join(' '),
      ghost: [
        'bg-transparent text-[var(--text-muted)]',
        'hover:text-[var(--text)] hover:bg-[var(--surface-2)]',
        'border border-transparent',
        'focus-visible:ring-inset',
      ].join(' '),
      danger: [
        'bg-[var(--danger-bg)] text-[var(--danger)]',
        'border border-transparent',
        'shadow-xs hover:shadow-sm',
        'hover:bg-[var(--danger)] hover:text-white',
      ].join(' '),
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
        className={[
          'flex h-11 w-full rounded-md',
          'border-[1.5px] border-[var(--input-border)]',
          'bg-[var(--input-bg)]',
          'px-3 py-2 text-sm text-[var(--text)]',
          'placeholder:text-[var(--placeholder)]',
          'transition-colors duration-150',
          'focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-2 focus:ring-[var(--focus-ring)]',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'font-sans',
          className,
        ].join(' ')}
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
        className={[
          'flex w-full rounded-md',
          'border-[1.5px] border-[var(--input-border)]',
          'bg-[var(--input-bg)]',
          'px-3 py-2 text-sm text-[var(--text)]',
          'placeholder:text-[var(--placeholder)]',
          'transition-colors duration-150',
          'focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-2 focus:ring-[var(--focus-ring)]',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'font-sans resize-none',
          className,
        ].join(' ')}
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

// className applies to the outer wrapper (for sizing); the <select> is always w-full inside.
export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <div className={`relative ${className}`}>
        <select
          ref={ref}
          className={[
            'appearance-none h-11 w-full rounded-md',
            'border-[1.5px] border-[var(--input-border)]',
            'bg-[var(--input-bg)]',
            'px-3 pr-10 text-sm text-[var(--text)]',
            'transition-colors duration-150',
            'focus:outline-none focus:border-[var(--input-border-focus)] focus:ring-2 focus:ring-[var(--focus-ring)]',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'font-sans cursor-pointer',
          ].join(' ')}
          {...props}
        >
          {children}
        </select>
        <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--icon)]" />
      </div>
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

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'default', className = '' }) => {
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
      className={[
        'inline-flex items-center px-2 py-0.5 rounded-md whitespace-nowrap',
        'text-xs font-medium font-sans tracking-wide',
        'border',
        variants[variant],
        className,
      ].join(' ')}
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

export const Card: React.FC<CardProps> = ({ children, className = '', variant = 'default' }) => {
  const variants: Record<string, string> = {
    default: 'bg-[var(--surface-1)] border border-[var(--border)] shadow-xs',
    elevated: 'bg-[var(--surface-2)] border border-[var(--border)] shadow-md',
  };

  return (
    <div className={['rounded-xl', variants[variant], className].join(' ')}>
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
  orientation = 'horizontal',
}) => (
  <div
    className={[
      orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]',
      'bg-[var(--border)]',
      className,
    ].join(' ')}
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
      className={[
        'inline-flex items-center gap-2 cursor-pointer',
        disabled ? 'opacity-50 cursor-not-allowed' : '',
        className,
      ].join(' ')}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange?.(!checked)}
        className={[
          'relative inline-flex h-6 w-11 shrink-0',
          'cursor-pointer rounded-full border-2 border-transparent',
          'transition-colors duration-200 [transition-timing-function:cubic-bezier(.5,1.5,.5,1)]',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)]',
          checked ? 'bg-[var(--accent)]' : 'bg-[var(--surface-3)]',
          disabled ? 'cursor-not-allowed' : '',
        ].join(' ')}
      >
        <span
          className={[
            'pointer-events-none inline-block h-5 w-5',
            'transform rounded-full bg-white ring-0',
            'shadow-sm',
            'transition-[transform,box-shadow] duration-200 [transition-timing-function:cubic-bezier(.5,1.5,.5,1)]',
            checked ? 'translate-x-5 shadow-md' : 'translate-x-0',
          ].join(' ')}
        />
      </button>
      {label && <span className="text-sm text-[var(--text)]">{label}</span>}
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

export const CodeBlock: React.FC<CodeBlockProps> = ({ children, language, className = '' }) => {
  return (
    <div
      className={[
        'rounded-lg overflow-hidden',
        'border border-[var(--code-border)]',
        'bg-[var(--code-bg)]',
        className,
      ].join(' ')}
    >
      {language && (
        <div className="px-4 py-2 border-b border-[var(--code-border)] text-xs text-[var(--text-faint)] font-mono">
          {language}
        </div>
      )}
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm font-mono text-[var(--text)]">{children}</code>
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

export const ChatBubble: React.FC<ChatBubbleProps> = ({ children, variant, className = '' }) => {
  if (variant === 'user') {
    return (
      <div
        className={[
          'max-w-content px-4 py-3 text-sm',
          'leading-[var(--leading-relaxed)]',
          'bg-[var(--bubble-user-bg)] text-[var(--text)]',
          'rounded-2xl shadow-xs',
          className,
        ].join(' ')}
      >
        {children}
      </div>
    );
  }

  return (
    <div
      className={[
        'max-w-content px-4 py-3 pl-5 text-sm',
        'leading-[var(--leading-relaxed)]',
        'bg-[var(--bubble-assistant-bg)] text-[var(--text)]',
        'rounded-2xl',
        'border-l-2 border-[var(--accent-subtle)]',
        className,
      ].join(' ')}
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
      className={[
        'rounded-full overflow-hidden flex items-center justify-center',
        'bg-[var(--surface-2)] border border-[var(--border)]',
        'text-[var(--text-muted)] font-medium',
        sizes[size],
        className,
      ].join(' ')}
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
  <div className={['rounded-md animate-shimmer', className].join(' ')} />
);

// ============================================
// TOOLTIP
// ============================================

export interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  side = 'top',
  className = '',
}) => {
  const positions: Record<string, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <div className={`relative inline-flex group ${className}`}>
      {children}
      <div
        role="tooltip"
        className={[
          'pointer-events-none absolute z-50 whitespace-nowrap',
          'px-2.5 py-1.5 rounded-md text-xs font-medium',
          'bg-[var(--surface-3)] text-[var(--text)] border border-[var(--border)]',
          'shadow-md',
          'opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0',
          'transition-all duration-150 ease-out',
          positions[side],
        ].join(' ')}
      >
        {content}
      </div>
    </div>
  );
};

// ============================================
// ICON BUTTON
// ============================================

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  'aria-label': string;
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ size = 'md', ...props }, ref) => (
    <Button ref={ref} variant="ghost" size="icon" {...props} />
  )
);
IconButton.displayName = 'IconButton';

// ============================================
// KBD
// ============================================

export interface KbdProps {
  children: React.ReactNode;
  className?: string;
}

export const Kbd: React.FC<KbdProps> = ({ children, className = '' }) => (
  <kbd
    className={[
      'inline-flex items-center justify-center',
      'px-1.5 py-0.5 rounded',
      'text-[11px] font-mono font-medium',
      'bg-[var(--surface-2)] text-[var(--text-muted)]',
      'border border-[var(--border)] border-b-[var(--border-strong)]',
      'shadow-xs',
      className,
    ].join(' ')}
  >
    {children}
  </kbd>
);
