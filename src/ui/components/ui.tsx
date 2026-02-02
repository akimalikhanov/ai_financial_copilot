import React from 'react';
import { Loader2 } from 'lucide-react';

// Button
export const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'danger', size?: 'sm' | 'md' | 'icon' }>(
  ({ className = '', variant = 'primary', size = 'md', children, disabled, ...props }, ref) => {
    
    const baseStyle = "inline-flex items-center justify-center font-sans font-medium transition-all focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2 focus:ring-offset-zinc-950 disabled:opacity-50 disabled:pointer-events-none";
    
    const variants = {
      primary: "bg-zinc-100 text-zinc-900 hover:bg-zinc-200 border border-transparent",
      secondary: "bg-zinc-900 text-zinc-300 border border-zinc-700 hover:bg-zinc-800 hover:text-zinc-100",
      ghost: "bg-transparent text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50",
      danger: "bg-red-900/20 text-red-400 border border-red-900/50 hover:bg-red-900/40"
    };

    const sizes = {
      sm: "h-8 px-3 text-xs rounded-sm",
      md: "h-10 px-4 text-sm rounded-sm",
      icon: "h-9 w-9 p-0 rounded-sm"
    };

    return (
      <button
        ref={ref}
        className={`${baseStyle} ${variants[variant]} ${sizes[size]} ${className}`}
        disabled={disabled}
        {...props}
      >
        {disabled && variant === 'primary' ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

// Input
export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className = '', ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`flex h-10 w-full rounded-sm border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-accent-500 focus:ring-1 focus:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50 font-sans ${className}`}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

// Badge
export const Badge = ({ children, variant = 'default', className = '' }: { children?: React.ReactNode, variant?: 'default' | 'outline' | 'success' | 'warning', className?: string }) => {
  const styles = {
    default: "bg-zinc-800 text-zinc-300",
    outline: "border border-zinc-700 text-zinc-400",
    success: "bg-emerald-950/30 text-emerald-400 border border-emerald-900/50",
    warning: "bg-amber-950/30 text-amber-400 border border-amber-900/50"
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium font-mono ${styles[variant]} ${className}`}>
      {children}
    </span>
  );
};

// Card
export const Card = ({ children, className = '' }: { children?: React.ReactNode, className?: string }) => (
  <div className={`bg-zinc-900/40 border border-zinc-800 backdrop-blur-sm ${className}`}>
    {children}
  </div>
);

// Separator
export const Separator = ({ className = '', orientation = 'horizontal' }: { className?: string, orientation?: 'horizontal' | 'vertical' }) => (
  <div className={`${orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]'} bg-zinc-800 ${className}`} />
);