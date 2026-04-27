import React, { useState } from 'react';
import * as api from '../services/api';
import { Button, Input, Card } from './ui';

type Mode = 'login' | 'register';

interface LoginPageProps {
  onSuccess: (accessToken: string) => void;
}

export function LoginPage({ onSuccess }: LoginPageProps) {
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === 'login') {
        const { access_token } = await api.login(email, password);
        onSuccess(access_token);
      } else {
        const { access_token } = await api.register(email, password, displayName || undefined);
        onSuccess(access_token);
      }
    } catch (err) {
      const apiErr = err as api.ApiError;
      const friendlyMessage =
        apiErr.statusCode === 401
          ? 'Invalid email or password.'
          : apiErr.statusCode === 400
            ? 'Please check your email and password and try again.'
            : apiErr.message ?? 'Authentication failed. Please try again.';
      setError(friendlyMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleModeSwitch = () => {
    setMode(m => m === 'login' ? 'register' : 'login');
    setError(null);
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4"
      style={{ background: 'radial-gradient(ellipse at 50% 40%, var(--surface-1) 0%, var(--bg) 70%)' }}
    >
      <div className="w-full max-w-sm animate-fade-in">
        {/* Wordmark above card */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 bg-[var(--accent)] rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-[var(--shadow-md)]">
            <span className="font-bold text-white font-mono text-xl">AI</span>
          </div>
          <h1 className="text-2xl font-semibold text-[var(--text)] tracking-tight">Financial Copilot</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Analyze financial documents with AI precision.</p>
        </div>

        <Card className="p-8 shadow-[var(--shadow-lg)]" variant="elevated">
          <p className="text-sm text-[var(--text-muted)] text-center mb-6">
            {mode === 'login' ? 'Sign in to continue' : 'Create an account'}
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5">Email</label>
              <Input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoComplete="email"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5">Display name (optional)</label>
                <Input
                  type="text"
                  value={displayName}
                  onChange={e => setDisplayName(e.target.value)}
                  placeholder="Your name"
                  autoComplete="name"
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-[var(--text-muted)] mb-1.5">Password</label>
              <Input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                minLength={6}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
            </div>

            {error && (
              <div className="text-sm text-[var(--danger)] bg-[var(--danger-bg)] px-3 py-2 rounded-lg border border-[var(--danger)] border-opacity-20">
                {error}
              </div>
            )}

            <Button type="submit" className="w-full" loading={loading} disabled={loading}>
              {mode === 'login' ? 'Sign in' : 'Create account'}
            </Button>
          </form>

          <button
            type="button"
            onClick={handleModeSwitch}
            className="w-full mt-5 text-sm text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors"
          >
            {mode === 'login' ? "Don't have an account? Register" : 'Already have an account? Sign in'}
          </button>
        </Card>
      </div>
    </div>
  );
}
