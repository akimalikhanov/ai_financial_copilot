import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import * as api from '../services/api';

type AuthContextValue = {
  accessToken: string | null;
  setAccessToken: (token: string | null) => void;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
  refreshTokens: () => Promise<string | null>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const refreshTokens = useCallback(async (): Promise<string | null> => {
    try {
      const data = await api.refreshTokens();
      setAccessToken(data.access_token);
      return data.access_token;
    } catch {
      setAccessToken(null);
      return null;
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setAccessToken(null);
    }
  }, []);

  useEffect(() => {
    api.setAccessTokenGetter(() => accessToken);
    api.setAccessTokenSetter(setAccessToken);
    api.setOnRefreshFailure(() => setAccessToken(null));
  }, [accessToken]);

  useEffect(() => {
    if (accessToken !== null) return;
    refreshTokens();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      accessToken,
      setAccessToken,
      logout,
      isAuthenticated: accessToken !== null,
      refreshTokens,
    }),
    [accessToken, setAccessToken, logout, refreshTokens]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
