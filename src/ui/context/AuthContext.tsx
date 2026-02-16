import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import * as api from '../services/api';

type AuthContextValue = {
  accessToken: string | null;
  setAccessToken: (token: string | null) => void;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
  authChecked: boolean;
  refreshTokens: () => Promise<string | null>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

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
    if (accessToken !== null) {
      setAuthChecked(true);
      return;
    }
    refreshTokens().finally(() => setAuthChecked(true));
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      accessToken,
      setAccessToken,
      logout,
      isAuthenticated: accessToken !== null,
      authChecked,
      refreshTokens,
    }),
    [accessToken, setAccessToken, logout, authChecked, refreshTokens]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
