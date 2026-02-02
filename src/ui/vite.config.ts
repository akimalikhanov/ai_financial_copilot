import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Resolve env directory relative to UI root (goes up to repo root)
const ENV_DIR = path.resolve(__dirname, '../..');

export default defineConfig({
  // Load .env files from repo root (VITE_* vars exposed to client)
  envDir: ENV_DIR,

  server: {
    port: 3000,
    host: '0.0.0.0',
  },

  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
});
