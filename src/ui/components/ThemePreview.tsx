import React, { useState } from 'react';
import { Sun, Moon, Check, AlertTriangle, Info, X, Send, Search, User, Bot, Code, FileText } from 'lucide-react';
import {
  Button,
  Input,
  Badge,
  Card,
  Separator,
  Toggle,
  CodeBlock,
  ChatBubble,
  Avatar,
  Skeleton,
  Textarea
} from './ui';

/**
 * Theme Preview Component
 *
 * A comprehensive showcase of all design tokens and components.
 * Use this to verify the theme looks correct in both dark and light modes.
 */

export const ThemePreview: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [toggleValue, setToggleValue] = useState(true);

  // Toggle theme
  React.useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    document.documentElement.classList.toggle('light', !isDarkMode);
  }, [isDarkMode]);

  return (
    <div className="min-h-screen bg-[var(--bg)] p-8 transition-colors">
      <div className="max-w-5xl mx-auto space-y-12">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-[var(--text)] mb-2">Theme Preview</h1>
            <p className="text-[var(--text-muted)]">
              ChatGPT-inspired design system with OpenAI green accent
            </p>
          </div>
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--surface-2)] border border-[var(--border)] text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors"
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            <span className="text-sm font-medium">{isDarkMode ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>

        <Separator />

        {/* Color Palette */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Color Palette</h2>

          {/* Backgrounds */}
          <div className="mb-8">
            <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Backgrounds</h3>
            <div className="grid grid-cols-4 gap-4">
              {[
                { name: 'bg', label: 'Base' },
                { name: 'surface-1', label: 'Surface 1' },
                { name: 'surface-2', label: 'Surface 2' },
                { name: 'surface-3', label: 'Surface 3' },
              ].map(({ name, label }) => (
                <div key={name} className="space-y-2">
                  <div
                    className="h-20 rounded-lg border border-[var(--border)]"
                    style={{ backgroundColor: `var(--${name})` }}
                  />
                  <div className="text-xs">
                    <div className="font-medium text-[var(--text)]">{label}</div>
                    <div className="text-[var(--text-faint)] font-mono">--{name}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Text Colors */}
          <div className="mb-8">
            <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Text</h3>
            <div className="space-y-2 bg-[var(--surface-1)] p-4 rounded-lg border border-[var(--border)]">
              <p className="text-[var(--text)]">Primary text (--text) - High contrast for main content</p>
              <p className="text-[var(--text-muted)]">Muted text (--text-muted) - Secondary information</p>
              <p className="text-[var(--text-faint)]">Faint text (--text-faint) - Tertiary/placeholder text</p>
              <p className="text-[var(--accent)]">Accent text (--accent) - Links and highlights</p>
            </div>
          </div>

          {/* Accent & Semantic */}
          <div className="mb-8">
            <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Accent & Semantic</h3>
            <div className="grid grid-cols-5 gap-4">
              {[
                { name: 'accent', label: 'Accent' },
                { name: 'accent-hover', label: 'Accent Hover' },
                { name: 'success', label: 'Success' },
                { name: 'warning', label: 'Warning' },
                { name: 'danger', label: 'Danger' },
              ].map(({ name, label }) => (
                <div key={name} className="space-y-2">
                  <div
                    className="h-16 rounded-lg"
                    style={{ backgroundColor: `var(--${name})` }}
                  />
                  <div className="text-xs">
                    <div className="font-medium text-[var(--text)]">{label}</div>
                    <div className="text-[var(--text-faint)] font-mono">--{name}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <Separator />

        {/* Buttons */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Buttons</h2>
          <div className="space-y-6">

            {/* Variants */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Variants</h3>
              <div className="flex flex-wrap gap-4">
                <Button variant="primary">Primary</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="danger">Danger</Button>
              </div>
            </div>

            {/* Sizes */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Sizes</h3>
              <div className="flex flex-wrap items-center gap-4">
                <Button size="sm">Small</Button>
                <Button size="md">Medium</Button>
                <Button size="lg">Large</Button>
                <Button size="icon"><Send size={16} /></Button>
              </div>
            </div>

            {/* States */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">States</h3>
              <div className="flex flex-wrap gap-4">
                <Button>Default</Button>
                <Button disabled>Disabled</Button>
                <Button loading>Loading</Button>
              </div>
            </div>
          </div>
        </section>

        <Separator />

        {/* Form Elements */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Form Elements</h2>
          <div className="grid grid-cols-2 gap-8">

            {/* Inputs */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Inputs</h3>
              <Input placeholder="Default input" />
              <div className="relative">
                <Search className="absolute left-3 top-2.5 text-[var(--text-faint)]" size={16} />
                <Input placeholder="With icon" className="pl-10" />
              </div>
              <Input placeholder="Disabled input" disabled />
            </div>

            {/* Textarea & Toggle */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wide">Textarea & Toggle</h3>
              <Textarea placeholder="Write something..." rows={3} />
              <div className="flex items-center gap-4 pt-2">
                <Toggle checked={toggleValue} onChange={setToggleValue} label="Toggle switch" />
              </div>
            </div>
          </div>
        </section>

        <Separator />

        {/* Badges */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Badges</h2>
          <div className="flex flex-wrap gap-3">
            <Badge variant="default">Default</Badge>
            <Badge variant="outline">Outline</Badge>
            <Badge variant="accent">Accent</Badge>
            <Badge variant="success">Success</Badge>
            <Badge variant="warning">Warning</Badge>
            <Badge variant="danger">Danger</Badge>
          </div>
        </section>

        <Separator />

        {/* Cards */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Cards</h2>
          <div className="grid grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="font-semibold text-[var(--text)] mb-2">Default Card</h3>
              <p className="text-sm text-[var(--text-muted)]">
                A standard card with subtle border and surface-1 background.
              </p>
            </Card>
            <Card variant="elevated" className="p-6">
              <h3 className="font-semibold text-[var(--text)] mb-2">Elevated Card</h3>
              <p className="text-sm text-[var(--text-muted)]">
                A card with shadow elevation for modal-like appearances.
              </p>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Chat Bubbles */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Chat Bubbles</h2>
          <div className="max-w-2xl space-y-4">
            <div className="flex gap-3 justify-end">
              <ChatBubble variant="user">
                What are the key financial metrics for Q3 2024?
              </ChatBubble>
            </div>
            <div className="flex gap-3">
              <Avatar fallback={<Bot size={16} />} size="sm" />
              <ChatBubble variant="assistant">
                Based on the quarterly report, here are the key metrics:
                <br /><br />
                • Revenue: $45.2B (+12% YoY)<br />
                • Operating Margin: 28.5%<br />
                • Free Cash Flow: $8.1B
              </ChatBubble>
            </div>
          </div>
        </section>

        <Separator />

        {/* Code Block */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Code Block</h2>
          <CodeBlock language="typescript">
{`// Example: Theme configuration
const theme = {
  dark: {
    bg: '#0B0F14',
    accent: '#10A37F',
    text: '#E6EDF5',
  },
  light: {
    bg: '#FFFFFF',
    accent: '#10A37F',
    text: '#0B1220',
  },
};`}
          </CodeBlock>
        </section>

        <Separator />

        {/* Avatars & Skeleton */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Avatars & Skeleton</h2>
          <div className="flex items-center gap-8">
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide">Avatars</h3>
              <div className="flex items-center gap-3">
                <Avatar size="sm" fallback={<User size={14} />} />
                <Avatar size="md" fallback={<User size={16} />} />
                <Avatar size="lg" fallback={<User size={20} />} />
              </div>
            </div>
            <div className="flex-1 space-y-3">
              <h3 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide">Skeleton Loaders</h3>
              <div className="flex items-center gap-3">
                <Skeleton className="h-10 w-10 rounded-full" />
                <div className="space-y-2 flex-1">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            </div>
          </div>
        </section>

        <Separator />

        {/* Focus States Demo */}
        <section>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-6">Focus States</h2>
          <p className="text-[var(--text-muted)] mb-4 text-sm">
            Tab through these elements to see the green focus ring (WCAG compliant).
          </p>
          <div className="flex flex-wrap gap-4">
            <Button>Focus me</Button>
            <Input placeholder="Focus me" className="w-48" />
            <Toggle checked={false} onChange={() => {}} />
          </div>
        </section>

        {/* Footer */}
        <div className="pt-8 pb-4 text-center text-sm text-[var(--text-faint)]">
          Theme tokens are defined in <code className="font-mono bg-[var(--surface-2)] px-1.5 py-0.5 rounded">index.css</code> and <code className="font-mono bg-[var(--surface-2)] px-1.5 py-0.5 rounded">theme.ts</code>
        </div>

      </div>
    </div>
  );
};

export default ThemePreview;
