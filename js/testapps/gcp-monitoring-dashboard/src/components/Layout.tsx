import { Activity } from 'lucide-react';
import { Link, Outlet } from 'react-router-dom';

export function Layout() {
  return (
    <div className="min-h-screen bg-surface-0">
      {/* Header */}
      <header className="border-b border-border bg-surface-1 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="flex items-center gap-2 hover:opacity-80">
              <Activity className="h-5 w-5 text-accent-blue" />
              <h1 className="text-lg font-semibold text-text-primary">
                Genkit
              </h1>
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs text-text-tertiary">
              Monitoring Dashboard
            </span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="p-6">
        <Outlet />
      </main>
    </div>
  );
}
