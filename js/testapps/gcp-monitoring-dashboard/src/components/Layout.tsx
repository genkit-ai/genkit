import { Activity } from 'lucide-react';
import { Link, Outlet } from 'react-router-dom';
import { ProjectSelector } from './ProjectSelector';
import { TimeRangeSelector } from './TimeRangeSelector';

export function Layout() {
  return (
    <div className="min-h-screen bg-surface-0">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-surface-1/95 backdrop-blur px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="flex items-center gap-2 hover:opacity-80">
              <Activity className="h-5 w-5 text-accent-blue" />
              <h1 className="text-lg font-semibold text-text-primary">
                Genkit
              </h1>
            </Link>
            <span className="text-xs text-text-tertiary">
              Monitoring Dashboard
            </span>
          </div>
          <div className="flex items-center gap-3">
            <TimeRangeSelector />
            <ProjectSelector />
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
