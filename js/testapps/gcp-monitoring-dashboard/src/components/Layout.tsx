import { Activity } from 'lucide-react';
import { Link, Outlet } from 'react-router-dom';
import { ProjectSelector } from './ProjectSelector';
import { TimeRangeSelector } from './TimeRangeSelector';

export function Layout() {
  return (
    <div className="min-h-screen bg-surface-0">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-surface-1/95 backdrop-blur-md px-6 py-3.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
              <Activity className="h-5 w-5 text-accent-blue" />
              <h1 className="text-lg font-semibold text-text-primary">
                Genkit
              </h1>
            </Link>
            <div className="h-4 w-px bg-border/60" />
            <span className="text-xs font-medium text-text-tertiary">
              Google Cloud Monitoring
            </span>
          </div>
          <div className="flex items-center gap-3">
            <TimeRangeSelector />
            <ProjectSelector />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
