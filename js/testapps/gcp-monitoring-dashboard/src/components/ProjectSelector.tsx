import { useQuery } from '@tanstack/react-query';
import { ChevronDown, FolderOpen } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { useDashboard } from '../contexts/DashboardContext';

export function ProjectSelector() {
  const { projectId, setProjectId } = useDashboard();
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  // Fetch projects from backend
  const { data: projectsData } = useQuery({
    queryKey: ['projects'],
    queryFn: () => api.projects(),
    staleTime: 600_000,
  });

  // Fetch auth status to auto-fill project
  const { data: authData } = useQuery({
    queryKey: ['auth', 'status'],
    queryFn: () => api.authStatus(),
    staleTime: 600_000,
  });

  // Auto-set project from auth if not already set
  useEffect(() => {
    if (!projectId && authData?.projectId) {
      setProjectId(authData.projectId);
    }
  }, [projectId, authData, setProjectId]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const projects = projectsData?.projects || [];

  const handleSubmit = () => {
    const val = inputValue.trim();
    if (val) {
      setProjectId(val);
      setInputValue('');
      setOpen(false);
    }
  };

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 rounded-md border border-border bg-surface-2 px-3 py-1.5 text-xs text-text-primary hover:bg-surface-3 transition-colors"
      >
        <FolderOpen className="h-3.5 w-3.5 text-text-tertiary" />
        <span className="max-w-[180px] truncate">
          {projectId || 'Select project...'}
        </span>
        <ChevronDown className="h-3 w-3 text-text-tertiary" />
      </button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-72 rounded-md border border-border bg-surface-2 shadow-lg">
          <div className="p-2">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSubmit();
              }}
            >
              <input
                type="text"
                placeholder="Enter project ID..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                className="w-full rounded border border-border bg-surface-1 px-2 py-1.5 text-xs text-text-primary placeholder-text-tertiary focus:border-accent-blue focus:outline-none"
                autoFocus
              />
            </form>
          </div>

          {projects.length > 0 && (
            <div className="border-t border-border">
              {projects.map((p) => (
                <button
                  key={p.projectId}
                  onClick={() => {
                    setProjectId(p.projectId);
                    setOpen(false);
                  }}
                  className={`flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors ${
                    p.projectId === projectId
                      ? 'bg-accent-blue/10 text-accent-blue'
                      : 'text-text-secondary hover:bg-surface-3 hover:text-text-primary'
                  }`}
                >
                  <FolderOpen className="h-3 w-3 flex-shrink-0" />
                  <span className="truncate">{p.projectId}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
