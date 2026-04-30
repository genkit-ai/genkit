import { NavLink, Outlet } from 'react-router-dom';

// ---------------------------------------------------------------------------
// App shell — sidebar nav + content outlet for each demo page.
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { to: '/weather', icon: '🌤️', label: 'Weather Chat' },
  { to: '/client-state', icon: '🌤️', label: 'Weather Chat (Stateless)' },
  { to: '/translator', icon: '🌐', label: 'Translator' },
  { to: '/banking', icon: '🏦', label: 'Banking (Interrupt)' },
  { to: '/workspace', icon: '🛠️', label: 'Workspace Builder' },
  { to: '/background', icon: '⏳', label: 'Background (Detach)' },
  { to: '/branching', icon: '🔀', label: 'Branching (Variants)' },
];

export default function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="sidebar-title">🔥 Genkit Agents</h1>
        <p className="sidebar-subtitle">Session Flow Demos</p>

        <nav className="nav-list">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-hint">
            Each page is a self-contained sample showing how to use the{' '}
            <code>genkit/beta/client</code> library.
          </span>
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
