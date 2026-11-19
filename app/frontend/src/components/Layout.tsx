import { Link, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">Salud</Link>
        <nav>
          <Link to="/calls">Agent Call Reviews</Link>
          <Link to="/phase2">Payment Propensity</Link>
        </nav>
      </header>
      <main className="page"><Outlet /></main>
    </div>
  );
}
