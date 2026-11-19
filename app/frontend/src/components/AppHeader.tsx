import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { LOGO } from "../api/client";

/** The team dashboard's header (logo, title, subtitle, chips on the right); the logo leads home. */
export function AppHeader({ title, sub, children }: { title: string; sub: string; children?: ReactNode }) {
  return (
    <header>
      <div className="brand">
        <Link to="/" title="Home">
          <img src={LOGO} alt="Salud Revenue Partners" onError={(e) => e.currentTarget.remove()} />
          <div><h1>{title}</h1><div className="sub">{sub}</div></div>
        </Link>
      </div>
      <span className="spacer" />
      <Link className="navlink" to="/">Home</Link>
      {children}
    </header>
  );
}
