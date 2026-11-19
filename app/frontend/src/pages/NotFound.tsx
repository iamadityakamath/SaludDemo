import { Link } from "react-router-dom";
import { AppHeader } from "../components/AppHeader";

export function NotFound() {
  return (
    <div className="page-shell">
      <AppHeader title="Salud" sub="Page not found" />
      <main className="page"><h2 className="title">Page not found</h2><p><Link className="navlink" to="/">Back to the home page</Link></p></main>
    </div>
  );
}
