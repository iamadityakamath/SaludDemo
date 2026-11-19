import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { AppHeader } from "../components/AppHeader";

export function Home() {
  const health = useApi(api.health, []);
  return (
    <div className="page-shell">
      <AppHeader title="Salud" sub="Call reviews & payment propensity" />
      <main className="page">
        <h2 className="title">What would you like to open?</h2>
        <p className="lead">Two parts of the project, in one place.</p>
        <div className="home-cards">
          <Link to="/calls" className="home-card">
            <h3>Agent Call Reviews</h3>
            <p>Every call's transcript with caller sentiment, the AI's QA checklist with quotes, summary, outcome and
              coaching tips. Upload new calls and re-analyze.</p>
            <span className="go">Open →</span>
          </Link>
          <Link to="/phase2" className="home-card">
            <h3>Payment Propensity</h3>
            <p>Phase 2: the training data, the features that matter and the models that predict who pays within 30 days.</p>
            <span className="go">Open →</span>
          </Link>
        </div>
        {health.data && !health.data.database && (
          <p className="error">The backend can't reach the database. Check DATABASE_URL in the repository .env.</p>)}
        {health.error && <p className="error">The backend isn't running: start it first (see app/README.md).</p>}
      </main>
    </div>
  );
}
