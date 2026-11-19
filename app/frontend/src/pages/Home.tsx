import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";

export function Home() {
  const health = useApi(api.health, []);
  return (
    <section>
      <h1>Salud</h1>
      <p className="lead">Call reviews and payment propensity in one place.</p>
      <div className="cards">
        <Link to="/calls" className="card">
          <h2>Agent Call Reviews</h2>
          <p>Every call's transcript with caller and agent sentiment, the AI review and Salud's call record.</p>
          <span className="card-cta">Open →</span>
        </Link>
        <Link to="/phase2" className="card">
          <h2>Payment Propensity</h2>
          <p>Phase 2: the training data, the features that matter and the models that predict who pays.</p>
          <span className="card-cta">Open →</span>
        </Link>
      </div>
      {health.data && health.data.status !== "ok" && (
        <p className="error">The backend can't reach the database. Check DATABASE_URL in the repository .env.</p>
      )}
      {health.error && <p className="error">The backend isn't running: start it first (see app/README.md).</p>}
    </section>
  );
}
