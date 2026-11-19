import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { AppHeader } from "../components/AppHeader";

const pct = (v: number | null) => (v === null ? "—" : `${Math.round(v * 100)}%`);

export function Phase2() {
  const s = useApi(api.phase2Summary, []);
  return (
    <div className="page-shell">
      <AppHeader title="Payment Propensity" sub="Who is likely to pay within 30 days" />
      <main className="page">
        <h2 className="title">Payment propensity</h2>
        <p className="lead">Before a call (who to call first) and after it (follow-up priority).</p>
        {s.loading && <p className="lead"><span className="spin" />Loading…</p>}
        {s.error && <p className="error">Couldn't load: {s.error}</p>}
        {s.data && !s.data.available && (
          <p className="lead">The Phase 2 master file isn't on this machine (Phase 2/reports/pre_training_data.xlsx).</p>)}
        {s.data?.available && (
          <>
            <h3 className="section-h">Training data · {s.data.rows.toLocaleString()} calls · {s.data.date_range.first_call} to {s.data.date_range.last_call}</h3>
            <table className="table">
              <thead><tr><th>Data</th><th>Split</th><th>Calls</th><th>Patients</th><th>Usable</th>
                <th>Paid within 30 days</th><th>Paid after the call</th></tr></thead>
              <tbody>
                {s.data.by_part.map((p) => (
                  <tr key={`${p.data_type}-${p.split}`}>
                    <td>{p.data_type}</td><td>{p.split}</td><td>{p.calls.toLocaleString()}</td><td>{p.patients.toLocaleString()}</td>
                    <td>{p.usable.toLocaleString()}</td><td>{pct(p.pay_rate_pre_call)}</td><td>{pct(p.pay_rate_post_call)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3 className="section-h">Feature sets</h3>
            <div className="sets">
              {Object.entries(s.data.feature_sets).map(([model, sets]) => (
                <div className="card" key={model}>
                  <div className="card-head">{model}</div>
                  <ul>{Object.entries(sets).map(([name, feats]) => (
                    <li key={name}><b>{name}</b> ({feats.length}): {feats.slice(0, 6).join(", ")}{feats.length > 6 ? ", …" : ""}</li>
                  ))}</ul>
                </div>
              ))}
            </div>
            <p className="reason">Model results will appear here as the six model-family tasks come back.</p>
          </>
        )}
      </main>
    </div>
  );
}
