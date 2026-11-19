import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { Status } from "../components/Badges";

const pct = (v: number | null) => (v === null ? "—" : `${Math.round(v * 100)}%`);

export function Phase2() {
  const s = useApi(api.phase2Summary, []);
  return (
    <section>
      <h1>Payment Propensity</h1>
      <p className="lead">Who is likely to pay within 30 days: before a call (who to call first) and after it.</p>
      <Status loading={s.loading} error={s.error} />
      {s.data && !s.data.available && (
        <p className="muted">The Phase 2 master file isn't on this machine (Phase 2/reports/pre_training_data.xlsx).</p>
      )}
      {s.data?.available && (
        <>
          <h2>Training data</h2>
          <p className="muted">{s.data.rows.toLocaleString()} calls · {s.data.date_range.first_call} to {s.data.date_range.last_call}</p>
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
          <h2>Feature sets</h2>
          <div className="grid">
            {Object.entries(s.data.feature_sets).map(([model, sets]) => (
              <div className="panel" key={model}>
                <h3>{model}</h3>
                <ul>{Object.entries(sets).map(([name, feats]) => (
                  <li key={name}><b>{name}</b> ({feats.length}): {feats.slice(0, 6).join(", ")}{feats.length > 6 ? ", …" : ""}</li>
                ))}</ul>
              </div>
            ))}
          </div>
          <p className="muted">Model results will appear here as the six model-family tasks come back.</p>
        </>
      )}
    </section>
  );
}
