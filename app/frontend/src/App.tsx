import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AgentDashboard } from "./pages/AgentDashboard";
import { Home } from "./pages/Home";
import { NotFound } from "./pages/NotFound";
import { Phase2 } from "./pages/Phase2";

// No login: the app is local only for now (see app/README.md).
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route index element={<Home />} />
        <Route path="calls" element={<AgentDashboard />} />
        <Route path="phase2" element={<Phase2 />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}
