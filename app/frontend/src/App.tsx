import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CallDetail } from "./pages/CallDetail";
import { CallReviews } from "./pages/CallReviews";
import { Home } from "./pages/Home";
import { NotFound } from "./pages/NotFound";
import { Phase2 } from "./pages/Phase2";

// No login: the app is local only for now (see app/README.md).
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="calls" element={<CallReviews />} />
          <Route path="calls/:key" element={<CallDetail />} />
          <Route path="phase2" element={<Phase2 />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
