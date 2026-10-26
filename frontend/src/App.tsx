import { Link, Navigate, Route, Routes } from "react-router-dom";
import AdminPage from "./pages/AdminPage";
import CustomerPage from "./pages/CustomerPage";

export default function App() {
  return (
    <div className="min-h-screen font-sans text-ink">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="border-b border-sand/80 bg-mist/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <Link to="/" className="font-display text-xl tracking-tight">
            Atelier Returns
          </Link>
          <nav className="flex items-center gap-5 text-sm">
            <Link className="hover:text-clay" to="/">
              Start a return
            </Link>
            <Link className="hover:text-clay" to="/admin">
              Reviewer desk
            </Link>
          </nav>
        </div>
      </header>
      <main id="main">
        <Routes>
          <Route path="/" element={<CustomerPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
