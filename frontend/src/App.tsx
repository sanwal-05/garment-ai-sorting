import { Link, NavLink, Navigate, Route, Routes } from "react-router-dom";
import AdminPage from "./pages/AdminPage";
import CustomerPage from "./pages/CustomerPage";

export default function App() {
  return (
    <div className="min-h-screen font-sans text-charcoal">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2 focus:shadow-md"
      >
        Skip to content
      </a>
      <header className="border-b border-sand/70 bg-white/70 backdrop-blur-md sticky top-0 z-30 transition-all">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3.5">
          <Link to="/" className="group flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-sage/20 text-sage-dark font-display font-semibold transition-transform group-hover:scale-105">
              A
            </span>
            <div className="leading-tight">
              <span className="font-display text-lg tracking-tight text-charcoal block">Atelier Returns</span>
              <span className="text-[10px] uppercase tracking-wider text-stone-500 font-medium">AI Garment Vetting</span>
            </div>
          </Link>
          <nav className="flex items-center gap-1.5 rounded-full border border-sand/80 bg-sand/35 p-1 text-xs sm:text-sm font-medium shadow-xs">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `rounded-full px-4 py-1.5 transition-all flex items-center gap-1.5 ${
                  isActive
                    ? "bg-white text-charcoal font-bold shadow-xs border border-sand/70"
                    : "text-stone-600 hover:text-charcoal hover:bg-white/60"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span className={`h-2 w-2 rounded-full ${isActive ? "bg-sage ring-2 ring-sage/30" : "bg-stone-300"}`} />
                  <span>Start a return</span>
                </>
              )}
            </NavLink>
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                `rounded-full px-4 py-1.5 transition-all flex items-center gap-1.5 ${
                  isActive
                    ? "bg-white text-charcoal font-bold shadow-xs border border-sand/70"
                    : "text-stone-600 hover:text-charcoal hover:bg-white/60"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span className={`h-2 w-2 rounded-full ${isActive ? "bg-dusty ring-2 ring-dusty/30" : "bg-stone-300"}`} />
                  <span>Reviewer desk</span>
                </>
              )}
            </NavLink>
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
