import {
  Link,
  useLocation,
  useNavigate,
} from "react-router-dom";

import { useEffect, useState } from "react";

import {
  LogOut,
  Menu,
  Satellite,
  X,
} from "lucide-react";

const links = [
  ["/", "Overview"],
  ["/dashboard", "Dashboard"],
  ["/events", "Events"],
  ["/map", "Map"],
  ["/analytics", "Analytics"],
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const [authenticated, setAuthenticated] = useState(() =>
    Boolean(localStorage.getItem("firms_auth_session"))
  );

  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    setAuthenticated(
      Boolean(localStorage.getItem("firms_auth_session"))
    );
  }, [location.pathname]);

  function closeMenu() {
    setOpen(false);
  }

  function signOut() {
    localStorage.removeItem("firms_auth_session");
    setAuthenticated(false);
    closeMenu();
    navigate("/");
  }

  return (
    <header
      className={`nav ${
        location.pathname === "/" ? "nav-hero" : ""
      }`}
    >
      <Link className="brand" to="/" onClick={closeMenu}>
        <span className="brandmark">
          <Satellite size={17} />
        </span>

        <span>
          FIRMS <b>Fire Intelligence</b>
        </span>
      </Link>

      <nav className={open ? "mobile-open" : ""}>
        {authenticated && (
          <>
            {links.map(([to, label]) => (
              <Link
                key={to}
                to={to}
                onClick={closeMenu}
                className={
                  location.pathname === to ? "active" : ""
                }
              >
                {label}
              </Link>
            ))}

            <Link
              to="/ai-prediction"
              onClick={closeMenu}
              className={
                location.pathname === "/ai-prediction"
                  ? "active"
                  : ""
              }
            >
              AI Prediction
            </Link>

            <button
              className="nav-signout"
              onClick={signOut}
              type="button"
            >
              <LogOut size={14} />
              Sign out
            </button>
          </>
        )}

        {!authenticated && (
          <>
            <Link
              className="nav-auth"
              to="/login"
              onClick={closeMenu}
            >
              Sign in
            </Link>

            <Link
              className="nav-cta"
              to="/signup"
              onClick={closeMenu}
            >
              Register
            </Link>
          </>
        )}
      </nav>

      <button
        className="icon-btn menu"
        onClick={() => setOpen((current) => !current)}
        aria-label="Menu"
        type="button"
      >
        {open ? <X /> : <Menu />}
      </button>
    </header>
  );
}