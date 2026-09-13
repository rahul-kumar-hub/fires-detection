



// cgt3


import { ArrowUp, ChevronDown, Flame, Globe2, MapPin } from "lucide-react";
import { Link } from "react-router-dom";
import "./footer.css";

const footerColumns = [
  {
    title: "Get to Know Us",
    links: [
      { label: "About FIRMS", to: "/about" },
      { label: "Our Mission", to: "/about" },
      { label: "Technology", to: "/model/1" },
      { label: "Careers", to: "/" },
    ],
  },
  {
    title: "Connect with Us",
    links: [
      { label: "Contact Us", to: "/" },
      { label: "GitHub", to: "/" },
      { label: "LinkedIn", to: "/" },
      { label: "Community", to: "/" },
    ],
  },
  {
    title: "Platform & Models",
    links: [
      { label: "Fire Detection", to: "/dashboard" },
      { label: "NASA FIRMS", to: "/events" },
      { label: "Random Forest ML", to: "/model/2" },
      { label: "Spatial Analytics", to: "/analytics" },
    ],
  },
  {
    title: "Let Us Help You",
    links: [
      { label: "Help Center", to: "/" },
      { label: "How It Works", to: "/about" },
      { label: "Report an Issue", to: "/" },
      { label: "Privacy & Security", to: "/" },
    ],
  },
];

const technicalServices = [
  {
    title: "NASA FIRMS",
    description: "Satellite-based active fire detection data",
  },
  {
    title: "Snorkel AI",
    description: "Data labeling and machine learning workflows",
  },
  {
    title: "Random Forest ML",
    description: "Fire-risk prediction and classification",
  },
  {
    title: "Spatial Analytics",
    description: "Geospatial analysis and risk visualization",
  },
];

export default function Footer() {
  const handleBackToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  return (
    <footer className="firms-footer">
      {/* Tier 1: Back to top */}
      <button
        type="button"
        className="footer-back-to-top"
        onClick={handleBackToTop}
      >
        <ArrowUp size={15} strokeWidth={2} />
        <span>Back to top</span>
      </button>

      {/* Tier 2: Main footer columns */}
      <section className="footer-main">
        <div className="footer-main-inner">
          {footerColumns.map((column) => (
            <div className="footer-column" key={column.title}>
              <h3>{column.title}</h3>

              <ul>
                {column.links.map((link) => (
                  <li key={link.label}>
                    <Link to={link.to}>{link.label}</Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Tier 3: Brand and localization */}
      <section className="footer-localization">
        <div className="footer-localization-inner">
          <Link to="/" className="footer-brand">
            <span className="footer-brand-icon">
              <Flame size={20} strokeWidth={2.2} />
            </span>

            <span>
              <strong>FIRMS</strong>
              <span> Fire Intelligence</span>
            </span>
          </Link>

          <div className="footer-localization-actions">
            <button type="button" className="footer-locale-button">
              <Globe2 size={15} />
              <span>English</span>
              <ChevronDown size={14} />
            </button>

            <button type="button" className="footer-locale-button">
              <MapPin size={15} />
              <span>India</span>
            </button>
          </div>
        </div>
      </section>

      {/* Tier 4: Technical services and legal */}
      <section className="footer-bottom">
        <div className="footer-bottom-inner">
          {/* <div className="footer-services-grid">
            {technicalServices.map((service) => (
              <div className="footer-service" key={service.title}>
                <h4>{service.title}</h4>
                <p>{service.description}</p>
              </div>
            ))}
          </div> */}

          <div className="footer-legal">
            <p>
              © {new Date().getFullYear()} FIRMS Fire Intelligence. All rights
              reserved.
            </p>

            <div className="footer-legal-links">
              <Link to="/">Privacy Notice</Link>
              <Link to="/">Terms of Use</Link>
              <Link to="/">Cookie Policy</Link>
            </div>
          </div>
        </div>
      </section>
    </footer>
  );
}


