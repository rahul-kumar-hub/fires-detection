import {
  Flame,
  Factory,
  Wheat,
  Pickaxe,
  Satellite,
  ArrowRight,
  BrainCircuit,
  Network,
} from "lucide-react";

import { Link } from "react-router-dom";
import { useEffect, useState } from "react";

import Page from "../components/common/Page";
import GlassCard from "../components/common/GlassCard";
import StatCard from "../components/common/StatCard";
import FireMap from "../components/map/FireMap";
import PredictionCard from "../components/dashboard/PredictionCard";
import { useDashboard } from "../hooks/useDashboard";
import { demoEvents } from "../data/demo/events";
import { checkBackendHealth } from "../services/api";

export default function Dashboard() {
  const { data } = useDashboard();

  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "offline"
  >("checking");

  useEffect(() => {
    let mounted = true;

    async function verifyBackend() {
      try {
        await checkBackendHealth();

        if (mounted) {
          setBackendStatus("connected");
        }
      } catch {
        if (mounted) {
          setBackendStatus("offline");
        }
      }
    }

    verifyBackend();

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <Page
      title="Fire Intelligence"
      subtitle="Monitor and analyze classified fire events."
    >
      <section className="dashboard-model-choice">
        <div>
          <div className="eyebrow">CHOOSE YOUR MODEL VIEW</div>

          <h2>Two ways to read the same fire signal.</h2>

          <p>
            Use Model 1 for contextual source evidence or Model 2 for
            event-based weak supervision and final classification.
          </p>
        </div>

        <div className="model-options">
          <Link className="model-option" to="/model/1">
            <span className="model-option-icon">
              <BrainCircuit size={19} />
            </span>

            <span>
              <b>Model 1</b>
              <small>Contextual source classifier</small>
            </span>

            <ArrowRight size={16} />
          </Link>

          <Link className="model-option active" to="/model">
            <span className="model-option-icon">
              <Network size={19} />
            </span>

            <span>
              <b>Model 2</b>
              <small>Snorkel event intelligence</small>
            </span>

            <ArrowRight size={16} />
          </Link>
        </div>
      </section>

      <div className="kpis">
        <StatCard
          label="Total Fire Events"
          value={data?.totalEvents.toLocaleString() ?? "302,070"}
          icon={<Satellite />}
        />

        <StatCard
          label="High Confidence"
          value={data?.highConfidence.toLocaleString() ?? "184,230"}
          icon={<Flame />}
        />

        <StatCard
          label="Wildfire"
          value={data?.wildfire.toLocaleString() ?? "109,820"}
          icon={<Flame />}
        />

        <StatCard
          label="Agriculture"
          value={data?.agriculture.toLocaleString() ?? "72,140"}
          icon={<Wheat />}
        />

        <StatCard
          label="Industrial"
          value={data?.industrial.toLocaleString() ?? "49,380"}
          icon={<Factory />}
        />

        <StatCard
          label="Mining"
          value={data?.mining.toLocaleString() ?? "36,110"}
          icon={<Pickaxe />}
        />

        <StatCard
          label="Gas"
          value={data?.gas.toLocaleString() ?? "34,620"}
          icon={<Flame />}
        />
      </div>

      <div className="dashboard-grid">
        <GlassCard className="map-card">
          <div className="card-head">
            <div>
              <div className="eyebrow">LIVE GEOSPATIAL VIEW</div>
              <h3>Fire events across the region</h3>
            </div>

            <span className="mode-pill">
              <i /> DEMO
            </span>
          </div>

          <FireMap events={demoEvents} />
        </GlassCard>

        <PredictionCard event={demoEvents[0]} />
      </div>

      <div className="backend-note">
        BACKEND STATUS ·{" "}
        <span>
          {backendStatus === "checking" && "CHECKING CONNECTION..."}
          {backendStatus === "connected" && "FASTAPI CONNECTED"}
          {backendStatus === "offline" && "FASTAPI OFFLINE"}
        </span>
      </div>
    </Page>
  );
}