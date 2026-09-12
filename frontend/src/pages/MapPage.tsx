import { useState } from "react";
import Page from "../components/common/Page";
import GlassCard from "../components/common/GlassCard";
import FireMap from "../components/map/FireMap";
import { ClassBadge } from "../components/common/Badges";
import { demoEvents } from "../data/demo/events";
import { CLASS_ORDER } from "../config/classes";
import type { FireClass } from "../types/fire";

type MapFilter = "ALL" | FireClass;

export default function MapPage() {
  const [filter, setFilter] = useState<MapFilter>("ALL");

  const events =
    filter === "ALL"
      ? demoEvents
      : demoEvents.filter(
          (event) => event.prediction.predictedClass === filter
        );

  return (
    <Page
      title="Fire Map"
      subtitle="Explore classified fire activity geographically."
      wide
    >
      <div className="map-layout">
        <GlassCard className="map-filters">
          <div className="eyebrow">FILTERS</div>

          {["ALL", ...CLASS_ORDER].map((currentFilter) => (
            <button
              className={
                filter === currentFilter
                  ? "map-filter active"
                  : "map-filter"
              }
              onClick={() =>
                setFilter(currentFilter as MapFilter)
              }
              key={currentFilter}
            >
              {currentFilter === "ALL" ? (
                "All sources"
              ) : (
                <ClassBadge value={currentFilter as FireClass} />
              )}

              <span>
                {currentFilter === "ALL"
                  ? demoEvents.length
                  : demoEvents.filter(
                      (event) =>
                        event.prediction.predictedClass ===
                        currentFilter
                    ).length}
              </span>
            </button>
          ))}

          <hr />

          <div className="eyebrow">CONFIDENCE</div>

          <label>
            <input type="checkbox" defaultChecked />
            High confidence
          </label>

          <label>
            <input type="checkbox" />
            Medium confidence
          </label>

          <label>
            <input type="checkbox" />
            Low confidence
          </label>
        </GlassCard>

        <div className="fullscreen-map">
          <FireMap
            events={events}
            height="calc(100vh - 235px)"
          />
        </div>
      </div>

      <div className="backend-note">
        DEMO MAP ·{" "}
        <span>BACKEND TODO:</span> use paginated or viewport-based events
        rather than sending all 302,070 events to the browser.
      </div>
    </Page>
  );
}