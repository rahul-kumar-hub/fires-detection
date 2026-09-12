import { FormEvent, useState } from "react";
import Page from "../components/common/Page";
import GlassCard from "../components/common/GlassCard";
import { comparePredictions } from "../services/api";
import "../style.css"

type ModelResult = {
  predicted_class: string;
  confidence: number;
  probabilities?: Record<string, number>;
};

type FinalPrediction = {
  predicted_class: string;
  confidence: number;
  selected_model: string;
  reason: string;
};

type ComparisonResult = {
  model1: ModelResult;
  model2: ModelResult;
  final_prediction: FinalPrediction;
};

function formatConfidence(value: number) {
  const percentage = value <= 1 ? value * 100 : value;
  return `${percentage.toFixed(2)}%`;
}

function ConfidenceBar({ value }: { value: number }) {
  const percentage = Math.max(
    0,
    Math.min(100, value <= 1 ? value * 100 : value)
  );

  return (
    <div className="ai-confidence-track">
      <div
        className="ai-confidence-fill"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

function ModelResultCard({
  title,
  model,
  accent,
}: {
  title: string;
  model: ModelResult;
  accent: "blue" | "orange";
}) {
  return (
    <GlassCard className={`ai-model-card ${accent}`}>
      <div className="ai-model-card-top">
        <div>
          <span className="ai-card-kicker">MODEL OUTPUT</span>
          <h3>{title}</h3>
        </div>

        <span className="ai-model-badge">
          <span />
          ONLINE
        </span>
      </div>

      <div className="ai-prediction-label">Predicted class</div>

      <div className="ai-prediction-value">
        {model.predicted_class || "Unknown"}
      </div>

      <div className="ai-confidence-header">
        <span>Confidence score</span>
        <strong>{formatConfidence(model.confidence)}</strong>
      </div>

      <ConfidenceBar value={model.confidence} />

      {model.probabilities &&
        Object.keys(model.probabilities).length > 0 && (
          <div className="ai-probability-list">
            <div className="ai-probability-heading">
              <span>Class probabilities</span>
              <span>Score</span>
            </div>

            {Object.entries(model.probabilities).map(
              ([label, probability]) => (
                <div className="ai-probability-row" key={label}>
                  <span>{label}</span>
                  <strong>{formatConfidence(probability)}</strong>
                </div>
              )
            )}
          </div>
        )}
    </GlassCard>
  );
}

export default function AIPrediction() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setError("");
    setResult(null);

    const lat = Number(latitude);
    const lon = Number(longitude);

    if (!latitude || !longitude) {
      setError("Please enter both latitude and longitude.");
      return;
    }

    if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
      setError("Latitude must be between -90 and 90.");
      return;
    }

    if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
      setError("Longitude must be between -180 and 180.");
      return;
    }

    try {
      setLoading(true);

      // Your API expects two separate arguments.
      const response = await comparePredictions(lat, lon);

      setResult(response);
    } catch (requestError) {
      console.error(requestError);

      setError(
        "Unable to generate prediction. Please check the backend connection."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Page
      title="AI Prediction"
      subtitle="Compare both intelligence models and generate a final land-cover prediction."
      wide
    >
      <section className="ai-prediction-page">
        <GlassCard className="ai-command-card">
          <div className="ai-command-header">
            <div>
              <span className="ai-card-kicker">PREDICTION ENGINE</span>

              <h2>Run a location analysis</h2>

              <p>
                Enter a geographic point to compare Model 1 and Model 2
                predictions.
              </p>
            </div>

            <div className="ai-command-icon">⌁</div>
          </div>

          <form
            onSubmit={handleSubmit}
            className="ai-prediction-form"
            noValidate
          >
            <div className="ai-location-grid">
              <label className="ai-field">
                <span>Latitude</span>

                <input
                  type="number"
                  step="any"
                  min="-90"
                  max="90"
                  placeholder="28.6139"
                  value={latitude}
                  onChange={(event) => setLatitude(event.target.value)}
                />

                <small>Range: -90 to 90</small>
              </label>

              <label className="ai-field">
                <span>Longitude</span>

                <input
                  type="number"
                  step="any"
                  min="-180"
                  max="180"
                  placeholder="77.2090"
                  value={longitude}
                  onChange={(event) => setLongitude(event.target.value)}
                />

                <small>Range: -180 to 180</small>
              </label>
            </div>

            {error && (
              <div className="ai-error-message" role="alert">
                {error}
              </div>
            )}

            <div className="ai-form-footer">
              <div className="ai-form-status">
                <span />
                Ready for analysis
              </div>

              <button
                type="submit"
                className="ai-run-button"
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Run prediction"}
                <span>↗</span>
              </button>
            </div>
          </form>
        </GlassCard>

        {result && (
          <section className="ai-results-section">
            <div className="ai-results-heading">
              <div>
                <span className="ai-card-kicker">ANALYSIS RESULTS</span>
                <h2>Model comparison</h2>
              </div>

              <div className="ai-coordinate-pill">
                {Number(latitude).toFixed(4)},{" "}
                {Number(longitude).toFixed(4)}
              </div>
            </div>

            <div className="ai-model-grid">
              <ModelResultCard
                title="Model 1"
                model={result.model1}
                accent="blue"
              />

              <ModelResultCard
                title="Model 2"
                model={result.model2}
                accent="orange"
              />
            </div>

            <GlassCard className="ai-final-card">
              <div className="ai-final-icon">✓</div>

              <div>
                <span className="ai-card-kicker">FINAL DECISION</span>

                <h2>{result.final_prediction.predicted_class}</h2>

                <p>{result.final_prediction.reason}</p>

                <div className="ai-final-details">
                  <span>
                    Confidence:{" "}
                    <strong>
                      {formatConfidence(
                        result.final_prediction.confidence
                      )}
                    </strong>
                  </span>

                  <span>
                    Selected model:{" "}
                    <strong>
                      {result.final_prediction.selected_model}
                    </strong>
                  </span>
                </div>
              </div>

              <span className="ai-final-badge">CONFIRMED</span>
            </GlassCard>
          </section>
        )}
      </section>
    </Page>
  );
}