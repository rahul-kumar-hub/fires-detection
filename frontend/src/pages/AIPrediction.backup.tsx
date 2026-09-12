import { FormEvent, useState } from "react";
import Page from "../components/common/Page";
import GlassCard from "../components/common/GlassCard";
import { comparePredictions } from "../services/api";

type PredictionResult = {
  predicted_class: string;
  confidence: number;
  probabilities?: Record<string, number>;
};

type ModelResult = PredictionResult & {
  raw_result?: unknown;
};

type ComparisonResult = {
  success: boolean;
  latitude: number;
  longitude: number;
  model1: ModelResult;
  model2: ModelResult;
  final_prediction: {
    predicted_class: string;
    confidence: number;
    selected_model: string;
    reason: string;
  };
};

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function ModelResultCard({
  title,
  result,
  highlighted = false,
}: {
  title: string;
  result: ModelResult;
  highlighted?: boolean;
}) {
  return (
    <GlassCard
      className={`ai-model-result ${
        highlighted ? "ai-model-result-selected" : ""
      }`}
    >
      <div className="ai-result-header">
        <div>
          <div className="eyebrow">{title}</div>
          <h2>{result.predicted_class}</h2>
        </div>

        {highlighted && (
          <span className="ai-selected-badge">
            Highest confidence
          </span>
        )}
      </div>

      <div className="ai-confidence">
        <span>Confidence</span>
        <strong>{formatConfidence(result.confidence)}</strong>
      </div>

      {result.probabilities &&
        Object.keys(result.probabilities).length > 0 && (
          <div className="ai-probabilities">
            <div className="eyebrow">CLASS PROBABILITIES</div>

            {Object.entries(result.probabilities).map(
              ([className, probability]) => (
                <div className="ai-probability-row" key={className}>
                  <span>{className}</span>
                  <strong>
                    {formatProbability(Number(probability))}
                  </strong>
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

  const [result, setResult] =
    useState<ComparisonResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setError("");
    setResult(null);

    const parsedLatitude = Number(latitude);
    const parsedLongitude = Number(longitude);

    if (!latitude.trim() || !longitude.trim()) {
      setError("Please enter both latitude and longitude.");
      return;
    }

    if (
      !Number.isFinite(parsedLatitude) ||
      parsedLatitude < -90 ||
      parsedLatitude > 90
    ) {
      setError("Latitude must be between -90 and 90.");
      return;
    }

    if (
      !Number.isFinite(parsedLongitude) ||
      parsedLongitude < -180 ||
      parsedLongitude > 180
    ) {
      setError("Longitude must be between -180 and 180.");
      return;
    }

    try {
      setLoading(true);

      const response = await comparePredictions(
        parsedLatitude,
        parsedLongitude
      );

      setResult(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Prediction failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  const selectedModel =
    result?.final_prediction.selected_model === "model1"
      ? "Model 1"
      : "Model 2";

  return (
    <Page
      title="AI Prediction"
      subtitle="Run both fire-source models on the same geographic coordinates."
      wide
    >
      <section className="ai-prediction-page">
        <GlassCard className="ai-input-card">
          <div className="eyebrow">LOCATION INPUT</div>

          <h2>Enter fire location</h2>

          <p className="ai-description">
            Both AI models will run separately using the same
            latitude and longitude. The final prediction is selected
            from the model with the highest confidence.
          </p>

          <form className="ai-input-form" onSubmit={handleSubmit}>
            <label>
              Latitude
              <input
                type="number"
                step="any"
                min="-90"
                max="90"
                placeholder="Example: 21.92127"
                value={latitude}
                onChange={(event) =>
                  setLatitude(event.target.value)
                }
                required
              />
            </label>

            <label>
              Longitude
              <input
                type="number"
                step="any"
                min="-180"
                max="180"
                placeholder="Example: 83.3507"
                value={longitude}
                onChange={(event) =>
                  setLongitude(event.target.value)
                }
                required
              />
            </label>

            <button
              className="primary-button ai-run-button"
              type="submit"
              disabled={loading}
            >
              {loading
                ? "Running both models..."
                : "Run AI Prediction"}
            </button>
          </form>

          {error && (
            <div className="ai-error" role="alert">
              {error}
            </div>
          )}
        </GlassCard>

        {loading && (
          <GlassCard className="ai-loading-card">
            <div className="eyebrow">PROCESSING</div>
            <h3>
              Model 1 and Model 2 are running...
            </h3>
            <p>
              This may take some time because both models collect
              geographic and satellite-related features.
            </p>
          </GlassCard>
        )}

        {result && !loading && (
          <>
            <div className="ai-location-summary">
              <span>
                Latitude: <strong>{result.latitude}</strong>
              </span>
              <span>
                Longitude: <strong>{result.longitude}</strong>
              </span>
            </div>

            <section className="ai-model-grid">
              <ModelResultCard
                title="MODEL 1"
                result={result.model1}
                highlighted={
                  result.final_prediction.selected_model ===
                  "model1"
                }
              />

              <ModelResultCard
                title="MODEL 2"
                result={result.model2}
                highlighted={
                  result.final_prediction.selected_model ===
                  "model2"
                }
              />
            </section>

            <GlassCard className="ai-final-card">
              <div className="eyebrow">FINAL AI PREDICTION</div>

              <h2>{result.final_prediction.predicted_class}</h2>

              <div className="ai-final-details">
                <div>
                  <span>Selected model</span>
                  <strong>{selectedModel}</strong>
                </div>

                <div>
                  <span>Final confidence</span>
                  <strong>
                    {formatConfidence(
                      result.final_prediction.confidence
                    )}
                  </strong>
                </div>
              </div>

              <p>{result.final_prediction.reason}</p>
            </GlassCard>
          </>
        )}
      </section>
    </Page>
  );
}
