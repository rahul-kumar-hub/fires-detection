import { API_BASE_URL } from "../config/app";

import {
  demoEvents,
  getDemoEvent,
} from "../data/demo/events";

import { demoDashboardStats } from "../data/demo/dashboard";
import { demoAnalytics } from "../data/demo/analytics";
import { demoModelInfo } from "../data/demo/model";

import type { AnalysisResult } from "../types/api";

export interface HealthResponse {
  success: boolean;
  message: string;
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }

  return response.json();
}

export async function getEvents() {
  /*
   * BACKEND TODO:
   * Replace demo events with GET /api/events response.
   */
  return demoEvents;
}

export async function getEventById(id: string) {
  /*
   * BACKEND TODO:
   * Connect event details to GET /api/events/{event_id}.
   */
  return getDemoEvent(id);
}

export async function getDashboardStats() {
  /*
   * BACKEND TODO:
   * Replace with dashboard statistics endpoint.
   */
  return demoDashboardStats;
}

export async function getAnalytics() {
  /*
   * BACKEND TODO:
   * Replace demo analytics with aggregated backend data.
   */
  return demoAnalytics;
}

export async function getModelInfo() {
  /*
   * BACKEND TODO:
   * Connect to GET /api/model/info.
   */
  return demoModelInfo;
}

export async function analyzeFile(
  file: File
): Promise<AnalysisResult> {
  /*
   * BACKEND TODO:
   * Replace simulated processing with multipart POST to FastAPI.
   */
  await new Promise((resolve) => setTimeout(resolve, 1800));

  return {
    eventsProcessed: Math.max(
      120,
      Math.round(file.size / 1024)
    ),
    averageConfidence: 0.84,
    highConfidence: 0.62,
    predictionsReady: true,
  };
}


export type ComparisonApiResult = {
  success: boolean;
  latitude: number;
  longitude: number;
  model1: {
    predicted_class: string;
    confidence: number;
    probabilities?: Record<string, number>;
    raw_result?: unknown;
  };
  model2: {
    predicted_class: string;
    confidence: number;
    probabilities?: Record<string, number>;
    raw_result?: unknown;
  };
  final_prediction: {
    predicted_class: string;
    confidence: number;
    selected_model: string;
    reason: string;
  };
};

type PredictionApiResponse = {
  success: boolean;
  model: "model1" | "model2";
  result: {
    prediction?: {
      predicted_class_name?: string;
      class_name?: string;
      predicted_class?: string;
      confidence?: number;
      probabilities?: Record<string, number>;
    };
    final_class?: string;
    class_name?: string;
    confidence?: number;
    probabilities?: Record<string, number>;
  };
};

function normalizePrediction(
  response: PredictionApiResponse
): ComparisonApiResult["model1"] {
  const prediction = response.result.prediction;

  return {
    predicted_class:
      prediction?.predicted_class_name ??
      prediction?.class_name ??
      prediction?.predicted_class ??
      response.result.final_class ??
      response.result.class_name ??
      "Unknown",
    confidence: prediction?.confidence ?? response.result.confidence ?? 0,
    probabilities:
      prediction?.probabilities ?? response.result.probabilities ?? {},
  };
}

async function predictModel(
  model: "model1" | "model2",
  latitude: number,
  longitude: number
) {
  const response = await fetch(`${API_BASE_URL}/api/predict/${model}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ latitude, longitude }),
  });

  if (!response.ok) {
    throw new Error(`Model ${model} prediction failed.`);
  }

  return normalizePrediction(
    (await response.json()) as PredictionApiResponse
  );
}

export function predictModel1(latitude: number, longitude: number) {
  return predictModel("model1", latitude, longitude);
}

export function predictModel2(latitude: number, longitude: number) {
  return predictModel("model2", latitude, longitude);
}

export async function comparePredictions(
  latitude: number,
  longitude: number
): Promise<ComparisonApiResult> {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/compare`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        latitude,
        longitude,
      }),
    }
  );

  if (!response.ok) {
    let message = "Prediction request failed.";

    try {
      const errorBody = await response.json();

      if (typeof errorBody.detail === "string") {
        message = errorBody.detail;
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message);
  }

  return response.json();
}