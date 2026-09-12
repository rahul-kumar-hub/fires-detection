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