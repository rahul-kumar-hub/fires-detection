export const APP_NAME="FIRMS Fire Intelligence";
export const TAGLINE="From satellite fire detections to source intelligence.";
export const DEMO_MODE=(import.meta.env.VITE_DEMO_MODE ?? "true") !== "false";
export const API_BASE_URL=import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";