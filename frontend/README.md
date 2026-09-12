# FIRMS Fire Intelligence

Premium frontend-only React/Vite/TypeScript demo for **FIRMS Fire Intelligence**.

## Run
```bash
npm install
npm run dev
```

## Included
- Landing / Dashboard / Events / Event Details / Map / Analytics
- Model 1 frontend experience
- Model 2 — Snorkel-Assisted Random Forest
- Detailed Model 2 pipeline
- Batch analysis demo flow
- About and Settings
- Responsive layout
- Dark/light/system-ready appearance controls
- Framer Motion page, card, pipeline and micro animations
- Centralized demo data
- FastAPI-ready service layer

## Model 1 note
The source specification supplied for this build defines Model 2 in detail but does not define Model 1's actual algorithm, features, classes, or metrics. Therefore the Model 1 page is intentionally a **frontend integration shell** rather than fabricated scientific/model details. It is ready for the backend team's actual Model 1 response.

## Backend handoff
Replace implementations in `src/services/api.ts`. Pages consume service functions/hooks rather than importing demo data directly.

No backend, database, authentication server, ML inference, model training, or real file processing is included.
