# System wspomagania klasyfikacji zmian skórnych

Praca inżynierska — system wspomagający klasyfikację zmian skórnych na obrazach dermatoskopowych
z wykorzystaniem głębokich sieci neuronowych (transfer learning, EfficientNet), udostępniany
personelowi medycznemu poprzez aplikację webową.

## Struktura repozytorium

- `ml/` — przygotowanie danych, trening i ewaluacja modelu klasyfikacyjnego (PyTorch)
- `backend/` — REST API (FastAPI) obsługujące inferencję modelu i historię przypadków (PostgreSQL)
- `frontend/` — aplikacja webowa (React) dla personelu medycznego

## Status

Projekt w fazie wczesnej implementacji.

- ✅ Dataset (HAM10000, podział po `lesion_id`) i model bazowy (EfficientNet-B0, transfer
  learning, dwuetapowy trening) — `ml/`
- ✅ Grad-CAM (skrypt eksperymentalny, jeszcze niepodłączony do backendu)
- 🔧 Backend (FastAPI + PostgreSQL) — w budowie: `GET /health`, `POST /predict` (inferencja +
  zapis przypadku), `GET/DELETE /cases` (historia przypadków), przesłane zdjęcia serwowane pod
  `/storage/uploads`
- ⬜ Frontend (React), Grad-CAM w API, konteneryzacja (Docker Compose), testy i dokumentacja

Aktualna roadmapa: dataset → model bazowy → Grad-CAM → backend → frontend → konteneryzacja →
testy i dokumentacja.
