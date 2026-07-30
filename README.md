# System wspomagania klasyfikacji zmian skórnych

Praca inżynierska — system wspomagający klasyfikację zmian skórnych na obrazach dermatoskopowych
z wykorzystaniem głębokich sieci neuronowych (transfer learning, EfficientNet), udostępniany
personelowi medycznemu poprzez aplikację webową.

## Struktura repozytorium

- `ml/` — przygotowanie danych, trening i ewaluacja modelu klasyfikacyjnego (PyTorch)
- `backend/` — REST API (FastAPI) obsługujące inferencję modelu i historię przypadków (PostgreSQL)
- `frontend/` — aplikacja webowa (React) dla personelu medycznego

## Status

Projekt w fazie wczesnej implementacji. Aktualna roadmapa: dataset (HAM10000) → model bazowy →
Grad-CAM → backend → frontend → konteneryzacja (Docker Compose) → testy i dokumentacja.
