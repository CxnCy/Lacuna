# Project Lacuna — Architecture

## Purpose

Lacuna is a temporal machine-learning research system that investigates whether historical structural and temporal patterns in scientific literature can predict future research activity between previously weakly connected or underexplored research topics.

## Core Pipeline

OpenAlex Data
→ Temporal Knowledge Network
→ Candidate Lacuna Generation
→ Historical Feature Engineering
→ Temporal ML Training and Backtesting
→ Prediction Engine
→ Backend API
→ Interactive Atlas

## Backend Modules

- `data` — Scientific data acquisition, cleaning, validation, and caching.
- `network` — Temporal scientific knowledge-network construction.
- `candidates` — Generation of plausible candidate topic pairs.
- `features` — Historical feature engineering and label construction.
- `models` — Model training, evaluation, and temporal backtesting.
- `prediction` — Prediction ranking, inference, and explanations.
- `api` — Backend endpoints consumed by the Atlas.
- `core` — Shared configuration and infrastructure.

## Data Storage

- `data/raw` — Original acquired data.
- `data/interim` — Intermediate transformed data.
- `data/processed` — Cleaned datasets ready for analysis.

## Generated Artifacts

- `artifacts/graphs` — Historical network snapshots.
- `artifacts/datasets` — ML-ready datasets.
- `artifacts/models` — Trained models.
- `artifacts/results` — Evaluation and experiment results.

## Frontend

The frontend is built with React, TypeScript, and Vite.

The final Atlas rendering technology will be selected later based on actual network size, spatial-layout requirements, and performance testing.

## Architectural Principle

All information used to make a prediction at historical cutoff T must have been available at or before T.

The scientific network used by the machine-learning system and the structure visualised by the Atlas must derive from the same underlying scientific data.