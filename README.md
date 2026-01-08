# Multi-Agent Clinical Decision Support System (Type 2 Diabetes)

This repository contains a **Multi-Agent System (MAS)** for clinical decision support in **Type 2 Diabetes (T2D) risk assessment and triage**, developed as part of an **MSc Artificial Intelligence dissertation**.

The system integrates multiple specialised agents (data handling, clinical risk assessment, laboratory reasoning, diagnostic reasoning, and explanation generation) into a single orchestrated pipeline with a web-based frontend.

---

## 1. Project Overview

**Goal**: Demonstrate that a multi-agent architecture provides advantages over a traditional single-model baseline in:

* Predictive performance
* Robustness to incomplete/noisy data
* Interpretability and clinical reasoning transparency

**Target users**: Clinicians (not patients).

**Primary output**: A structured clinical assessment with explanations suitable for clinical review.

---

## 2. System Architecture

**Backend (Python)**

* Orchestrated via a LangGraph-based workflow
* Modular agents:

  * `DataHandlingAgent` – schema validation & preprocessing
  * `ClinicalAssessmentAgent` – ML-based T2D risk estimation + triage
  * `LaboratoryAgent` – lab recency & diagnostic relevance reasoning
  * `DiagnosticAgent` – guideline-driven diagnostic interpretation
  * `ExplanationAgent` – clinician-oriented explanation synthesis

**Frontend (React + TypeScript)**

* Patient data input form
* Results panel rendering structured outputs
* Explicit display of reasoning steps, lab evidence, and clinical alignment

---

## 3. Repository Structure

```
.
├── backend/
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── clinical_agent.py
│   │   ├── lab_agent.py
│   │   ├── diagnostic_agent.py
│   │   └── explanation_agent.py
│   ├── orchestrator.py
│   ├── llm_client.py
│   ├── main.py
│   └── evaluation/
│       └── evaluate_mas.ipynb
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── PatientForms.tsx
│   │   │   └── ResultsPanel.tsx
│   │   └── types.ts
│   └── index.css
│
├── artifacts/
│   ├── clinical_model.joblib
│   └── eval/
│
├── data/
│   └── raw/
│
├── README.md
└── requirements.txt
```

---

## 4. Running the System

### 4.1 Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py
```

The backend exposes an inference pipeline callable via the orchestrator.

---

### 4.2 Frontend

```bash
npm install
npm run dev
```

The frontend consumes the backend response and renders structured clinical results.

---

## 5. Input & Output Contract

### Input (example)

```json
{
  "age": 45,
  "gender": "male",
  "bmi": 31.2,
  "glucose": 158,
  "hba1c": 52,
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "never"
}
```

### Output (simplified shape)

```json
{
  "run_id": "run_local_001",
  "final_output": {
    "clinical_assessment": { ... },
    "laboratory_assessment": { ... },
    "diagnostic_assessment": { ... },
    "explanation": { ... },
    "meta": {
      "api_version": "1.0",
      "model_version": "clinical_v1",
      "data_completeness": 0.87
    }
  }
}
```

---

## 6. Evaluation

The evaluation framework supports:

* MAS performance (Accuracy, Precision, Recall, F1, ROC–AUC)
* Comparison against a single-model baseline
* Robustness testing (missing/noisy inputs)

Run evaluation via:

```bash
jupyter notebook backend/evaluation/evaluate_mas.ipynb
```

All results are saved to:

```
artifacts/eval/
```

---

## 7. Reproducibility

* Fixed random seeds for training and evaluation
* Identical dataset splits for MAS and baseline
* Versioned model and pipeline metadata included in outputs

---

## 8. Scope & Limitations

* Research prototype only (not for clinical deployment)
* Not a diagnostic tool; supports clinical decision-making
* Evaluation limited to retrospective datasets

---

## 9. Dissertation Context

This system was developed as part of an **MSc Artificial Intelligence dissertation** and is intended to demonstrate:

* Practical application of multi-agent systems in healthcare AI
* Trade-offs between monolithic ML models and agent-based architectures
* The role of explainability in clinical AI systems

---

## 10. License

Academic use only.
