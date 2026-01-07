import { useState } from "react";
import Header from "./components/header";
import PatientForm from "./components/patientforms";
import ResultsPanel from "./components/resultspanel";
// ========================================================

// Replace with real backend call later
async function fakeInfer(_payload: any) {
  await new Promise((r) => setTimeout(r, 700));
  return {
    run_id: "run_2026_01_07_001",
    final_output: {
      clinical_assessment: {
        risk_T2D_now: 0.82,
        triage_label: "critical",
        top_contributors: {
          hba1c: 0.31,
          glucose: 0.27,
          bmi: 0.14,
          age: 0.06,
          hypertension: 0.04,
        },
        raw_proba_vector: [0.18, 0.82],
      },
      laboratory_assessment: {
        lab_evidence: [
          { test: "HbA1c", value: 52, unit: "mmol/mol", interpreted_as: "diabetes_range", is_recent: true },
          { test: "FPG", value: 7.4, unit: "mmol/L", interpreted_as: "diabetes_range", is_recent: true },
        ],
        urgency: "priority",
        recommend_repeat_test: false,
      },
      diagnostic_assessment: {
        diagnosis_label: "T2D",
        diagnostic_basis: "HbA1c",
        confidence_level: "high",
        recommended_next_step: "confirm_diagnosis_and_initiate_management",
      },
      explanation: {
        summary: "The patient demonstrates a high probability of Type 2 Diabetes.",
        reasoning_steps: [
          "HbA1c is in the diagnostic range for diabetes (≥48 mmol/mol).",
          "Fasting plasma glucose exceeds diagnostic thresholds.",
          "BMI and age increase baseline metabolic risk.",
        ],
        clinical_alignment: "Consistent with NICE and ADA diagnostic criteria.",
      },
      meta: {
        model_version: "clinical_v1.2",
        assessment_timestamp: "2026-01-07T14:32:10Z",
        data_completeness: 0.93,
        flags: [],
      },
    },
  };
}

export default function App() {
  const [isLoading, setIsLoading] = useState(false);

  // Only becomes true AFTER we have a response (success or error)
  const [showResults, setShowResults] = useState(false);

  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const [formKey, setFormKey] = useState(0);

  const run = async (formData: any) => {
    setIsLoading(true);
    setError(null);

    try {
      const out = await fakeInfer(formData); // swap to backend later
      setResult(out);
      setShowResults(true); // IMPORTANT: animate only after results arrive
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setResult(null);
      setShowResults(true); // still slide to show error panel
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setShowResults(false);
    setResult(null);
    setError(null);
    setIsLoading(false);
    setFormKey((k) => k + 1); // force PatientForm remount (clears inputs)
  };

  // =======================================================
  // UI starts here
  return (
    <div className="min-h-screen w-full bg-gray-50">
      <Header />

      {/* header is fixed h-16; pt-16 offsets content */}
      <main className="w-full pt-16">
        {/* Full-height workspace under header */}
        <div className="h-[calc(100vh-64px)] w-full px-5 py-6">
          {/* Push-style split layout */}
          <div className="flex h-full w-full items-start gap-6">
            {/* LEFT: Patient form (always mounted) */}
            <section
              className={[
                "transition-all duration-500 ease-in-out flex justify-center overflow-y-auto max-h-[calc(100vh-64px-48px)] pr-1",
                showResults ? "w-[30%]" : "w-full",
              ].join(" ")}
            >
              <div
                className={[
                  "transition-all duration-500 ease-in-out",
                  showResults ? "w-full" : "w-full max-w-3xl",
                ].join(" ")}
              >
                <PatientForm key={formKey} isLoading={isLoading} onSubmit={run} />
              </div>
            </section>

            {/* RIGHT: Results panel (slides in + grows to 70%) */}
            <section
              className={[
                "transition-all duration-500 ease-in-out overflow-y-auto max-h-[calc(100vh-64px-48px)] pr-1",
                showResults ? "w-[70%]" : "w-0",
              ].join(" ")}
            >
              <div
                className={[
                  "h-full transition-all duration-500 ease-in-out",
                  showResults ? "translate-x-0 opacity-100" : "translate-x-10 opacity-0 pointer-events-none",
                ].join(" ")}
              >
                <ResultsPanel isLoading={isLoading} data={result} error={error} onReset={reset} />
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
