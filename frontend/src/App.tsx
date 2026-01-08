import { useState } from "react";
import Header from "./components/header";
import PatientForm from "./components/patientforms";
import ResultsPanel from "./components/resultspanel";
// ========================================================

// Replace with real backend call later
async function infer(payload: any) {
  const base = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

  const res = await fetch(`${base}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ payload}),
  });

  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      msg = err?.detail ?? err?.message ?? JSON.stringify(err);
    } catch {
      msg = await res.text();
    }
    throw new Error(msg || `Request failed (${res.status})`);
  }

  return res.json();
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
      const out = await infer(formData);
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

console.log("API base:", import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000");
