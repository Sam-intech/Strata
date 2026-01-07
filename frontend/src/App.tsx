import { useState } from "react";
import Header from "./components/header";
import PatientForm from "./components/patientforms";
import ResultsPanel from "./components/resultspanel";

// Replace with real backend call later
async function fakeInfer(payload: any) {
  await new Promise((r) => setTimeout(r, 700));
  return {
    risk_T2D_now: 0.42,
    triage_label: "medium",
    next_steps: ["order_diagnostic_labs"],
    data_flags: { hba1c_missing: true },
    raw: payload,
  };
}

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<"center" | "split">("center");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (formPayload: any) => {
    setIsLoading(true);
    setError(null);
    setStage("split");

    try {
      // Later: replace with real backend call
      const out = await fakeInfer(formPayload);
      setResult(out);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setStage("center");
    setResult(null);
    setError(null);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen w-full bg-gray-50">
      <Header />

      {/* header fixed at top (h-16) */}
      <main className="w-full px-15 pt-16">
        <div className="relative h-[calc(100vh-64px)] w-full overflow-hidden">
          {/* CENTER STAGE: scrollable + centered */}
          <section
            className={[
              "absolute inset-0 overflow-y-auto transition-transform duration-500 ease-in-out",
              stage === "center" ? "translate-x-0" : "-translate-x-full",
            ].join(" ")}
          >
            <div className="flex min-h-full items-center justify-center px-4 py-10">
              <PatientForm isLoading={isLoading} onSubmit={run} />
            </div>
          </section>

          {/* SPLIT STAGE: left scroll + right scroll */}
          <section
            className={[
              "absolute inset-0 flex w-full gap-6 px-4 transition-transform duration-500 ease-in-out",
              stage === "split" ? "translate-x-0" : "translate-x-full",
            ].join(" ")}
          >
            {/* Left: form (scrollable column) */}
            <div className="h-full w-full overflow-y-auto py-10">
              <div className="flex justify-start">
                <PatientForm isLoading={isLoading} onSubmit={run} />
              </div>
            </div>

            {/* Right: results (desktop) */}
            <div className="hidden h-full w-full overflow-y-auto py-10 md:block">
              <div className="flex justify-end">
                <ResultsPanel
                  isLoading={isLoading}
                  data={result}
                  error={error}
                  onReset={reset}
                />
              </div>
            </div>
          </section>

          {/* Mobile results drawer */}
          <section
            className={[
              "absolute inset-x-0 bottom-0 md:hidden transition-transform duration-500 ease-in-out",
              stage === "split" ? "translate-y-0" : "translate-y-full",
            ].join(" ")}
          >
            <div className="border-t border-zinc-200 bg-white p-4">
              <ResultsPanel
                isLoading={isLoading}
                data={result}
                error={error}
                onReset={reset}
              />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
