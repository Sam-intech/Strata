export default function ResultsPanel({ isLoading, data, error, onReset }: { isLoading?: boolean; data?: any; error?: string | null; onReset: () => void;}) {
  const runId = data?.run_id ?? data?.runId ?? "—";
  const root = data?.final_output ?? data?.finalOutput ?? data ?? {};

  // accept BOTH shapes (old + new)
  const clinical =
    root?.clinical_assessment ??
    root?.clinicalAssessment ??
    root?.clinical ??
    null;

  const labs =
    root?.laboratory_assessment ??
    root?.laboratoryAssessment ??
    root?.laboratory ??
    null;

  const diag =
    root?.diagnostic_assessment ??
    root?.diagnosticAssessment ??
    root?.diagnostic ??
    null;

  const explRoot =
    root?.explanation ??
    root?.explanation_output ??
    root?.explanationOutput ??
    null;

  // --------------------------------------------------------------------------  
  // NEW backend shape: explanation.clinician_report
  const expl = explRoot?.clinician_report ?? explRoot ?? null;

  // -- --------------------------------------------------------------------------
  // meta/system info (new backend currently returns info under `info`)
  const sys =
    root?.meta ??
    root?.system_info ??
    root?.systemInfo ??
    root?.info ??
    null;

  const risk = clinical?.risk_T2D_now;
  const triage = clinical?.triage_label ?? "—";
  const contributors = clinical?.top_contributors ?? {};

  // --------------------------------------------------------------------------
  // NEW backend labs: no lab_evidence; leave table empty for now
  const labEvidence = labs?.lab_evidence ?? [];

  // -- --------------------------------------------------------------------------
  // NEW backend urgency in: laboratory.test_plan.urgency
  const urgency =
    labs?.urgency ??
    labs?.test_plan?.urgency ??
    null;

  // --------------------------------------------------------------------------
  // NEW backend “repeat test” in: laboratory.test_plan.need_retest
  const recommendRepeat =
    typeof labs?.recommend_repeat_test === "boolean"
      ? labs.recommend_repeat_test
      : typeof labs?.test_plan?.need_retest === "boolean"
        ? labs.test_plan.need_retest
        : undefined;

  // --------------------------------------------------------------------------
  // --- normalize old/new keys so UI renders ---
  const assessmentTs =
    sys?.assessment_timestamp ??
    sys?.assessmentTimestamp ??
    sys?.assessment_time ??
    sys?.assessmentTime ??
    null;

  const dataCompleteness =
    sys?.data_completeness ??
    sys?.dataCompleteness ??
    null;

  // diagnostic: old vs new
  const diagLabel =
    diag?.diagnosis_label ?? diag?.diagnosisLabel ?? diag?.label ?? "—";

  const diagBasis =
    diag?.diagnostic_basis ?? diag?.diagnosticBasis ?? "—";

  const diagConfidence =
    diag?.confidence_level ?? diag?.confidenceLevel ?? diag?.confidence ?? null;

  const diagNextStep =
    diag?.recommended_next_step ??
    diag?.recommendedNextStep ??
    diag?.next_step ??
    diag?.nextStep ??
    "—";

  // explanation: old vs new (backend has clinician_report.evidence[])
  const explSummary = expl?.summary ?? "—";
  const explSteps =
    expl?.reasoning_steps ??
    expl?.reasoningSteps ??
    expl?.evidence ?? // clinician_report.evidence[]
    [];
  const explAlignment =
    expl?.clinical_alignment ?? expl?.clinicalAlignment ?? "—";



  // ===========================================================================
  // result ui starts here
  return (
    <div className="w-full max-w-none rounded-xl border border-zinc-200 bg-white p-6 shadow-sm text-left flex flex-col gap-10">
      {/* Header */}
      {/* <div className="flex flex-col gap-2"> */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold">Results</h2>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
              <span className="rounded-md border border-zinc-200 px-2 py-1">
                Run ID: <span className="font-medium text-zinc-700">{runId}</span>
              </span>
              {sys?.assessment_timestamp && (
                <span className="rounded-md border border-zinc-200 px-2 py-1">
                  {formatIso(sys.assessment_timestamp)}
                </span>
              )}
            </div>
          </div>
          <button type="button" onClick={onReset} className="rounded-lg border border-[var(--brand-200)] bg-[var(--brand-100)] px-3 py-2 text-sm hover:bg-[var(--brand-200)]">
            New patient
          </button>
        </div>
      {/* </div> */}


      {/* Main Results Section */}
      <div className="flex flex-col gap-6">
        <div className="mt-0 space-y-6">
          {isLoading && (
            <div className="rounded-lg border border-zinc-200 p-4 text-sm text-zinc-600">
              Running diagnosis…
            </div>
          )}

          {!isLoading && error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              {error}
            </div>
          )}

          {!isLoading && !error && !data && (
            <div className="rounded-lg border border-zinc-200 p-4 text-sm text-zinc-600">
              No result yet.
            </div>
          )}

          {!isLoading && !error && data && (
            <>
              {/* Urgency banner */}
              {urgency && (
                <div
                  className={[
                    "rounded-lg border p-4",
                    urgencyClass(urgency),
                  ].join(" ")}
                >
                  <div className="text-sm font-semibold">
                    {urgencyLabel(urgency)}
                  </div>
                  <div className="mt-1 text-sm opacity-90">
                    {urgencyHint(urgency)}
                  </div>
                </div>
              )}

              {/* Top row: Risk + Triage + Data completeness */}
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-xl border border-zinc-200 p-4">
                  <div className="text-xs text-zinc-500">T2D risk</div>
                  <div className="mt-3 flex items-center gap-4">
                    <RiskRing value={typeof risk === "number" ? risk : null} />
                    <div>
                      <div className="text-2xl font-semibold">
                        {fmtPct(risk)}
                      </div>
                      <div className="mt-1 text-xs text-zinc-500">
                        Model probability
                      </div>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-zinc-200 p-4">
                  <div className="text-xs text-zinc-500">Triage</div>
                  <div className="mt-3">
                    <span
                      className={[
                        "inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold",
                        triageBadgeClass(triage),
                      ].join(" ")}
                    >
                      {String(triage)}
                    </span>
                  </div>
                  {/* {diag?.confidence_level && (
                    <div className="mt-3 text-xs text-zinc-500">
                      Confidence:{" "}
                      <span className="font-medium text-zinc-700">
                        {String(diag.confidence_level)}
                      </span>
                    </div>
                  )} */}
                  {typeof diagConfidence === "number" && (
                    <div className="mt-3 text-xs text-zinc-500">
                      Confidence:{" "}
                      <span className="font-medium text-zinc-700">
                        {(diagConfidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}                    
                </div>

                <div className="rounded-xl border border-zinc-200 p-4">
                  <div className="text-xs text-zinc-500">Data completeness</div>
                  <div className="mt-3 text-2xl font-semibold">
                    {dataCompleteness != null ? fmtPct(dataCompleteness) : "—"}
                  </div>
                  <div className="mt-1 text-xs text-zinc-500">
                    Missing/unknown fields reduce confidence.
                  </div>
                </div>
              </div>


              {/* Explanation */}
              <div className="rounded-xl border border-zinc-200 p-5">
                <div className="text-sm font-semibold text-zinc-900">
                  Explanation
                </div>
                <div className="mt-1 text-xs text-zinc-500">
                  Structured narrative for clinician review.
                </div>
                <div className="mt-4 space-y-4">
                  <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
                    <div className="text-xs font-semibold text-zinc-500">
                      Summary
                    </div>
                    <div className="mt-1 text-sm text-zinc-800">
                      {String(explSummary)}
                    </div>
                  </div>
                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-lg border border-zinc-200 p-4">
                      <div className="text-xs font-semibold text-zinc-500">
                        Reasoning steps
                      </div>
                      {Array.isArray(explSteps) && explSteps.length > 0 ? (
                        <ul className="mt-2 space-y-2 text-sm text-zinc-700">
                          {explSteps.map((s: any, i: number) => (
                            <li key={i} className="flex gap-2">
                              <span className="mt-1 h-2 w-2 flex-shrink-0 rounded-full bg-zinc-900" />
                              <span>{String(s)}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="mt-2 text-sm text-zinc-600">—</div>
                      )}
                    </div>
                    <div className="rounded-lg border border-zinc-200 p-4">
                      <div className="text-xs font-semibold text-zinc-500">
                        Clinical alignment
                      </div>
                      <div className="mt-2 text-sm text-zinc-700">
                        {explAlignment ? String(explAlignment) : "—"}
                      </div>
                    </div>
                  </div>
                </div>
              </div>


              {/* Diagnosis card */}
              <div className="rounded-xl border border-zinc-200 p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-zinc-900">
                      Diagnostic assessment
                    </div>
                    <div className="mt-1 text-xs text-zinc-500">
                      Clinician-facing summary
                    </div>
                  </div>
                  {/* {diag?.diagnosis_label && (
                    <span className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-sm font-semibold text-zinc-900">
                      {String(diag.diagnosis_label)}
                    </span>
                  )} */}
                  {diagLabel && diagLabel !== "—" && (
                    <span className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-sm font-semibold text-zinc-900">
                      {String(diagLabel)}
                    </span>
                  )}
                </div>
                <div className="mt-4 grid gap-4 sm:grid-cols-3">
                  {/* <MiniStat label="Basis" value={diag?.diagnostic_basis ?? "—"} />
                  <MiniStat
                    label="Confidence"
                    value={diag?.confidence_level ?? "—"}
                  />
                  <MiniStat
                    label="Next step"
                    value={prettySnake(diag?.recommended_next_step ?? "—")}
                  /> */}
                  <MiniStat label="Basis" value={diagBasis} />
                  <MiniStat label="Confidence" value={typeof diagConfidence === "number" ? `${(diagConfidence * 100).toFixed(0)}%` : "—"} />
                  <MiniStat label="Next step" value={prettySnake(diagNextStep)} />
                </div>
              </div>


              {/* Contributors + Labs */}
              <div className="grid gap-4 lg:grid-cols-2">
                {/* Top contributors */}
                <div className="rounded-xl border border-zinc-200 p-5">
                  <div className="text-sm font-semibold text-zinc-900">
                    Drivers of risk
                  </div>
                  <div className="mt-1 text-xs text-zinc-500">
                    Relative contribution (not causal).
                  </div>
                  <div className="mt-4 space-y-3">
                    {Object.keys(contributors).length === 0 ? (
                      <div className="text-sm text-zinc-600">—</div>
                    ) : (
                      normalizeContrib(contributors).map(({ key, val }) => (
                        <ContribBar key={key} name={prettyKey(key)} value={val} />
                      ))
                    )}
                  </div>
                </div>

                {/* Labs evidence */}
                <div className="rounded-xl border border-zinc-200 p-5">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold text-zinc-900">
                        Lab evidence
                      </div>
                      <div className="mt-1 text-xs text-zinc-500">
                        Highlighted rows suggest abnormal ranges.
                      </div>
                    </div>
                    {/* {typeof labs?.recommend_repeat_test === "boolean" && (
                      <span
                        className={[
                          "rounded-full border px-3 py-1 text-xs font-semibold",
                          labs.recommend_repeat_test
                            ? "border-amber-200 bg-amber-50 text-amber-800"
                            : "border-emerald-200 bg-emerald-50 text-emerald-800",
                        ].join(" ")}
                      >
                        {labs.recommend_repeat_test
                          ? "Repeat test recommended"
                          : "No repeat needed"}
                      </span>
                    )} */}
                    {typeof recommendRepeat === "boolean" && (
                      <span
                        className={[
                          "rounded-full border px-3 py-1 text-xs font-semibold",
                          recommendRepeat
                            ? "border-amber-200 bg-amber-50 text-amber-800"
                            : "border-emerald-200 bg-emerald-50 text-emerald-800",
                        ].join(" ")}
                      >
                        {recommendRepeat
                          ? "Repeat test recommended"
                          : "No repeat needed"}
                      </span>
                    )}
                  </div>
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-xs text-zinc-500">
                          <th className="pb-2">Test</th>
                          <th className="pb-2">Value</th>
                          <th className="pb-2">Interpretation</th>
                          <th className="pb-2">Recency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.isArray(labEvidence) && labEvidence.length > 0 ? (
                          labEvidence.map((r: any, idx: number) => {
                            const interp = String(r?.interpreted_as ?? "—");
                            const abnormal = isAbnormalInterp(interp);
                            return (
                              <tr
                                key={`${r?.test ?? "lab"}_${idx}`}
                                className={[
                                  "border-t",
                                  abnormal
                                    ? "bg-red-50/50 border-red-100"
                                    : "border-zinc-100",
                                ].join(" ")}
                              >
                                <td className="py-2 font-medium text-zinc-900">
                                  {String(r?.test ?? "—")}
                                </td>
                                <td className="py-2 text-zinc-700">
                                  {String(r?.value ?? "—")}{" "}
                                  <span className="text-xs text-zinc-500">
                                    {String(r?.unit ?? "")}
                                  </span>
                                </td>
                                <td className="py-2">
                                  <span
                                    className={[
                                      "rounded-full border px-2 py-1 text-xs font-semibold",
                                      abnormal
                                        ? "border-red-200 bg-red-50 text-red-800"
                                        : "border-zinc-200 bg-zinc-50 text-zinc-800",
                                    ].join(" ")}
                                  >
                                    {prettySnake(interp)}
                                  </span>
                                </td>
                                <td className="py-2">
                                  {r?.is_recent ? (
                                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-800">
                                      Recent
                                    </span>
                                  ) : (
                                    <span className="rounded-full border border-zinc-200 bg-zinc-50 px-2 py-1 text-xs font-semibold text-zinc-700">
                                      Unknown
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr className="border-t border-zinc-100">
                            <td className="py-3 text-sm text-zinc-600" colSpan={4}>
                              No lab evidence provided.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>


              {/* System info */}
              {/* <div className="rounded-xl border border-zinc-200 p-5">
                <div className="text-sm font-semibold text-zinc-900">
                  System info
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-3">
                  <MiniStat label="Model" value={sys?.model_version ?? "—"} />
                  <MiniStat
                    label="Timestamp"
                    value={sys?.assessment_timestamp ? formatIso(sys.assessment_timestamp) : "—"}
                  />
                  <MiniStat
                    label="Flags"
                    value={
                      Array.isArray(sys?.flags) ? String(sys.flags.length) : "—"
                    }
                  />
                </div>
              </div> */}
            </>
          )}
        </div>
      </div>


      {/* Disclaimer */}
      <p className="w-[70%] text-center text-xs text-zinc-500 mx-auto">
        Disclaimer: This is a Research prototype and should not be used for
        final clinical decision but for decision support only.
      </p>
    </div>
  );
}

/* ---------- small UI helpers ---------- */

function MiniStat({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="mt-1 text-sm font-semibold text-zinc-900">
        {String(value ?? "—")}
      </div>
    </div>
  );
}

function ContribBar({ name, value }: { name: string; value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <div className="font-medium text-zinc-800">{name}</div>
        <div className="text-xs text-zinc-500">{(pct * 100).toFixed(0)}%</div>
      </div>
      <div className="mt-2 h-2 w-full rounded-full bg-zinc-100">
        <div
          className="h-2 rounded-full bg-[var(--brand-600)]"
          style={{ width: `${pct * 100}%` }}
        />
      </div>
    </div>
  );
}

function RiskRing({ value }: { value: number | null }) {
  const v = typeof value === "number" ? Math.max(0, Math.min(1, value)) : 0;
  const r = 18;
  const c = 2 * Math.PI * r;
  const dash = c * v;

  // color by severity
  const ringClass =
    v >= 0.75 ? "stroke-red-500" : v >= 0.5 ? "stroke-amber-500" : "stroke-emerald-500";

  return (
    <div className="relative h-12 w-12">
      <svg viewBox="0 0 44 44" className="h-12 w-12 -rotate-90">
        <circle
          cx="22"
          cy="22"
          r={r}
          fill="none"
          strokeWidth="6"
          className="stroke-zinc-200"
        />
        <circle
          cx="22"
          cy="22"
          r={r}
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c - dash}`}
          className={ringClass}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-zinc-900">
        {typeof value === "number" ? `${Math.round(v * 100)}` : "—"}
      </div>
    </div>
  );
}

/* ---------- formatting + mapping ---------- */

function fmtPct(x: any) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function prettySnake(s: any) {
  if (!s) return "—";
  return String(s).replaceAll("_", " ");
}

function prettyKey(k: string) {
  // contributor keys tend to be short; make them look nicer
  const map: Record<string, string> = {
    hba1c: "HbA1c",
    glucose: "Glucose",
    bmi: "BMI",
    age: "Age",
    hypertension: "Hypertension",
  };
  return map[k] ?? prettySnake(k);
}

function normalizeContrib(obj: Record<string, number>) {
  const entries = Object.entries(obj)
  .filter(([, v]) => typeof v === "number" && !Number.isNaN(v))
  .map(([k, v]) => [k, Math.abs(v)] as const)
  .sort((a, b) => b[1] - a[1]);

  const max = Math.max(...entries.map(([, v]) => v), 0);
  return entries.map(([key, v]) => ({
    key,
    val: max > 0 ? v / max : 0,
  }));
}

function triageBadgeClass(label: string) {
  const s = String(label).toLowerCase();
  if (s.includes("critical") || s.includes("high")) {
    return "border-red-200 bg-red-50 text-red-800";
  }
  if (s.includes("medium") || s.includes("moderate")) {
    return "border-amber-200 bg-amber-50 text-amber-800";
  }
  if (s.includes("low")) {
    return "border-emerald-200 bg-emerald-50 text-emerald-800";
  }
  return "border-zinc-200 bg-zinc-50 text-zinc-800";
}

function urgencyClass(u: string) {
  const s = String(u).toLowerCase();
  if (s.includes("priority") || s.includes("urgent")) {
    return "border-amber-200 bg-amber-50 text-amber-900";
  }
  if (s.includes("routine") || s.includes("low")) {
    return "border-emerald-200 bg-emerald-50 text-emerald-900";
  }
  return "border-zinc-200 bg-zinc-50 text-zinc-900";
}

function urgencyLabel(u: string) {
  const s = String(u).toLowerCase();
  if (s.includes("priority")) return "Priority review required";
  if (s.includes("urgent")) return "Urgent review required";
  if (s.includes("routine")) return "Routine review";
  return prettySnake(u);
}

function urgencyHint(u: string) {
  const s = String(u).toLowerCase();
  if (s.includes("priority")) return "Recommend clinician review and action planning today.";
  if (s.includes("urgent")) return "Recommend clinician review as soon as possible.";
  if (s.includes("routine")) return "Continue routine monitoring and follow-up.";
  return "Review recommended.";
}

function isAbnormalInterp(interpretedAs: string) {
  const s = interpretedAs.toLowerCase();
  return s.includes("diabetes") || s.includes("abnormal") || s.includes("high") || s.includes("positive");
}

function formatIso(iso: string) {
  // keep it simple: show date + time without seconds if present
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
  } catch {
    return String(iso);
  }
}
