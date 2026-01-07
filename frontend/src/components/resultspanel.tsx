export default function ResultsPanel({
  isLoading,
  data,
  error,
  onReset,
}: {
  isLoading?: boolean;
  data?: any;
  error?: string | null;
  onReset: () => void;
}) {
  return (
    <div className="w-full max-w-3xl rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Results</h2>
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm hover:bg-zinc-50"
        >
          New patient
        </button>
      </div>

      <div className="mt-4">
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
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Kpi label="Risk" value={fmt(data?.risk_T2D_now)} />
              <Kpi label="Triage" value={String(data?.triage_label ?? "—")} />
            </div>

            <div>
              <div className="text-sm font-semibold">Next steps</div>
              <ul className="mt-2 list-disc pl-5 text-sm text-zinc-700">
                {(data?.next_steps ?? []).map((x: string) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
            </div>

            <details className="rounded-lg border border-zinc-200 p-3">
              <summary className="cursor-pointer text-sm font-semibold">
                Raw JSON
              </summary>
              <pre className="mt-3 max-h-80 overflow-auto text-xs leading-5">
                {JSON.stringify(data, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>

      <p className="mt-4 text-xs text-zinc-500">
        Research prototype. Decision support only.
      </p>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-4">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}

function fmt(x: any) {
  return typeof x === "number" ? x.toFixed(3) : "—";
}
