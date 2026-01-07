import { useMemo, useState } from "react";
// ==============================================

type Sex = "male" | "female" | "not_say";
type Smoking = "never" | "former" | "current" | "unknown";
type YesNo = 0 | 1;

type LabKey = "hba1c" | "fpg" | "ogtt" | "egfr";

export default function PatientForm({
  isLoading,
  onSubmit,
}: {
  isLoading?: boolean;
  onSubmit: (data: any) => void;
}) {
  // Core
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<Sex>("male");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [bmi, setBmi] = useState("");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [smoking, setSmoking] = useState<Smoking>("never");
  const [hypertension, setHypertension] = useState<YesNo>(0);
  const [heartDisease, setHeartDisease] = useState<YesNo>(0);

  // Labs selector + collapse
  const [labsOpen, setLabsOpen] = useState(false);
  const [selectedLabs, setSelectedLabs] = useState<LabKey[]>([]);

  // Labs fields
  const [hba1c, setHba1c] = useState("");
  const [hba1cUnit, setHba1cUnit] = useState("%");
  const [fpg, setFpg] = useState("");
  const [fpgUnit, setFpgUnit] = useState("mmol/L");
  const [ogtt, setOgtt] = useState("");
  const [ogttUnit, setOgttUnit] = useState("mmol/L");
  const [egfr, setEgfr] = useState("");

  function toggleLab(k: LabKey) {
    setSelectedLabs((prev) =>
      prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k]
    );
  }
  const has = (k: LabKey) => selectedLabs.includes(k);

  // Calculate BMI if height & weight entered
  function handleHeightWeightChange(h: string, w: string) {
    setHeight(h);
    setWeight(w);
    if (h && w) {
      const hM = parseFloat(h) / 100;
      const bmiVal = parseFloat(w) / (hM * hM);
      setBmi(bmiVal ? bmiVal.toFixed(1) : "");
    } else {
      setBmi("");
    }
  }

  // Basic guardrails (optional but useful)
  const canSubmit = useMemo(() => {
    if (!age) return false;
    // allow partial core, but require age at least
    return true;
  }, [age]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    // This object is the "payload" leaving the form.
    // App.tsx receives it as "formPayload" in run(formPayload).
    const payload = {
      // core
      age,
      sex,
      height,
      weight,
      bmi,
      bp_systolic: bpSys,
      bp_diastolic: bpDia,
      smoking,
      hypertension,
      heartDisease,

      // labs
      labs_selected: selectedLabs,
      labs: {
        ...(has("hba1c") ? { hba1c, hba1cUnit } : {}),
        ...(has("fpg") ? { fpg, fpgUnit } : {}),
        ...(has("ogtt") ? { ogtt, ogttUnit } : {}),
        ...(has("egfr") ? { egfr } : {}),
      },
    };

    onSubmit(payload);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-3xl rounded-xl border border-zinc-200 bg-white p-6 shadow-sm space-y-6 text-left"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Patient inputs</h2>
        <span className="text-xs text-zinc-500">Core + optional labs</span>
      </div>

      {/* CORE */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Age">
          <input
            type="number"
            className="input"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            disabled={isLoading}
          />
        </Field>

        <Field label="Sex">
          <select
            className="input"
            value={sex}
            onChange={(e) => setSex(e.target.value as Sex)}
            disabled={isLoading}
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="not_say">Rather not say</option>
          </select>
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Height (cm)">
          <input
            type="number"
            className="input"
            value={height}
            onChange={(e) => handleHeightWeightChange(e.target.value, weight)}
            disabled={isLoading}
          />
        </Field>

        <Field label="Weight (kg)">
          <input
            type="number"
            className="input"
            value={weight}
            onChange={(e) => handleHeightWeightChange(height, e.target.value)}
            disabled={isLoading}
          />
        </Field>
      </div>

      <Field label="BMI">
        <input className="input" value={bmi} readOnly disabled />
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="BP Systolic (mmHg)">
          <input
            type="number"
            className="input"
            value={bpSys}
            onChange={(e) => setBpSys(e.target.value)}
            disabled={isLoading}
          />
        </Field>

        <Field label="BP Diastolic (mmHg)">
          <input
            type="number"
            className="input"
            value={bpDia}
            onChange={(e) => setBpDia(e.target.value)}
            disabled={isLoading}
          />
        </Field>
      </div>

      <Field label="Smoking Status">
        <select
          className="input"
          value={smoking}
          onChange={(e) => setSmoking(e.target.value as Smoking)}
          disabled={isLoading}
        >
          <option value="never">Never</option>
          <option value="former">Former</option>
          <option value="current">Current</option>
          <option value="unknown">Unknown</option>
        </select>
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Toggle
          label="Hypertension"
          value={hypertension}
          onChange={setHypertension}
          disabled={isLoading}
        />
        <Toggle
          label="Heart Disease"
          value={heartDisease}
          onChange={setHeartDisease}
          disabled={isLoading}
        />
      </div>

      {/* LABS (collapsible + selected fields only) */}
      <details
        className="rounded-xl border border-zinc-200 p-4"
        open={labsOpen}
        onToggle={(e) => setLabsOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
          Labs (optional)
          <span className="ml-2 text-xs font-normal text-zinc-500">
            select what’s available
          </span>
        </summary>

        <div className="mt-4 space-y-6">
          <div className="rounded-lg border border-zinc-200 p-3">
            <div className="text-sm font-semibold text-zinc-800">
              Available labs
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <LabCheck
                label="HbA1c"
                checked={has("hba1c")}
                onChange={() => toggleLab("hba1c")}
              />
              <LabCheck
                label="FPG / Random Glucose"
                checked={has("fpg")}
                onChange={() => toggleLab("fpg")}
              />
              <LabCheck
                label="OGTT 2h"
                checked={has("ogtt")}
                onChange={() => toggleLab("ogtt")}
              />
              <LabCheck
                label="Creatinine / eGFR"
                checked={has("egfr")}
                onChange={() => toggleLab("egfr")}
              />
            </div>
          </div>

          {has("hba1c") && (
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="HbA1c">
                <input
                  type="number"
                  className="input"
                  value={hba1c}
                  onChange={(e) => setHba1c(e.target.value)}
                  disabled={isLoading}
                />
              </Field>

              <Field label="Unit">
                <select
                  className="input"
                  value={hba1cUnit}
                  onChange={(e) => setHba1cUnit(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="%">%</option>
                  <option value="mmol/mol">mmol/mol</option>
                </select>
              </Field>
            </div>
          )}

          {has("fpg") && (
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="FPG / Random Glucose">
                <input
                  type="number"
                  className="input"
                  value={fpg}
                  onChange={(e) => setFpg(e.target.value)}
                  disabled={isLoading}
                />
              </Field>

              <Field label="Unit">
                <select
                  className="input"
                  value={fpgUnit}
                  onChange={(e) => setFpgUnit(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="mmol/L">mmol/L</option>
                  <option value="mg/dL">mg/dL</option>
                </select>
              </Field>
            </div>
          )}

          {has("ogtt") && (
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="OGTT 2h">
                <input
                  type="number"
                  className="input"
                  value={ogtt}
                  onChange={(e) => setOgtt(e.target.value)}
                  disabled={isLoading}
                />
              </Field>

              <Field label="Unit">
                <select
                  className="input"
                  value={ogttUnit}
                  onChange={(e) => setOgttUnit(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="mmol/L">mmol/L</option>
                  <option value="mg/dL">mg/dL</option>
                </select>
              </Field>
            </div>
          )}

          {has("egfr") && (
            <Field label="Creatinine / eGFR">
              <input
                type="number"
                className="input"
                value={egfr}
                onChange={(e) => setEgfr(e.target.value)}
                disabled={isLoading}
              />
            </Field>
          )}
        </div>
      </details>

      <button
        type="submit"
        disabled={isLoading || !canSubmit}
        className="w-full rounded-lg bg-black px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
      >
        {isLoading ? "Running…" : "Run assessment"}
      </button>

      <p className="text-center text-xs text-zinc-500">
        Research prototype. Decision support only.
      </p>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-zinc-700">{label}</span>
      {children}
    </label>
  );
}

function Toggle({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: 0 | 1;
  onChange: (v: 0 | 1) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-zinc-200 p-3">
      <span className="text-sm text-zinc-700">{label}</span>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange(1)}
          className={[
            "rounded-md px-3 py-1 text-sm",
            value === 1 ? "bg-zinc-900 text-white" : "border border-zinc-200 text-zinc-700",
          ].join(" ")}
        >
          Yes
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange(0)}
          className={[
            "rounded-md px-3 py-1 text-sm",
            value === 0 ? "bg-zinc-900 text-white" : "border border-zinc-200 text-zinc-700",
          ].join(" ")}
        >
          No
        </button>
      </div>
    </div>
  );
}

function LabCheck({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-700">
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-zinc-300"
        checked={checked}
        onChange={onChange}
      />
      {label}
    </label>
  );
}
