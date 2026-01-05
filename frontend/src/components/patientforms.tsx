import { useState } from "react";
import type { PatientRaw, SmokingHistory } from "../api/types";
// ===============================================================


type Props = {
  isLoading?: boolean;
  onSubmit: (patient: PatientRaw) => void;
};

const smokingOptions: SmokingHistory[] = [
  "never",
  "former",
  "current",
  "not current",
  "ever",
  "unknown",
];

export default function PatientForm({ isLoading, onSubmit }: Props) {
  const [age, setAge] = useState("");
  const [bmi, setBmi] = useState("");
  const [glucose, setGlucose] = useState("");
  const [smoking, setSmoking] = useState<SmokingHistory>("never");
  const [hypertension, setHypertension] = useState<0 | 1>(0);
  const [heartDisease, setHeartDisease] = useState<0 | 1>(0);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const patient: PatientRaw = {
      age: Number(age),
      bmi: Number(bmi),
      glucose: Number(glucose),
      smoking_history: smoking,
      hypertension,
      heart_disease: heartDisease,
    };

    onSubmit(patient);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm"
    >
      <h2 className="text-lg font-semibold">Patient inputs</h2>

      <div className="mt-4 grid gap-4">
        <Field label="Age">
          <input
            type="number"
            className="input"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            disabled={isLoading}
          />
        </Field>

        <Field label="BMI">
          <input
            type="number"
            step="0.1"
            className="input"
            value={bmi}
            onChange={(e) => setBmi(e.target.value)}
            disabled={isLoading}
          />
        </Field>

        <Field label="Glucose">
          <input
            type="number"
            className="input"
            value={glucose}
            onChange={(e) => setGlucose(e.target.value)}
            disabled={isLoading}
          />
        </Field>

        <Field label="Smoking history">
          <select
            className="input"
            value={smoking}
            onChange={(e) => setSmoking(e.target.value as SmokingHistory)}
            disabled={isLoading}
          >
            {smokingOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </Field>

        <Toggle
          label="Hypertension"
          value={hypertension}
          onChange={setHypertension}
          disabled={isLoading}
        />

        <Toggle
          label="Heart disease"
          value={heartDisease}
          onChange={setHeartDisease}
          disabled={isLoading}
        />
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="mt-4 w-full rounded-lg bg-black px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
      >
        {isLoading ? "Running…" : "Run assessment"}
      </button>

      <p className="mt-2 text-xs text-zinc-500">
        Research prototype. Decision support only.
      </p>
    </form>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
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
          className={`rounded-md px-3 py-1 text-sm ${
            value === 1
              ? "bg-zinc-900 text-white"
              : "border border-zinc-200 text-zinc-700"
          }`}
        >
          Yes
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange(0)}
          className={`rounded-md px-3 py-1 text-sm ${
            value === 0
              ? "bg-zinc-900 text-white"
              : "border border-zinc-200 text-zinc-700"
          }`}
        >
          No
        </button>
      </div>
    </div>
  );
}
