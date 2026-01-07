import { useMemo, useState } from "react";
// ==============================================

type Gender = "male" | "female" | "not_say";
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
  const [gender, setGender] = useState<Gender>("male");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [bmi, setBmi] = useState("");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [smoking, setSmoking] = useState<Smoking>("never");
  const [hypertension, setHypertension] = useState<YesNo>(0);
  const [heartDisease, setHeartDisease] = useState<YesNo>(0);
  const [familyHistoryDiabetes, setFamilyHistoryDiabetes] = useState<YesNo>(0);
  const [prevGestationalDiabetes, setPrevGestationalDiabetes] = useState<YesNo>(0);

  // Labs selector + collapse
  const [labsOpen, setLabsOpen] = useState(false);
  const [selectedLabs, setSelectedLabs] = useState<LabKey[]>([]);

  // Labs fields
  const [hba1c, setHba1c] = useState("");
  const [hba1cUnit, setHba1cUnit] = useState("%");
  const [hba1cDate, setHba1cDate] = useState("");
  const [fpg, setFpg] = useState("");
  const [fpgUnit, setFpgUnit] = useState("mmol/L");
  const [fpgDate, setFpgDate] = useState("");
  const [ogtt, setOgtt] = useState("");
  const [ogttUnit, setOgttUnit] = useState("mmol/L");
  const [ogttDate, setOgttDate] = useState("");
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
      gender,
      height,
      weight,
      bmi,
      bp_systolic: bpSys,
      bp_diastolic: bpDia,
      smoking,
      hypertension,
      heartDisease,
      familyHistoryDiabetes,
      prevGestationalDiabetes,

      // labs
      labs_selected: selectedLabs,
      labs: {
        // ✅ NEW: include dates where you already collect them
        ...(has("hba1c") ? { hba1c, hba1cUnit, hba1cDate } : {}),
        ...(has("fpg") ? { fpg, fpgUnit, fpgDate } : {}),
        ...(has("ogtt") ? { ogtt, ogttUnit, ogttDate } : {}),
        ...(has("egfr") ? { egfr } : {}),
      },
    };

    onSubmit(payload);
  }

  // ======================================================================================
  // form ui starts here
  return (
    <form
      onSubmit={handleSubmit}
      className="w-full max-w-3xl flex flex-col gap-5 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm space-y-6 text-left"
    >
      {/* Header */}
      <div className="flex items-end justify-between">
        <h2 className="text-2xl font-semibold">Patient inputs</h2>
        <span className="text-xs text-zinc-500">Core + optional labs</span>
      </div>

      {/* Main Form Fields (including Labs Section) */}
      <div className="flex flex-col gap-10">
        {/* Patient ID */}
        <div className="flex justify-between gap-4 sm:flex-row">
          <Field label="Patient ID (optional)" className="flex-1">
            <input type="text" className="input" placeholder="e.g. 12345" disabled={isLoading} />
          </Field>
          <Field label="Time of assessment (optional)" className="flex-1">
            <input type="date" className="input" placeholder="e.g. 2023-01-01" disabled={isLoading} />
          </Field>
        </div>

        {/* Demographics */}
        <div className="flex flex-col gap-4">
          <div className="space-y-4">
            <div className="flex justify-between gap-4 sm:flex-row">
              <Field label="Age" className="flex-1">
                <input type="number" className="input" value={age} onChange={(e) => setAge(e.target.value)} disabled={isLoading}/>
              </Field>
              <Field label="Gender" className="flex-1">
                <select className="input" value={gender} onChange={(e) => setGender(e.target.value as Gender)} disabled={isLoading}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="not_say">Rather not say</option>
                </select>
              </Field>
            </div>
            <div className="flex justify-between gap-4 sm:flex-row">
              <Field label="Height (cm)" className="flex-1">
                <input type="number" className="input" value={height} onChange={(e) => handleHeightWeightChange(e.target.value, weight)} disabled={isLoading} />
              </Field>
              <Field label="Weight (kg)" className="flex-1">
                <input type="number" className="input" value={weight} onChange={(e) => handleHeightWeightChange(height, e.target.value)} disabled={isLoading} />
              </Field>
            </div>
          </div>
          <Field label="BMI">
            <input className="input" value={bmi} readOnly disabled />
          </Field>
        </div>

        {/* Vitals */}
        <div className="flex flex-col gap-4">
          <div className="flex gap-4 sm:flex-row">
            <Field label="Systolic Blood Pressure (mmHg)" className="flex-1">
              <input type="number" className="input" value={bpSys} onChange={(e) => setBpSys(e.target.value)} disabled={isLoading} />
            </Field>
            <Field label="Diastolic Blood Pressure (mmHg)" className="flex-1">
              <input type="number" className="input" value={bpDia} onChange={(e) => setBpDia(e.target.value)} disabled={isLoading} />
            </Field>
          </div>
          <Field label="Blood Pressure">
            <input className="input" value={bpSys && bpDia ? `${bpSys}/${bpDia}` : ""} readOnly disabled />
          </Field>
        </div>

        {/* Clinical History and risk factors */}
        <div className="flex flex-col gap-4">
          <div className="flex flex-col justify-between gap-2">
            <Toggle label="Hypertension" value={hypertension} onChange={setHypertension} disabled={isLoading} />
            <Toggle label="Heart Disease" value={heartDisease} onChange={setHeartDisease} disabled={isLoading} />
            <Toggle label="Family History of Diabetes" value={familyHistoryDiabetes} onChange={setFamilyHistoryDiabetes} disabled={isLoading} />
            <Toggle label="Prev. Gestational Diabetes" value={prevGestationalDiabetes} onChange={setPrevGestationalDiabetes} disabled={isLoading} />
          </div>
          <Field label="Smoking Status" className="w-full">
            <select className="input" value={smoking} onChange={(e) => setSmoking(e.target.value as Smoking)} disabled={isLoading}>
              <option value="never">Never</option>
              <option value="former">Former</option>
              <option value="current">Current</option>
              {/* <option value="unknown">Unknown</option> */}
            </select>
          </Field>
          <div className="flex justify-between gap-4 sm:flex-row">
            <Field label="Physical Activity Level" className="flex-1">
              <select className="input" disabled={isLoading}>
                <option>Sedentary</option>
                <option>Lightly Active</option>
                <option>Moderate</option>
                <option>Active</option>
                <option>Very Active</option>
              </select>
            </Field>
            <Field label="Alcohol Consumption" className="flex-1">
              <select className="input" disabled={isLoading}>
                <option>Never</option>
                <option>Occasionally</option>
                <option>Regularly</option>
                <option>Heavy</option>
              </select>
            </Field>
            {/* <Field label="Prev. Gestational Diabetes" className="flex-1">
              <input type="radio" className="input" disabled />
            </Field> */}
          </div>
        </div>

        {/* Labs Section */}
        <details className="rounded-xl border space-y-6 border-zinc-200 p-4" open={labsOpen}
          onToggle={(e) =>
            setLabsOpen((e.target as HTMLDetailsElement).open)
          }
        >
          <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
            Labs (optional)
            <span className="ml-2 text-xs font-normal text-zinc-500">
              select what’s available
            </span>
          </summary>

          <div className="mt-4 space-y-4">
            <div className="rounded-lg border border-zinc-200 p-3">
              <div className="text-sm font-semibold text-zinc-800">
                Available labs
              </div>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:flex-wrap">
                <LabCheck label="HbA1c" checked={has("hba1c")} onChange={() => toggleLab("hba1c")} />
                <LabCheck label="FPG / Random Glucose" checked={has("fpg")} onChange={() => toggleLab("fpg")} />
                <LabCheck label="OGTT 2h" checked={has("ogtt")} onChange={() => toggleLab("ogtt")} />
                <LabCheck label="Creatinine / eGFR" checked={has("egfr")} onChange={() => toggleLab("egfr")} />
              </div>
            </div>

            {has("hba1c") && (
              <div className="flex gap-4 sm:flex-row">
                <Field label="HbA1c" className="flex-1">
                  <input type="number" className="input" value={hba1c} onChange={(e) => setHba1c(e.target.value)} disabled={isLoading} />
                </Field>
                <Field label="Unit">
                  <select className="input" value={hba1cUnit} onChange={(e) => setHba1cUnit(e.target.value)} disabled={isLoading}>
                    <option value="%">%</option>
                    <option value="mmol/mol">mmol/mol</option>
                  </select>
                </Field>
                <Field label="Date of Lab (optional)">
                  <input type="date" className="input" value={hba1cDate} onChange={(e) => setHba1cDate(e.target.value)} disabled={isLoading}/>
                </Field>
              </div>
            )}

            {has("fpg") && (
              <div className="flex gap-4 sm:flex-row">
                <Field label="FPG / Random Glucose" className="flex-1">
                  <input type="number" className="input" value={fpg} onChange={(e) => setFpg(e.target.value)} disabled={isLoading} />
                </Field>
                <Field label="Unit">
                  <select className="input" value={fpgUnit} onChange={(e) => setFpgUnit(e.target.value)} disabled={isLoading}>
                    <option value="mmol/L">mmol/L</option>
                    <option value="mg/dL">mg/dL</option>
                  </select>
                </Field>
                <Field label="Date of Lab (optional)">
                  <input type="date" className="input" value={fpgDate} onChange={(e) => setFpgDate(e.target.value)} disabled={isLoading}/>
                </Field>
              </div>
            )}

            {has("ogtt") && (
              <div className="flex gap-4 sm:flex-row">
                <Field label="OGTT 2h" className="flex-1">
                  <input type="number" className="input" value={ogtt} onChange={(e) => setOgtt(e.target.value)} disabled={isLoading} />
                </Field>
                <Field label="Unit">
                  <select className="input" value={ogttUnit} onChange={(e) => setOgttUnit(e.target.value)} disabled={isLoading}>
                    <option value="mmol/L">mmol/L</option>
                    <option value="mg/dL">mg/dL</option>
                  </select>
                </Field>
                <Field label="Date of Lab (optional)">
                  <input type="date" className="input" value={ogttDate} onChange={(e) => setOgttDate(e.target.value)} disabled={isLoading}/>
                </Field>
              </div>
            )}

            {has("egfr") && (
              <Field label="Creatinine / eGFR" className="w-1/2">
                <input type="number" className="input" value={egfr} onChange={(e) => setEgfr(e.target.value)} disabled={isLoading} />
              </Field>
            )}
          </div>
        </details>
      </div>

      {/* Footer: Button and Disclaimer */}
      <div className="space-y-5 flex flex-col items-center">
        <button
          type="submit"
          disabled={isLoading || !canSubmit}
          className="w-full h-[50px] rounded-lg bg-black px-4 py-2 text-md font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
        >
          {isLoading ? "Running…" : "Run assessment"}
        </button>
        <p className="w-[70%] text-center text-xs text-zinc-500">
          Disclaimer: This is a Research prototype and should not be used for
          final clinical decision but for decision support only.
        </p>
      </div>
    </form>
  );
}
// form ui ends here
// ======================================================================================

function Field({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label
      className={["flex flex-col gap-1 text-sm", className]
        .filter(Boolean)
        .join(" ")}
    >
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
        <button type="button" disabled={disabled} onClick={() => onChange(1)}
          className={[
            "rounded-md px-3 py-1 text-sm",
            value === 1
              ? "bg-zinc-900 text-white"
              : "border border-zinc-200 text-zinc-700",
          ].join(" ")}
        >
          Yes
        </button>
        <button type="button" disabled={disabled} onClick={() => onChange(0)}
          className={[
            "rounded-md px-3 py-1 text-sm",
            value === 0
              ? "bg-zinc-900 text-white"
              : "border border-zinc-200 text-zinc-700",
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
      <input type="checkbox" className="h-4 w-4 rounded border-zinc-300" checked={checked} onChange={onChange}/>
      {label}
    </label>
  );
}
