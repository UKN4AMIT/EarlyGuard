import { useEffect, useMemo, useState, type FormEvent } from 'react'
import {
  Activity, ArrowRight, CheckCircle2, CircleAlert, HeartPulse,
  LoaderCircle, RefreshCw, ShieldCheck, Stethoscope, Wifi, WifiOff
} from 'lucide-react'

type ApiStatus = 'checking' | 'online' | 'offline'
type Field = {
  key: string
  label: string
  unit?: string
  kind: 'number' | 'select'
  options?: string[]
  step?: string
  placeholder?: string
  group: string
}

type PatientForm = Record<string, string>
type Prediction = {
  prediction: 'CKD' | 'Not CKD'
  predicted_class: number
  ckd_probability: number
  not_ckd_probability: number
  model: string
  disclaimer: string
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

const fields: Field[] = [
  { key: 'age', label: 'Age', unit: 'years', kind: 'number', step: '1', placeholder: 'e.g. 48', group: 'Patient details' },
  { key: 'bp', label: 'Blood pressure', unit: 'mmHg', kind: 'number', step: 'any', placeholder: 'e.g. 80', group: 'Patient details' },
  { key: 'sg', label: 'Specific gravity', kind: 'number', step: 'any', placeholder: 'e.g. 1.020', group: 'Urine examination' },
  { key: 'al', label: 'Albumin', kind: 'number', step: 'any', placeholder: 'e.g. 1', group: 'Urine examination' },
  { key: 'su', label: 'Sugar', kind: 'number', step: 'any', placeholder: 'e.g. 0', group: 'Urine examination' },
  { key: 'rbc', label: 'Red blood cells', kind: 'select', options: ['normal', 'abnormal'], group: 'Urine examination' },
  { key: 'pc', label: 'Pus cell', kind: 'select', options: ['normal', 'abnormal'], group: 'Urine examination' },
  { key: 'pcc', label: 'Pus cell clumps', kind: 'select', options: ['present', 'notpresent'], group: 'Urine examination' },
  { key: 'ba', label: 'Bacteria', kind: 'select', options: ['present', 'notpresent'], group: 'Urine examination' },
  { key: 'bgr', label: 'Blood glucose random', unit: 'mg/dL', kind: 'number', step: 'any', placeholder: 'e.g. 121', group: 'Blood investigations' },
  { key: 'bu', label: 'Blood urea', unit: 'mg/dL', kind: 'number', step: 'any', placeholder: 'e.g. 36', group: 'Blood investigations' },
  { key: 'sc', label: 'Serum creatinine', unit: 'mg/dL', kind: 'number', step: 'any', placeholder: 'e.g. 1.2', group: 'Blood investigations' },
  { key: 'sod', label: 'Sodium', unit: 'mEq/L', kind: 'number', step: 'any', placeholder: 'e.g. 138', group: 'Blood investigations' },
  { key: 'pot', label: 'Potassium', unit: 'mEq/L', kind: 'number', step: 'any', placeholder: 'e.g. 4.5', group: 'Blood investigations' },
  { key: 'hemo', label: 'Hemoglobin', unit: 'g/dL', kind: 'number', step: 'any', placeholder: 'e.g. 13.5', group: 'Blood investigations' },
  { key: 'pcv', label: 'Packed cell volume', unit: '%', kind: 'number', step: 'any', placeholder: 'e.g. 40', group: 'Blood investigations' },
  { key: 'wc', label: 'White blood cell count', unit: 'cells/cumm', kind: 'number', step: 'any', placeholder: 'e.g. 8000', group: 'Blood investigations' },
  { key: 'rc', label: 'Red blood cell count', unit: 'millions/cumm', kind: 'number', step: 'any', placeholder: 'e.g. 5.2', group: 'Blood investigations' },
  { key: 'htn', label: 'Hypertension', kind: 'select', options: ['yes', 'no'], group: 'Medical history' },
  { key: 'dm', label: 'Diabetes mellitus', kind: 'select', options: ['yes', 'no'], group: 'Medical history' },
  { key: 'cad', label: 'Coronary artery disease', kind: 'select', options: ['yes', 'no'], group: 'Medical history' },
  { key: 'appet', label: 'Appetite', kind: 'select', options: ['good', 'poor'], group: 'Medical history' },
  { key: 'pe', label: 'Pedal edema', kind: 'select', options: ['yes', 'no'], group: 'Medical history' },
  { key: 'ane', label: 'Anemia', kind: 'select', options: ['yes', 'no'], group: 'Medical history' },
]

const initialForm = (): PatientForm => Object.fromEntries(fields.map((field) => [field.key, '']))

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [modelInfo, setModelInfo] = useState<{ loaded: boolean; model?: string; model_features?: number } | null>(null)
  const [form, setForm] = useState<PatientForm>(initialForm)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const groups = useMemo(() => [...new Set(fields.map((field) => field.group))], [])
  const completed = fields.filter((field) => form[field.key].trim() !== '').length

  async function checkApi() {
    setApiStatus('checking')
    try {
      const [healthResponse, modelResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`),
        fetch(`${API_BASE}/api/model/status`),
      ])
      if (!healthResponse.ok || !modelResponse.ok) throw new Error('API check failed')
      const model = await modelResponse.json()
      setModelInfo(model)
      setApiStatus('online')
    } catch {
      setApiStatus('offline')
      setModelInfo(null)
    }
  }

  useEffect(() => { void checkApi() }, [])

  function updateField(key: string, value: string) {
    setForm((current) => ({ ...current, [key]: value }))
    setResult(null)
    setError('')
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setResult(null)

    if (completed === 0) {
      setError('Please enter at least one patient parameter before submitting.')
      return
    }
    if (apiStatus !== 'online') {
      setError('The backend API is offline. Start the FastAPI server and try again.')
      return
    }

    const payload: Record<string, string | number | null> = {}
    for (const field of fields) {
      const value = form[field.key].trim()
      payload[field.key] = value === '' ? null : field.kind === 'number' ? Number(value) : value
    }

    setSubmitting(true)
    try {
      const response = await fetch(`${API_BASE}/api/predict/kidney`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()
      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item: { msg?: string }) => item.msg).join(', ')
          : data.detail
        throw new Error(detail || `Request failed (${response.status})`)
      }
      setResult(data as Prediction)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to get a prediction. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  function resetForm() {
    setForm(initialForm())
    setResult(null)
    setError('')
  }

  const ckdPercent = result ? Math.max(0, Math.min(100, result.ckd_probability * 100)) : 0

  return (
    <main className="app-shell">
      <nav className="topbar">
        <a className="brand" href="#"><span className="brand-icon"><Activity size={19} /></span><span>earlyguard<span className="brand-period">.</span></span></a>
        <div className="topbar-right">
          <span className={`api-pill ${apiStatus}`}><span className="api-dot" />{apiStatus === 'checking' ? 'Checking API' : apiStatus === 'online' ? 'API connected' : 'API offline'}</span>
          <button className="icon-button" onClick={() => void checkApi()} aria-label="Refresh API status" title="Refresh API status"><RefreshCw size={16} /></button>
        </div>
      </nav>

      <section className="intro-section">
        <div className="eyebrow"><span className="eyebrow-line" /> EXPLAINABLE AI · RENAL HEALTH RESEARCH</div>
        <div className="intro-grid">
          <div>
            <h1>Understand the signals.<br /><em>Explore the patterns.</em></h1>
            <p className="intro-copy">An academic machine-learning prototype that demonstrates how selected patient parameters can be processed by a kidney disease classification model.</p>
          </div>
          <aside className="model-card">
            <div className="model-card-top"><span className="model-symbol"><HeartPulse size={19} /></span><span className="model-label">MODEL STATUS</span></div>
            <strong>{modelInfo?.loaded ? 'Model ready' : apiStatus === 'offline' ? 'Not connected' : 'Checking...'}</strong>
            <span className="model-detail">{modelInfo?.model || 'Waiting for backend response'}</span>
            {modelInfo?.loaded && <span className="model-features">{modelInfo.model_features ?? '—'} processed features</span>}
          </aside>
        </div>
      </section>

      <section className="workspace">
        <div className="form-heading">
          <div><div className="eyebrow">KIDNEY HEALTH MODULE / 01</div><h2>Patient assessment</h2><p>Enter available parameters. Leave unknown values blank.</p></div>
          <div className="completion"><span>{completed} / 24</span><div className="progress-track"><div style={{ width: `${(completed / 24) * 100}%` }} /></div><small>fields completed</small></div>
        </div>

        <div className="content-grid">
          <form className="assessment-form" onSubmit={handleSubmit}>
            {groups.map((group) => (
              <section className="field-section" key={group}>
                <div className="field-section-title"><span>{group}</span><span className="section-count">{fields.filter((f) => f.group === group && form[f.key].trim()).length} entered</span></div>
                <div className="field-grid">
                  {fields.filter((field) => field.group === group).map((field) => (
                    <label className="field" key={field.key}>
                      <span className="field-label">{field.label}{field.unit && <small>{field.unit}</small>}</span>
                      {field.kind === 'number' ? (
                        <input
                          type="number"
                          min="0"
                          max={field.key === 'age' ? 120 : field.key === 'bp' ? 300 : undefined}
                          step={field.step}
                          placeholder={field.placeholder}
                          value={form[field.key]}
                          onChange={(event) => updateField(field.key, event.target.value)}
                        />
                      ) : (
                        <select value={form[field.key]} onChange={(event) => updateField(field.key, event.target.value)}>
                          <option value="">Select if known</option>
                          {field.options?.map((option) => <option key={option} value={option}>{option === 'notpresent' ? 'Not present' : option.charAt(0).toUpperCase() + option.slice(1)}</option>)}
                        </select>
                      )}
                    </label>
                  ))}
                </div>
              </section>
            ))}
            <div className="form-actions">
              <button className="submit-button" type="submit" disabled={submitting || apiStatus !== 'online'}>
                {submitting ? <><LoaderCircle className="spin" size={17} /> Processing...</> : <>Run research prediction <ArrowRight size={17} /></>}
              </button>
              <button className="reset-button" type="button" onClick={resetForm}>Clear form</button>
            </div>
            {apiStatus === 'offline' && <div className="inline-alert"><WifiOff size={17} /><span>Backend not reachable at <code>{API_BASE}</code>. Start FastAPI on port 8000.</span></div>}
            {error && <div className="error-alert"><CircleAlert size={17} /><span>{error}</span></div>}
          </form>

          <aside className="result-column">
            <div className="result-card">
              <div className="result-card-head"><span className="result-icon"><Activity size={18} /></span><span>MODEL OUTPUT</span></div>
              {!result ? (
                <div className="empty-result"><div className="empty-graphic"><Stethoscope size={25} /></div><h3>Results will appear here</h3><p>Complete the available fields and run the model to view its classification output.</p><div className="empty-note"><ShieldCheck size={15} /> Research prototype only</div></div>
              ) : (
                <div className="prediction-result">
                  <span className="result-overline">PREDICTED CLASS</span>
                  <h3 className={result.predicted_class === 1 ? 'ckd-text' : 'no-ckd-text'}>{result.prediction}</h3>
                  <p className="model-used">Generated by {result.model}</p>
                  <div className="probability-block">
                    <div className="probability-heading"><span>CKD model probability</span><strong>{ckdPercent.toFixed(2)}%</strong></div>
                    <div className="probability-track"><div style={{ width: `${ckdPercent}%` }} /></div>
                    <div className="probability-ends"><span>0%</span><span>100%</span></div>
                  </div>
                  <div className="probability-row"><span>Not CKD model probability</span><strong>{(result.not_ckd_probability * 100).toFixed(2)}%</strong></div>
                  <div className="result-disclaimer"><ShieldCheck size={16} /><p>{result.disclaimer}</p></div>
                </div>
              )}
            </div>
            <div className="safety-card"><ShieldCheck size={19} /><div><strong>Important limitation</strong><p>This is an educational research prototype, not a clinically validated diagnostic or screening tool. Model probabilities are not calibrated individual risk estimates. Do not use this output for medical decisions.</p></div></div>
          </aside>
        </div>
      </section>

      <footer className="footer"><span>EARLYGUARD · B.TECH AI/ML RESEARCH PROJECT</span><span><Wifi size={13} /> LOCAL DEVELOPMENT BUILD · v0.2.0</span></footer>
    </main>
  )
}

export default App
