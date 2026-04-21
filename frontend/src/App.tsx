import { useCallback, useMemo, useState } from "react";

type PerClass = {
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

type GroundTruthSource = {
  source?: "uploaded" | "auto" | "none";
  filename?: string;
  matched_stem?: string | null;
  annotators?: string[];
  files?: string[];
  n_articles?: number;
  per_annotator?: Record<string, Record<string, string>>;
};

type Evaluation = {
  hallucination?: {
    total_evaluations?: number;
    hallucination_flags?: number;
    hallucination_rate?: number;
  };
  classification?: {
    support?: number;
    macro_f1?: number | null;
    cohens_kappa?: number | null;
    accuracy?: number | null;
    per_class?: Record<string, PerClass> | null;
    confusion_matrix?: number[][] | null;
    labels?: string[];
    kappa_interpretation?: string | null;
    kappa_note?: string | null;
    note?: string;
    coverage?: {
      ground_truth_total?: number;
      predictions_total?: number;
      overlap_total?: number;
      missing_in_predictions?: number;
      extra_predictions?: number;
    };
    scoring_scope?: {
      mode?: string;
      regulation_scope?: string[];
      focus_articles?: string[];
    };
  };
  n_predictions?: number;
  ground_truth_provided?: boolean;
  ground_truth_source?: GroundTruthSource;
};

type AnalyzeResponse = {
  doc_id: string;
  filename?: string;
  doc_type?: string;
  regulations?: string[];
  risk_score?: number;
  risk_level?: string;
  articles_evaluated?: number;
  hallucination_rate?: number;
  evaluation?: Evaluation;
  evaluation_url?: string;
  violations?: ViolationReport["violations"];
  violation_report_url: string;
  assessment_report_url: string;
  remediation_report_url: string;
  pipeline_log_url: string | null;
};

type ViolationReport = {
  doc_id?: string;
  doc_type?: string;
  regulations?: string[];
  risk_score?: number;
  risk_level?: string;
  articles_evaluated?: number;
  hallucination_rate?: number;
  violations?: Array<{
    article_id: string;
    article_title?: string;
    regulation?: string;
    verdict: string;
    /** Residual exposure given verdict (backend-combined). */
    risk_level?: string;
    /** Statutory / article importance from regulation metadata. */
    article_priority?: string;
    reasoning?: string;
  }>;
};

const API_BASE = "/api/v1";

function riskClass(level?: string): string {
  const l = (level || "").toLowerCase();
  if (l === "low") return "risk-low";
  if (l === "medium") return "risk-med";
  if (l === "high" || l === "critical") return "risk-high";
  return "";
}

function verdictBadge(verdict: string): string {
  const v = verdict.toLowerCase();
  if (v === "full") return "badge full";
  if (v === "partial") return "badge partial";
  return "badge missing";
}

function riskPillClass(level?: string): string {
  const l = (level || "").toLowerCase();
  if (l === "low") return "risk-pill low";
  if (l === "medium") return "risk-pill med";
  if (l === "high") return "risk-pill high";
  if (l === "critical") return "risk-pill critical";
  return "risk-pill";
}

/* -------------------------- Icons -------------------------- */

const Icon = {
  Shield: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2l8 4v6c0 5-3.5 9-8 10-4.5-1-8-5-8-10V6l8-4z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  ),
  Upload: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  ),
  Play: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="6 4 20 12 6 20 6 4" fill="currentColor" stroke="none" />
    </svg>
  ),
  Doc: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M8 13h8M8 17h6M8 9h2" />
    </svg>
  ),
  PdfFile: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <path d="M14 2v6h6" />
      <text x="7" y="18" fontSize="6" fontWeight="700" stroke="none" fill="currentColor">PDF</text>
    </svg>
  ),
  JsonFile: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M9 13c-1 0-1 1-1 2s0 2 1 2M15 13c1 0 1 1 1 2s0 2-1 2" />
    </svg>
  ),
  Terminal: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="M7 9l3 3-3 3M13 15h4" />
    </svg>
  ),
  ArrowUpRight: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7 17L17 7M8 7h9v9" />
    </svg>
  ),
};

/* -------------------------- Risk gauge -------------------------- */

function RiskGauge({ score, level }: { score?: number; level?: string }) {
  const max = 4;
  const pct = score != null ? Math.max(0, Math.min(1, score / max)) : 0;
  const r = 66;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - pct);
  const lvl = (level || "").toLowerCase();

  return (
    <div className="gauge" data-level={lvl}>
      <svg viewBox="0 0 160 160">
        <circle className="track" cx="80" cy="80" r={r} strokeWidth="12" fill="none" />
        <circle
          className="progress"
          cx="80"
          cy="80"
          r={r}
          strokeWidth="12"
          fill="none"
          strokeDasharray={c}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="inner">
        <div className="inner-stack">
          <div className="score">
            {score ?? "—"}
            <span className="unit">/ {max}</span>
          </div>
          <div className="tier">{level || "Unknown"}</div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------- Evaluation -------------------------- */

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function invertTone(v: number | null | undefined, good: number, ok: number): string {
  // For metrics where LOWER is better (e.g. hallucination rate).
  if (v == null) return "neutral";
  if (v <= good) return "good";
  if (v <= ok) return "ok";
  return "bad";
}

function MetricCard({
  label,
  value,
  tone,
  caption,
  footnote,
}: {
  label: string;
  value: string;
  tone: string;
  caption?: string;
  footnote?: string;
}) {
  return (
    <div className={`eval-card tone-${tone}`}>
      <div className="eval-card-label">{label}</div>
      <div className="eval-card-value">{value}</div>
      {caption && <div className="eval-card-caption">{caption}</div>}
      {footnote && <div className="eval-card-footnote">{footnote}</div>}
    </div>
  );
}

function GroundTruthBadge({ source }: { source?: GroundTruthSource }) {
  if (!source || source.source === "none" || !source.source) {
    return (
      <div className="gt-source gt-source-none">
        <span className="gt-source-kind">No ground truth</span>
        <span className="gt-source-detail">
          Upload a JSON or place an annotation PDF under
          <code className="mono"> test_datasets/&lt;reg&gt;/annotations/</code> to enable classification metrics.
        </span>
      </div>
    );
  }
  if (source.source === "uploaded") {
    return (
      <div className="gt-source gt-source-uploaded">
        <span className="gt-source-kind">Ground truth · uploaded</span>
        <span className="gt-source-detail">
          <code className="mono">{source.filename}</code>
          {source.n_articles != null ? (
            <span> · {source.n_articles} article{source.n_articles === 1 ? "" : "s"}</span>
          ) : null}
        </span>
      </div>
    );
  }
  // auto
  const annotators = source.annotators || [];
  return (
    <div className="gt-source gt-source-auto">
      <span className="gt-source-kind">Ground truth · auto-matched</span>
      <span className="gt-source-detail">
        {source.matched_stem ? (
          <>
            Matched <code className="mono">{source.matched_stem}</code>
          </>
        ) : null}
        {annotators.length > 0 ? (
          <>
            {" "}· aggregated from {annotators.length} annotator{annotators.length === 1 ? "" : "s"}:{" "}
            {annotators.map((a) => (
              <span className="gt-annotator" key={a}>
                {a}
              </span>
            ))}
          </>
        ) : null}
      </span>
    </div>
  );
}

function EvaluationSection({ evaluation }: { evaluation: Evaluation }) {
  const { hallucination, classification } = evaluation;
  const hallRate = hallucination?.hallucination_rate ?? null;

  const hasGT = evaluation.ground_truth_provided && classification?.support;
  const labels = classification?.labels || ["Full", "Partial", "Missing"];
  const cm = classification?.confusion_matrix || null;
  const coverage = classification?.coverage || null;
  const scoringScope = classification?.scoring_scope;

  return (
    <>
      <div className="section-title">Evaluation metrics</div>
      <p className="section-hint">
        Hallucination rate reflects canonical reporter verdicts. Pred-vs-GT confusion matrix and
        per-class precision/recall/F1 are computed against ground truth, uploaded or auto-resolved from
        <code className="mono"> test_datasets/</code>.
      </p>
      <div className="metric-legend">
        <p><strong>Hallucination rate:</strong> fraction of evaluated articles flagged as unsupported by retrieved evidence. Lower is better.</p>
        <p><strong>Confusion matrix:</strong> compares predicted labels vs ground truth labels (Full/Partial/Missing). Diagonal cells are correct predictions.</p>
      </div>

      <section className="card">
        <GroundTruthBadge source={evaluation.ground_truth_source} />

        <div className="eval-grid">
          <MetricCard
            label="Hallucination rate"
            value={pct(hallRate, 2)}
            tone={invertTone(hallRate, 0.05, 0.15)}
            caption={
              hallucination?.total_evaluations != null
                ? `${hallucination.hallucination_flags ?? 0}/${hallucination.total_evaluations} flagged`
                : "Debate-flagged turns"
            }
            footnote="Lower is better"
          />
        </div>

        {hasGT && cm ? (
          <>
            <div className="section-title" style={{ marginTop: "1.75rem" }}>
              Confusion matrix (Ground truth × Predicted)
            </div>
            <div className="eval-coverage">
              <span>
                Scoring scope: <strong>{scoringScope?.mode === "focus_articles_only" ? "focus articles only" : "all GT"}</strong>
              </span>
              <span>GT articles: <strong>{coverage?.ground_truth_total ?? "—"}</strong></span>
              <span>Predicted articles: <strong>{coverage?.predictions_total ?? "—"}</strong></span>
              <span>Scored overlap: <strong>{coverage?.overlap_total ?? "—"}</strong></span>
              <span>GT missing in prediction: <strong>{coverage?.missing_in_predictions ?? 0}</strong></span>
              <span>Extra predicted (no GT): <strong>{coverage?.extra_predictions ?? 0}</strong></span>
            </div>
            <div className="table-wrap" style={{ marginTop: "0.2rem" }}>
              <table>
                <thead>
                  <tr>
                    <th>Ground truth \\ Pred</th>
                    {labels.map((lbl) => (
                      <th key={lbl} style={{ textAlign: "right" }}>{lbl}</th>
                    ))}
                    <th style={{ textAlign: "right" }}>Row total</th>
                  </tr>
                </thead>
                <tbody>
                  {labels.map((rowLbl, rowI) => {
                    const badge =
                      rowLbl === "Full" ? "badge full" : rowLbl === "Partial" ? "badge partial" : "badge missing";
                    const row = cm[rowI] || [];
                    const rowTotal = row.reduce((a, b) => a + (b || 0), 0);
                    return (
                      <tr key={rowLbl}>
                        <td>
                          <span className={badge}>{rowLbl}</span>
                        </td>
                        {labels.map((_, colI) => {
                          const v = row[colI] || 0;
                          const onDiag = rowI === colI;
                          return (
                            <td
                              key={`${rowLbl}-${colI}`}
                              style={{ textAlign: "right" }}
                              className={`mono-cell ${onDiag ? "cm-diag" : ""}`}
                            >
                              {v}
                            </td>
                          );
                        })}
                        <td style={{ textAlign: "right" }} className="mono-cell">{rowTotal}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="section-hint" style={{ marginTop: "0.65rem" }}>
              Metric meaning: row totals show how many ground-truth articles exist for each class; off-diagonal cells indicate misclassifications.
            </p>
          </>
        ) : (
          <div className="eval-note">
            <strong>Tip:</strong> upload a ground-truth JSON (e.g.{" "}
            <code className="mono">{"{\"art_5\":\"Full\",\"art_13\":\"Partial\"}"}</code>) with your next run to unlock
            confusion matrix and class-level quality metrics.
          </div>
        )}
      </section>
    </>
  );
}

/* -------------------------- Component -------------------------- */

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [gtFile, setGtFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [summary, setSummary] = useState<AnalyzeResponse | null>(null);
  const [detail, setDetail] = useState<ViolationReport | null>(null);

  const accept = ".pdf,.docx,.txt";

  const onPick = useCallback((f: File | null) => {
    setError(null);
    setWarning(null);
    setSummary(null);
    setDetail(null);
    setFile(f);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files?.[0];
      if (f) onPick(f);
    },
    [onPick],
  );

  const runAnalysis = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setWarning(null);
    setSummary(null);
    setDetail(null);
    const body = new FormData();
    body.append("file", file);
    if (gtFile) body.append("ground_truth", gtFile);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        let msg = `Request failed (${res.status})`;
        try {
          const err = await res.json();
          if (err.detail) msg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const data: AnalyzeResponse = await res.json();
      setSummary(data);
      if (data.violations && data.violations.length > 0) {
        setDetail({
          doc_id: data.doc_id,
          doc_type: data.doc_type,
          regulations: data.regulations,
          risk_score: data.risk_score,
          risk_level: data.risk_level,
          articles_evaluated: data.articles_evaluated,
          hallucination_rate: data.hallucination_rate,
          violations: data.violations,
        });
      }

      const rep = await fetch(`${API_BASE}/reports/${data.doc_id}`);
      if (rep.ok) {
        const json: ViolationReport = await rep.json();
        setDetail(json);
      } else if (!data.violations?.length) {
        setWarning("Run completed, but detailed article rows could not be loaded.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [file, gtFile]);

  const downloads = useMemo(() => {
    if (!summary) return null;
    const items: {
      href: string;
      label: string;
      hint: string;
      iconKind: "pdf" | "json" | "log";
    }[] = [
      {
        href: summary.assessment_report_url,
        label: "Assessment report",
        hint: "PDF · audit findings",
        iconKind: "pdf",
      },
      {
        href: summary.remediation_report_url,
        label: "Remediation report",
        hint: "PDF · actions and acceptance criteria",
        iconKind: "pdf",
      },
      {
        href: summary.violation_report_url,
        label: "Violation report",
        hint: "JSON · machine-readable results",
        iconKind: "json",
      },
    ];
    if (summary.pipeline_log_url) {
      items.push({
        href: summary.pipeline_log_url,
        label: "Pipeline log",
        hint: "JSON · prompts, responses, thinking traces",
        iconKind: "log",
      });
    }
    return items;
  }, [summary]);

  return (
    <div className="app-shell">
      <header className="brand">
        <div className="brand-mark">
          <Icon.Shield />
        </div>
        <div>
          <h1>Agentic Audit</h1>
          <p>Upload an enterprise document and receive POA&amp;M-ready compliance deliverables.</p>
        </div>
        <div className="brand-badges">
          <span className="chip">AI compliance</span>
          <span className="chip">POA&amp;M ready</span>
        </div>
      </header>

      <section className="card">
        <h2>
          <Icon.Upload /> Document upload
        </h2>
        <p className="sub">
          Supported formats: PDF, Word (.docx), or plain text. The full pipeline runs on the server
          (classification, RAG retrieval, adversarial debate, reporting).
        </p>

        <div
          className={`dropzone${drag ? " drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById("file-input")?.click()}
          role="presentation"
        >
          <input
            id="file-input"
            type="file"
            accept={accept}
            onChange={(e) => onPick(e.target.files?.[0] ?? null)}
          />
          <div className="dropzone-icon">
            <Icon.Upload />
          </div>
          <strong>{file ? file.name : "Drop a file here, or click to browse"}</strong>
          <span>{file ? `${(file.size / 1024).toFixed(1)} KB` : "Drag & drop or browse to select"}</span>
          {!file && (
            <div className="dropzone-formats">
              <span>PDF</span>
              <span>DOCX</span>
              <span>TXT</span>
            </div>
          )}
        </div>

        <div className="gt-row">
          <label className="gt-label" htmlFor="gt-input">
            <span className="gt-label-title">Ground truth (optional)</span>
            <span className="gt-label-hint">
              JSON <code>{"{article_id: \"Full|Partial|Missing\"}"}</code> enables confusion matrix and
              per-class precision/recall/F1.
            </span>
          </label>
          <div className="gt-input-row">
            <input
              id="gt-input"
              type="file"
              accept=".json,application/json"
              onChange={(e) => setGtFile(e.target.files?.[0] ?? null)}
            />
            {gtFile && (
              <button type="button" className="ghost ghost-sm" onClick={() => setGtFile(null)}>
                Clear
              </button>
            )}
          </div>
        </div>

        <div className="btn-row">
          <button type="button" className="primary" disabled={!file || loading} onClick={runAnalysis}>
            {loading ? (
              <>
                <span className="spinner" aria-hidden style={{ width: 14, height: 14, borderWidth: 2 }} />
                Analyzing…
              </>
            ) : (
              <>
                <Icon.Play /> Run compliance analysis
              </>
            )}
          </button>
          {file && !loading && (
            <button type="button" className="ghost" onClick={() => onPick(null)}>
              Clear
            </button>
          )}
          {file && <span className="file-pill mono">{file.name}</span>}
          {gtFile && <span className="file-pill mono" title="Ground truth">GT · {gtFile.name}</span>}
        </div>

        {loading && (
          <div className="status loading">
            <div className="spinner" aria-hidden />
            <span>
              Running the compliance pipeline. This may take several minutes on first load (models and
              indexes).
            </span>
          </div>
        )}

        {error && <div className="status error">{error}</div>}
        {warning && <div className="status" style={{ borderColor: "var(--line)" }}>{warning}</div>}
      </section>

      {summary && (
        <>
          <div className="section-title">Run summary</div>
          <section className="card">
            <h2>
              <Icon.Doc /> Results
            </h2>
            <div className="run-meta">
              <span className="tag">
                ID <span className="mono">{summary.doc_id}</span>
              </span>
              {summary.filename ? (
                <span className="tag">
                  File <span className="mono">{summary.filename}</span>
                </span>
              ) : null}
              {summary.doc_type ? (
                <span className="tag">{summary.doc_type.replace(/_/g, " ")}</span>
              ) : null}
            </div>

            <div className="results-grid">
              <div className="risk-panel">
                <div className="label">Overall risk</div>
                <RiskGauge score={summary.risk_score} level={summary.risk_level} />
              </div>

              <div className="grid-metrics">
                <div className="metric">
                  <label>Risk level</label>
                  <div className={`val ${riskClass(summary.risk_level)}`}>
                    {summary.risk_level ?? "—"}
                  </div>
                </div>
                <div className="metric">
                  <label>Document type</label>
                  <div className="val" style={{ fontSize: "1rem", textTransform: "capitalize" }}>
                    {(summary.doc_type || "—").replace(/_/g, " ")}
                  </div>
                </div>
                <div className="metric">
                  <label>Articles evaluated</label>
                  <div className="val">{summary.articles_evaluated ?? "—"}</div>
                </div>
                <div className="metric">
                  <label>Hallucination rate</label>
                  <div className="val">
                    {summary.hallucination_rate != null
                      ? `${(Number(summary.hallucination_rate) * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                </div>
                <div className="metric" style={{ gridColumn: "span 2" }}>
                  <label>Regulations</label>
                  <div className="val" style={{ fontSize: "0.95rem", textTransform: "uppercase" }}>
                    {(summary.regulations || []).join(" · ") || "—"}
                  </div>
                </div>
              </div>
            </div>

            <div className="section-title" style={{ marginTop: "1.75rem" }}>
              Deliverables
            </div>
            <div className="downloads">
              {downloads?.map((d) => {
                const iconEl =
                  d.iconKind === "pdf" ? <Icon.PdfFile /> : d.iconKind === "log" ? <Icon.Terminal /> : <Icon.JsonFile />;
                return (
                  <a key={d.href} className="btn-link" href={d.href} target="_blank" rel="noreferrer">
                    <div className={`dl-icon ${d.iconKind}`}>{iconEl}</div>
                    <div className="dl-body">
                      <span className="title">{d.label}</span>
                      <span className="hint">{d.hint}</span>
                    </div>
                    <span className="dl-arrow">
                      <Icon.ArrowUpRight />
                    </span>
                  </a>
                );
              })}
            </div>
          </section>

          {summary.evaluation && (
            <EvaluationSection evaluation={summary.evaluation} />
          )}

          {detail?.violations && detail.violations.length > 0 && (
            <>
              <div className="section-title">Article findings (preview)</div>
              <p className="section-hint">
                Verdict = coverage. Risk = residual exposure given that verdict (not raw statute weight alone).
                {detail.violations.some((x) => x.article_priority) ? (
                  <> Hover a risk value for statutory article priority where available.</>
                ) : null}
              </p>
              <section className="card" style={{ padding: 0, overflow: "hidden" }}>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Article</th>
                        <th>Verdict</th>
                        <th title="Residual exposure after combining coverage with article importance">Risk</th>
                        <th>Summary</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.violations.map((v) => (
                        <tr key={`${v.article_id}-${v.regulation}`}>
                          <td className="mono-cell">
                            {v.article_id}
                            {v.article_title ? (
                              <div className="article-title">{v.article_title}</div>
                            ) : null}
                          </td>
                          <td>
                            <span className={verdictBadge(v.verdict)}>{v.verdict}</span>
                          </td>
                          <td>
                            <span
                              className={riskPillClass(v.risk_level)}
                              title={
                                v.article_priority
                                  ? `Statutory article priority (weight): ${v.article_priority}`
                                  : undefined
                              }
                            >
                              {v.risk_level ?? "—"}
                            </span>
                          </td>
                          <td style={{ maxWidth: "420px", color: "var(--muted)" }}>
                            {(v.reasoning || "").slice(0, 220)}
                            {(v.reasoning || "").length > 220 ? "…" : ""}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </>
      )}

      <p className="footer-note">
        API documentation: <a href="/docs">/docs</a> · Health: <a href="/api/v1/health">/api/v1/health</a>
      </p>
    </div>
  );
}
