import { memo } from "react";
import { fmtDate } from "../geometry";
import { MEDIA_LABEL } from "../i18n";
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronUpDownIcon,
  LayersIcon,
  PlusIcon,
  SidebarLeftIcon,
  SidebarRightIcon,
  StatusGlyph,
} from "../icons";
import type { Async, MediaName, RunSummary } from "../types";
import { describeJob, type JobState } from "../useAnalysisJob";
import Menu from "./Menu";
import { LAYER_LABELS, type Layers } from "./VideoStage";

type Props = {
  runs: Async<RunSummary[]>;
  runId: string | null;
  onSelectRun: (id: string) => void;
  media: MediaName[];
  view: MediaName;
  onViewChange: (v: MediaName) => void;
  layers: Layers;
  onToggleLayer: (k: keyof Layers) => void;
  job: JobState;
  onPickFile: () => void;
  onCancelJob: () => void;
  onDismissJob: () => void;
  mode: "review" | "analysis";
  onModeChange: (m: "review" | "analysis") => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
};

const VIEWS: MediaName[] = ["source", "classic", "pro"];

function stem(name: string): string {
  return name.replace(/\.[a-z0-9]+$/i, "");
}

function ProgressRing({ fraction }: { fraction: number }) {
  const r = 7;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 20 20" width={18} height={18} className="ring" aria-hidden="true">
      <circle cx="10" cy="10" r={r} fill="none" strokeWidth="2.2" className="ring-track" />
      <circle
        cx="10"
        cy="10"
        r={r}
        fill="none"
        strokeWidth="2.2"
        strokeLinecap="round"
        className="ring-fill"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - Math.max(0.02, fraction))}
      />
    </svg>
  );
}

/** Kaynak / Klasik / Pro picker (toolbar on wide windows, above the video on phones). */
export function ViewPicker({ media, view, onChange }: { media: MediaName[]; view: MediaName; onChange: (v: MediaName) => void }) {
  return (
    <div className="segmented" role="group" aria-label="Video görünümü">
      {VIEWS.map((v) => {
        const ok = media.includes(v);
        return (
          <button
            key={v}
            type="button"
            aria-pressed={view === v}
            disabled={!ok}
            title={ok ? undefined : `${MEDIA_LABEL[v]} videosu bu analizde yok`}
            onClick={() => onChange(v)}
          >
            {MEDIA_LABEL[v]}
          </button>
        );
      })}
    </div>
  );
}

function ToolbarImpl(p: Props) {
  const runList = p.runs.status === "ready" ? p.runs.data : [];
  const run = runList.find((r) => r.id === p.runId) ?? null;
  const overlayOn = p.view === "source";
  const job = describeJob(p.job);

  let titleMain = "CourtVision";
  let titleSub = "İnceleme Odası";
  if (run) {
    titleMain = stem(run.video);
    titleSub = `${fmtDate(run.created)} · ${run.n_flights} vuruş · ${run.n_calls} karar`;
  } else if (p.runs.status === "loading" || p.runs.status === "idle") {
    titleSub = "Analizler yükleniyor…";
  } else if (p.runs.status === "error") {
    titleSub = "Analizler alınamadı";
  }

  return (
    <header className="toolbar">
      <div className="tb-leading">
        <div className="tb-group">
          <button
            type="button"
            className="tb-btn"
            aria-pressed={p.sidebarOpen}
            onClick={p.onToggleSidebar}
            title={p.sidebarOpen ? "Kenar çubuğunu gizle" : "Kenar çubuğunu göster"}
            aria-label="Kenar çubuğu"
          >
            <SidebarLeftIcon />
          </button>
        </div>
      </div>

      <div className="tb-center">
        <div className="doc-title">
          <div className="doc-title-text">
            <span className="doc-title-main">{titleMain}</span>
            <span className="doc-title-sub">{titleSub}</span>
          </div>
          {runList.length > 1 && <ChevronUpDownIcon size={14} />}
          {runList.length > 0 && (
            <select
              className="doc-title-select"
              aria-label="Analiz seç"
              title="Analiz seç"
              value={p.runId ?? ""}
              onChange={(e) => p.onSelectRun(e.target.value)}
            >
              {runList.map((r) => (
                <option key={r.id} value={r.id}>
                  {stem(r.video)} — {fmtDate(r.created)}
                </option>
              ))}
            </select>
          )}
        </div>

        <span className="tb-spacer" />

        {run && (
          <div className="segmented" role="group" aria-label="Görünüm">
            <button type="button" aria-pressed={p.mode === "review"} onClick={() => p.onModeChange("review")}>
              İnceleme
            </button>
            <button type="button" aria-pressed={p.mode === "analysis"} onClick={() => p.onModeChange("analysis")}>
              Analiz
            </button>
          </div>
        )}

        {run && p.mode === "review" && (
          <>
            <ViewPicker media={p.media} view={p.view} onChange={p.onViewChange} />
            <div className="tb-group">
              <Menu
                title="Katmanlar"
                align="end"
                label={
                  <>
                    <LayersIcon />
                    <ChevronDownIcon size={12} />
                  </>
                }
              >
                {() => (
                  <>
                    <div className="menu-label">Videonun üstünde göster</div>
                    {LAYER_LABELS.map((l) => (
                      <button
                        key={l.key}
                        type="button"
                        role="menuitemcheckbox"
                        aria-checked={p.layers[l.key]}
                        className="menu-item"
                        disabled={!overlayOn}
                        onClick={() => p.onToggleLayer(l.key)}
                      >
                        <span className="menu-check">{p.layers[l.key] && <CheckIcon size={13} />}</span>
                        {l.label}
                      </button>
                    ))}
                    {!overlayOn && (
                      <>
                        <div className="menu-sep" />
                        <p className="menu-note">
                          Katmanlar tarayıcıda yalnızca Kaynak görünümünde çizilir; Klasik ve Pro videolarda katman videoya
                          işlenmiştir.
                        </p>
                      </>
                    )}
                  </>
                )}
              </Menu>
            </div>
          </>
        )}
      </div>

      <div className="tb-trailing">
        <div className="tb-group">
          {p.job.phase === "idle" ? (
            <button type="button" className="tb-btn" onClick={p.onPickFile} title="Video analiz et…" aria-label="Video analiz et">
              <PlusIcon />
            </button>
          ) : (
            <Menu
              title={`Analiz: ${job.text}`}
              align="end"
              label={
                job.tone === "error" ? (
                  <span className="glyph--out" style={{ display: "grid" }}>
                    <StatusGlyph kind="out" size={17} />
                  </span>
                ) : job.tone === "ok" ? (
                  <span className="glyph--in" style={{ display: "grid" }}>
                    <StatusGlyph kind="in" size={17} />
                  </span>
                ) : (
                  <ProgressRing fraction={job.fraction ?? 0} />
                )
              }
            >
              {(close) => (
                <div className={`job job--${job.tone}`} role="status" aria-live="polite">
                  <span className="job-file" title={"file" in p.job ? p.job.file : ""}>
                    {"file" in p.job ? p.job.file : ""}
                  </span>
                  <span className="job-stage">{job.text}</span>
                  {job.fraction !== null && (
                    <div
                      className="bar"
                      role="progressbar"
                      aria-label="Analiz ilerlemesi"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Math.round(job.fraction * 100)}
                    >
                      <div className="bar-fill" style={{ width: `${(job.fraction * 100).toFixed(1)}%` }} />
                    </div>
                  )}
                  <div className="job-actions">
                    {p.job.phase === "uploading" && (
                      <button type="button" className="btn" onClick={p.onCancelJob}>
                        Yüklemeyi Durdur
                      </button>
                    )}
                    {(p.job.phase === "done" || p.job.phase === "error") && (
                      <>
                        <button
                          type="button"
                          className="btn"
                          onClick={() => {
                            p.onDismissJob();
                            close();
                          }}
                        >
                          Kapat
                        </button>
                        <button
                          type="button"
                          className="btn btn--primary"
                          onClick={() => {
                            p.onDismissJob();
                            close();
                            p.onPickFile();
                          }}
                        >
                          Başka Video…
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </Menu>
          )}
          <button
            type="button"
            className="tb-btn"
            aria-pressed={p.inspectorOpen}
            onClick={p.onToggleInspector}
            title={p.inspectorOpen ? "Denetçiyi gizle" : "Denetçiyi göster"}
            aria-label="Denetçi"
          >
            <SidebarRightIcon />
          </button>
        </div>
      </div>
    </header>
  );
}

const Toolbar = memo(ToolbarImpl);
export default Toolbar;
