import { Component, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, errorText, isAbort } from "./api";
import Analysis from "./components/Analysis";
import Inspector from "./components/Inspector";
import Sidebar from "./components/Sidebar";
import Timeline from "./components/Timeline";
import Toolbar, { ViewPicker } from "./components/Toolbar";
import VideoStage, { loadLayers, saveLayers, type Layers, type VideoStageHandle } from "./components/VideoStage";
import { bounceFrameOf } from "./geometry";
import { halfLabel, playerName } from "./i18n";
import { Spinner, VideoIcon } from "./icons";
import type { Async, Frames, Health, LabelRecord, LabelSummary, MediaName, Report, Review, RunSummary } from "./types";
import { useAnalysisJob } from "./useAnalysisJob";

// three.js is large: load the 3D replay in its own chunk so the review UI paints first.
const Replay3D = lazy(() => import("./components/Replay3D"));

/** Keeps a failing panel (e.g. WebGL / chunk load error) from taking the whole window down. */
class PanelBoundary extends Component<{ title: string; children: ReactNode }, { error: string | null }> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(err: unknown) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
  render() {
    if (this.state.error !== null) {
      return (
        <section className="section" aria-label={this.props.title}>
          <h2 className="section-title">{this.props.title}</h2>
          <div className="inline-error" role="alert">
            <span>Bu bölüm yüklenemedi: {this.state.error}</span>
            <button type="button" className="link" onClick={() => this.setState({ error: null })}>
              Yeniden dene
            </button>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

/** Minimal shape checks so a malformed payload yields an inline error instead of a crash. */
function checkReport(r: Report): Report {
  if (!r || typeof r !== "object") throw new Error("Rapor bir nesne değil.");
  if (!r.video || !(r.video.fps > 0)) throw new Error("Raporda geçerli bir video.fps yok.");
  if (!Array.isArray(r.flights) || !Array.isArray(r.hits)) throw new Error("Raporda vuruş listeleri eksik.");
  if (!r.players?.near || !r.players?.far) throw new Error("Raporda oyuncu özetleri eksik.");
  if (!r.calibration || !r.ball) throw new Error("Raporda kalibrasyon ya da top özeti eksik.");
  if (!Array.isArray(r.limitations)) throw new Error("Raporda sınırlamalar listesi eksik.");
  return r;
}

function checkFrames(f: Frames): Frames {
  if (!f || typeof f !== "object") throw new Error("Kare verisi bir nesne değil.");
  if (!Array.isArray(f.H) || !Array.isArray(f.ball) || !f.players?.near || !f.players?.far) {
    throw new Error("Kare verisinde H / top / oyuncu dizileri eksik.");
  }
  return f;
}

function sortRuns(list: RunSummary[]): RunSummary[] {
  const ts = (s: string) => {
    const t = Date.parse(s);
    return Number.isNaN(t) ? 0 : t;
  };
  return [...list].sort((a, b) => ts(b.created) - ts(a.created));
}

const UI_KEY = "courtvision.ui";

function loadUi(): { sidebar: boolean; inspector: boolean } {
  try {
    const v = JSON.parse(window.localStorage.getItem(UI_KEY) ?? "{}") as Record<string, unknown>;
    return { sidebar: v.sidebar !== false, inspector: v.inspector !== false };
  } catch {
    return { sidebar: true, inspector: true };
  }
}

function saveUi(ui: { sidebar: boolean; inspector: boolean }) {
  try {
    window.localStorage.setItem(UI_KEY, JSON.stringify(ui));
  } catch {
    /* storage unavailable — keep in memory only */
  }
}

function PageState({ children, alert }: { children: ReactNode; alert?: boolean }) {
  return (
    <div className="page-state">
      <div className="empty" role={alert ? "alert" : undefined}>
        {children}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Async<Health>>({ status: "loading" });
  const [runs, setRuns] = useState<Async<RunSummary[]>>({ status: "loading" });
  const [runId, setRunId] = useState<string | null>(null);
  const [report, setReport] = useState<Async<Report>>({ status: "idle" });
  const [frames, setFrames] = useState<Async<Frames>>({ status: "idle" });
  const [reviews, setReviews] = useState<Async<Review[]>>({ status: "idle" });
  const [currentFrame, setCurrentFrame] = useState(0);
  const [selectedFlightId, setSelectedFlightId] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [view, setView] = useState<MediaName>("source");
  const [mode, setMode] = useState<"review" | "analysis">("review");
  const [labels, setLabels] = useState<LabelRecord[]>([]);
  const [labelSummary, setLabelSummary] = useState<LabelSummary>({ n: 0 });
  const [labelingId, setLabelingId] = useState<number | null>(null);
  const [labelError, setLabelError] = useState<string | null>(null);
  const [layers, setLayers] = useState<Layers>(loadLayers);
  const [ui, setUi] = useState(loadUi);
  const stageRef = useRef<VideoStageHandle>(null);
  const runsCtrl = useRef<AbortController | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadRuns = useCallback(async (preferId?: string) => {
    runsCtrl.current?.abort();
    const ctrl = new AbortController();
    runsCtrl.current = ctrl;
    setRuns((prev) => (prev.status === "ready" ? prev : { status: "loading" }));
    try {
      const list = sortRuns(await api.runs(ctrl.signal));
      setRuns({ status: "ready", data: list });
      setRunId((prev) => {
        if (preferId && list.some((r) => r.id === preferId)) return preferId;
        if (prev && list.some((r) => r.id === prev)) return prev;
        return list[0]?.id ?? null;
      });
    } catch (err) {
      if (isAbort(err)) return;
      setRuns({ status: "error", error: errorText(err) });
    }
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    api.health(ctrl.signal).then(
      (h) => setHealth({ status: "ready", data: h }),
      (err: unknown) => {
        if (!isAbort(err)) setHealth({ status: "error", error: errorText(err) });
      },
    );
    void loadRuns();
    return () => {
      ctrl.abort();
      runsCtrl.current?.abort();
    };
  }, [loadRuns]);

  // Load report, frames and reviews for the selected run (independently; one failing does not blank the others).
  useEffect(() => {
    setSelectedFlightId(null);
    setCurrentFrame(0);
    if (!runId) {
      setReport({ status: "idle" });
      setFrames({ status: "idle" });
      setReviews({ status: "idle" });
      return;
    }
    const ctrl = new AbortController();
    setReport({ status: "loading" });
    setFrames({ status: "loading" });
    setReviews({ status: "loading" });
    api.report(runId, ctrl.signal).then(
      (r) => {
        try {
          setReport({ status: "ready", data: checkReport(r) });
        } catch (err) {
          setReport({ status: "error", error: errorText(err) });
        }
      },
      (err: unknown) => {
        if (!isAbort(err)) setReport({ status: "error", error: errorText(err) });
      },
    );
    api.frames(runId, ctrl.signal).then(
      (f) => {
        try {
          setFrames({ status: "ready", data: checkFrames(f) });
        } catch (err) {
          setFrames({ status: "error", error: errorText(err) });
        }
      },
      (err: unknown) => {
        if (!isAbort(err)) setFrames({ status: "error", error: errorText(err) });
      },
    );
    api.reviews(runId, ctrl.signal).then(
      (list) => setReviews({ status: "ready", data: Array.isArray(list) ? list : [] }),
      (err: unknown) => {
        if (!isAbort(err)) setReviews({ status: "error", error: errorText(err) });
      },
    );
    setLabels([]);
    setLabelSummary({ n: 0 });
    setLabelingId(null);
    api.labels(runId, ctrl.signal).then(setLabels, () => undefined);
    api.labelSummary(runId, ctrl.signal).then(setLabelSummary, () => undefined);
    return () => ctrl.abort();
  }, [runId, reloadKey]);

  const reviewsCtrl = useRef<AbortController | null>(null);
  const reloadReviews = useCallback(() => {
    if (!runId) return;
    reviewsCtrl.current?.abort();
    const ctrl = new AbortController();
    reviewsCtrl.current = ctrl;
    api.reviews(runId, ctrl.signal).then(
      (list) => setReviews({ status: "ready", data: Array.isArray(list) ? list : [] }),
      (err: unknown) => {
        if (!isAbort(err)) setReviews({ status: "error", error: errorText(err) });
      },
    );
  }, [runId]);

  useEffect(() => () => reviewsCtrl.current?.abort(), []);

  const onReviewPosted = useCallback(
    (r: Review) => {
      // Show the new review immediately, then re-read the authoritative history from GET /reviews.
      setReviews((prev) =>
        prev.status === "ready" ? { status: "ready", data: [...prev.data.filter((x) => x.id !== r.id), r] } : prev,
      );
      reloadReviews();
    },
    [reloadReviews],
  );

  const reportData = report.status === "ready" ? report.data : null;
  const framesData = frames.status === "ready" ? frames.data : null;
  const selectedRun = runs.status === "ready" ? (runs.data.find((r) => r.id === runId) ?? null) : null;
  const media = useMemo(() => selectedRun?.media ?? [], [selectedRun]);

  // A run without the chosen video falls back to one it has.
  useEffect(() => {
    if (media.length > 0 && !media.includes(view)) setView(media.includes("source") ? "source" : media[0]);
  }, [media, view]);

  const onSelectFlight = useCallback(
    (id: number | null) => {
      setSelectedFlightId(id);
      if (id === null || !reportData) return;
      const f = reportData.flights.find((x) => x.id === id);
      if (!f) return;
      stageRef.current?.seekToFrame(bounceFrameOf(f) ?? f.start_frame, true);
    },
    [reportData],
  );

  const onTimelineSelect = useCallback((id: number) => setSelectedFlightId(id), []);
  const onOpenShot = useCallback(
    (id: number) => {
      setMode("review");
      onSelectFlight(id);
    },
    [onSelectFlight],
  );
  const onSeek = useCallback((frame: number) => stageRef.current?.seekToFrame(frame), []);
  const onSidebarSelect = useCallback((id: number) => onSelectFlight(id), [onSelectFlight]);

  const onStartLabel = useCallback(
    (id: number) => {
      setLabelError(null);
      setMode("review");
      onSelectFlight(id);
      setLabelingId(id);
    },
    [onSelectFlight],
  );
  const onCancelLabel = useCallback(() => setLabelingId(null), []);
  const onLabel = useCallback(
    (frame: number, px: [number, number]) => {
      if (!runId || labelingId === null) return;
      let labeler = "";
      try {
        labeler = window.localStorage.getItem("courtvision.reviewer")?.trim() ?? "";
      } catch {
        /* storage unavailable */
      }
      api.postLabel(runId, { flight_id: labelingId, frame, px, labeler: labeler || "Hakem", note: "" }).then(
        (rec) => {
          setLabels((prev) => [...prev, rec]);
          setLabelingId(null);
          api.labelSummary(runId).then(setLabelSummary, () => undefined);
        },
        (err: unknown) => setLabelError(errorText(err)),
      );
    },
    [runId, labelingId],
  );

  const onJobDone = useCallback((newRunId: string) => void loadRuns(newRunId), [loadRuns]);
  const job = useAnalysisJob(onJobDone);
  const pickFile = useCallback(() => fileRef.current?.click(), []);

  const onViewChange = useCallback((v: MediaName) => {
    stageRef.current?.rememberPosition();
    setView(v);
  }, []);

  const onToggleLayer = useCallback((k: keyof Layers) => {
    setLayers((prev) => {
      const next = { ...prev, [k]: !prev[k] };
      saveLayers(next);
      return next;
    });
  }, []);

  const toggleSidebar = useCallback(
    () =>
      setUi((u) => {
        const next = { ...u, sidebar: !u.sidebar };
        saveUi(next);
        return next;
      }),
    [],
  );
  const toggleInspector = useCallback(
    () =>
      setUi((u) => {
        const next = { ...u, inspector: !u.inspector };
        saveUi(next);
        return next;
      }),
    [],
  );

  const selectedFlight = useMemo(
    () => (reportData && selectedFlightId !== null ? (reportData.flights.find((f) => f.id === selectedFlightId) ?? null) : null),
    [reportData, selectedFlightId],
  );
  const shotLabel =
    selectedFlight && reportData
      ? `Vuruş ${selectedFlight.id + 1} · ${playerName(reportData.players[selectedFlight.hitter].label)} → ${halfLabel(selectedFlight.receiver)}`
      : null;

  let body: ReactNode;
  if (runs.status === "loading" || runs.status === "idle") {
    body = (
      <PageState>
        <Spinner size={22} />
        <p className="empty-text">Sunucuya bağlanılıyor…</p>
      </PageState>
    );
  } else if (runs.status === "error") {
    body = (
      <PageState alert>
        <p className="empty-title">Analizler yüklenemedi</p>
        <p className="empty-text">{runs.error}</p>
        <p className="empty-text">
          Sunucuyu <code>courtvision serve</code> ile başlatın (8765 portunu dinler).
        </p>
        <button type="button" className="btn" onClick={() => void loadRuns()}>
          Yeniden Dene
        </button>
      </PageState>
    );
  } else if (runs.data.length === 0) {
    body = (
      <PageState>
        <VideoIcon size={44} />
        <p className="empty-title">Henüz analiz yok</p>
        <p className="empty-text">
          Bir maç videosu seçin; analiz bitince burada açılır. Komut satırından da çalıştırabilirsiniz:{" "}
          <code>courtvision analyze &lt;video&gt;</code>
        </p>
        <button type="button" className="btn btn--primary" onClick={pickFile}>
          Video Seç…
        </button>
      </PageState>
    );
  } else if (!runId || !selectedRun || report.status === "loading" || report.status === "idle") {
    body = (
      <PageState>
        <Spinner size={22} />
        <p className="empty-text">Rapor yükleniyor…</p>
      </PageState>
    );
  } else if (report.status === "error") {
    body = (
      <PageState alert>
        <p className="empty-title">Rapor açılamadı</p>
        <p className="empty-text">{report.error}</p>
        <button type="button" className="btn" onClick={() => setReloadKey((k) => k + 1)}>
          Yeniden Dene
        </button>
      </PageState>
    );
  } else {
    const rep = report.data;
    body = (
      <>
        <Sidebar
          report={rep}
          reviews={reviews}
          selectedFlightId={selectedFlightId}
          onSelect={onSidebarSelect}
          health={health}
        />
        <div className="content-stack">
          {mode === "analysis" && <Analysis report={rep} onOpenShot={onOpenShot} labelSummary={labelSummary} />}
          <main className="content" hidden={mode === "analysis"}>
            <div className="phone-bar">
              <ViewPicker media={media} view={view} onChange={onViewChange} />
            </div>
            <VideoStage
              key={runId}
              ref={stageRef}
              runId={runId}
              media={media}
              view={view}
              layers={layers}
              report={rep}
              frames={frames}
              currentFrame={currentFrame}
              selectedFlightId={selectedFlightId}
              onFrame={setCurrentFrame}
              labelMode={labelingId !== null && labelingId === selectedFlightId}
              onLabel={onLabel}
              onCancelLabel={onCancelLabel}
            />
            <Timeline
              report={rep}
              calibValid={framesData?.calib_valid ?? null}
              currentFrame={currentFrame}
              selectedFlightId={selectedFlightId}
              onSeek={onSeek}
              onSelectFlight={onTimelineSelect}
            />
            <PanelBoundary title="3B Tekrar">
              <Suspense
                fallback={
                  <section className="section">
                    <h2 className="section-title">3B Tekrar</h2>
                    <div className="replay">
                      <div className="replay-cover">
                        <Spinner />
                      </div>
                    </div>
                  </section>
                }
              >
                <Replay3D flight={selectedFlight} shotLabel={shotLabel} />
              </Suspense>
            </PanelBoundary>
          </main>
          <Inspector
            runId={runId}
            report={rep}
            frames={framesData}
            currentFrame={currentFrame}
            reviews={reviews}
            selectedFlightId={selectedFlightId}
            onSelectFlight={onSelectFlight}
            onReviewPosted={onReviewPosted}
            onReloadReviews={reloadReviews}
            labels={labels}
            labelingId={labelingId}
            labelError={labelError}
            onStartLabel={onStartLabel}
            onCancelLabel={onCancelLabel}
          />
        </div>
      </>
    );
  }

  return (
    <div className={`app${ui.sidebar ? "" : " app--no-sidebar"}${ui.inspector ? "" : " app--no-inspector"}`}>
      <Toolbar
        runs={runs}
        runId={runId}
        onSelectRun={setRunId}
        media={media}
        view={view}
        onViewChange={onViewChange}
        layers={layers}
        onToggleLayer={onToggleLayer}
        job={job.state}
        onPickFile={pickFile}
        onCancelJob={job.cancel}
        onDismissJob={job.dismiss}
        mode={mode}
        onModeChange={setMode}
        sidebarOpen={ui.sidebar}
        onToggleSidebar={toggleSidebar}
        inspectorOpen={ui.inspector}
        onToggleInspector={toggleInspector}
      />
      <input
        ref={fileRef}
        type="file"
        accept="video/*,.mp4,.mov,.mkv,.avi"
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) job.start(f);
        }}
      />
      <div className="window">{body}</div>
    </div>
  );
}
