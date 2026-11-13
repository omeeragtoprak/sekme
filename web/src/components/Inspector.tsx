import { memo, useEffect, useMemo, useState, type CSSProperties } from "react";
import { api, errorText } from "../api";
import {
  DASH,
  fmtClock,
  fmtDate,
  fmtNum,
  fmtProb,
  fmtSigmaCm,
  fmtSignedCm,
  verdictColor,
  verdictKind,
} from "../geometry";
import { DECISIONS, decisionShort, halfLabel, hitterLabel, lineLabel, strokeLabel, verdictLabel } from "../i18n";
import { CheckIcon, CourtIcon, StatusGlyph } from "../icons";
import type {
  Async,
  Call,
  Decision,
  Flight,
  Frames,
  LabelRecord,
  Report,
  Review,
  ServeSummary,
  SystemCallSnapshot,
} from "../types";
import MiniCourt from "./MiniCourt";
import SessionInfo from "./SessionInfo";

type Tab = "shot" | "court" | "session";
const TABS: { key: Tab; label: string }[] = [
  { key: "shot", label: "Vuruş" },
  { key: "court", label: "Saha" },
  { key: "session", label: "Oturum" },
];

type Props = {
  runId: string;
  report: Report;
  frames: Frames | null;
  currentFrame: number;
  reviews: Async<Review[]>;
  selectedFlightId: number | null;
  onSelectFlight: (id: number) => void;
  onReviewPosted: (r: Review) => void;
  onReloadReviews: () => void;
  labels: LabelRecord[];
  labelingId: number | null;
  labelError: string | null;
  onStartLabel: (id: number) => void;
  onCancelLabel: () => void;
  serveSummary: ServeSummary | null;
  serveError: string | null;
  onServeMark: (frame: number) => void;
  onServeRemove: (id: string) => void;
  onServeComplete: (complete: boolean) => void;
};

function InspectorImpl(p: Props) {
  const [tab, setTab] = useState<Tab>("shot");
  const flight = useMemo(
    () => (p.selectedFlightId === null ? null : (p.report.flights.find((f) => f.id === p.selectedFlightId) ?? null)),
    [p.report.flights, p.selectedFlightId],
  );

  // Choosing a shot while the session facts are showing brings its details forward.
  useEffect(() => {
    if (p.selectedFlightId !== null) setTab((t) => (t === "session" ? "shot" : t));
  }, [p.selectedFlightId]);

  const flightReviews = useMemo(() => {
    if (p.reviews.status !== "ready" || !flight) return [];
    return p.reviews.data
      .filter((r) => r.flight_id === flight.id)
      .sort((a, b) => Date.parse(b.created) - Date.parse(a.created));
  }, [p.reviews, flight]);

  return (
    <aside className="inspector" aria-label="Denetçi">
      <div className="insp-tabs">
        <div className="segmented segmented--fill" role="tablist" aria-label="Denetçi bölümleri">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={tab === t.key}
              aria-pressed={tab === t.key}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <div className="insp-body" role="tabpanel">
        {tab === "shot" &&
          (flight ? (
            <>
              <ShotPane
                key={flight.id}
                runId={p.runId}
                report={p.report}
                flight={flight}
                reviews={flightReviews}
                reviewsState={p.reviews}
                onReviewPosted={p.onReviewPosted}
                onReloadReviews={p.onReloadReviews}
              />
              {flight.call && (
                <LabelPane
                  labels={p.labels.filter((l) => l.flight_id === flight.id)}
                  active={p.labelingId === flight.id}
                  error={p.labelingId === flight.id ? p.labelError : null}
                  onStart={() => p.onStartLabel(flight.id)}
                  onCancel={p.onCancelLabel}
                />
              )}
            </>
          ) : (
            <div className="empty">
              <CourtIcon size={40} />
              <p className="empty-title">Vuruş seçilmedi</p>
              <p className="empty-text">Vuruş listesinden, zaman çizelgesinden ya da sahadan bir vuruş seçin.</p>
            </div>
          ))}
        {tab === "shot" && (
          <ServeLabelPane
            summary={p.serveSummary}
            error={p.serveError}
            frame={p.currentFrame}
            fps={p.report.video.fps}
            onMark={p.onServeMark}
            onRemove={p.onServeRemove}
            onComplete={p.onServeComplete}
          />
        )}
        {tab === "court" && (
          <MiniCourt
            report={p.report}
            frames={p.frames}
            currentFrame={p.currentFrame}
            selectedFlightId={p.selectedFlightId}
            onSelectFlight={p.onSelectFlight}
          />
        )}
        {tab === "session" && <SessionInfo report={p.report} />}
      </div>
    </aside>
  );
}

// ---------------------------------------------------------------------------------------------

function snapshotText(s: SystemCallSnapshot): string {
  return `${verdictLabel(s.verdict)} · ${fmtProb(s.p_in)} · ${fmtSignedCm(s.margin_m)} ± ${fmtSigmaCm(s.sigma_m)} · ${lineLabel(s.line)}`;
}

function textClass(v: Call["verdict"]): string {
  return v === "IN" ? "text-in" : v === "OUT" ? "text-out" : "text-review";
}

type PaneProps = {
  runId: string;
  report: Report;
  flight: Flight;
  reviews: Review[];
  reviewsState: Async<Review[]>;
  onReviewPosted: (r: Review) => void;
  onReloadReviews: () => void;
};

function ShotPane({ runId, report, flight: f, reviews, reviewsState, onReviewPosted, onReloadReviews }: PaneProps) {
  const c = f.call;
  const route = `${hitterLabel(f.hitter_name, report.players[f.hitter].label)} → ${halfLabel(f.receiver)}`;
  const unreliable = f.rejected ? "güvenilmez" : null;
  const radarErr =
    f.board_speed_kmh && f.hit_speed_kmh ? (100 * (f.hit_speed_kmh - f.board_speed_kmh)) / f.board_speed_kmh : null;
  const bounceFrame = f.bounce?.frame ?? f.bounce_frame;

  return (
    <>
      <div className="hero">
        <span className={`glyph--${verdictKind(c?.verdict)}`} style={{ display: "grid" }}>
          <StatusGlyph kind={verdictKind(c?.verdict)} size={34} />
        </span>
        <span className="hero-title">{verdictLabel(c?.verdict)}</span>
        <span className="hero-sub">
          Vuruş {f.id + 1}
          {f.kind === "serve" ? " · Servis" : f.stroke ? ` · ${strokeLabel(f.stroke, f.stroke_wing)}` : ""} · {route}
          {bounceFrame !== null && bounceFrame !== undefined ? ` · sekme karesi ${Math.round(bounceFrame)}` : ""}
        </span>
      </div>

      {c && (
        <section className="block">
          <h3 className="group-title">Karar</h3>
          <div className="group">
            <div className="row">
              <span className="row-label">İçeride olma olasılığı</span>
              <span className="row-value">
                <strong className={textClass(c.verdict)}>{fmtProb(c.p_in)}</strong>
              </span>
            </div>
            <div className="gauge-wrap">
              <div
                className="gauge"
                role="meter"
                aria-label="Topun içeride olma olasılığı"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(c.p_in * 1000) / 10}
                aria-valuetext={fmtProb(c.p_in)}
              >
                <div
                  className="gauge-fill"
                  style={{ width: `${Math.max(1, Math.min(100, c.p_in * 100))}%`, "--c": verdictColor(c.verdict) } as CSSProperties}
                />
                <i className="gauge-tick" style={{ left: "5%" }} />
                <i className="gauge-tick" style={{ left: "95%" }} />
              </div>
            </div>
            <div className="row">
              <span className="row-label">Çizgiye mesafe</span>
              <span className="row-value num" title="Belirleyen çizginin dış kenarına işaretli mesafe ve 1σ belirsizliği">
                <strong>{fmtSignedCm(c.margin_m)}</strong> ± {fmtSigmaCm(c.sigma_m)}
              </span>
            </div>
            <div className="row">
              <span className="row-label">Belirleyen çizgi</span>
              <span className="row-value">{lineLabel(c.line)}</span>
            </div>
            {c.conflict && (
              <div className="row-stack">
                <span className="text-warn" style={{ fontWeight: 600 }}>
                  Kanıt çelişkisi
                </span>
                <p className="row-sub">{c.conflict}</p>
              </div>
            )}
          </div>
          <p className="group-foot">
            Olasılık %95 ve üstündeyse İçeride, %5 ve altındaysa Dışarıda; arası her zaman insan incelemesine gider. Mesafe
            artıysa top içeride.
          </p>
        </section>
      )}

      <section className="block">
        <h3 className="group-title">Ölçüm</h3>
        <div className="group">
          <div className="row">
            <span className="row-label">Vuruş hızı</span>
            <span className="row-value num">
              {unreliable ?? (f.hit_speed_kmh !== null ? `${fmtNum(f.hit_speed_kmh, 0)} km/sa` : DASH)}
            </span>
          </div>
          {f.board_speed_kmh ? (
            <div className="row">
              <span className="row-label">Radar panosu</span>
              <span className="row-value num" title="Stadyum radar panosundan OCR ile okundu">
                {fmtNum(f.board_speed_kmh, 0)} km/sa
                {radarErr !== null ? ` · fark ${radarErr > 0 ? "+" : ""}%${fmtNum(radarErr, 1)}` : ""}
              </span>
            </div>
          ) : null}
          <div className="row">
            <span className="row-label">Sekmedeki hız</span>
            <span className="row-value num">
              {unreliable ?? (f.bounce_speed_kmh !== null ? `${fmtNum(f.bounce_speed_kmh, 0)} km/sa` : DASH)}
            </span>
          </div>
          <div className="row">
            <span className="row-label">File üstü</span>
            <span className="row-value num" title="3B uydurmadan; ± değeri 1σ">
              {f.net_clearance_m !== null
                ? `${fmtNum(f.net_clearance_m, 2)}${f.net_clearance_sd_m != null ? ` ± ${fmtNum(f.net_clearance_sd_m, 2)}` : ""} m`
                : DASH}
            </span>
          </div>
          {f.hit_height_m !== null && f.kind === "serve" && (
            <div className="row">
              <span className="row-label">Temas yüksekliği</span>
              <span className="row-value num">{unreliable ?? `${fmtNum(f.hit_height_m, 2)} m`}</span>
            </div>
          )}
          {f.stroke !== undefined && f.kind !== "serve" && (
            <div className="row">
              <span className="row-label">Vuruş türü</span>
              <span className="row-value" title={f.stroke_cue}>
                {strokeLabel(f.stroke, f.stroke_wing) ?? "belirlenemedi"}
                {f.stroke && !f.stroke_confident ? " (düşük güven)" : ""}
              </span>
            </div>
          )}
          {f.hand && f.kind === "serve" && (
            <div className="row">
              <span className="row-label">El</span>
              <span className="row-value">{f.hand === "right" ? "sağ" : "sol"} el</span>
            </div>
          )}
          <div className="row">
            <span className="row-label">Uydurma hatası</span>
            <span className="row-value num">{f.fit ? `${fmtNum(f.fit.rms_px, 2)} px · ${f.fit.n_obs} gözlem` : DASH}</span>
          </div>
        </div>
        {f.rejected && (
          <p className="group-foot">3B uydurma fizik kontrollerinden geçmedi; bu değerler gösterilmiyor (nedeni aşağıdaki notlarda).</p>
        )}
        {f.play_continued === true && c?.verdict === "IN" && (
          <p className="group-foot">Karşı oyuncu bu sekmeden sonra topa vurdu; bu, İçeride kararıyla tutarlı.</p>
        )}
      </section>

      {f.notes.length > 0 && (
        <section className="block">
          <h3 className="group-title">{c ? "Analiz notları" : "Neden karar yok"}</h3>
          <div className="group">
            {f.notes.map((n, i) => (
              <div key={i} className="row-stack">
                <p>{n}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {c && (
        <ReviewForm
          runId={runId}
          flight={f}
          call={c}
          reviews={reviews}
          reviewsState={reviewsState}
          onReviewPosted={onReviewPosted}
          onReloadReviews={onReloadReviews}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------------------------

const REVIEWER_KEY = "courtvision.reviewer";

function loadReviewer(): string {
  try {
    return window.localStorage.getItem(REVIEWER_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveReviewer(name: string) {
  try {
    window.localStorage.setItem(REVIEWER_KEY, name);
  } catch {
    /* storage unavailable — keep in memory only */
  }
}

type FormProps = {
  runId: string;
  flight: Flight;
  call: Call;
  reviews: Review[];
  reviewsState: Async<Review[]>;
  onReviewPosted: (r: Review) => void;
  onReloadReviews: () => void;
};

function ReviewForm({ runId, flight, call, reviews, reviewsState, onReviewPosted, onReloadReviews }: FormProps) {
  const [decision, setDecision] = useState<Decision | null>(null);
  const [reviewer, setReviewer] = useState(loadReviewer);
  const [note, setNote] = useState("");
  const [posting, setPosting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const name = reviewer.trim();
  const nameOk = name.length >= 1 && name.length <= 64;
  const latest = reviews[0] ?? null;
  const idBase = `review-${flight.id}`;

  const submit = async () => {
    if (!decision || !nameOk) return;
    setPosting(true);
    setError(null);
    setFlash(null);
    try {
      const r = await api.postReview(runId, {
        flight_id: flight.id,
        decision,
        reviewer: name,
        note: note.trim().slice(0, 500),
      });
      onReviewPosted(r);
      setDecision(null);
      setNote("");
      setFlash(`Kaydedildi: ${decisionShort(r.decision)}`);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setPosting(false);
    }
  };

  return (
    <>
      <section className="block">
        <h3 className="group-title">Hakem kararı</h3>
        {reviewsState.status === "error" && (
          <div className="inline-error" role="alert" style={{ marginBottom: 8 }}>
            <span>Karar geçmişi alınamadı: {reviewsState.error}</span>
            <button type="button" className="link" onClick={onReloadReviews}>
              Yeniden dene
            </button>
          </div>
        )}
        {latest && (
          <div className="group" style={{ marginBottom: 10 }}>
            <div className="row">
              <span className="row-label">Son karar</span>
              <span className="row-value">
                <strong>{decisionShort(latest.decision)}</strong>
              </span>
            </div>
            <div className="row">
              <span className="row-label">Hakem</span>
              <span className="row-value">
                {latest.reviewer} · {fmtDate(latest.created)}
              </span>
            </div>
            {latest.decision !== "confirm" && (
              <div className="row">
                <span className="row-label">Sistem kararı</span>
                <span className="row-value">{verdictLabel(call.verdict)} olarak saklanıyor</span>
              </div>
            )}
          </div>
        )}
        <div className="group" role="radiogroup" aria-label="Karar">
          {DECISIONS.map((d) => (
            <button
              key={d.value}
              type="button"
              role="radio"
              aria-checked={decision === d.value}
              className="choice"
              disabled={posting}
              onClick={() => setDecision(decision === d.value ? null : d.value)}
            >
              <span>{d.label}</span>
              <span className="choice-check">{decision === d.value && <CheckIcon size={15} />}</span>
            </button>
          ))}
        </div>
        <div className="group" style={{ marginTop: 10 }}>
          <div className="field-row">
            <label htmlFor={`${idBase}-name`}>Ad</label>
            <input
              id={`${idBase}-name`}
              type="text"
              value={reviewer}
              maxLength={64}
              autoComplete="name"
              placeholder="Adınız"
              onChange={(e) => {
                setReviewer(e.target.value);
                saveReviewer(e.target.value);
              }}
            />
          </div>
          <div className="field-stack">
            <div className="field-head">
              <label htmlFor={`${idBase}-note`}>Not</label>
              <span className="field-count num">{note.length}/500</span>
            </div>
            <textarea
              id={`${idBase}-note`}
              className="textarea"
              value={note}
              maxLength={500}
              rows={2}
              placeholder="İsteğe bağlı — ne gördünüz?"
              onChange={(e) => setNote(e.target.value)}
            />
          </div>
        </div>
        <div className="submit-row">
          <button
            type="button"
            className="btn btn--primary btn--large btn--wide"
            disabled={!decision || !nameOk || posting}
            onClick={() => void submit()}
          >
            {posting ? "Kaydediliyor…" : "Kararı Kaydet"}
          </button>
          {flash && (
            <span className="flash" role="status">
              <CheckIcon size={14} /> {flash}
            </span>
          )}
          {error && (
            <p className="inline-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <p className="group-foot">
          {!decision
            ? "Bir karar seçin. "
            : !nameOk
              ? "Kararın kime ait olduğu görünsün diye adınızı yazın. "
              : ""}
          Karar, o anki sistem kararıyla birlikte yalnızca eklenebilen denetim kaydına yazılır.
        </p>
      </section>

      {(reviews.length > 0 || reviewsState.status === "loading") && (
        <section className="block">
          <h3 className="group-title">Geçmiş</h3>
          {reviewsState.status === "loading" ? (
            <p className="group-foot">Yükleniyor…</p>
          ) : (
            <div className="group">
              {reviews.map((r) => (
                <div key={r.id} className="row-stack">
                  <div className="row-line">
                    <strong>{decisionShort(r.decision)}</strong>
                    <time className="row-sub num" dateTime={r.created}>
                      {fmtDate(r.created)}
                    </time>
                  </div>
                  <p className="row-sub">
                    {r.reviewer}
                    {r.note ? ` — “${r.note}”` : ""}
                  </p>
                  <p className="row-sub row-sub--tertiary">O anki sistem kararı: {snapshotText(r.system_call)}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </>
  );
}

function LabelPane({
  labels,
  active,
  error,
  onStart,
  onCancel,
}: {
  labels: LabelRecord[];
  active: boolean;
  error: string | null;
  onStart: () => void;
  onCancel: () => void;
}) {
  const last = labels[labels.length - 1] ?? null;
  const e = last?.eval;
  return (
    <section className="block">
      <h3 className="group-title">Doğruluk etiketi</h3>
      {last && e && (
        <div className="group" style={{ marginBottom: 10 }}>
          <div className="row">
            <span className="row-label">Sistemden uzaklık</span>
            <span className="row-value num">
              <strong>{fmtNum(e.error_cm, 0)} cm</strong> · 1 piksel ≈ {fmtNum(last.px_cm, 1)} cm
            </span>
          </div>
          <div className="row">
            <span className="row-label">%95 elipsin içinde</span>
            <span className="row-value">{e.inside_95 ? "evet" : "hayır"}</span>
          </div>
          <div className="row">
            <span className="row-label">Etikete göre</span>
            <span className="row-value">
              <strong className={e.truth === "IN" ? "text-in" : "text-out"}>{verdictLabel(e.truth)}</strong> ·{" "}
              {fmtSignedCm(e.truth_margin_cm / 100)}
            </span>
          </div>
          <div className="row">
            <span className="row-label">Sistem kararı</span>
            <span className="row-value">
              {verdictLabel(e.system)}
              {e.agrees === null ? " · insana gitti" : e.agrees ? " · tutuyor" : " · tutmuyor"}
            </span>
          </div>
          <div className="row">
            <span className="row-label">Etiketleyen</span>
            <span className="row-value">
              {last.labeler} · kare {last.frame} · {fmtDate(last.created)}
            </span>
          </div>
        </div>
      )}
      {active ? (
        <button type="button" className="btn btn--wide" onClick={onCancel}>
          İşaretlemeyi Bırak
        </button>
      ) : (
        <button type="button" className="btn btn--wide" onClick={onStart}>
          {last ? "Yeniden İşaretle" : "Sekme Noktasını İşaretle"}
        </button>
      )}
      {error && (
        <p className="inline-error" role="alert" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
      <p className="group-foot">
        {active
          ? "Videoda topun yere değdiği noktaya tıklayın (topun alt kenarı). ← → ile kareyi ayarlayın, büyüteç hassas tıklamaya yardım eder; Esc vazgeçer."
          : "Gerçek sekme noktasını işaretlemek, sistemin doğruluğunu ölçmenin tek yolu. Nokta, sistemin kullandığı aynı kuralla değerlendirilir; her etiket ayrı ve silinemez biçimde kaydedilir. Ad, Hakem kararı bölümündeki addır."}
      </p>
    </section>
  );
}

function ServeLabelPane({
  summary,
  error,
  frame,
  fps,
  onMark,
  onRemove,
  onComplete,
}: {
  summary: ServeSummary | null;
  error: string | null;
  frame: number;
  fps: number;
  onMark: (frame: number) => void;
  onRemove: (id: string) => void;
  onComplete: (complete: boolean) => void;
}) {
  const rows = summary ? [...summary.rows].sort((a, b) => a.frame - b.frame) : [];
  return (
    <section className="block">
      <h3 className="group-title">Servis etiketi</h3>
      <button type="button" className="btn btn--wide" onClick={() => onMark(frame)}>
        Servis Temasını İşaretle (bu kare)
      </button>
      {error && (
        <p className="inline-error" role="alert" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
      {rows.length > 0 && (
        <div className="group" style={{ marginTop: 10 }}>
          {rows.map((r) => (
            <div className="row" key={r.id}>
              <span className="row-label num">
                kare {r.frame} · {fmtClock(r.frame / fps)}
              </span>
              <span className="row-value">
                <span className={r.matched_system_frame === null ? "text-review" : "text-in"}>
                  {r.matched_system_frame === null ? "sistem bulamadı" : "sistem buldu"}
                </span>{" "}
                <button type="button" className="btn" onClick={() => onRemove(r.id)}>
                  Kaldır
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
      <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
        <input type="checkbox" checked={summary?.complete ?? false} onChange={(e) => onComplete(e.target.checked)} />
        Bu koşudaki tüm servisleri işaretledim
      </label>
      <p className="group-foot">
        Servis teması, servis atışında raketin topa değdiği andır. Videoda gördüğünüz her servisi, hatalı olanlar dahil, temas
        karesinde işaretleyin (← → ile kareyi ayarlayın). Kutuyu yalnızca hiçbir servisi atlamadığınızdan eminseniz işaretleyin;
        fazladan servis oranı ancak o zaman hesaplanır. Her kayıt ayrı ve silinemez biçimde tutulur; Kaldır yeni bir kayıt ekler.
      </p>
    </section>
  );
}

const Inspector = memo(InspectorImpl);
export default Inspector;
