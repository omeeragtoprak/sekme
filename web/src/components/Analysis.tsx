import { memo, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { COURT, COURT_LINES, fmtNum, fmtPct, fmtProb, fmtSignedCm, verdictColor, verdictKind } from "../geometry";
import { displayName, halfLabel, hitterLabel, servePlacement, verdictLabel } from "../i18n";
import { StatusGlyph } from "../icons";
import type { Flight, LabelSummary, Report } from "../types";

type Props = { report: Report; onOpenShot: (id: number) => void; labelSummary: LabelSummary };

type Tip = { x: number; y: number; text: string } | null;

/** Players as entities: broadcast names when most shots are named, else the side labels. */
function usePlayers(report: Report) {
  return useMemo(() => {
    const flights = report.flights;
    const named = flights.filter((f) => f.hitter_name).length;
    const bc = report.broadcast && report.broadcast.ocr ? report.broadcast : null;
    const byName = !!bc?.names && named >= 0.5 * flights.length;
    const keys: string[] = byName && bc?.names ? bc.names.map(displayName) : [hitterLabel(null, "P1"), hitterLabel(null, "P2")];
    const keyOf = (f: Flight): string | null =>
      byName ? (f.hitter_name ? displayName(f.hitter_name) : null) : hitterLabel(null, report.players[f.hitter].label);
    return { keys, keyOf, byName, unnamed: byName ? flights.length - named : 0 };
  }, [report]);
}

const SLOT = ["var(--series-1)", "var(--series-2)"];

/** A contact speed is shown only when its 3D fit passed the plausibility gates. */
function trustedSpeed(f: Flight): number | null {
  return f.rejected ? null : f.hit_speed_kmh;
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

function Card({ title, sub, children, wide, tall }: { title: string; sub?: string; children: ReactNode; wide?: boolean; tall?: boolean }) {
  return (
    <section className={`card${wide ? " card--wide" : ""}${tall ? " card--tall" : ""}`}>
      <header className="card-head">
        <h2 className="card-title">{title}</h2>
        {sub && <p className="card-sub">{sub}</p>}
      </header>
      {children}
    </section>
  );
}

type ShowFn = (e: { clientX: number; clientY: number; currentTarget: Element }, text: string) => void;
type ShowAtFn = (el: Element, text: string) => void;

function AnalysisImpl({ report, onOpenShot, labelSummary }: Props) {
  const [tip, setTip] = useState<Tip>(null);
  const players = usePlayers(report);
  const flights = report.flights;
  const calls = flights.filter((f) => f.call);
  const count = (v: string) => calls.filter((f) => f.call?.verdict === v).length;
  const speeds = flights.map(trustedSpeed).filter((v): v is number => v !== null);
  const serves = flights.filter((f) => f.kind === "serve");
  const bc = report.broadcast && report.broadcast.ocr ? report.broadcast : null;
  const radar = bc?.serve_speed;

  const place = (host: Element | null, x: number, y: number, text: string) => {
    if (!(host instanceof HTMLElement)) return;
    const r = host.getBoundingClientRect();
    setTip({ x: x - r.left + host.scrollLeft, y: y - r.top + host.scrollTop, text });
  };
  const show: ShowFn = (e, text) => place(e.currentTarget.closest(".analysis"), e.clientX, e.clientY, text);
  const showAt: ShowAtFn = (el, text) => {
    const b = el.getBoundingClientRect();
    place(el.closest(".analysis"), b.left + b.width / 2, b.top, text);
  };
  const hide = () => setTip(null);
  const measurable = report.calibration.total_frames ? report.calibration.valid_frames / report.calibration.total_frames : null;

  return (
    <div className="analysis" onMouseLeave={hide}>
      <header className="analysis-head">
        <h1 className="analysis-title">{players.byName ? players.keys.join(" – ") : "Maç analizi"}</h1>
        <p className="analysis-sub">
          {fmtNum(report.video.duration_s, 0)} saniyelik klip · {flights.length} vuruş · kalibre kareler {fmtPct(measurable, 0)}
        </p>
      </header>

      <div className="stats">
        <Stat label="Vuruş" value={flights.length} sub={`${calls.length} karar · ${flights.length - calls.length} karar yok`} />
        <Stat
          label="Kararlar"
          value={calls.length}
          sub={
            <span className="stat-verdicts">
              <span className="glyph--in">
                <StatusGlyph kind="in" size={12} />
              </span>
              {count("IN")} içeride
              <span className="glyph--out">
                <StatusGlyph kind="out" size={12} />
              </span>
              {count("OUT")} dışarıda
              <span className="glyph--review">
                <StatusGlyph kind="review" size={12} />
              </span>
              {count("REVIEW")} inceleme
            </span>
          }
        />
        <Stat
          label="Vuruş hızı (ortanca)"
          value={speeds.length ? `${fmtNum(median(speeds), 0)} km/sa` : "—"}
          sub={speeds.length ? `en yüksek ${fmtNum(Math.max(...speeds), 0)} km/sa · ${speeds.length} vuruş` : undefined}
        />
        <Stat
          label="Servis"
          value={serves.length}
          sub={
            radar && radar.n > 0 && radar.mean_error_pct !== null
              ? `radara göre ${radar.mean_error_pct > 0 ? "+" : ""}%${fmtNum(radar.mean_error_pct, 1)} fark (${radar.n} servis)`
              : "radar karşılaştırması yok"
          }
        />
      </div>

      <div className="cards">
        <Card title="Sekme haritası" sub="Kararı olan her sekme; renk kararı gösterir. Noktaya tıklayınca vuruş açılır.">
          <BounceMap flights={calls} onOpen={onOpenShot} showAt={showAt} hide={hide} />
          <div className="legend" aria-hidden="true">
            {(["IN", "OUT", "REVIEW"] as const).map((v) => (
              <span key={v}>
                <span className={`glyph--${verdictKind(v)}`}>
                  <StatusGlyph kind={verdictKind(v)} size={12} />
                </span>
                {verdictLabel(v)}
              </span>
            ))}
          </div>
        </Card>

        <div className="card-stack">
          <Card title="Derinlik" sub="Ralli vuruşlarının karşı yarıda düştüğü bölge, oyuncu başına.">
            <Depth flights={flights} players={players} show={show} hide={hide} />
          </Card>

          <Card title="Servisler" sub="Servis kutusundaki yön, ölçtüğümüz hız ve stadyum radar panosu.">
            <ServeTable serves={serves} report={report} onOpen={onOpenShot} />
          </Card>
        </div>
        <Card title="Vuruş türleri" sub="Vurucunun pozu ve topun gövdeye göre tarafından; hız yalnızca güvenilir 3B uydurmalardan." wide>
          <StrokeTable flights={flights} players={players} />
        </Card>

        <Card title="Vuruş hızı" sub="Her nokta bir vuruş (temas anındaki hız, 3B uydurmadan); dikey çizgi ortanca." wide>
          <SpeedStrip flights={flights} players={players} onOpen={onOpenShot} show={show} showAt={showAt} hide={hide} />
        </Card>

        <Card
          title="Etiketli doğruluk"
          sub="Hakemin işaretlediği gerçek sekme noktalarına göre sistemin hatası. Etiket İnceleme'de, vuruş denetçisinden eklenir."
          wide
        >
          <LabelCard s={labelSummary} />
        </Card>

        <Card
          title="Yayın doğrulaması"
          sub="Skor tabelası ve radar panosu, sistemin kendi ölçümlerini bağımsız olarak kontrol etmek için okunur."
          wide
        >
          <BroadcastCheck report={report} />
        </Card>
      </div>

      {tip && (
        <div className="viz-tip" style={{ left: tip.x, top: tip.y }} role="status">
          {tip.text}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------

const U = 100;
const sx = (x: number) => x * U;
const sy = (y: number) => -y * U;

function BounceMap({ flights, onOpen, showAt, hide }: { flights: Flight[]; onOpen: (id: number) => void; showAt: ShowAtFn; hide: () => void }) {
  return (
    <svg className="map-svg" viewBox="-760 -1300 1520 2600" role="img" aria-label="Kort üzerinde sekme noktaları">
      <rect x={-760} y={-1300} width={1520} height={2600} rx={40} style={{ fill: "var(--court-surround)" }} />
      <rect
        x={sx(-COURT.doublesX)}
        y={sy(COURT.baselineY)}
        width={2 * COURT.doublesX * U}
        height={2 * COURT.baselineY * U}
        style={{ fill: "var(--court-surface)" }}
      />
      <g stroke="#f7f9fc" strokeOpacity={0.9} strokeWidth={1.2} vectorEffect="non-scaling-stroke">
        {COURT_LINES.map((s, i) => (
          <line key={i} x1={sx(s.a[0])} y1={sy(s.a[1])} x2={sx(s.b[0])} y2={sy(s.b[1])} vectorEffect="non-scaling-stroke" />
        ))}
      </g>
      <line x1={-640} y1={0} x2={640} y2={0} stroke="#fff" strokeOpacity={0.75} strokeWidth={2} vectorEffect="non-scaling-stroke" />
      {flights.map((f) => {
        const c = f.call;
        if (!c) return null;
        const text = `Vuruş ${f.id + 1} · ${verdictLabel(c.verdict)} · ${fmtProb(c.p_in)} · ${fmtSignedCm(c.margin_m)}`;
        return (
          <g
            key={f.id}
            className="map-dot"
            role="button"
            tabIndex={0}
            aria-label={text}
            onMouseEnter={(e) => showAt(e.currentTarget, text)}
            onFocus={(e) => showAt(e.currentTarget, text)}
            onMouseLeave={hide}
            onBlur={hide}
            onClick={() => onOpen(f.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onOpen(f.id);
              }
            }}
          >
            <circle cx={sx(c.x)} cy={sy(c.y)} r={52} fill="transparent" />
            <circle
              cx={sx(c.x)}
              cy={sy(c.y)}
              r={22}
              fill={verdictColor(c.verdict)}
              stroke="var(--court-surface)"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      })}
    </svg>
  );
}

type PlayersInfo = ReturnType<typeof usePlayers>;

const ZONES = [
  { key: "short", label: "Kısa", hint: "servis karesi içi", test: (d: number) => d < 6.4 },
  { key: "mid", label: "Orta", hint: "servis çizgisinin gerisi", test: (d: number) => d >= 6.4 && d < 8.9 },
  { key: "deep", label: "Derin", hint: "dip çizgiye son 3 m", test: (d: number) => d >= 8.9 },
];

function Legend({ keys, counts }: { keys: string[]; counts?: number[] }) {
  return (
    <div className="legend">
      {keys.map((k, i) => (
        <span key={k}>
          <i className="legend-dot" style={{ background: SLOT[i] }} aria-hidden="true" />
          {k}
          {counts ? <span className="legend-n"> · {counts[i]}</span> : null}
        </span>
      ))}
    </div>
  );
}

function Depth({ flights, players, show, hide }: { flights: Flight[]; players: PlayersInfo; show: ShowFn; hide: () => void }) {
  const rally = flights.filter((f) => f.kind === "rally" && f.call && f.call.verdict !== "OUT");
  const rows = players.keys.map((k) => {
    const mine = rally.filter((f) => players.keyOf(f) === k);
    return {
      key: k,
      n: mine.length,
      pct: ZONES.map((z) => (mine.length ? mine.filter((f) => z.test(Math.abs(f.call?.y ?? 0))).length / mine.length : 0)),
    };
  });
  if (rows.every((r) => r.n === 0)) return <p className="card-empty">Bu klipte kararı olan ralli vuruşu yok.</p>;
  return (
    <>
      <div className="bars" role="table" aria-label="Derinlik dağılımı">
        {ZONES.map((z, zi) => (
          <div className="bars-row" role="row" key={z.key}>
            <span className="bars-label" role="rowheader">
              {z.label}
              <small>{z.hint}</small>
            </span>
            <div className="bars-group">
              {rows.map((r, ri) => (
                <div
                  className="bar-line"
                  key={r.key}
                  role="cell"
                  aria-label={`${r.key}: ${fmtPct(r.pct[zi], 0)}`}
                  onMouseMove={(e) => show(e, `${r.key} · ${z.label}: ${fmtPct(r.pct[zi], 0)} (${r.n} vuruştan)`)}
                  onMouseLeave={hide}
                >
                  <span className="bar-mark" style={{ width: `${Math.max(0.5, r.pct[zi] * 100)}%`, background: SLOT[ri] }} />
                  <span className="bar-value">{fmtPct(r.pct[zi], 0)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <Legend keys={players.keys} counts={rows.map((r) => r.n)} />
    </>
  );
}

function SpeedStrip({
  flights,
  players,
  onOpen,
  show,
  showAt,
  hide,
}: {
  flights: Flight[];
  players: PlayersInfo;
  onOpen: (id: number) => void;
  show: ShowFn;
  showAt: ShowAtFn;
  hide: () => void;
}) {
  const [table, setTable] = useState(false);
  const max = 220;
  const rows = players.keys.map((k) => {
    const mine = flights.filter((f) => players.keyOf(f) === k && trustedSpeed(f) !== null);
    const v = mine.map((f) => trustedSpeed(f) as number);
    return { key: k, shots: mine, med: median(v), max: v.length ? Math.max(...v) : null };
  });
  const ticks = [0, 50, 100, 150, 200];
  return (
    <>
      <div className="strip">
        {rows.map((r, ri) => (
          <div className="strip-row" key={r.key}>
            <span className="strip-label">{r.key}</span>
            <div className="strip-track">
              {ticks.map((t) => (
                <i key={t} className="strip-grid" style={{ left: `${(t / max) * 100}%` }} aria-hidden="true" />
              ))}
              {r.med !== null && (
                <i className="strip-median" style={{ left: `${(r.med / max) * 100}%` }} aria-hidden="true">
                  <span>{fmtNum(r.med, 0)}</span>
                </i>
              )}
              {r.shots.map((f) => {
                const text = `Vuruş ${f.id + 1}${f.kind === "serve" ? " (servis)" : ""} · ${fmtNum(f.hit_speed_kmh, 0)} km/sa`;
                return (
                  <button
                    key={f.id}
                    type="button"
                    className={`strip-dot${f.kind === "serve" ? " strip-dot--serve" : ""}`}
                    style={{ left: `${Math.min(100, ((f.hit_speed_kmh as number) / max) * 100)}%`, "--c": SLOT[ri] } as CSSProperties}
                    aria-label={text}
                    onMouseMove={(e) => show(e, text)}
                    onMouseLeave={hide}
                    onFocus={(e) => showAt(e.currentTarget, text)}
                    onBlur={hide}
                    onClick={() => onOpen(f.id)}
                  />
                );
              })}
            </div>
          </div>
        ))}
        <div className="strip-row strip-axis" aria-hidden="true">
          <span className="strip-label" />
          <div className="strip-ticks">
            {ticks.map((t) => (
              <span key={t} style={{ left: `${(t / max) * 100}%` }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="card-foot">
        <Legend keys={players.keys} counts={rows.map((r) => r.shots.length)} />
        <span className="legend-extra">
          <i className="strip-dot strip-dot--serve strip-dot--key" aria-hidden="true" /> servis · km/sa
        </span>
        <button type="button" className="link" onClick={() => setTable((t) => !t)} aria-expanded={table}>
          {table ? "Tabloyu gizle" : "Tablo olarak göster"}
        </button>
      </div>
      {table && (
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Oyuncu</th>
              <th scope="col">Vuruş</th>
              <th scope="col">Ortanca</th>
              <th scope="col">En yüksek</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <th scope="row">{r.key}</th>
                <td>{r.shots.length}</td>
                <td>{r.med !== null ? `${fmtNum(r.med, 0)} km/sa` : "—"}</td>
                <td>{r.max !== null ? `${fmtNum(r.max, 0)} km/sa` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="card-note">
        Yalnızca fizik kontrollerinden geçen 3B uydurmalar gösterilir
        {flights.some((f) => f.rejected && f.hit_speed_kmh !== null)
          ? ` (${flights.filter((f) => f.rejected && f.hit_speed_kmh !== null).length} reddedilen uydurma dışarıda)`
          : ""}
        .{players.unnamed > 0 ? ` ${players.unnamed} vuruşta tabela yakında görünmediği için oyuncu adı bilinmiyor; bunlar da gösterilmez.` : ""}
      </p>
    </>
  );
}

const STROKE_ROWS: { key: string; test: (f: Flight) => boolean }[] = [
  { key: "Forehand", test: (f) => f.stroke === "forehand" },
  { key: "Backhand", test: (f) => f.stroke === "backhand" },
  { key: "Vole", test: (f) => f.stroke === "volley" },
  { key: "Smaç", test: (f) => f.stroke === "overhead" },
  { key: "Servis", test: (f) => f.kind === "serve" },
  { key: "Belirlenemedi", test: (f) => f.kind !== "serve" && !f.stroke },
];

function StrokeTable({ flights, players }: { flights: Flight[]; players: PlayersInfo }) {
  if (!flights.some((f) => f.stroke !== undefined)) return <p className="card-empty">Bu analizde vuruş türü hesaplanmamış.</p>;
  const cell = (k: string, row: (typeof STROKE_ROWS)[number]) => {
    const mine = flights.filter((f) => players.keyOf(f) === k && row.test(f));
    const med = median(mine.map(trustedSpeed).filter((v): v is number => v !== null));
    return mine.length ? `${mine.length}${med !== null ? ` · ${fmtNum(med, 0)} km/sa` : ""}` : "—";
  };
  const hands = flights.find((f) => f.hand_source === "servis") ? "servisten belirlendi" : "varsayılan: sağ";
  return (
    <>
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Vuruş türü</th>
            {players.keys.map((k, i) => (
              <th scope="col" key={k}>
                <i className="legend-dot" style={{ background: SLOT[i], display: "inline-block", marginRight: 6 }} aria-hidden="true" />
                {k} <span className="muted">(adet · ortanca hız)</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {STROKE_ROWS.map((row) => (
            <tr key={row.key}>
              <th scope="row">{row.key}</th>
              {players.keys.map((k) => (
                <td key={k}>{cell(k, row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="card-note">
        Oyuncunun eli {hands}. Forehand ve backhand ayrımı US Open klibinde 16 vuruşta gözle kontrol edildi, 16'sı doğruydu; bu küçük
        bir örneklem, bir doğruluk oranı değil.
      </p>
    </>
  );
}

function ServeTable({ serves, report, onOpen }: { serves: Flight[]; report: Report; onOpen: (id: number) => void }) {
  if (serves.length === 0) return <p className="card-empty">Bu klipte servis tespit edilmedi.</p>;
  return (
    <table className="data-table data-table--click">
      <thead>
        <tr>
          <th scope="col">Vuruş</th>
          <th scope="col">Servisi atan</th>
          <th scope="col">Yön</th>
          <th scope="col">Ölçülen</th>
          <th scope="col">Radar panosu</th>
          <th scope="col">Fark</th>
          <th scope="col">Karar</th>
        </tr>
      </thead>
      <tbody>
        {serves.map((f) => {
          const err =
            f.board_speed_kmh && f.hit_speed_kmh ? (100 * (f.hit_speed_kmh - f.board_speed_kmh)) / f.board_speed_kmh : null;
          return (
            <tr key={f.id} onClick={() => onOpen(f.id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onOpen(f.id)}>
              <th scope="row">{f.id + 1}</th>
              <td>
                {hitterLabel(f.server_name ?? f.hitter_name, report.players[f.hitter].label)}{" "}
                <span className="muted">({halfLabel(f.hitter).replace(" yarı", "")})</span>
              </td>
              <td>{f.call ? servePlacement(f.call.x) : "—"}</td>
              <td>{trustedSpeed(f) !== null ? `${fmtNum(f.hit_speed_kmh, 0)} km/sa` : "—"}</td>
              <td>{f.board_speed_kmh ? `${fmtNum(f.board_speed_kmh, 0)} km/sa` : "—"}</td>
              <td>{err !== null ? `${err > 0 ? "+" : ""}%${fmtNum(err, 1)}` : "—"}</td>
              <td>
                {f.call ? (
                  <span className="cell-verdict">
                    <span className={`glyph--${verdictKind(f.call.verdict)}`}>
                      <StatusGlyph kind={verdictKind(f.call.verdict)} size={12} />
                    </span>
                    {verdictLabel(f.call.verdict)}
                  </span>
                ) : (
                  "Karar yok"
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function LabelCard({ s }: { s: LabelSummary }) {
  if (!("error_cm_median" in s)) {
    return (
      <p className="card-empty">
        Henüz etiket yok. Bir vuruş seçip denetçideki “Sekme Noktasını İşaretle” ile topun yere değdiği noktaya tıklayın; hata, %95
        elipsin tutarlılığı ve karar uyumu burada birikir. Etiket olmadan sistemin gerçek görüntüdeki doğruluğu bilinmez.
      </p>
    );
  }
  return (
    <div className="check-grid">
      <div className="group">
        <div className="row">
          <span className="row-label">Etiketli vuruş</span>
          <span className="row-value num">{s.n}</span>
        </div>
        <div className="row">
          <span className="row-label">Sekme hatası (ortanca · %90)</span>
          <span className="row-value num">
            <strong>{fmtNum(s.error_cm_median, 0)} cm</strong> · {fmtNum(s.error_cm_p90, 0)} cm
          </span>
        </div>
        <div className="row">
          <span className="row-label">%95 elipsin içinde</span>
          <span className="row-value num">
            {fmtPct(s.inside_95_share, 0)} <span className="muted">(hedef ≈ %95)</span>
          </span>
        </div>
      </div>
      <div className="group">
        <div className="row">
          <span className="row-label">Kesin kararlarda uyum</span>
          <span className="row-value num">
            <strong>
              {s.agreements}/{s.decisive}
            </strong>
          </span>
        </div>
        <div className="row">
          <span className="row-label">İnsana giden kararlar</span>
          <span className="row-value num">
            {s.reviews} · etikete göre {s.review_truth.IN} içeride, {s.review_truth.OUT} dışarıda
          </span>
        </div>
      </div>
      <p className="card-note">
        Her vuruş için son etiket sayılır. {s.n < 30 ? "Örneklem küçük; bu değerler henüz bir doğruluk oranı değildir." : ""}
      </p>
    </div>
  );
}

function BroadcastCheck({ report }: { report: Report }) {
  const b = report.broadcast;
  if (!b || !b.ocr) {
    const why = b && "reason" in b ? (b.reason === "disabled" ? " (kapalı)" : " (Apple Vision OCR kullanılamıyor)") : "";
    return <p className="card-empty">Yayın okunmadı{why}.</p>;
  }
  const ss = b.serve_speed;
  const sides = b.server_sides;
  const fps = report.video.fps;
  const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  return (
    <div className="check-grid">
      <div className="group">
        <div className="row">
          <span className="row-label">Okunan kare</span>
          <span className="row-value num">
            {b.frames_read} (her {fmtNum(b.step_frames / fps, 1)} sn) · {b.score_readings} tabela
          </span>
        </div>
        <div className="row">
          <span className="row-label">Tabeladaki oyuncular</span>
          <span className="row-value">{b.names ? b.names.map(displayName).join(" – ") : "okunamadı"}</span>
        </div>
        <div className="row">
          <span className="row-label">Servis hızı, radara göre</span>
          <span className="row-value num">
            {ss.n > 0 && ss.mean_error_pct !== null ? (
              <>
                <strong>
                  {ss.mean_error_pct > 0 ? "+" : ""}%{fmtNum(ss.mean_error_pct, 1)}
                </strong>{" "}
                ortalama · {ss.n} servis
              </>
            ) : (
              "eşleşen servis yok"
            )}
          </span>
        </div>
        <div className="row">
          <span className="row-label">Servis tarafı tutarlılığı</span>
          <span className="row-value">
            {sides.consistent === null ? "kontrol edilemedi" : sides.consistent ? `tutarlı (${sides.serves.length} servis)` : "tutarsız"}
          </span>
        </div>
        {sides.start_end && (
          <div className="row">
            <span className="row-label">Maç başındaki taraflar</span>
            <span className="row-value">
              {Object.entries(sides.start_end)
                .map(([n, s]) => `${displayName(n)}: ${s === "near" ? "yakın" : "uzak"}`)
                .join(" · ")}
            </span>
          </div>
        )}
      </div>
      <div className="group">
        {ss.unmatched_updates.length === 0 ? (
          <div className="row-stack">
            <p className="row-sub">Her radar güncellemesi tespit edilen bir servisle eşleşti.</p>
          </div>
        ) : (
          ss.unmatched_updates.map((u, i) => (
            <div className="row-stack" key={i}>
              <div className="row-line">
                <span className="num">
                  {clock(u.t[0])}–{clock(u.t[1])} · pano {u.board} {b.speed_unit ?? ""}
                </span>
                <span className={u.reason === "missed" ? "row-sub text-out" : "row-sub"}>
                  {u.reason === "missed" ? "kaçırılmış servis" : u.reason === "ambiguous" ? "belirsiz" : "klip kesintisi"}
                </span>
              </div>
              <p className="row-sub">
                {u.reason === "missed"
                  ? "Arama penceresinin tamamı ölçülebilir görüntüde ama sistem servis bulamadı: tespit hatası."
                  : u.reason === "ambiguous"
                    ? "Pencerede birden çok servis var; hangisine ait olduğu söylenemiyor."
                    : "Servis, klibin kesilen ya da kalibre edilemeyen kısmında kalıyor; sistemin görebileceği bir yerde değil."}
              </p>
            </div>
          ))
        )}
      </div>
      <p className="card-note">
        Pano değeri servisten birkaç saniye sonra güncellenir; her güncelleme ondan 1–6,5 sn önceki tek servisle eşleştirilir. Sayı
        kazananı yalnızca tabela tek sayı arayla okunduğunda çıkarılır
        {b.score_changes.some((c) => c.winner) ? "." : "; bu klipte tabela sayılar arasında atladığı için çıkarılamadı."}
      </p>
    </div>
  );
}

const Analysis = memo(AnalysisImpl);
export default Analysis;
