import { memo } from "react";
import { DASH, fmtDate, fmtNum, fmtPct } from "../geometry";
import { playerName, SIDE_LABEL, timingLabel } from "../i18n";
import type { ModelCard, Report, Side } from "../types";

const SIDES: Side[] = ["near", "far"];
const MODELS: { key: keyof Report["models"]; label: string }[] = [
  { key: "court", label: "Kort anahtar noktaları" },
  { key: "ball", label: "Top dedektörü" },
  { key: "players", label: "Oyuncu dedektörü" },
  { key: "pose", label: "Poz (vuruş türü)" },
];

/** Licences that restrict use are flagged (not judged) so reviewers notice them. */
function licence(l: string): { text: string; warn: boolean } {
  const s = l.toLowerCase();
  const text = s.includes("none published") ? "Yayımlanmış lisans yok (yalnızca araştırma)" : l;
  const warn = ["none", "research", "unknown", "agpl", "non-commercial"].some((k) => s.includes(k));
  return { text, warn };
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value num" title={title}>
        {value}
      </span>
    </div>
  );
}

function ModelRow({ label, card }: { label: string; card: ModelCard | undefined }) {
  if (!card) return <Row label={label} value="Bildirilmedi" />;
  const lic = licence(card.license);
  return (
    <div className="row-stack">
      <div className="row-line">
        <span>{label}</span>
        <span className={lic.warn ? "row-sub text-warn" : "row-sub"} title="Sunucunun bildirdiği lisans">
          {lic.text}
        </span>
      </div>
      <p className="row-sub">{card.source}</p>
      {card.trained_on && (
        <p className="row-sub row-sub--tertiary" title={card.selected_on ? `Tur seçimi: ${card.selected_on}` : undefined}>
          Eğitim: {card.trained_on}
          {card.epoch != null && ` · ${card.epoch}. tur`}
        </p>
      )}
      <p className="row-sub row-sub--tertiary num" title={`${card.file}\nsha256 ${card.sha256}`}>
        {card.file} · sha256 {card.sha256.slice(0, 12)}…
      </p>
    </div>
  );
}

function SessionInfoImpl({ report }: { report: Report }) {
  const cal = report.calibration;
  const ball = report.ball;
  const ballTotal = ball.detected_frames + ball.interpolated_frames + ball.missing_frames;
  const timings = Object.entries(report.timings_s ?? {});
  const totalTime = timings.reduce((a, [, v]) => a + (Number.isFinite(v) ? v : 0), 0);
  const cc = cal.camera_center_m;

  return (
    <>
      <section className="block">
        <h3 className="group-title">Ölçüm kalitesi</h3>
        <div className="group">
          <Row
            label="Kalibre kareler"
            value={`${cal.valid_frames}/${cal.total_frames} · ${fmtPct(cal.total_frames > 0 ? cal.valid_frames / cal.total_frames : null)}`}
          />
          <Row label="Çizgi uyumu (ortanca)" value={cal.line_rmse_px_median !== null ? `${fmtNum(cal.line_rmse_px_median, 2)} px` : DASH} />
          <Row label="Çizgi desteği (ortanca)" value={fmtPct(cal.line_support_median)} />
          <Row
            label="Top tespiti"
            value={fmtPct(ballTotal > 0 ? ball.detected_frames / ballTotal : null)}
            title={`${ball.detected_frames} tespit · ${ball.interpolated_frames} ara değer · ${ball.missing_frames} kayıp`}
          />
        </div>
        <p className="group-foot">
          Kalibre olmayan karelerde (yakın çekim, tribün) ölçüm yapılmaz. Top {ball.detected_frames} karede tespit edildi,{" "}
          {ball.interpolated_frames} karede ara değerle dolduruldu, {ball.missing_frames} karede yok.
        </p>
      </section>

      {SIDES.map((side) => {
        const pl = report.players[side];
        return (
          <section className="block" key={side}>
            <h3 className="group-title">
              {playerName(pl.label)} · {SIDE_LABEL[side]}
            </h3>
            <div className="group">
              <Row label="Koşulan mesafe" value={`${fmtNum(pl.distance_m, 1)} m`} />
              <Row label="En yüksek hız" value={`${fmtNum(pl.peak_kmh, 1)} km/sa`} />
              <Row label="Ortalama hız" value={`${fmtNum(pl.mean_kmh, 1)} km/sa`} />
              <Row label="Takip edilen kareler" value={fmtPct(pl.coverage, 0)} />
            </div>
          </section>
        );
      })}

      <section className="block">
        <h3 className="group-title">Kamera</h3>
        <div className="group">
          <Row label="Odak uzaklığı" value={cal.focal_px_median !== null ? `${fmtNum(cal.focal_px_median, 0)} px` : DASH} />
          <Row
            label="Konum (x; y; z)"
            value={cc ? `${fmtNum(cc[0], 1)}; ${fmtNum(cc[1], 1)}; ${fmtNum(cc[2], 1)} m` : DASH}
            title="Kortun ortasına göre metre; y uzak dip çizgiye doğru, z yukarı"
          />
          <Row label="Anahtar kare" value={String(cal.keyframes)} />
          <Row label="Kamera geçişi" value={String(cal.shot_cuts.length)} />
        </div>
      </section>

      <section className="block">
        <h3 className="group-title">Video</h3>
        <div className="group">
          <Row label="Çözünürlük" value={`${report.video.width} × ${report.video.height}`} />
          <Row label="Kare hızı" value={`${fmtNum(report.video.fps, 2)} fps`} />
          <Row label="Süre" value={`${fmtNum(report.video.duration_s, 1)} sn · ${report.video.n_frames} kare`} />
          <div className="row-stack">
            <span>Dosya</span>
            <p className="row-sub" title={`sha256 ${report.video.sha256}`}>
              {report.video.file}
            </p>
          </div>
        </div>
      </section>

      <section className="block">
        <h3 className="group-title">Modeller</h3>
        <div className="group">
          {MODELS.map((m) => (
            <ModelRow key={m.key} label={m.label} card={report.models?.[m.key]} />
          ))}
        </div>
      </section>

      <section className="block">
        <h3 className="group-title">İşlem süreleri</h3>
        <div className="group">
          {timings.length === 0 ? (
            <Row label="Süre bilgisi" value="Bildirilmedi" />
          ) : (
            <>
              {timings.map(([k, v]) => (
                <Row
                  key={k}
                  label={timingLabel(k) + ((k === "perception_s" || k === "broadcast_s") && v < 1 ? " (önbellekten)" : "")}
                  value={`${fmtNum(v, 1)} sn`}
                />
              ))}
              <Row label="Toplam" value={`${fmtNum(totalTime, 1)} sn`} />
            </>
          )}
        </div>
      </section>

      <section className="block">
        <h3 className="group-title">Bu sistemin henüz yapamadıkları</h3>
        <div className="group">
          {report.limitations.length === 0 ? (
            <Row label="Sınırlama" value="Raporda listelenmemiş" />
          ) : (
            report.limitations.map((l, i) => (
              <div key={i} className="row-stack">
                <p>{l}</p>
              </div>
            ))
          )}
        </div>
        <p className="group-foot num" title={`${report.schema} · ${report.created}`}>
          Rapor sürümü {report.version} · {fmtDate(report.created)}
        </p>
      </section>
    </>
  );
}

const SessionInfo = memo(SessionInfoImpl);
export default SessionInfo;
