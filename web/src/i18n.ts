// Turkish wording for the Review Room. Data identifiers (verdict codes, line names, sides) stay
// English in the API; this module is the only place that turns them into text people read.
import type { Decision, MediaName, PlayerLabel, Side, Verdict } from "./types";

export const VERDICT_LABEL: Record<Verdict, string> = { IN: "İçeride", OUT: "Dışarıda", REVIEW: "İnceleme" };

export function verdictLabel(v: Verdict | null | undefined): string {
  return v ? VERDICT_LABEL[v] : "Karar yok";
}

export function playerName(label: PlayerLabel | string): string {
  const m = /^P(\d)$/.exec(label);
  return m ? `Oyuncu ${m[1]}` : label;
}

export const SIDE_LABEL: Record<Side, string> = { near: "yakın", far: "uzak" };

export function halfLabel(s: Side): string {
  return `${SIDE_LABEL[s]} yarı`;
}

const LINES: Record<string, string> = {
  "far baseline": "uzak dip çizgi",
  "near baseline": "yakın dip çizgi",
  "far service line": "uzak servis çizgisi",
  "near service line": "yakın servis çizgisi",
  "centre service line": "orta servis çizgisi",
  net: "file",
};

/** Turkish name of a court line identifier (mirrors courtvision/i18n.py); unknown names pass through. */
export function lineLabel(line: string): string {
  if (LINES[line]) return LINES[line];
  const m = /^(left|right) (singles|doubles) sideline$/.exec(line);
  if (m) return `${m[1] === "left" ? "sol" : "sağ"} ${m[2] === "singles" ? "tekler" : "çiftler"} yan çizgisi`;
  return line;
}

export const DECISIONS: { value: Decision; label: string; short: string }[] = [
  { value: "confirm", label: "Sistem kararını onayla", short: "Onaylandı" },
  { value: "overturn_in", label: "İçeride olarak değiştir", short: "İçeride’ye çevrildi" },
  { value: "overturn_out", label: "Dışarıda olarak değiştir", short: "Dışarıda’ya çevrildi" },
  { value: "unsure", label: "Emin değilim", short: "Emin değil" },
];

export function decisionShort(d: Decision): string {
  return DECISIONS.find((x) => x.value === d)?.short ?? d;
}

export const MEDIA_LABEL: Record<MediaName, string> = { source: "Kaynak", classic: "Klasik", pro: "Pro" };

const STAGES: Record<string, string> = {
  queued: "Sırada",
  perception: "Görüntü işleniyor",
  analysis: "Analiz ediliyor",
  render: "Videolar hazırlanıyor",
  done: "Tamamlandı",
};

export function stageLabel(stage: string): string {
  return STAGES[stage] ?? stage;
}

const TIMINGS: Record<string, string> = {
  perception_s: "Görüntü işleme",
  calibration_s: "Kalibrasyon",
  analysis_s: "Analiz",
  render_s: "Video hazırlama",
  broadcast_s: "Yayın okuma (OCR)",
};

export function timingLabel(key: string): string {
  return TIMINGS[key] ?? key.replace(/_s$/, "").replace(/_/g, " ");
}

/** "DJOKOVIC" -> "Djokovic" (Latin names use English casing so that "I" does not become "ı"). */
export function displayName(name: string): string {
  const turkish = /[İIŞĞÜÖÇ]/.test(name) && /[İŞĞ]/.test(name);
  const loc = turkish ? "tr" : "en";
  return name
    .toLocaleLowerCase(loc)
    .replace(/(^|[\s'-])(\p{L})/gu, (_m, sep: string, ch: string) => sep + ch.toLocaleUpperCase(loc));
}

/** Who hit a shot: the broadcast name when known, otherwise the side-based player label. */
export function hitterLabel(name: string | null | undefined, label: string): string {
  return name ? displayName(name) : playerName(label);
}

const STROKES: Record<string, string> = {
  serve: "Servis",
  forehand: "Forehand",
  backhand: "Backhand",
  volley: "Vole",
  overhead: "Smaç",
};

/** "Forehand", "Backhand vole", ... ; null when the stroke could not be determined. */
export function strokeLabel(kind: string | null | undefined, wing?: string | null): string | null {
  if (!kind) return null;
  if (kind === "volley" && wing) return `${STROKES[wing]} vole`;
  return STROKES[kind] ?? kind;
}

/** Serve direction in the service box from the bounce's distance to the centre line. */
export function servePlacement(x: number): string {
  const r = Math.abs(x) / 4.115;
  return r < 1 / 3 ? "T" : r < 2 / 3 ? "Gövde" : "Geniş";
}
