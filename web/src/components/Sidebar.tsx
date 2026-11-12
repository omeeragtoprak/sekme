import { memo, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { bounceFrameRaw, fmtProb, verdictKind } from "../geometry";
import { displayName, halfLabel, playerName, strokeLabel, verdictLabel } from "../i18n";
import { StatusGlyph } from "../icons";
import type { Async, Flight, Health, Report, Review } from "../types";

type Props = {
  report: Report;
  reviews: Async<Review[]>;
  selectedFlightId: number | null;
  onSelect: (id: number) => void;
  health: Async<Health>;
};

type Filter = "all" | "review" | "out" | "in" | "none";

const TILES: { key: Exclude<Filter, "all">; label: string; title: string; color: string; test: (f: Flight) => boolean }[] = [
  { key: "review", label: "İnceleme", title: "İncelenecek vuruşlar", color: "var(--review)", test: (f) => f.call?.verdict === "REVIEW" },
  { key: "out", label: "Dışarıda", title: "Dışarıda kararları", color: "var(--out)", test: (f) => f.call?.verdict === "OUT" },
  { key: "in", label: "İçeride", title: "İçeride kararları", color: "var(--in)", test: (f) => f.call?.verdict === "IN" },
  { key: "none", label: "Karar yok", title: "Kararsız vuruşlar", color: "var(--gray)", test: (f) => !f.call },
];

const ALL = { title: "Tüm vuruşlar", test: (_f: Flight) => true };

function shortClock(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s - m * 60)).padStart(2, "0")}`;
}

function SidebarImpl({ report, reviews, selectedFlightId, onSelect, health }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const listRef = useRef<HTMLUListElement>(null);
  const fps = report.video.fps > 0 ? report.video.fps : 30;
  const flights = useMemo(() => [...report.flights].sort((a, b) => a.id - b.id), [report.flights]);
  const active = filter === "all" ? ALL : (TILES.find((t) => t.key === filter) ?? ALL);
  const visible = flights.filter(active.test);

  const reviewed = useMemo(() => {
    const s = new Set<number>();
    if (reviews.status === "ready") for (const r of reviews.data) s.add(r.flight_id);
    return s;
  }, [reviews]);

  // A shot selected elsewhere (timeline, court) must never be hidden by the filter.
  useEffect(() => {
    if (selectedFlightId === null) return;
    const f = flights.find((x) => x.id === selectedFlightId);
    if (f && !active.test(f)) setFilter("all");
  }, [selectedFlightId, flights, active]);

  useEffect(() => {
    if (selectedFlightId === null) return;
    document.getElementById(`shot-${selectedFlightId}`)?.scrollIntoView({ block: "nearest" });
  }, [selectedFlightId, filter]);

  const onKey = (e: KeyboardEvent<HTMLUListElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    if (visible.length === 0) return;
    const i = visible.findIndex((f) => f.id === selectedFlightId);
    let next: number;
    if (e.key === "Home") next = 0;
    else if (e.key === "End") next = visible.length - 1;
    else if (i < 0) next = 0;
    else next = Math.max(0, Math.min(visible.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
    onSelect(visible[next].id);
  };

  const ok = health.status === "ready" && health.data.status === "ok";

  return (
    <aside className="sidebar" aria-label="Vuruşlar">
      <div className="sidebar-scroll">
        <div className="tiles" role="group" aria-label="Karar filtresi">
          {TILES.map((t) => {
            const n = flights.filter(t.test).length;
            return (
              <button
                key={t.key}
                type="button"
                className="tile"
                style={{ "--tile-color": t.color } as CSSProperties}
                aria-pressed={filter === t.key}
                disabled={n === 0 && filter !== t.key}
                onClick={() => setFilter(filter === t.key ? "all" : t.key)}
                title={filter === t.key ? "Filtreyi kaldır" : `Yalnızca ${t.label.toLocaleLowerCase("tr")} vuruşları göster`}
              >
                <span className="tile-icon">
                  <StatusGlyph kind={t.key} size={24} />
                </span>
                <span className="tile-count">{n}</span>
                <span className="tile-label">{t.label}</span>
              </button>
            );
          })}
        </div>

        <div className="list-head">
          <span className="list-title">{active.title}</span>
          {filter === "all" ? (
            <span className="list-meta num">{flights.length}</span>
          ) : (
            <button type="button" className="link list-meta" onClick={() => setFilter("all")}>
              Tümünü göster
            </button>
          )}
        </div>

        {flights.length === 0 ? (
          <div className="empty">
            <p className="empty-title">Vuruş bulunamadı</p>
            <p className="empty-text">Analiz bu klipte top uçuşu bulmadı; karar verilecek bir şey yok.</p>
          </div>
        ) : (
          <ul
            ref={listRef}
            className="shots"
            role="listbox"
            tabIndex={0}
            aria-label={active.title}
            aria-activedescendant={selectedFlightId !== null ? `shot-${selectedFlightId}` : undefined}
            onKeyDown={onKey}
          >
            {visible.map((f) => {
              const c = f.call;
              const at = (bounceFrameRaw(f) ?? f.start_frame) / fps;
              const route = f.hitter === "near" ? "yakından uzağa" : "uzaktan yakına";
              const who = `${f.hitter_name ? displayName(f.hitter_name) : playerName(report.players[f.hitter].label)} → ${halfLabel(f.receiver)}`;
              const sub = c ? `${verdictLabel(c.verdict)} · ${fmtProb(c.p_in)} · ${route}` : `Karar yok · ${route}`;
              return (
                <li
                  key={f.id}
                  id={`shot-${f.id}`}
                  role="option"
                  aria-selected={f.id === selectedFlightId}
                  className="shot"
                  title={who}
                  onClick={() => {
                    listRef.current?.focus({ preventScroll: true });
                    onSelect(f.id);
                  }}
                >
                  <span className="shot-glyph">
                    <StatusGlyph kind={verdictKind(c?.verdict)} size={16} />
                  </span>
                  <span className="shot-title">
                    Vuruş {f.id + 1}
                    {f.hitter_name && <span className="shot-tag">{displayName(f.hitter_name)}</span>}
                    {(f.kind === "serve" || f.stroke) && (
                      <span className="shot-tag">{f.kind === "serve" ? "Servis" : strokeLabel(f.stroke, f.stroke_wing)}</span>
                    )}
                  </span>
                  <span className="shot-time">{shortClock(at)}</span>
                  <span className="shot-sub">
                    {sub}
                    {reviewed.has(f.id) ? " · incelendi" : ""}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="sidebar-foot" role="status">
        <span className={`status-dot ${ok ? "status-dot--ok" : health.status === "error" ? "status-dot--bad" : ""}`} />
        {health.status === "ready"
          ? `Sunucuya bağlı · sürüm ${health.data.version}`
          : health.status === "error"
            ? "Sunucuya ulaşılamıyor"
            : "Bağlanılıyor…"}
      </div>
    </aside>
  );
}

const Sidebar = memo(SidebarImpl);
export default Sidebar;
