// Small line icons drawn in the spirit of SF Symbols (SF Symbols themselves are licensed for Apple
// platforms only, so they are not shipped on the web). 20x20 grid, round caps, currentColor.
import type { ReactNode, SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 18, children, ...rest }: P & { children: ReactNode }) {
  return (
    <svg viewBox="0 0 20 20" width={size} height={size} aria-hidden="true" focusable="false" {...rest}>
      {children}
    </svg>
  );
}

const line = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const SidebarLeftIcon = (p: P) => (
  <Svg {...p}>
    <rect x="2.75" y="3.75" width="14.5" height="12.5" rx="3" {...line} />
    <path d="M7.75 3.75v12.5M4.75 7h1M4.75 9.5h1" {...line} />
  </Svg>
);

export const SidebarRightIcon = (p: P) => (
  <Svg {...p}>
    <rect x="2.75" y="3.75" width="14.5" height="12.5" rx="3" {...line} />
    <path d="M12.25 3.75v12.5M14.25 7h1M14.25 9.5h1" {...line} />
  </Svg>
);

export const PlayIcon = (p: P) => (
  <Svg {...p}>
    <path d="M6.2 4.1c0-.8.9-1.3 1.6-.9l8.3 5.2c.6.4.6 1.3 0 1.7l-8.3 5.2c-.7.4-1.6 0-1.6-.9z" fill="currentColor" />
  </Svg>
);

export const PauseIcon = (p: P) => (
  <Svg {...p}>
    <rect x="5" y="3.5" width="3.4" height="13" rx="1.1" fill="currentColor" />
    <rect x="11.6" y="3.5" width="3.4" height="13" rx="1.1" fill="currentColor" />
  </Svg>
);

export const FrameBackIcon = (p: P) => (
  <Svg {...p}>
    <rect x="3.5" y="4.5" width="2" height="11" rx="1" fill="currentColor" />
    <path d="M16 5.3v9.4c0 .7-.8 1.1-1.4.7l-6.5-4.7a.85.85 0 0 1 0-1.4l6.5-4.7c.6-.4 1.4 0 1.4.7z" fill="currentColor" />
  </Svg>
);

export const FrameForwardIcon = (p: P) => (
  <Svg {...p}>
    <rect x="14.5" y="4.5" width="2" height="11" rx="1" fill="currentColor" />
    <path d="M4 5.3v9.4c0 .7.8 1.1 1.4.7l6.5-4.7a.85.85 0 0 0 0-1.4L5.4 4.6C4.8 4.2 4 4.6 4 5.3z" fill="currentColor" />
  </Svg>
);

export const PlusIcon = (p: P) => (
  <Svg {...p}>
    <path d="M10 4.5v11M4.5 10h11" {...line} strokeWidth={1.7} />
  </Svg>
);

export const ChevronUpDownIcon = (p: P) => (
  <Svg {...p}>
    <path d="M7 8l3-3 3 3M7 12l3 3 3-3" {...line} />
  </Svg>
);

export const ChevronDownIcon = (p: P) => (
  <Svg {...p}>
    <path d="M6 8l4 4 4-4" {...line} />
  </Svg>
);

export const LayersIcon = (p: P) => (
  <Svg {...p}>
    <path d="M10 3.5l6.5 3.3L10 10.1 3.5 6.8z" {...line} />
    <path d="M3.5 10l6.5 3.3 6.5-3.3M3.5 13.2l6.5 3.3 6.5-3.3" {...line} />
  </Svg>
);

export const CheckIcon = (p: P) => (
  <Svg {...p}>
    <path d="M4.5 10.5l3.5 3.5 7.5-8" {...line} strokeWidth={1.9} />
  </Svg>
);

export const RefreshIcon = (p: P) => (
  <Svg {...p}>
    <path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9M15.5 3.8v3.4h-3.4" {...line} />
  </Svg>
);

export const VideoIcon = (p: P) => (
  <Svg {...p}>
    <rect x="2.75" y="5.25" width="10.5" height="9.5" rx="2.5" {...line} />
    <path d="M13.25 8.6l3.1-2c.4-.2.9 0 .9.5v5.8c0 .5-.5.7-.9.5l-3.1-2" {...line} />
  </Svg>
);

export const CourtIcon = (p: P) => (
  <Svg {...p}>
    <rect x="5" y="2.75" width="10" height="14.5" rx="1.5" {...line} />
    <path d="M5 10h10M7 6.2h6M7 13.8h6M10 6.2v7.6" {...line} strokeWidth={1.2} />
  </Svg>
);

export const InfoIcon = (p: P) => (
  <Svg {...p}>
    <circle cx="10" cy="10" r="7.25" {...line} />
    <path d="M10 9v4.6" {...line} strokeWidth={1.7} />
    <circle cx="10" cy="6.4" r="1" fill="currentColor" />
  </Svg>
);

/** Filled status symbol: a coloured disc with a white glyph (checkmark / xmark / exclamation / minus). */
export function StatusGlyph({ kind, size = 16 }: { kind: "in" | "out" | "review" | "none"; size?: number }) {
  const glyph =
    kind === "in" ? (
      <path d="M6.3 10.3l2.5 2.5 4.9-5.4" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    ) : kind === "out" ? (
      <path d="M7.2 7.2l5.6 5.6M12.8 7.2l-5.6 5.6" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round" />
    ) : kind === "review" ? (
      <>
        <path d="M10 5.9v5" fill="none" stroke="#fff" strokeWidth={2.1} strokeLinecap="round" />
        <circle cx="10" cy="14" r="1.25" fill="#fff" />
      </>
    ) : (
      <path d="M6.6 10h6.8" fill="none" stroke="#fff" strokeWidth={2.1} strokeLinecap="round" />
    );
  return (
    <svg viewBox="0 0 20 20" width={size} height={size} aria-hidden="true" focusable="false" className={`glyph glyph--${kind}`}>
      <circle cx="10" cy="10" r="9" fill="currentColor" />
      {glyph}
    </svg>
  );
}

/** The macOS activity indicator: eight spokes fading around the clock. */
export function Spinner({ size = 18 }: { size?: number }) {
  return (
    <svg viewBox="0 0 20 20" width={size} height={size} className="spinner" role="presentation" aria-hidden="true">
      {Array.from({ length: 8 }, (_, i) => (
        <rect
          key={i}
          x="9.1"
          y="2"
          width="1.8"
          height="4.6"
          rx="0.9"
          fill="currentColor"
          opacity={0.18 + (0.82 * i) / 7}
          transform={`rotate(${i * 45} 10 10)`}
        />
      ))}
    </svg>
  );
}
