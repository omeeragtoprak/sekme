import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

type Props = {
  /** Toolbar button content. */
  label: ReactNode;
  title: string;
  align?: "start" | "end";
  disabled?: boolean;
  className?: string;
  children: (close: () => void) => ReactNode;
};

/**
 * A pull-down menu anchored to a toolbar button (macOS NSMenu look). It is never modal: a click
 * outside, Escape or choosing an item closes it, and focus returns to the button.
 */
export default function Menu({ label, title, align = "start", disabled, className = "tb-btn", children }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setOpen(false);
        btnRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    // Move focus into the menu so arrow/tab navigation starts there.
    rootRef.current?.querySelector<HTMLElement>(".menu button:not(:disabled)")?.focus({ preventScroll: true });
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const close = () => {
    setOpen(false);
    btnRef.current?.focus();
  };

  const onMenuKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = Array.from(e.currentTarget.querySelectorAll<HTMLElement>("button:not(:disabled)"));
    if (items.length === 0) return;
    const i = items.indexOf(document.activeElement as HTMLElement);
    const next = e.key === "ArrowDown" ? (i + 1) % items.length : (i - 1 + items.length) % items.length;
    items[next]?.focus();
  };

  return (
    <div className="menu-anchor" ref={rootRef}>
      <button
        ref={btnRef}
        type="button"
        className={className}
        title={title}
        aria-label={title}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        {label}
      </button>
      {open && (
        <div id={id} className={`menu menu--${align}`} role="menu" onKeyDown={onMenuKey}>
          {children(close)}
        </div>
      )}
    </div>
  );
}
