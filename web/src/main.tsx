import { Component, StrictMode, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

/** Last-resort boundary: a rendering bug must never leave a blank screen. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("İnceleme Odası çöktü:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="page-state" style={{ height: "100%" }}>
          <div className="empty" role="alert">
            <p className="empty-title">Beklenmeyen bir hata oluştu</p>
            <p className="empty-text">{this.state.error.message}</p>
            <button type="button" className="btn" onClick={() => window.location.reload()}>
              Yeniden Yükle
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>,
  );
}
