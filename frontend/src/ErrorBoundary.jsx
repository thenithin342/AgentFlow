import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            background: "var(--af-bg-app)",
            height: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "var(--af-font-sans)",
            color: "var(--af-text-primary)",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <h2 style={{ color: "var(--af-synthesizer)", marginBottom: 16 }}>
              Something went wrong.
            </h2>
            <button
              onClick={() => window.location.reload()}
              style={{
                background: "var(--af-bg-panel)",
                border: "1px solid var(--af-border)",
                color: "var(--af-text-primary)",
                padding: "8px 16px",
                borderRadius: 4,
                cursor: "pointer",
              }}
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
