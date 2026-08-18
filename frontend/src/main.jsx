import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import AuthScreen from "./auth/AuthScreen.jsx";
import { AuthProvider, useAuth } from "./auth/AuthContext.jsx";
import "./index.css";

/**
 * The gate lives here rather than inside App so App keeps its single job:
 * running one transcription session. It never has to ask whether there is a
 * user, because it does not mount until there is one.
 */
function Gate() {
  const { user, loading } = useAuth();

  if (loading) {
    // Shown only while /auth/me resolves an existing token. Rendering the
    // sign-in form first would flash a login screen at someone who is
    // already signed in.
    return (
      <div className="flex h-full items-center justify-center bg-neutral-50">
        <span className="text-sm text-neutral-400">Loading…</span>
      </div>
    );
  }

  return user ? <App /> : <AuthScreen />;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <Gate />
    </AuthProvider>
  </React.StrictMode>,
);
