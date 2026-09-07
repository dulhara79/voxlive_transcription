import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

/**
 * AuthContext — who is signed in, and the token every WebSocket needs.
 *
 * WHERE THE TOKEN LIVES
 * ---------------------
 * localStorage. That is readable by any script running on this origin, so an
 * XSS bug becomes a stolen session. The alternative — an httpOnly cookie —
 * is not available here for a real reason: the browser WebSocket API cannot
 * set headers, so the backend reads the credential from `?token=`, which
 * means JavaScript has to be able to read it too.
 *
 * What that buys, and what it costs, is worth being explicit about:
 *   + a reload or a reopened tab keeps you signed in
 *   - the token is only as safe as the app's dependency tree
 *
 * The mitigations are the token's short TTL (AUTH_ACCESS_TTL_SEC, 12 h by
 * default) and the fact that it grants nothing beyond one organization's
 * transcription sessions.
 *
 * `getStoredToken` is exported separately because `useAudioStream` needs the
 * token when it builds the socket URL, and that happens inside a callback
 * where reading it from React state would capture a stale value.
 */

const TOKEN_KEY = "voxlive.access_token";

const API_BASE = (
  import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export function getStoredToken() {
  try {
    return window.localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    // Private browsing modes throw on storage access rather than returning
    // null. Signing in still works; it just will not survive a reload.
    return "";
  }
}

function storeToken(token) {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* see above */
  }
}

/** POST/GET against the backend, surfacing FastAPI's `detail` as the message. */
async function api(path, { method = "GET", body, token } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    // fetch only rejects on a network-level failure, so this is genuinely
    // "the backend is not reachable" rather than any 4xx or 5xx.
    throw new Error(
      `Can't reach the server at ${API_BASE}. Check that the backend is running.`,
    );
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : // 422 from Pydantic arrives as a list of field errors; show the
          // first one rather than "[object Object]".
          detail?.[0]?.msg || `Request failed (${response.status}).`,
    );
  }
  return payload;
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(getStoredToken);
  // `loading` covers only the initial /auth/me call. Rendering the sign-in
  // screen before it resolves would flash a login form at someone who is
  // already signed in.
  const [loading, setLoading] = useState(Boolean(getStoredToken()));

  const applySession = useCallback((session) => {
    storeToken(session.access_token);
    setToken(session.access_token);
    setUser({
      id: session.user_id,
      email: session.email,
      name: session.name,
      role: session.role,
      organizationId: session.organization_id,
      organizationName: session.organization_name,
      planCode: session.plan_code,
    });
    return session;
  }, []);

  const signOut = useCallback(() => {
    storeToken("");
    setToken("");
    setUser(null);
  }, []);

  // Rehydrate on load. This re-reads the user server-side rather than
  // decoding the token, so a role change or a disabled account takes effect
  // on the next page load instead of at token expiry.
  useEffect(() => {
    const stored = getStoredToken();
    if (!stored) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    api("/auth/me", { token: stored })
      .then((session) => {
        if (!cancelled) applySession(session);
      })
      .catch(() => {
        // Expired, revoked, or the backend is down. Either way there is no
        // usable session, and keeping a dead token only produces a confusing
        // 1008 close on the first recording attempt.
        if (!cancelled) signOut();
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [applySession, signOut]);

  const signIn = useCallback(
    async ({ email, password, organizationId }) => {
      const result = await api("/auth/login", {
        method: "POST",
        body: {
          email,
          password,
          ...(organizationId ? { organization_id: organizationId } : {}),
        },
      });
      // The same email can exist in several organizations, so a successful
      // password check does not always identify one account. The caller
      // renders a chooser and calls chooseOrganization next.
      if (result.needs_organization) return result;
      return applySession(result);
    },
    [applySession],
  );

  const chooseOrganization = useCallback(
    async ({ orgSelectToken, organizationId }) =>
      applySession(
        await api("/auth/login/organization", {
          method: "POST",
          body: {
            org_select_token: orgSelectToken,
            organization_id: organizationId,
          },
        }),
      ),
    [applySession],
  );

  const signUp = useCallback(
    async ({ organizationName, name, email, password }) =>
      applySession(
        await api("/auth/signup", {
          method: "POST",
          body: {
            organization_name: organizationName,
            name,
            email,
            password,
          },
        }),
      ),
    [applySession],
  );

  const value = useMemo(
    () => ({
      user,
      token,
      loading,
      signIn,
      signUp,
      signOut,
      chooseOrganization,
    }),
    [user, token, loading, signIn, signUp, signOut, chooseOrganization],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
