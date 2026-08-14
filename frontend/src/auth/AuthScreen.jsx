import { useCallback, useState } from "react";
import { useAuth } from "./AuthContext.jsx";

/**
 * AuthScreen — sign in, or create an organization.
 *
 * THREE MODES, NOT THREE PAGES
 * ----------------------------
 *   signin   email + password
 *   signup   organization name + your name + email + password
 *   choose   which organization — reached only when one email and password
 *            matched accounts in more than one of them
 *
 * `choose` exists because the database makes email unique PER ORGANIZATION.
 * It is rare, so it is a step that appears when needed rather than a field
 * everyone fills in. Asking for the organization up front would also be a
 * privacy problem: it would let anyone test whether a given person holds an
 * account at a given company.
 *
 * The visual language is the transcript view's — neutral greys, pill buttons,
 * one accent — because this is the same product, not a marketing page in
 * front of it.
 */

const MIN_PASSWORD_LENGTH = 10;

export default function AuthScreen() {
  const { signIn, signUp, chooseOrganization } = useAuth();

  const [mode, setMode] = useState("signin");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [organizationName, setOrganizationName] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Set when the backend needs an organization chosen.
  const [choice, setChoice] = useState(null);

  const switchMode = useCallback((next) => {
    setMode(next);
    setError("");
    setChoice(null);
    setPassword("");
  }, []);

  const run = useCallback(async (work) => {
    setBusy(true);
    setError("");
    try {
      return await work();
    } catch (err) {
      setError(err.message || "Something went wrong.");
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  const onSubmit = useCallback(async () => {
    if (mode === "signup") {
      if (password.length < MIN_PASSWORD_LENGTH) {
        setError(
          `Use at least ${MIN_PASSWORD_LENGTH} characters. Length matters more than symbols.`,
        );
        return;
      }
      await run(() => signUp({ organizationName, name, email, password }));
      return;
    }
    const result = await run(() => signIn({ email, password }));
    if (result?.needs_organization) {
      setChoice({
        token: result.org_select_token,
        organizations: result.organizations,
      });
    }
  }, [mode, run, signIn, signUp, organizationName, name, email, password]);

  const onChoose = useCallback(
    async (organizationId) => {
      const done = await run(() =>
        chooseOrganization({
          orgSelectToken: choice.token,
          organizationId,
        }),
      );
      // The chooser token is one-use in practice and short-lived. If it has
      // expired, send the person back to the password step rather than
      // leaving them on a list of buttons that will keep failing.
      if (!done) setChoice(null);
    },
    [choice, chooseOrganization, run],
  );

  // Enter should submit. There is no <form> element here on purpose: a form
  // in this app would trigger a full page navigation on submit in some
  // browsers, which discards the audio worklet and the socket.
  const onKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !busy) onSubmit();
    },
    [busy, onSubmit],
  );

  return (
    <div className="flex min-h-full items-center justify-center bg-neutral-50 px-6 py-12 text-neutral-900">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">VoxLive</h1>
          <p className="mt-1 text-xs text-neutral-500">
            Sinhala · English · Tamil
          </p>
        </div>

        <div className="rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm">
          {choice ? (
            <ChooseOrganization
              organizations={choice.organizations}
              busy={busy}
              error={error}
              onChoose={onChoose}
              onCancel={() => switchMode("signin")}
            />
          ) : (
            <>
              <h2 className="text-base font-medium">
                {mode === "signup" ? "Create an organization" : "Sign in"}
              </h2>
              <p className="mt-1 text-xs text-neutral-500">
                {mode === "signup"
                  ? "You'll be its owner and can add your team afterwards."
                  : "Use the account your organization gave you."}
              </p>

              <div className="mt-5 space-y-3">
                {mode === "signup" && (
                  <>
                    <Field
                      label="Organization name"
                      value={organizationName}
                      onChange={setOrganizationName}
                      onKeyDown={onKeyDown}
                      autoComplete="organization"
                      placeholder="Sri Lanka Telecom"
                    />
                    <Field
                      label="Your name"
                      value={name}
                      onChange={setName}
                      onKeyDown={onKeyDown}
                      autoComplete="name"
                      placeholder="Dulhara Kaushalya"
                    />
                  </>
                )}
                <Field
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  onKeyDown={onKeyDown}
                  autoComplete="email"
                  placeholder="you@example.com"
                />
                <Field
                  label="Password"
                  type="password"
                  value={password}
                  onChange={setPassword}
                  onKeyDown={onKeyDown}
                  autoComplete={
                    mode === "signup" ? "new-password" : "current-password"
                  }
                  hint={
                    mode === "signup"
                      ? `At least ${MIN_PASSWORD_LENGTH} characters`
                      : undefined
                  }
                />
              </div>

              {error && <ErrorNote>{error}</ErrorNote>}

              <button
                type="button"
                onClick={onSubmit}
                disabled={busy || !email || !password}
                className="mt-5 w-full rounded-full bg-neutral-900 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {busy
                  ? "Working…"
                  : mode === "signup"
                    ? "Create organization"
                    : "Sign in"}
              </button>

              <p className="mt-4 text-center text-xs text-neutral-500">
                {mode === "signup" ? (
                  <>
                    Already have an account?{" "}
                    <LinkButton onClick={() => switchMode("signin")}>
                      Sign in
                    </LinkButton>
                  </>
                ) : (
                  <>
                    Setting up a new team?{" "}
                    <LinkButton onClick={() => switchMode("signup")}>
                      Create an organization
                    </LinkButton>
                  </>
                )}
              </p>
            </>
          )}
        </div>

        <p className="mt-4 text-center text-xs text-neutral-400">
          Transcripts stay in your browser until you download them.
        </p>
      </div>
    </div>
  );
}

function ChooseOrganization({
  organizations,
  busy,
  error,
  onChoose,
  onCancel,
}) {
  return (
    <>
      <h2 className="text-base font-medium">Choose an organization</h2>
      <p className="mt-1 text-xs text-neutral-500">
        This email is registered with more than one.
      </p>

      <div className="mt-5 space-y-2">
        {organizations.map((org) => (
          <button
            key={org.organization_id}
            type="button"
            disabled={busy}
            onClick={() => onChoose(org.organization_id)}
            className="flex w-full items-center justify-between rounded-xl border border-neutral-200 px-4 py-3 text-left transition-colors hover:border-neutral-400 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <span className="text-sm font-medium">{org.organization_name}</span>
            <span className="text-xs uppercase tracking-wide text-neutral-400">
              {org.role}
            </span>
          </button>
        ))}
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}

      <button
        type="button"
        onClick={onCancel}
        className="mt-4 w-full text-center text-xs text-neutral-500 underline underline-offset-2 hover:text-neutral-800"
      >
        Use a different account
      </button>
    </>
  );
}

function Field({ label, hint, value, onChange, type = "text", ...rest }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-neutral-600">{label}</span>
      <input
        {...rest}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm outline-none transition-colors placeholder:text-neutral-400 focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900"
      />
      {hint && (
        <span className="mt-1 block text-xs text-neutral-400">{hint}</span>
      )}
    </label>
  );
}

function ErrorNote({ children }) {
  return (
    <div
      role="alert"
      className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800"
    >
      {children}
    </div>
  );
}

function LinkButton({ onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="font-medium text-neutral-900 underline underline-offset-2 hover:text-neutral-600"
    >
      {children}
    </button>
  );
}
