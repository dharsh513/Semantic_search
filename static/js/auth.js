/* =============================================================================
   Auth screen controller — tab switching, validation, strength meter, submit
   ========================================================================== */
(function () {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const CFG = window.AUTH_CONFIG || {};
  const el = {
    seg:        $(".seg"),
    tabLogin:   $("#tab-login"),
    tabSignup:  $("#tab-signup"),
    title:      $("#auth-title"),
    sub:        $("#auth-sub"),
    error:      $("#auth-error"),
    loginForm:  $("#login-form"),
    signupForm: $("#signup-form"),
    pwMeter:    $("#pw-meter"),
    pwLabel:    $("#pw-label"),
    themeBtn:   $("#theme-btn"),
    toast:      $("#toast"),
    rotator:    $("#rotator"),
    dots:       $("#dots"),
    motes:      $("#motes")
  };

  const COPY = {
    login:  { title: "Welcome back",
              sub: "Sign in to pick up your searches where you left them." },
    signup: { title: "Create your account",
              sub: "Your history, exports and paper conversations stay private to you." }
  };

  let mode = "login";
  let busy = false;

  /* ------------------------------------------------------------ helpers */
  function showError(message) {
    if (!message) { el.error.hidden = true; return; }
    el.error.textContent = message;
    el.error.hidden = false;
    // restart the shake animation
    el.error.style.animation = "none";
    void el.error.offsetWidth;
    el.error.style.animation = "";
  }

  function fieldError(form, name, message) {
    const slot = $(`[data-err="${name}"]`, form);
    if (!slot) return;
    const field = slot.closest(".field");
    slot.textContent = message || "";
    if (field) {
      field.classList.toggle("invalid", Boolean(message));
      if (message) field.classList.remove("valid");
    }
  }

  function clearErrors(form) {
    showError("");
    $$(".field", form).forEach(f => f.classList.remove("invalid"));
  }

  function markValid(input, ok) {
    const field = input.closest(".field");
    if (!field) return;
    field.classList.toggle("valid", ok);
    if (ok) field.classList.remove("invalid");
  }

  let toastTimer = null;
  function toast(message, isError) {
    el.toast.className = "toast" + (isError ? " error" : "");
    el.toast.textContent = message;
    el.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, 3600);
  }

  const EMAIL_RE = /^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$/;

  /* -------------------------------------------------------------- tabs */
  function setMode(next, focus) {
    if (!el.signupForm) return;             // signups disabled
    mode = next;
    const isSignup = next === "signup";

    el.seg.classList.toggle("on-signup", isSignup);
    el.tabLogin.classList.toggle("active", !isSignup);
    el.tabSignup.classList.toggle("active", isSignup);
    el.tabLogin.setAttribute("aria-selected", String(!isSignup));
    el.tabSignup.setAttribute("aria-selected", String(isSignup));

    el.loginForm.hidden = isSignup;
    el.signupForm.hidden = !isSignup;

    el.title.textContent = COPY[next].title;
    el.sub.textContent = COPY[next].sub;
    showError("");

    history.replaceState(null, "", isSignup ? "/signup" : "/login");

    if (focus) {
      const first = isSignup ? $("#signup-name") : $("#login-email");
      if (first) first.focus();
    }
  }

  if (el.tabLogin)  el.tabLogin.addEventListener("click", () => setMode("login", true));
  if (el.tabSignup) el.tabSignup.addEventListener("click", () => setMode("signup", true));

  /* ------------------------------------------------------ password eyes */
  $$(".pw-eye").forEach(btn =>
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.toggle);
      if (!input) return;
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
      btn.classList.toggle("on", !showing);
      btn.setAttribute("aria-label", showing ? "Show password" : "Hide password");
      input.focus();
    })
  );

  /* --------------------------------------------------- strength meter */
  // Mirrors rag/auth.py::password_strength so the browser and the server
  // agree on what counts as strong. The server still re-checks on submit.
  const COMMON = new Set([
    "password", "password1", "password123", "12345678", "123456789",
    "1234567890", "qwerty123", "qwertyuiop", "letmein1", "welcome1",
    "abc12345", "iloveyou", "admin123", "root1234", "changeme",
    "passw0rd", "p@ssword", "football", "baseball", "sunshine",
    "princess", "trustno1", "starwars", "monkey12", "dragon123",
    "pubmed123", "research", "12341234", "11111111", "00000000"
  ]);

  function strength(pw) {
    pw = pw || "";
    let score = 0;
    if (pw.length >= 8) score++;
    if (pw.length >= 12) score++;
    const classes = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/]
      .filter(re => re.test(pw)).length;
    if (classes >= 2) score++;
    if (classes >= 3 && pw.length >= 10) score++;
    if (COMMON.has(pw.toLowerCase())) score = 0;
    const labels = ["Very weak", "Weak", "Fair", "Good", "Strong"];
    return { score: score, label: labels[Math.min(score, 4)] };
  }

  function passwordProblem(pw, email, name) {
    pw = pw || "";
    if (pw.length < 8) return "Use at least 8 characters.";
    if (pw.length > 200) return "Passwords cannot exceed 200 characters.";
    if (COMMON.has(pw.toLowerCase()))
      return "That password is too common — pick something less guessable.";
    if (/^\d+$/.test(pw)) return "Use more than just numbers.";
    if (new Set(pw).size < 4) return "Use a wider mix of characters.";
    const local = (email || "").split("@")[0].toLowerCase();
    if (local.length > 2 && pw.toLowerCase().includes(local))
      return "Do not use your email address in your password.";
    const flat = (name || "").toLowerCase().replace(/\s+/g, "");
    if (flat.length > 2 && pw.toLowerCase().includes(flat))
      return "Do not use your name in your password.";
    return null;
  }

  const pwInput = $("#signup-password");
  if (pwInput) {
    pwInput.addEventListener("input", () => {
      const value = pwInput.value;
      el.pwMeter.hidden = value.length === 0;
      const s = strength(value);
      el.pwMeter.dataset.score = String(s.score);
      el.pwLabel.textContent = s.label;

      const problem = value ? passwordProblem(
        value, $("#signup-email").value, $("#signup-name").value) : null;
      fieldError(el.signupForm, "password", value ? problem : "");
      markValid(pwInput, Boolean(value) && !problem);

      const confirm = $("#signup-confirm");
      if (confirm.value) checkConfirm();
    });
  }

  function checkConfirm() {
    const pw = $("#signup-password").value;
    const confirm = $("#signup-confirm");
    const ok = confirm.value === pw && confirm.value.length > 0;
    fieldError(el.signupForm, "confirm",
               confirm.value && !ok ? "Passwords do not match." : "");
    markValid(confirm, ok);
    return ok;
  }

  const confirmInput = $("#signup-confirm");
  if (confirmInput) confirmInput.addEventListener("input", checkConfirm);

  /* ------------------------------------------------- live field checks */
  function wireEmail(input, form) {
    if (!input) return;
    input.addEventListener("blur", () => {
      const value = input.value.trim();
      if (!value) { fieldError(form, "email", ""); markValid(input, false); return; }
      const ok = EMAIL_RE.test(value);
      fieldError(form, "email", ok ? "" : "That does not look like a valid email address.");
      markValid(input, ok);
    });
    input.addEventListener("input", () => {
      if (input.closest(".field").classList.contains("invalid")) {
        fieldError(form, "email", "");
      }
    });
  }
  wireEmail($("#login-email"), el.loginForm);
  wireEmail($("#signup-email"), el.signupForm);

  const nameInput = $("#signup-name");
  if (nameInput) {
    nameInput.addEventListener("blur", () => {
      const value = nameInput.value.trim();
      if (!value) return;
      const ok = value.length >= 2;
      fieldError(el.signupForm, "name",
                 ok ? "" : "Enter your name (at least 2 characters).");
      markValid(nameInput, ok);
    });
  }

  /* -------------------------------------------------------------- submit */
  async function post(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data: data };
  }

  function setBusy(button, on, labelBusy, labelIdle) {
    busy = on;
    button.disabled = on;
    const label = $(".btn-label", button);
    if (label) label.textContent = on ? labelBusy : labelIdle;
  }

  function go(target) {
    // The server already validated that `next` is a same-site path.
    window.location.href = target || "/";
  }

  if (el.loginForm) {
    el.loginForm.addEventListener("submit", async e => {
      e.preventDefault();
      if (busy) return;
      clearErrors(el.loginForm);

      const email = $("#login-email").value.trim();
      const password = $("#login-password").value;

      if (!EMAIL_RE.test(email)) {
        fieldError(el.loginForm, "email", "Enter a valid email address.");
        $("#login-email").focus();
        return;
      }
      if (!password) {
        fieldError(el.loginForm, "password", "Enter your password.");
        $("#login-password").focus();
        return;
      }

      const button = $("#login-submit");
      setBusy(button, true, "Signing in…", "Sign in");

      const r = await post("/api/auth/login", {
        email: email,
        password: password,
        remember: $("#login-remember").checked,
        next: CFG.nextUrl || "/"
      }).catch(err => ({ ok: false, data: { error: "Network error: " + err.message } }));

      if (!r.ok) {
        setBusy(button, false, "", "Sign in");
        showError(r.data.error || "Sign-in failed.");
        if (r.data.field) fieldError(el.loginForm, r.data.field, " ");
        $("#login-password").select();
        return;
      }

      setBusy(button, true, "Welcome back…", "Sign in");
      go(r.data.next);
    });
  }

  if (el.signupForm) {
    el.signupForm.addEventListener("submit", async e => {
      e.preventDefault();
      if (busy) return;
      clearErrors(el.signupForm);

      const name = $("#signup-name").value.trim();
      const email = $("#signup-email").value.trim();
      const password = $("#signup-password").value;

      if (name.length < 2) {
        fieldError(el.signupForm, "name", "Enter your name (at least 2 characters).");
        $("#signup-name").focus(); return;
      }
      if (!EMAIL_RE.test(email)) {
        fieldError(el.signupForm, "email", "Enter a valid email address.");
        $("#signup-email").focus(); return;
      }
      const problem = passwordProblem(password, email, name);
      if (problem) {
        fieldError(el.signupForm, "password", problem);
        $("#signup-password").focus(); return;
      }
      if (!checkConfirm()) { $("#signup-confirm").focus(); return; }

      const button = $("#signup-submit");
      setBusy(button, true, "Creating account…", "Create account");

      const r = await post("/api/auth/signup", {
        name: name, email: email, password: password,
        remember: true, next: CFG.nextUrl || "/"
      }).catch(err => ({ ok: false, data: { error: "Network error: " + err.message } }));

      if (!r.ok) {
        setBusy(button, false, "", "Create account");
        showError(r.data.error || "Could not create the account.");
        if (r.data.field) fieldError(el.signupForm, r.data.field, " ");
        if (r.status === 409) setTimeout(() => setMode("login", true), 1400);
        return;
      }

      const adopted = r.data.adopted || {};

      // Reset the button so the user isn't stuck on a disabled state
      setBusy(button, false, "", "Create account");

      // Pre-fill the login form with the credentials the user just registered
      const loginEmailInput = $("#login-email");
      const loginPasswordInput = $("#login-password");
      if (loginEmailInput) loginEmailInput.value = email;
      if (loginPasswordInput) loginPasswordInput.value = password;

      if (adopted.searches) {
        toast("Account created! " + adopted.searches + " existing search(es) adopted. Now sign in.");
      } else {
        toast("Account created! Your credentials have been filled in — click Sign in to continue.");
      }

      // Switch to the login tab so the user signs in with their new credentials
      setMode("login", false);

      // Highlight the sign-in button so it's obvious what to do next
      const loginBtn = $("#login-submit");
      if (loginBtn) {
        loginBtn.focus();
        loginBtn.classList.add("btn-highlight");
        setTimeout(() => loginBtn.classList.remove("btn-highlight"), 2500);
      }
    });
  }

  /* ------------------------------------------------------------ rotator */
  const slides = $$(".rot", el.rotator);
  let slide = 0;
  let rotateTimer = null;

  if (slides.length) {
    slides.forEach((_, i) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "dot" + (i === 0 ? " on" : "");
      dot.setAttribute("aria-label", "Highlight " + (i + 1));
      dot.addEventListener("click", () => { showSlide(i); restart(); });
      el.dots.appendChild(dot);
    });

    const dotNodes = $$(".dot", el.dots);

    function showSlide(i) {
      slide = (i + slides.length) % slides.length;
      slides.forEach((s, k) => s.classList.toggle("active", k === slide));
      dotNodes.forEach((d, k) => d.classList.toggle("on", k === slide));
    }
    function restart() {
      clearInterval(rotateTimer);
      rotateTimer = setInterval(() => showSlide(slide + 1), 5200);
    }
    restart();
  }

  /* -------------------------------------------------------------- motes */
  if (el.motes && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    for (let i = 0; i < 16; i++) {
      const m = document.createElement("span");
      m.className = "mote";
      m.style.left = (Math.random() * 100).toFixed(2) + "%";
      m.style.animationDuration = (13 + Math.random() * 14).toFixed(1) + "s";
      m.style.animationDelay = (-Math.random() * 22).toFixed(1) + "s";
      m.style.setProperty("--sway", (Math.random() * 90 - 45).toFixed(0) + "px");
      m.style.opacity = (0.25 + Math.random() * 0.5).toFixed(2);
      el.motes.appendChild(m);
    }
  }

  /* -------------------------------------------------------------- theme */
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    window.__theme = t;
    el.themeBtn.textContent = t === "dark" ? "☀" : "◐";
  }
  setTheme(window.matchMedia &&
           window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  el.themeBtn.addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") === "dark"
             ? "light" : "dark"));

  /* --------------------------------------------------------------- boot */
  if (CFG.allowSignup && (CFG.startMode === "signup" || CFG.firstRun)) {
    setMode("signup", false);
  }
  const firstInput = mode === "signup" ? $("#signup-name") : $("#login-email");
  if (firstInput) firstInput.focus();
})();
