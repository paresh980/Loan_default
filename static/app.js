(function () {
  "use strict";

  const form = document.getElementById("applicant-form");
  const btn = document.getElementById("submit-btn");
  const result = document.getElementById("result");
  const panel = document.getElementById("panel");
  const formError = document.getElementById("form-error");
  const emptyView = panel.querySelector(".empty");
  const scoredView = panel.querySelector(".scored");

  const PRESETS = {
    low: {
      Age: 45, Education: "Master's", MaritalStatus: "Married", HasDependents: "No",
      EmploymentType: "Full-time", MonthsEmployed: 100, Income: 110000, CreditScore: 760,
      DTIRatio: 0.2, NumCreditLines: 2, HasMortgage: "Yes",
      LoanAmount: 20000, LoanPurpose: "Home", InterestRate: 6, LoanTerm: 24, HasCoSigner: "Yes",
    },
    high: {
      Age: 22, Education: "High School", MaritalStatus: "Single", HasDependents: "Yes",
      EmploymentType: "Unemployed", MonthsEmployed: 4, Income: 28000, CreditScore: 420,
      DTIRatio: 0.8, NumCreditLines: 4, HasMortgage: "No",
      LoanAmount: 180000, LoanPurpose: "Business", InterestRate: 22, LoanTerm: 60, HasCoSigner: "No",
    },
  };

  const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
  const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";

  function clearErrors() {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("invalid"));
    form.querySelectorAll(".error").forEach((e) => (e.textContent = ""));
    formError.hidden = true;
    formError.textContent = "";
  }

  function showFieldErrors(fields) {
    let first = null;
    Object.entries(fields).forEach(([name, message]) => {
      const wrap = form.querySelector(`[data-field="${name}"]`);
      const err = document.getElementById(`${name}-error`);
      if (!wrap || !err) return;
      wrap.classList.add("invalid");
      err.textContent = message;
      if (!first) first = wrap.querySelector("input, select");
    });
    if (first) first.focus();
  }

  function readForm() {
    const data = {};
    new FormData(form).forEach((value, key) => (data[key] = value));
    return data;
  }

  // Mirror of the server rules, so people get instant feedback.
  function validateClient(data) {
    const errors = {};
    form.querySelectorAll('input[type="number"]').forEach((input) => {
      const v = input.value.trim();
      const lo = parseFloat(input.min), hi = parseFloat(input.max);
      if (v === "" || Number.isNaN(Number(v))) errors[input.name] = "Enter a number.";
      else if (Number(v) < lo || Number(v) > hi) errors[input.name] = `Enter a value between ${lo} and ${hi}.`;
    });
    return errors;
  }

  function renderFactors(factors) {
    const list = document.getElementById("factors");
    list.innerHTML = "";
    const max = Math.max(...factors.map((f) => Math.abs(f.effect)), 0.0001);
    factors.forEach((f) => {
      const li = document.createElement("li");
      const width = Math.max((Math.abs(f.effect) / max) * 50, 2);

      const name = document.createElement("div");
      name.className = "name";
      name.textContent = f.label;
      const note = document.createElement("small");
      note.textContent = f.direction === "raises" ? "Raises the risk score" : "Lowers the risk score";
      name.appendChild(note);

      const bar = document.createElement("div");
      bar.className = "bar";
      bar.setAttribute("aria-hidden", "true");
      const fill = document.createElement("i");
      fill.className = f.direction === "raises" ? "up" : "down";
      fill.style.width = width + "%";
      bar.appendChild(fill);

      li.append(name, bar);
      list.appendChild(li);
    });
  }

  function renderResult(r) {
    const flagged = r.high_risk;
    panel.dataset.state = flagged ? "flag" : "ok";
    emptyView.hidden = true;
    scoredView.hidden = false;

    document.getElementById("verdict-title").textContent = flagged
      ? "Higher risk: review recommended"
      : "Lower risk";
    document.getElementById("verdict-text").textContent = flagged
      ? `The score is at or above the review threshold of ${pct(r.threshold)}.`
      : `The score is below the review threshold of ${pct(r.threshold)}.`;

    document.getElementById("score").textContent = pct(r.probability);
    document.getElementById("gauge").setAttribute(
      "aria-label",
      `Risk score ${pct(r.probability)} on a scale of 0 to 100, with the review threshold at ${pct(r.threshold)}.`
    );

    document.getElementById("zone-ok").style.width = pct(r.threshold, 1);
    document.getElementById("tick").style.left = pct(r.threshold, 1);
    const marker = document.getElementById("marker");
    marker.style.left = "0%";
    // Two frames so the browser registers the start position and animates to the score.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      marker.style.left = pct(Math.min(Math.max(r.probability, 0), 1), 1);
    }));

    renderFactors(r.factors);
    document.getElementById("fact-emi").textContent = "₹" + inr.format(r.monthly_payment);
    document.getElementById("fact-pti").textContent = pct(r.payment_to_income, 1);
  }

  function showError(message) {
    emptyView.hidden = !scoredView.hidden;
    formError.textContent = message;
    formError.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors();

    const data = readForm();
    const clientErrors = validateClient(data);
    if (Object.keys(clientErrors).length) {
      showFieldErrors(clientErrors);
      return;
    }

    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = "Scoring…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const body = await res.json().catch(() => ({}));

      if (res.status === 422 && body.fields) {
        showFieldErrors(body.fields);
      } else if (!res.ok) {
        showError(body.error || "The server could not score this application. Try again in a moment.");
      } else {
        renderResult(body);
        if (window.matchMedia("(max-width: 860px)").matches) {
          result.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    } catch (err) {
      showError("Could not reach the server. Check your connection and try again.");
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  });

  document.querySelectorAll("[data-preset]").forEach((b) => {
    b.addEventListener("click", () => {
      clearErrors();
      const values = PRESETS[b.dataset.preset];
      Object.entries(values).forEach(([name, value]) => {
        const el = form.elements[name];
        if (el) el.value = value;
      });
    });
  });
})();
