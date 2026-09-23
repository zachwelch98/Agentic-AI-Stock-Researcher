const STAGES = [
  { key: "planning", label: "Planning screening" },
  { key: "screening", label: "Screening industry" },
  { key: "insufficient_candidates", label: "Insufficient candidates" },
  { key: "researching", label: "Researching (4 domains)" },
  { key: "synthesizing", label: "Synthesizing report" },
  { key: "done", label: "Done" },
];

const POLL_INTERVAL_MS = 1500;

const form = document.getElementById("research-form");
const industryInput = document.getElementById("industry-input");
const tickerInput = document.getElementById("ticker-input");
const submitBtn = document.getElementById("submit-btn");
const statusPanel = document.getElementById("status-panel");
const reportPanel = document.getElementById("report-panel");

let pollTimer = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const industry = industryInput.value.trim();
  const ticker = tickerInput.value.trim();
  if (!industry && !ticker) return;
  if (industry && ticker) {
    renderError("Enter either an industry or a ticker, not both.");
    return;
  }

  stopPolling();
  reportPanel.innerHTML = "";
  submitBtn.disabled = true;

  try {
    const res = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ticker ? { ticker } : { industry }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      renderError(`Request failed (${res.status}): ${JSON.stringify(body)}`);
      submitBtn.disabled = false;
      return;
    }
    const { job_id } = await res.json();
    renderStatus({ status: "pending", graph_status: "planning", industry_query: industry, ticker, job_id });
    pollTimer = setInterval(() => pollJob(job_id), POLL_INTERVAL_MS);
    pollJob(job_id);
  } catch (err) {
    renderError(String(err));
    submitBtn.disabled = false;
  }
});

async function pollJob(jobId) {
  try {
    const res = await fetch(`/jobs/${jobId}`);
    if (!res.ok) {
      renderError(`Job lookup failed (${res.status})`);
      stopPolling();
      submitBtn.disabled = false;
      return;
    }
    const job = await res.json();
    renderStatus(job);

    if (job.status === "done" || job.status === "failed") {
      stopPolling();
      submitBtn.disabled = false;
      if (job.status === "done" && job.result) renderReport(job.result);
      if (job.status === "failed") renderError(job.error || "Job failed");
    }
  } catch (err) {
    renderError(String(err));
    stopPolling();
    submitBtn.disabled = false;
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderStatus(job) {
  const currentIdx = STAGES.findIndex((s) => s.key === job.graph_status);
  const stagesHtml = STAGES.filter((s) => {
    if (s.key === "insufficient_candidates") return job.graph_status === "insufficient_candidates";
    return true;
  })
    .map((s, idx) => {
      const stageIdx = STAGES.findIndex((x) => x.key === s.key);
      let cls = "";
      if (job.graph_status === s.key) cls = "active";
      else if (currentIdx >= 0 && stageIdx < currentIdx) cls = "done";
      return `<li class="${cls}">${s.label}</li>`;
    })
    .join("");

  statusPanel.innerHTML = `
    <div class="card">
      <strong>${escapeHtml(job.industry_query || job.ticker || "")}</strong>
      <ul class="stage-list">${stagesHtml}</ul>
      <div class="job-id">job: ${escapeHtml(job.job_id || "")} &middot; status: ${escapeHtml(job.status || "")}</div>
    </div>
  `;
}

function renderError(message) {
  statusPanel.innerHTML = `<div class="card error">${escapeHtml(message)}</div>`;
}

function renderReport(report) {
  if (report.outcome === "insufficient_candidates") {
    reportPanel.innerHTML = `<div class="card">${escapeHtml(report.message || "Insufficient candidates found.")}</div>`;
    return;
  }

  const note = report.data_provenance_note
    ? `<div class="note">${escapeHtml(report.data_provenance_note)}</div>`
    : "";

  const summary = report.overall_recommendation
    ? `<div class="card"><strong>Overall recommendation</strong><p>${escapeHtml(report.overall_recommendation)}</p></div>`
    : "";

  const tickers = (report.tickers || [])
    .map((t) => {
      const findings = (t.domain_findings || [])
        .map(
          (f) => `
            <div class="finding">
              <div class="domain">${escapeHtml(f.domain)}</div>
              <div>${escapeHtml(f.summary)}</div>
            </div>
          `
        )
        .join("");
      return `
        <div class="ticker-card">
          <h3>${escapeHtml(t.ticker)}</h3>
          <p>${escapeHtml(t.overall_take || "")}</p>
          ${t.strengths && t.strengths.length ? `<div><strong>Strengths</strong><ul class="plain">${t.strengths.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>` : ""}
          ${t.risks && t.risks.length ? `<div><strong>Risks</strong><ul class="plain">${t.risks.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>` : ""}
          <div class="findings">${findings}</div>
        </div>
      `;
    })
    .join("");

  reportPanel.innerHTML = `${note}${summary}${tickers}`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
