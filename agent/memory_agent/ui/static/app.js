const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

let unfinished = null;

function showScreen(name) {
  $$(".tab").forEach((t) => {
    const on = t.dataset.screen === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".screen").forEach((s) => {
    const on = s.dataset.screen === name;
    s.classList.toggle("active", on);
    s.hidden = !on;
  });
}

$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => showScreen(btn.dataset.screen));
});

function renderRecent(pages) {
  const ul = $("#recent");
  if (!pages.length) {
    ul.innerHTML = `<li class="muted">No pages yet. Sync history or browse, then ask.</li>`;
    return;
  }
  ul.innerHTML = pages
    .map(
      (p) => `
      <li>
        <div class="title">${escapeHtml(p.title || p.url)}</div>
        <div class="domain">${escapeHtml(p.domain || "")}</div>
        <div class="excerpt">${escapeHtml(p.url)}</div>
        <div class="muted">${escapeHtml(p.last_active_at || "")}${
          p.extraction_ok === 0 ? " · link only" : ""
        }</div>
      </li>`
    )
    .join("");
}

function renderAnswer(payload) {
  const box = $("#answer");
  const results = $("#results");
  box.hidden = false;
  box.classList.remove("hidden");
  const a = payload.answer || {};
  box.innerHTML = `
    <div>${formatAnswer(a.text || "")}</div>
    <div class="historical">Historical snapshot — content may have changed</div>
    <div class="meta">${a.found ? "Source-backed match" : "No strong match"}</div>
  `;
  results.innerHTML = (payload.results || [])
    .map(
      (r) => `
      <article class="result">
        <div class="title">${escapeHtml(r.title || r.url)}</div>
        <div class="domain">${escapeHtml(r.domain || "")} · score ${(r.score || 0).toFixed(2)}</div>
        <div class="excerpt">${escapeHtml(r.excerpt || "")}</div>
        <div class="muted">Captured ${escapeHtml(r.captured_at || "")}</div>
        <div class="row">
          <button class="btn small primary" data-open="${escapeAttr(r.url)}">Open</button>
        </div>
      </article>`
    )
    .join("");
  results.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => api(`/api/open?url=${encodeURIComponent(btn.dataset.open)}`));
  });
}

function formatAnswer(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", "&#39;");
}

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#ask-input").value.trim();
  if (!query) return;
  const data = await api("/api/ask", { method: "POST", body: JSON.stringify({ query }) });
  renderAnswer(data);
});

$("#sync-btn").addEventListener("click", async () => {
  $("#sync-btn").disabled = true;
  $("#sync-btn").textContent = "Syncing…";
  try {
    await api("/api/sync/history", { method: "POST" });
    await refreshRecent();
  } finally {
    $("#sync-btn").disabled = false;
    $("#sync-btn").textContent = "Sync browser history";
  }
});

async function refreshRecent() {
  const data = await api("/api/recent");
  renderRecent(data.pages || []);
}

function renderWorkspaces(data) {
  unfinished = data.unfinished;
  const banner = $("#resume-banner");
  if (unfinished && unfinished.next_step) {
    banner.hidden = false;
    banner.classList.remove("hidden");
    $("#resume-name").textContent = unfinished.name;
    $("#resume-next").textContent = unfinished.next_step;
    $("#resume-banner-btn").onclick = () => resumeWorkspace(unfinished.id);
  } else {
    banner.hidden = true;
    banner.classList.add("hidden");
  }

  const list = $("#workspace-list");
  const items = data.workspaces || [];
  if (!items.length) {
    list.innerHTML = `<li class="muted">No workspaces yet. Click Save my place.</li>`;
    return;
  }
  list.innerHTML = items
    .map(
      (w) => `
      <li class="workspace-item">
        <div class="title">${escapeHtml(w.name)}</div>
        <div class="excerpt">${escapeHtml(w.summary || "")}</div>
        <div class="muted">${w.item_count || 0} pages${
          w.next_step ? ` · Next: ${escapeHtml(w.next_step)}` : ""
        }</div>
        <div class="row">
          <button class="btn small primary" data-resume="${w.id}">Resume</button>
        </div>
      </li>`
    )
    .join("");
  list.querySelectorAll("[data-resume]").forEach((btn) => {
    btn.addEventListener("click", () => resumeWorkspace(btn.dataset.resume));
  });
}

async function resumeWorkspace(id) {
  const result = await api(`/api/workspaces/${id}/resume`, { method: "POST" });
  alert(
    `Resumed “${result.workspace?.name || "workspace"}”.\n` +
      `Reused ${result.reused?.length || 0} open tabs, opened ${
        (result.opened?.length || 0) + (result.pending_urls?.length || 0)
      }.\n\n${result.next_step ? "Next step: " + result.next_step : ""}\n` +
      (result.note || "")
  );
}

$("#save-place-btn").addEventListener("click", async () => {
  const proposal = await api("/api/workspaces/propose", {
    method: "POST",
    body: JSON.stringify({ next_step: "" }),
  });
  const panel = $("#save-panel");
  panel.hidden = false;
  panel.classList.remove("hidden");
  const tabs = proposal.tabs || [];
  panel.innerHTML = `
    <h3>Save my place</h3>
    <p class="muted">Observed tabs stay separate from your next-step note.</p>
    <label>Name<input id="ws-name" value="${escapeAttr(proposal.suggestion?.name || "")}" /></label>
    <label>Summary<textarea id="ws-summary" rows="2">${escapeHtml(
      proposal.suggestion?.summary || ""
    )}</textarea></label>
    <label>Next step (your intent)<input id="ws-next" placeholder="Tomorrow, compare delivery times." /></label>
    <div class="tab-picker">
      ${tabs
        .map(
          (t, i) => `
        <label>
          <input type="checkbox" checked data-page-id="${escapeAttr(t.page_id || "")}" data-tab-id="${escapeAttr(
            t.id || ""
          )}" data-keep="0" />
          <span>
            <strong>${escapeHtml(t.title || t.url)}</strong><br/>
            <span class="domain">${escapeHtml(t.domain || "")}</span>
          </span>
        </label>`
        )
        .join("") || "<p class='muted'>No open tabs via Chrome debugging; using recent memory.</p>"}
    </div>
    <div class="row">
      <button id="ws-save" class="btn primary">Save workspace</button>
      <button id="ws-save-close" class="btn">Save &amp; close selected tabs</button>
      <button id="ws-cancel" class="btn ghost">Cancel</button>
    </div>
    <p class="muted">Tabs that may have unsaved work should stay open — leave them unchecked for closing.</p>
  `;
  $("#ws-cancel").onclick = () => {
    panel.hidden = true;
    panel.classList.add("hidden");
  };
  const save = async (closeSelected) => {
    const checks = [...panel.querySelectorAll("input[type=checkbox]:checked")];
    const page_ids = checks.map((c) => c.dataset.pageId).filter(Boolean);
    const close_tab_ids = closeSelected
      ? checks.map((c) => c.dataset.tabId).filter(Boolean)
      : [];
    await api("/api/workspaces/save", {
      method: "POST",
      body: JSON.stringify({
        name: $("#ws-name").value.trim(),
        summary: $("#ws-summary").value.trim(),
        next_step: $("#ws-next").value.trim(),
        page_ids,
        close_tab_ids,
      }),
    });
    panel.hidden = true;
    panel.classList.add("hidden");
    await refreshWorkspaces();
    showScreen("workspaces");
  };
  $("#ws-save").onclick = () => save(false);
  $("#ws-save-close").onclick = () => save(true);
});

async function refreshWorkspaces() {
  const data = await api("/api/workspaces");
  renderWorkspaces(data);
}

async function refreshMemory() {
  const health = await api("/api/health");
  const stats = health.stats || {};
  $("#stats").innerHTML = ["pages", "snapshots", "visits", "workspaces", "exclusions"]
    .map(
      (k) => `<div><dt>${k}</dt><dd>${stats[k] ?? 0}</dd></div>`
    )
    .join("");
  $("#cdp-status").textContent = health.cdp
    ? "Chrome debugging connected — live tabs can be listed, opened, and closed."
    : "Chrome debugging offline — history sync and link restore still work.";
  const auto = health.autostart || {};
  $("#autostart-status").textContent = auto.enabled
    ? `Auto-start enabled (${auto.path})`
    : "Auto-start is off. Enable it so the agent deploys when you open your computer.";

  const ex = await api("/api/memory/exclusions");
  const ul = $("#exclusions");
  ul.innerHTML = (ex.exclusions || [])
    .map(
      (e) => `<li><span>${escapeHtml(e.pattern)}</span>
        <button class="btn small ghost" data-ex="${escapeAttr(e.pattern)}">Remove</button></li>`
    )
    .join("");
  ul.querySelectorAll("[data-ex]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/memory/exclusions?pattern=${encodeURIComponent(btn.dataset.ex)}`, {
        method: "DELETE",
      });
      refreshMemory();
    });
  });
}

$("#exclude-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const pattern = $("#exclude-input").value.trim();
  if (!pattern) return;
  await api("/api/memory/exclusions", {
    method: "POST",
    body: JSON.stringify({ pattern }),
  });
  $("#exclude-input").value = "";
  refreshMemory();
});

$("#autostart-on").addEventListener("click", async () => {
  await api("/api/autostart/install", { method: "POST" });
  refreshMemory();
});
$("#autostart-off").addEventListener("click", async () => {
  await api("/api/autostart/uninstall", { method: "POST" });
  refreshMemory();
});
$("#delete-all").addEventListener("click", async () => {
  if (!confirm("Delete all local browsing memory?")) return;
  await api("/api/memory/delete-all", { method: "POST" });
  await Promise.all([refreshRecent(), refreshWorkspaces(), refreshMemory()]);
});

async function boot() {
  await Promise.all([refreshRecent(), refreshWorkspaces(), refreshMemory()]);
  // Prefill flagship query for demo clarity
  if (!$("#ask-input").value) {
    $("#ask-input").placeholder = "What was that supplier with the minimum order of 500?";
  }
}

boot().catch((err) => {
  console.error(err);
  $("#recent").innerHTML = `<li class="muted">Agent UI failed to load: ${escapeHtml(
    err.message
  )}</li>`;
});
