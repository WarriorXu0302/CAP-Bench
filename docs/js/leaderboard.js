/**
 * Leaderboard renderer — fetches data/leaderboard.json and renders a
 * sortable table. Used by both index.html (top-5 preview) and
 * leaderboard.html (full table).
 */

const METRIC_COLS = [
  { key: "partial_completion", label: "Partial Completion", sortable: true,  highlight: true  },
  { key: "success_rate",       label: "Success Rate",       sortable: true,  highlight: false },
  { key: "complex_a",          label: "Complex-A",          sortable: true,  highlight: false },
  { key: "complex_p",          label: "Complex-P",          sortable: true,  highlight: false },
];

const EXTRA_COLS = [
  { key: "time_min", label: "Time (min)", sortable: true, format: (v) => v?.toFixed(1) ?? "—" },
  { key: "tokens_k", label: "Output (k tok)", sortable: true, format: (v) => v?.toFixed(1) ?? "—" },
];

function fmtPct(v, std) {
  if (v === null || v === undefined) return "—";
  const std_str = (std !== undefined && std !== null) ? `<span class="std">±${std.toFixed(1)}</span>` : "";
  return `${v.toFixed(1)}${std_str}`;
}

function categoryBadge(cat) {
  if (!cat) return "";
  const cls = cat.toLowerCase().includes("open") ? "opensource"
            : cat.toLowerCase().includes("reference") ? "reference"
            : "";
  return `<span class="cat-badge ${cls}">${cat}</span>`;
}

const MEDALS = ["🥇", "🥈", "🥉"];

function renderRows(entries, includeExtra = true) {
  return entries.map((e, idx) => {
    const refClass = e.category === "Reference" ? "reference" : "";
    const rankClass = (e.category !== "Reference" && idx < 3) ? `rank-${idx + 1}` : "";
    const cls = [refClass, rankClass].filter(Boolean).join(" ");

    let rankCell;
    if (e.category === "Reference") {
      rankCell = "—";
    } else if (idx < 3) {
      rankCell = `<span class="medal">${MEDALS[idx]}</span>${idx + 1}`;
    } else {
      rankCell = String(idx + 1);
    }

    let html = `<tr class="${cls}">`;
    html += `<td>${rankCell}</td>`;
    html += `<td><strong>${e.system}</strong>${categoryBadge(e.category)}</td>`;
    for (const col of METRIC_COLS) {
      html += `<td>${fmtPct(e[col.key], e[col.key + "_std"])}</td>`;
    }
    if (includeExtra) {
      for (const col of EXTRA_COLS) {
        html += `<td>${col.format(e[col.key])}</td>`;
      }
    }
    html += `</tr>`;
    return html;
  }).join("");
}

function renderHeader(includeExtra = true) {
  const cells = ["#", "System", ...METRIC_COLS.map((c) => c.label)];
  if (includeExtra) cells.push(...EXTRA_COLS.map((c) => c.label));
  return `<tr>${cells.map((c, i) => {
    const sortable = i >= 2;
    return `<th${sortable ? ' class="sortable"' : ""}>${c}${sortable ? ' <span class="arrow">↕</span>' : ""}</th>`;
  }).join("")}</tr>`;
}

function sortEntries(entries, key, descending = true) {
  return [...entries].sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return descending ? bv - av : av - bv;
  });
}

async function loadLeaderboard() {
  const resp = await fetch("data/leaderboard.json", { cache: "no-store" });
  if (!resp.ok) throw new Error(`leaderboard.json not found (${resp.status})`);
  return resp.json();
}

// ----- Preview (index.html) ---------------------------------------------------

async function renderPreview(targetId, topN = 5) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    const data = await loadLeaderboard();
    // Sort by Partial Completion descending; keep Reference (Human) at the bottom
    const ranked = data.entries.filter((e) => e.category !== "Reference");
    const ref    = data.entries.filter((e) => e.category === "Reference");
    const sorted = sortEntries(ranked, "partial_completion", true).slice(0, topN);
    const display = [...sorted, ...ref];

    target.innerHTML = `
      <table class="leaderboard">
        <thead>${renderHeader(false)}</thead>
        <tbody>${renderRows(display, false)}</tbody>
      </table>
    `;
  } catch (e) {
    target.innerHTML = `<p style="color:#b91c1c">Failed to load leaderboard: ${e.message}</p>`;
  }
}

// ----- Full leaderboard (leaderboard.html) -----------------------------------

async function renderFull(targetId, lastUpdatedId, submitLinkId) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    const data = await loadLeaderboard();

    if (lastUpdatedId) {
      const el = document.getElementById(lastUpdatedId);
      if (el && data.last_updated) el.textContent = data.last_updated;
    }
    if (submitLinkId && data.submission_url) {
      const el = document.getElementById(submitLinkId);
      if (el) el.href = data.submission_url;
    }

    let sortKey = "partial_completion";
    let sortDesc = true;

    function paint() {
      const ranked = data.entries.filter((e) => e.category !== "Reference");
      const ref    = data.entries.filter((e) => e.category === "Reference");
      const sorted = sortEntries(ranked, sortKey, sortDesc);
      target.innerHTML = `
        <table class="leaderboard">
          <thead>${renderHeader(true)}</thead>
          <tbody>${renderRows([...sorted, ...ref], true)}</tbody>
        </table>
      `;
      // Wire up sort handlers
      const ths = target.querySelectorAll("thead th.sortable");
      const allKeys = [...METRIC_COLS, ...EXTRA_COLS];
      ths.forEach((th, i) => {
        const key = allKeys[i].key;
        th.addEventListener("click", () => {
          if (sortKey === key) sortDesc = !sortDesc;
          else { sortKey = key; sortDesc = true; }
          paint();
        });
      });
    }
    paint();
  } catch (e) {
    target.innerHTML = `<p style="color:#b91c1c">Failed to load leaderboard: ${e.message}</p>`;
  }
}
