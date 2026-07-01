const numberFormatter = new Intl.NumberFormat("en-US");

const fallbackCatalog = {
  repository: {
    url: "https://github.com/Cazzy-Aporbo/velvet-python",
  },
  stats: {
    python_file_count: 6,
    function_count: 24,
    class_count: 8,
  },
  system: {
    proof_linked_count: 2,
    execution_linked_count: 2,
    hub_files: [],
    proof_files: [],
    execution_files: [],
  },
  files: [
    {
      id: "src--ai-py",
      path: "src/ai.py",
      title: "AI",
      category_label: "Core Systems",
      category_key: "core",
      headline: "Classifiers from first principles.",
      summary: "Three small model families for learning classification behavior from the inside.",
      why_it_matters: "This is where simple modeling ideas stay readable instead of disappearing behind a framework.",
      learning_moment: "Read it slowly, compare the interfaces, then test the tradeoffs against the rest of the repo.",
      best_for: "comparison-first learning",
      difficulty: "steady",
      depth_index: 71,
      system_role: "Core module",
      system_role_note: "This file shows how a model surface can stay small and still be instructive.",
      tags: ["Classification", "Core Engineering", "Probability"],
      anchors: ["NaiveBayesClassifier", "CosineSimilarityClassifier"],
      upstream_paths: [],
      downstream_paths: ["src/model_registry.py", "tests/test_ml_pipeline.py"],
      proof_paths: ["tests/test_ml_pipeline.py"],
      execution_paths: [],
      connectivity: {
        upstream_count: 0,
        downstream_count: 2,
        proof_count: 1,
        execution_count: 0,
        total_links: 2,
      },
      stats: {
        nonempty_lines: 180,
        function_count: 8,
        class_count: 2,
        import_count: 4,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/src/ai.py",
    },
    {
      id: "src--data_utils-py",
      path: "src/data_utils.py",
      title: "Data Utils",
      category_label: "Core Systems",
      category_key: "core",
      headline: "Validation, signatures, and split discipline.",
      summary: "Deterministic data contracts for inputs, profiles, and train-test boundaries.",
      why_it_matters: "This is where results stop being data-lucky and start becoming inspectable.",
      learning_moment: "Follow the validation path first, then see how everything downstream depends on it.",
      best_for: "input-contract thinking",
      difficulty: "deep",
      depth_index: 82,
      system_role: "Data contract",
      system_role_note: "This file carries data discipline into the rest of the repository.",
      tags: ["Data Quality", "Core Engineering", "Shared Module"],
      anchors: ["validate_dataset", "dataset_profile", "train_test_split"],
      upstream_paths: [],
      downstream_paths: ["src/pipeline.py", "scripts/dataset_audit.py", "CLI.py"],
      proof_paths: ["tests/test_data_pipeline_utils.py"],
      execution_paths: ["scripts/dataset_audit.py", "CLI.py"],
      connectivity: {
        upstream_count: 0,
        downstream_count: 3,
        proof_count: 1,
        execution_count: 2,
        total_links: 3,
      },
      stats: {
        nonempty_lines: 190,
        function_count: 9,
        class_count: 0,
        import_count: 9,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/src/data_utils.py",
    },
    {
      id: "src--pipeline-py",
      path: "src/pipeline.py",
      title: "Pipeline",
      category_label: "Core Systems",
      category_key: "core",
      headline: "Reproducible experimentation utilities for Velvet Python.",
      summary: "A deterministic run manifest layer that turns training behavior into evidence.",
      why_it_matters: "This is where experiments become something you can rerun, compare, and review.",
      learning_moment: "Look at the run payload before the training loop and see how much reliability is decided there.",
      best_for: "manifest-driven experimentation",
      difficulty: "deep",
      depth_index: 98,
      system_role: "Execution pipeline",
      system_role_note: "This file is the handoff between validated data, model execution, and evidence output.",
      tags: ["Pipelines", "Proof Linked", "Runnable Surface"],
      anchors: ["ExperimentRun", "run_classification_pipeline", "run_epochs"],
      upstream_paths: ["src/data_utils.py"],
      downstream_paths: ["src/evidence_ledger.py", "scripts/run_experiments.py", "CLI.py"],
      proof_paths: ["tests/test_pipeline_contracts.py"],
      execution_paths: ["scripts/run_experiments.py", "CLI.py"],
      connectivity: {
        upstream_count: 1,
        downstream_count: 3,
        proof_count: 1,
        execution_count: 2,
        total_links: 4,
      },
      stats: {
        nonempty_lines: 303,
        function_count: 12,
        class_count: 1,
        import_count: 7,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/src/pipeline.py",
    },
    {
      id: "scripts--run_experiments-py",
      path: "scripts/run_experiments.py",
      title: "Run Experiments",
      category_label: "Workflow Scripts",
      category_key: "scripts",
      headline: "CLI automation for repeatable experiment sweeps.",
      summary: "A script surface for exercising the pipeline and leaving behind artifacts you can inspect later.",
      why_it_matters: "This is where the repository stops being explanatory only and starts acting like a runnable system.",
      learning_moment: "Open this after the pipeline module so you can see how orchestration wraps the core behavior.",
      best_for: "automation and experiment surfaces",
      difficulty: "steady",
      depth_index: 67,
      system_role: "Workflow orchestrator",
      system_role_note: "This file packages core logic into something fast to rerun and compare.",
      tags: ["Automation", "Pipelines", "Runnable Surface"],
      anchors: ["main", "parse_args", "run_suite"],
      upstream_paths: ["src/pipeline.py", "src/model_registry.py", "src/data_utils.py"],
      downstream_paths: [],
      proof_paths: [],
      execution_paths: [],
      connectivity: {
        upstream_count: 3,
        downstream_count: 0,
        proof_count: 0,
        execution_count: 0,
        total_links: 3,
      },
      stats: {
        nonempty_lines: 122,
        function_count: 4,
        class_count: 0,
        import_count: 5,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/scripts/run_experiments.py",
    },
    {
      id: "src--evidence_ledger-py",
      path: "src/evidence_ledger.py",
      title: "Evidence Ledger",
      category_label: "Core Systems",
      category_key: "core",
      headline: "Drift checks, review packets, and reliability summaries.",
      summary: "The review layer that asks whether a result stays sturdy when you run it more than once.",
      why_it_matters: "This is where the repository shows that evidence is more than a metric; it is a reviewable shape.",
      learning_moment: "Read the recommendations and health summary logic to see how engineering judgment gets encoded.",
      best_for: "review and reliability thinking",
      difficulty: "deep",
      depth_index: 89,
      system_role: "Evidence reviewer",
      system_role_note: "This module turns run artifacts into a human-readable reliability surface.",
      tags: ["Proof Linked", "Auditing", "Core Engineering"],
      anchors: ["validate_manifest_payload", "build_evidence_ledger", "write_evidence_ledger"],
      upstream_paths: ["src/pipeline.py"],
      downstream_paths: ["CLI.py"],
      proof_paths: ["tests/test_evidence_ledger.py"],
      execution_paths: ["CLI.py"],
      connectivity: {
        upstream_count: 1,
        downstream_count: 1,
        proof_count: 1,
        execution_count: 1,
        total_links: 2,
      },
      stats: {
        nonempty_lines: 272,
        function_count: 14,
        class_count: 0,
        import_count: 5,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/src/evidence_ledger.py",
    },
    {
      id: "tests--test_pipeline_contracts-py",
      path: "tests/test_pipeline_contracts.py",
      title: "Test Pipeline Contracts",
      category_label: "Evidence & Tests",
      category_key: "tests",
      headline: "Contract tests for pipeline evidence and determinism.",
      summary: "Proof that the pipeline holds onto the fields and behavior the rest of the repo depends on.",
      why_it_matters: "This is where a learner can see exactly what the repo refuses to let drift.",
      learning_moment: "Pair this with the pipeline module and read the tests as the interface the code owes the rest of the system.",
      best_for: "proof-first reading",
      difficulty: "steady",
      depth_index: 63,
      system_role: "Proof surface",
      system_role_note: "This test file makes the pipeline promises explicit and rerunnable.",
      tags: ["Testing", "Proof Linked"],
      anchors: ["test_manifest_fields", "test_deterministic_split"],
      upstream_paths: ["src/pipeline.py"],
      downstream_paths: [],
      proof_paths: [],
      execution_paths: [],
      connectivity: {
        upstream_count: 1,
        downstream_count: 0,
        proof_count: 0,
        execution_count: 0,
        total_links: 1,
      },
      stats: {
        nonempty_lines: 96,
        function_count: 5,
        class_count: 0,
        import_count: 2,
      },
      github_url: "https://github.com/Cazzy-Aporbo/velvet-python/blob/main/tests/test_pipeline_contracts.py",
    },
  ],
};

const body = document.body;
const catalogUrl = body.dataset.catalog || "./catalog.json";
const nodeCluster = document.getElementById("nodeCluster");
const modal = document.getElementById("modal");
const modalSidebar = document.getElementById("modalSidebar");
const modalMain = document.getElementById("modalMain");
const repoStats = document.getElementById("repoStats");
const signalTape = document.getElementById("signalTape");
const fieldTitle = document.getElementById("fieldTitle");
const fieldSummary = document.getElementById("fieldSummary");
const fieldSystems = document.getElementById("fieldSystems");
const fieldTags = document.getElementById("fieldTags");
const shuffleButton = document.getElementById("shuffleButton");
const hubButton = document.getElementById("hubButton");
const proofButton = document.getElementById("proofButton");
const executionButton = document.getElementById("executionButton");
const repoLink = document.getElementById("repoLink");

let atlasPayload = fallbackCatalog;
let activePool = [];
let nodeModels = [];
let currentFrame = null;
let currentMode = "all";
let activeFileId = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    };
    return entities[character] || character;
  });
}

function uniqueByPath(files) {
  const seen = new Set();
  return files.filter((file) => {
    if (!file?.path || seen.has(file.path)) return false;
    seen.add(file.path);
    return true;
  });
}

function firstAvailable(...groups) {
  return uniqueByPath(groups.flat().filter(Boolean));
}

function buildPool(payload, mode = "all") {
  const system = payload.system || {};
  const featured = payload.featured || [];
  const files = payload.files || [];

  if (mode === "hub") {
    return firstAvailable(system.hub_files || [], featured, files.slice(0, 12)).slice(0, 10);
  }
  if (mode === "proof") {
    return firstAvailable(system.proof_files || [], files.filter((file) => (file.connectivity?.proof_count || 0) > 0), featured).slice(0, 10);
  }
  if (mode === "execution") {
    return firstAvailable(system.execution_files || [], files.filter((file) => (file.connectivity?.execution_count || 0) > 0), featured).slice(0, 10);
  }

  return firstAvailable(
    system.hub_files || [],
    system.execution_files || [],
    system.proof_files || [],
    payload.depth_leaders || [],
    featured,
    files.slice(0, 12),
  ).slice(0, 10);
}

function applyMode(mode) {
  currentMode = mode;
  activePool = buildPool(atlasPayload, mode);
  renderSignalTape();
  renderNodes(activePool);
  if (activePool.length) {
    hydrateField(activePool[0]);
  }
  updateControls();
}

function updateControls() {
  const map = {
    hub: hubButton,
    proof: proofButton,
    execution: executionButton,
  };
  Object.entries(map).forEach(([key, button]) => {
    if (!button) return;
    button.classList.toggle("active", currentMode === key);
  });
}

function renderStats(payload) {
  if (!repoStats) return;
  const stats = payload.stats || {};
  const system = payload.system || {};
  const chips = [
    ["Python files", numberFormatter.format(stats.python_file_count || 0)],
    ["Functions", numberFormatter.format(stats.function_count || 0)],
    ["Proof surfaces", numberFormatter.format(system.proof_linked_count || 0)],
    ["Runnable paths", numberFormatter.format(system.execution_linked_count || 0)],
  ];

  repoStats.innerHTML = chips
    .map(
      ([label, value]) => `
        <article class="stat-chip">
          <strong>${value}</strong>
          <span>${label}</span>
        </article>
      `,
    )
    .join("");
}

function renderSignalTape() {
  if (!signalTape) return;
  const tokens = activePool.slice(0, 6).map((file) => {
    const proofCount = file.connectivity?.proof_count || 0;
    const executionCount = file.connectivity?.execution_count || 0;
    const role = file.system_role || file.category_label;
    return `
      <button class="signal-token" type="button" data-file-id="${escapeHtml(file.id)}">
        <strong>${escapeHtml(file.title)}</strong>
        <span>${escapeHtml(role)}</span>
        <small>${proofCount} proof · ${executionCount} runnable</small>
      </button>
    `;
  });
  signalTape.innerHTML = tokens.join("");
}

function hydrateField(file) {
  if (!file) return;
  activeFileId = file.id;
  if (fieldTitle) fieldTitle.textContent = file.title;
  if (fieldSummary) {
    fieldSummary.textContent = file.why_it_matters || file.summary || file.headline || "";
  }
  if (fieldSystems) {
    const links = file.connectivity?.total_links || 0;
    const proof = file.connectivity?.proof_count || 0;
    const execution = file.connectivity?.execution_count || 0;
    fieldSystems.textContent = `${file.system_role || file.category_label} · ${links} internal links · ${proof} proof surfaces · ${execution} runnable paths`;
  }
  if (fieldTags) {
    fieldTags.innerHTML = (file.tags || [])
      .slice(0, 5)
      .map((tag) => `<span>${escapeHtml(tag)}</span>`)
      .join("");
  }
}

function listMarkup(items, emptyLabel) {
  if (!items?.length) {
    return `<p class="empty-line">${escapeHtml(emptyLabel)}</p>`;
  }
  return `<div class="path-list">${items.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function openModal(file) {
  if (!file || !modal || !modalSidebar || !modalMain) return;

  const proof = file.connectivity?.proof_count || 0;
  const execution = file.connectivity?.execution_count || 0;
  const links = file.connectivity?.total_links || 0;
  const tags = (file.tags || []).map((tag) => `<div class="modal-tag">${escapeHtml(tag)}</div>`).join("");

  modalSidebar.innerHTML = `
    <h3>Path</h3>
    <p>${escapeHtml(file.path)}</p>
    <h3>Role</h3>
    <p>${escapeHtml(file.system_role || file.category_label)}</p>
    <h3>Signals</h3>
    <p>${links} internal links</p>
    <p>${proof} proof surfaces</p>
    <p>${execution} runnable paths</p>
    <h3>Tags</h3>
    <div>${tags || '<p>No tags surfaced yet.</p>'}</div>
  `;

  modalMain.innerHTML = `
    <h2 class="modal-title">${escapeHtml(file.title)}</h2>
    <p style="font-size: 1rem; line-height: 1.8; margin-bottom: 1.35rem; opacity: 0.92;">${escapeHtml(file.summary || file.headline || "")}</p>
    <p style="font-size: 0.94rem; line-height: 1.8; margin-bottom: 2rem; opacity: 0.72;">${escapeHtml(file.system_role_note || file.learning_moment || "")}</p>

    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2rem; margin-bottom: 1rem; opacity: 0.7;">Visible anchors</h3>
    <div class="modal-code">${escapeHtml((file.anchors || []).join("\\n") || "No top-level anchors surfaced for this file.")}</div>

    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2rem; margin-bottom: 1rem; opacity: 0.7;">Pulls from</h3>
    ${listMarkup(file.upstream_paths, "This file starts mostly on its own.")}

    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2rem; margin-bottom: 1rem; opacity: 0.7;">Feeds into</h3>
    ${listMarkup(file.downstream_paths, "No downstream internal surfaces were captured.")}

    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2rem; margin-bottom: 1rem; opacity: 0.7;">Proof surfaces</h3>
    ${listMarkup(file.proof_paths, "No direct proof surfaces were captured for this file yet.")}

    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2rem; margin-bottom: 1rem; opacity: 0.7;">Runnable surfaces</h3>
    ${listMarkup(file.execution_paths, "This file is not directly exposed through a runnable surface.")}

    <div style="margin-top: 2rem; display: flex; gap: 0.8rem; flex-wrap: wrap;">
      <a class="modal-link" href="${escapeHtml(file.github_url || atlasPayload.repository?.url || '#')}" target="_blank" rel="noreferrer">Open on GitHub</a>
    </div>
  `;

  modal.classList.add("active");
}

function closeModal() {
  modal?.classList.remove("active");
}

function renderNodes(files) {
  if (!nodeCluster) return;
  nodeCluster.innerHTML = "";
  nodeModels = [];

  const sizeBase = window.innerWidth < 900 ? 96 : 122;

  files.forEach((file, index) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "node";
    node.dataset.fileId = file.id;

    const size = sizeBase + (index % 3) * 10;
    node.style.width = `${size}px`;
    node.style.height = `${size}px`;
    node.style.marginLeft = `${-size / 2}px`;
    node.style.marginTop = `${-size / 2}px`;

    node.innerHTML = `
      <div class="node-inner">
        <div class="node-front">
          <div class="node-label">${escapeHtml(file.name || file.path)}</div>
          <div class="node-info">${escapeHtml(file.category_label || file.system_role || "")}</div>
        </div>
        <div class="node-back">
          <div class="node-info">${escapeHtml(file.system_role || file.category_label || "")}</div>
          <div style="margin-top: 8px; font-size: 0.65rem; opacity: 0.7;">${escapeHtml(file.tags?.slice(0, 2).join(" • ") || file.difficulty || "")}</div>
        </div>
      </div>
    `;

    node.addEventListener("mouseenter", () => hydrateField(file));
    node.addEventListener("focus", () => hydrateField(file));
    node.addEventListener("click", () => openModal(file));
    nodeCluster.appendChild(node);

    nodeModels.push({
      file,
      element: node,
      angle: (Math.PI * 2 * index) / Math.max(files.length, 1),
      radius: 145 + (index % 3) * 48 + (index > 5 ? 18 : 0),
      speed: (0.00016 + (index % 4) * 0.00003) * (index % 2 === 0 ? 1 : -1),
      yScale: 0.64 + (index % 3) * 0.08,
      phase: index * 0.7,
      size,
    });
  });

  if (currentFrame) cancelAnimationFrame(currentFrame);
  currentFrame = requestAnimationFrame(animateNodes);
}

function animateNodes(time) {
  if (!nodeCluster || !nodeModels.length) return;
  const width = nodeCluster.clientWidth || 600;
  const height = nodeCluster.clientHeight || 600;
  const centerX = width / 2;
  const centerY = height / 2;

  nodeModels.forEach((model, index) => {
    const angle = model.angle + time * model.speed;
    const x = centerX + Math.cos(angle) * model.radius;
    const y = centerY + Math.sin(angle * model.yScale + model.phase) * (model.radius * 0.52);
    const scale = model.file.id === activeFileId ? 1.12 : 1;
    model.element.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale})`;
    model.element.style.zIndex = String(100 + Math.round(y));
  });

  currentFrame = requestAnimationFrame(animateNodes);
}

function shuffleField() {
  const modes = ["all", "hub", "proof", "execution"];
  const nextMode = modes[(modes.indexOf(currentMode) + 1) % modes.length];
  applyMode(nextMode);
}

async function loadCatalog() {
  try {
    const response = await fetch(catalogUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Unable to load catalog at ${catalogUrl}`);
    }
    atlasPayload = await response.json();
  } catch (_error) {
    atlasPayload = fallbackCatalog;
  }
}

async function init() {
  await loadCatalog();
  renderStats(atlasPayload);
  if (repoLink) {
    repoLink.href = atlasPayload.repository?.url || fallbackCatalog.repository.url;
  }
  applyMode("all");

  shuffleButton?.addEventListener("click", shuffleField);
  hubButton?.addEventListener("click", () => applyMode("hub"));
  proofButton?.addEventListener("click", () => applyMode("proof"));
  executionButton?.addEventListener("click", () => applyMode("execution"));

  signalTape?.addEventListener("click", (event) => {
    const target = event.target.closest("[data-file-id]");
    if (!target) return;
    const file = activePool.find((item) => item.id === target.dataset.fileId);
    if (file) {
      hydrateField(file);
      openModal(file);
    }
  });

  modal?.addEventListener("click", (event) => {
    if (event.target === modal) closeModal();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
  });
}

window.closeModal = closeModal;
window.addEventListener("resize", () => {
  if (activePool.length) {
    renderNodes(activePool);
  }
});

init();
