const state = {
  payload: null,
  files: [],
  filtered: [],
  activeCategory: "all",
  query: "",
  sortMode: "depth",
  spotlightIndex: 0,
};

const elements = {
  heroStats: document.querySelector("#hero-stats"),
  trackRail: document.querySelector("#track-rail"),
  clusterGrid: document.querySelector("#cluster-grid"),
  depthRail: document.querySelector("#depth-rail"),
  filterBar: document.querySelector("#filter-bar"),
  catalogGrid: document.querySelector("#catalog-grid"),
  catalogMeta: document.querySelector("#catalog-meta"),
  search: document.querySelector("#catalog-search"),
  sortSelect: document.querySelector("#sort-select"),
  surpriseButton: document.querySelector("#surprise-button"),
  generatedAt: document.querySelector("#generated-at"),
  nextFeatured: document.querySelector("#next-featured"),
  openSpotlight: document.querySelector("#open-spotlight"),
  spotlightTitle: document.querySelector("#spotlight-title"),
  spotlightPath: document.querySelector("#spotlight-path"),
  spotlightSummary: document.querySelector("#spotlight-summary"),
  spotlightTags: document.querySelector("#spotlight-tags"),
  spotlightStats: document.querySelector("#spotlight-stats"),
  spotlightWhy: document.querySelector("#spotlight-why"),
  spotlightLink: document.querySelector("#spotlight-link"),
  drawer: document.querySelector("#detail-drawer"),
  drawerBackdrop: document.querySelector("#drawer-backdrop"),
  drawerClose: document.querySelector("#drawer-close"),
  drawerCategory: document.querySelector("#drawer-category"),
  drawerTitle: document.querySelector("#drawer-title"),
  drawerPath: document.querySelector("#drawer-path"),
  drawerSummary: document.querySelector("#drawer-summary"),
  drawerTags: document.querySelector("#drawer-tags"),
  drawerStats: document.querySelector("#drawer-stats"),
  drawerWhy: document.querySelector("#drawer-why"),
  drawerLearning: document.querySelector("#drawer-learning"),
  drawerBestFor: document.querySelector("#drawer-best-for"),
  drawerAnchors: document.querySelector("#drawer-anchors"),
  drawerLink: document.querySelector("#drawer-link"),
  canvas: document.querySelector("#depth-field"),
};

async function loadCatalog() {
  const response = await fetch("./catalog.json");
  if (!response.ok) {
    throw new Error(`Unable to load catalog.json (${response.status})`);
  }
  return response.json();
}

function numberFormat(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function prettyDate(value) {
  try {
    return new Date(value).toLocaleString(undefined, {
      dateStyle: "long",
      timeStyle: "short",
    });
  } catch {
    return value;
  }
}

function renderStats() {
  const { stats } = state.payload;
  const chips = [
    ["Python files", numberFormat(stats.python_file_count)],
    ["Functions", numberFormat(stats.function_count)],
    ["Classes", numberFormat(stats.class_count)],
    ["Total lines", numberFormat(stats.total_lines)],
  ];
  elements.heroStats.innerHTML = chips
    .map(
      ([label, value]) => `
        <article class="metric-chip">
          <strong>${value}</strong>
          <span>${label}</span>
        </article>
      `,
    )
    .join("");
}

function renderTracks() {
  elements.trackRail.innerHTML = state.payload.tracks
    .map(
      (track) => `
        <article class="track-card glass-panel" data-reveal>
          <p class="eyebrow">${numberFormat(track.file_count)} files</p>
          <h3>${track.title}</h3>
          <p>${track.description}</p>
          <div class="tag-row">
            ${track.categories
              .map((category) => {
                const found = state.payload.categories.find((item) => item.key === category);
                return `<span>${found ? found.label : category}</span>`;
              })
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderClusters() {
  const maxCount = Math.max(...state.payload.categories.map((item) => item.file_count), 1);
  elements.clusterGrid.innerHTML = state.payload.categories
    .filter((category) => category.file_count > 0)
    .map(
      (category) => `
        <article class="cluster-card glass-panel" data-category="${category.key}" data-reveal>
          <p class="eyebrow">${numberFormat(category.file_count)} files</p>
          <h3>${category.label}</h3>
          <p>${category.description}</p>
          <div class="cluster-bar"><span style="width:${Math.max(
            12,
            (category.file_count / maxCount) * 100,
          )}%"></span></div>
        </article>
      `,
    )
    .join("");
}

function renderDepthLeaders() {
  elements.depthRail.innerHTML = state.payload.depth_leaders
    .map(
      (file, index) => `
        <article class="depth-card glass-panel" data-file-id="${file.id}" data-reveal>
          <span class="depth-rank">${index + 1}</span>
          <p class="eyebrow">${file.category_label}</p>
          <h3>${file.title}</h3>
          <p>${file.headline}</p>
          <div class="card-footer">
            <span class="depth-pill">Depth ${file.depth_index}</span>
            <button class="card-link card-button" type="button" data-file-id="${file.id}">
              Open file
            </button>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderFilterBar() {
  const buttons = [
    { key: "all", label: "All files", count: state.files.length },
    ...state.payload.categories
      .filter((category) => category.file_count > 0)
      .map((category) => ({
        key: category.key,
        label: category.label,
        count: category.file_count,
      })),
  ];

  elements.filterBar.innerHTML = buttons
    .map(
      (button) => `
        <button
          class="filter-button ${button.key === state.activeCategory ? "active" : ""}"
          type="button"
          data-category="${button.key}"
        >
          ${button.label} <span class="count">${numberFormat(button.count)}</span>
        </button>
      `,
    )
    .join("");
}

function sortedFiles(files) {
  const output = [...files];
  if (state.sortMode === "name") {
    output.sort((a, b) => a.title.localeCompare(b.title));
    return output;
  }
  if (state.sortMode === "category") {
    output.sort((a, b) => {
      const category = a.category_label.localeCompare(b.category_label);
      if (category !== 0) return category;
      return a.title.localeCompare(b.title);
    });
    return output;
  }
  output.sort((a, b) => {
    if (b.depth_index !== a.depth_index) return b.depth_index - a.depth_index;
    return b.stats.nonempty_lines - a.stats.nonempty_lines;
  });
  return output;
}

function filterFiles() {
  const query = state.query.trim().toLowerCase();
  let next = [...state.files];
  if (state.activeCategory !== "all") {
    next = next.filter((file) => file.category_key === state.activeCategory);
  }
  if (query) {
    next = next.filter((file) => {
      const haystack = [
        file.title,
        file.path,
        file.summary,
        file.why_it_matters,
        file.learning_moment,
        file.tags.join(" "),
        file.anchors.join(" "),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }
  state.filtered = sortedFiles(next);
}

function renderCatalogMeta() {
  const summary = `${numberFormat(state.filtered.length)} of ${numberFormat(state.files.length)} files visible`;
  const note =
    state.activeCategory === "all"
      ? "Atlas view is open across the whole repository."
      : `Focused on ${state.payload.categories.find((category) => category.key === state.activeCategory)?.label ?? state.activeCategory}.`;
  elements.catalogMeta.innerHTML = `<span>${summary}</span><span>${note}</span>`;
}

function cardMarkup(file) {
  return `
    <article class="catalog-card glass-panel" data-file-id="${file.id}" data-reveal>
      <div class="card-topline">
        <div>
          <p class="eyebrow">${file.category_label}</p>
          <p class="card-path">${file.path}</p>
        </div>
        <span class="depth-pill">Depth ${file.depth_index}</span>
      </div>
      <h3>${file.title}</h3>
      <p>${file.headline}</p>
      <div class="tag-row">
        ${file.tags.slice(0, 4).map((tag) => `<span>${tag}</span>`).join("")}
      </div>
      <div class="card-footer">
        <span class="difficulty-pill">${file.difficulty}</span>
        <button class="card-link card-button" type="button" data-file-id="${file.id}">
          Open file
        </button>
      </div>
    </article>
  `;
}

function renderCatalog() {
  filterFiles();
  renderCatalogMeta();
  elements.catalogGrid.innerHTML = state.filtered.map(cardMarkup).join("");
  bindCardButtons();
  revealVisible();
}

function renderSpotlight() {
  const featured = state.payload.featured;
  if (!featured.length) return;
  const file = featured[state.spotlightIndex % featured.length];

  elements.spotlightTitle.textContent = file.title;
  elements.spotlightPath.textContent = file.path;
  elements.spotlightSummary.textContent = file.headline;
  elements.spotlightWhy.textContent = file.why_it_matters;
  elements.spotlightLink.href = file.github_url;
  elements.spotlightLink.textContent = "Open on GitHub";
  elements.spotlightTags.innerHTML = file.tags
    .slice(0, 5)
    .map((tag) => `<span>${tag}</span>`)
    .join("");

  const stats = [
    ["Depth index", file.depth_index],
    ["Functions", file.stats.function_count],
    ["Classes", file.stats.class_count],
    ["Lines", file.stats.nonempty_lines],
  ];
  elements.spotlightStats.innerHTML = stats
    .map(
      ([label, value]) => `
        <div class="stat-box">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");

  elements.openSpotlight.onclick = () => openDrawer(file.id);
}

function renderDrawer(file) {
  elements.drawerCategory.textContent = file.category_label;
  elements.drawerTitle.textContent = file.title;
  elements.drawerPath.textContent = file.path;
  elements.drawerSummary.textContent = file.summary;
  elements.drawerWhy.textContent = file.why_it_matters;
  elements.drawerLearning.textContent = file.learning_moment;
  elements.drawerBestFor.textContent = file.best_for;
  elements.drawerLink.href = file.github_url;
  elements.drawerTags.innerHTML = file.tags.map((tag) => `<span>${tag}</span>`).join("");
  elements.drawerStats.innerHTML = [
    ["Depth index", file.depth_index],
    ["Difficulty", file.difficulty],
    ["Functions", file.stats.function_count],
    ["Classes", file.stats.class_count],
    ["Imports", file.stats.import_count],
    ["Non-empty lines", file.stats.nonempty_lines],
  ]
    .map(
      ([label, value]) => `
        <div class="stat-box">
          <strong>${value}</strong>
          <span>${label}</span>
        </div>
      `,
    )
    .join("");
  elements.drawerAnchors.innerHTML = file.anchors.length
    ? file.anchors.map((anchor) => `<li>${anchor}</li>`).join("")
    : "<li>No top-level symbols were captured for this file.</li>";
}

function openDrawer(fileId) {
  const file =
    state.files.find((candidate) => candidate.id === fileId) ||
    state.payload.featured.find((candidate) => candidate.id === fileId);
  if (!file) return;
  renderDrawer(file);
  elements.drawer.classList.add("is-open");
  elements.drawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeDrawer() {
  elements.drawer.classList.remove("is-open");
  elements.drawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function bindCardButtons() {
  document.querySelectorAll(".card-button, .catalog-card, .depth-card").forEach((node) => {
    node.addEventListener("click", (event) => {
      const target = event.currentTarget;
      const fileId = target.dataset.fileId;
      if (!fileId) return;
      openDrawer(fileId);
    });
  });
}

function bindControls() {
  elements.filterBar.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) return;
    state.activeCategory = button.dataset.category;
    renderFilterBar();
    renderCatalog();
  });

  elements.search.addEventListener("input", (event) => {
    state.query = event.target.value;
    renderCatalog();
  });

  elements.sortSelect.addEventListener("change", (event) => {
    state.sortMode = event.target.value;
    renderCatalog();
  });

  elements.surpriseButton.addEventListener("click", () => {
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
    const spotlightCategory = state.payload.featured[state.spotlightIndex].category_key;
    state.activeCategory = spotlightCategory;
    renderFilterBar();
    renderCatalog();
  });

  elements.nextFeatured.addEventListener("click", () => {
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
  });

  elements.drawerClose.addEventListener("click", closeDrawer);
  elements.drawerBackdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
}

function revealVisible() {
  document.querySelectorAll("[data-reveal]").forEach((node) => observer.observe(node));
}

const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.1 },
);

function renderGeneratedAt() {
  elements.generatedAt.textContent = `Catalog generated ${prettyDate(state.payload.generated_at)}`;
}

function animateDepthField(featured) {
  const canvas = elements.canvas;
  const context = canvas.getContext("2d");
  if (!context) return;

  const palette = {
    core: "rgba(127, 228, 245, 0.88)",
    cli: "rgba(255, 178, 135, 0.86)",
    scripts: "rgba(243, 165, 207, 0.84)",
    tests: "rgba(255, 216, 170, 0.82)",
    environments: "rgba(152, 185, 255, 0.88)",
    foundations: "rgba(183, 156, 255, 0.86)",
    games: "rgba(127, 228, 245, 0.86)",
    applied: "rgba(243, 165, 207, 0.86)",
    labs: "rgba(255, 178, 135, 0.82)",
    "legacy-env": "rgba(152, 185, 255, 0.8)",
  };

  let width = 0;
  let height = 0;
  let pointerX = 0;
  let pointerY = 0;

  const nodes = featured.slice(0, 18).map((file, index) => ({
    file,
    x: 0,
    y: 0,
    baseX: 0,
    baseY: 0,
    radius: 1.8 + (file.depth_index / 100) * 4.6,
    drift: 0.2 + (index % 5) * 0.08,
    phase: index * 0.9,
  }));

  function resize() {
    width = canvas.width = window.innerWidth * window.devicePixelRatio;
    height = canvas.height = window.innerHeight * window.devicePixelRatio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    context.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);

    nodes.forEach((node, index) => {
      const row = Math.floor(index / 6);
      const col = index % 6;
      node.baseX = window.innerWidth * (0.14 + col * 0.14);
      node.baseY = window.innerHeight * (0.18 + row * 0.2);
    });
  }

  function draw(time) {
    const t = time * 0.00035;
    context.clearRect(0, 0, window.innerWidth, window.innerHeight);

    nodes.forEach((node, index) => {
      const depthShiftX = (pointerX - window.innerWidth / 2) * (0.002 + (index % 4) * 0.0006);
      const depthShiftY = (pointerY - window.innerHeight / 2) * (0.0016 + (index % 4) * 0.0004);
      node.x = node.baseX + Math.cos(t + node.phase) * 24 * node.drift + depthShiftX;
      node.y = node.baseY + Math.sin(t * 1.1 + node.phase) * 22 * node.drift + depthShiftY;
    });

    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (distance > 240) continue;
        context.strokeStyle = `rgba(183, 156, 255, ${0.16 - distance / 2600})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
    }

    for (const node of nodes) {
      const color = palette[node.file.category_key] || "rgba(255,255,255,0.8)";
      const glow = context.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 10);
      glow.addColorStop(0, color);
      glow.addColorStop(1, "rgba(255,255,255,0)");
      context.fillStyle = glow;
      context.beginPath();
      context.arc(node.x, node.y, node.radius * 10, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = color;
      context.beginPath();
      context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      context.fill();
    }

    window.requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", (event) => {
    pointerX = event.clientX;
    pointerY = event.clientY;
  });

  resize();
  window.requestAnimationFrame(draw);
}

function autoRotateSpotlight() {
  window.setInterval(() => {
    if (!state.payload?.featured?.length) return;
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
  }, 6500);
}

async function init() {
  try {
    state.payload = await loadCatalog();
    state.files = state.payload.files;
    renderStats();
    renderTracks();
    renderClusters();
    renderDepthLeaders();
    renderFilterBar();
    renderGeneratedAt();
    renderCatalog();
    renderSpotlight();
    bindControls();
    autoRotateSpotlight();
    animateDepthField(state.payload.featured);
    revealVisible();
  } catch (error) {
    elements.catalogMeta.textContent = error.message;
  }
}

init();
