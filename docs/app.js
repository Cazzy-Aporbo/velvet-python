const state = {
  payload: null,
  files: [],
  filtered: [],
  categoryMetrics: [],
  activeCategory: "all",
  query: "",
  sortMode: "depth",
  spotlightIndex: 0,
};

const elements = {
  heroStats: document.querySelector("#hero-stats"),
  trackRail: document.querySelector("#track-rail"),
  clusterGrid: document.querySelector("#cluster-grid"),
  methodGrid: document.querySelector("#method-grid"),
  signalNotes: document.querySelector("#signal-notes"),
  orbitStrip: document.querySelector("#orbit-strip"),
  filterBar: document.querySelector("#filter-bar"),
  catalogGrid: document.querySelector("#catalog-grid"),
  catalogMeta: document.querySelector("#catalog-meta"),
  guideTitle: document.querySelector("#guide-title"),
  guideNote: document.querySelector("#guide-note"),
  guideTags: document.querySelector("#guide-tags"),
  guideList: document.querySelector("#guide-list"),
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
  orbitNodes: document.querySelector("#orbit-nodes"),
  noteLattice: document.querySelector("#note-lattice"),
  footerNoteField: document.querySelector("#footer-note-field"),
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

const categoryAccents = {
  core: "rose",
  cli: "cyan",
  scripts: "tangerine",
  tests: "gold",
  environments: "periwinkle",
  foundations: "lavender",
  games: "cyan",
  applied: "rose",
  labs: "tangerine",
  "legacy-env": "periwinkle",
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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[character] || character;
  });
}

function topValues(values, limit = 6) {
  const counts = new Map();
  values.filter(Boolean).forEach((value) => {
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit);
}

function getCategoryMetric(categoryKey) {
  return state.categoryMetrics.find((item) => item.key === categoryKey);
}

function buildCategoryMetrics() {
  state.categoryMetrics = state.payload.categories
    .filter((category) => category.file_count > 0)
    .map((category) => {
      const files = state.files.filter((file) => file.category_key === category.key);
      const averageDepth = Math.round(
        files.reduce((total, file) => total + file.depth_index, 0) / Math.max(files.length, 1),
      );
      const averageLines = Math.round(
        files.reduce((total, file) => total + file.stats.nonempty_lines, 0) / Math.max(files.length, 1),
      );
      const topTag = topValues(files.flatMap((file) => file.tags), 1)[0]?.[0] || "Python";
      const dominantDifficulty = topValues(files.map((file) => file.difficulty), 1)[0]?.[0] || "steady";

      return {
        ...category,
        averageDepth,
        averageLines,
        topTag,
        dominantDifficulty,
        accent: categoryAccents[category.key] || "lavender",
      };
    });
}

function scrollToCatalog() {
  document.querySelector("#catalog")?.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

function applyCategory(categoryKey, shouldScroll = false) {
  state.activeCategory = categoryKey;
  renderFilterBar();
  renderCatalog();
  if (shouldScroll) {
    scrollToCatalog();
  }
}

function applyTrack(trackId) {
  const track = state.payload.tracks.find((item) => item.id === trackId);
  if (!track) return;
  const firstVisibleCategory = track.categories.find((categoryKey) =>
    state.categoryMetrics.some((metric) => metric.key === categoryKey),
  );
  applyCategory(firstVisibleCategory || "all", true);
}

function sortedFiles(files) {
  const output = [...files];
  if (state.sortMode === "name") {
    output.sort((left, right) => left.title.localeCompare(right.title));
    return output;
  }
  if (state.sortMode === "category") {
    output.sort((left, right) => {
      const category = left.category_label.localeCompare(right.category_label);
      if (category !== 0) return category;
      return left.title.localeCompare(right.title);
    });
    return output;
  }

  output.sort((left, right) => {
    if (right.depth_index !== left.depth_index) {
      return right.depth_index - left.depth_index;
    }
    return right.stats.nonempty_lines - left.stats.nonempty_lines;
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
        file.headline,
        file.why_it_matters,
        file.learning_moment,
        file.best_for,
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

function renderStats() {
  if (!elements.heroStats) return;

  const stats = state.payload.stats;
  const testCount = getCategoryMetric("tests")?.file_count || 0;
  const runnableCount = (getCategoryMetric("scripts")?.file_count || 0) + (getCategoryMetric("cli")?.file_count || 0);
  const uniqueTagCount = new Set(state.files.flatMap((file) => file.tags)).size;

  const chips = [
    ["Python files", numberFormat(stats.python_file_count), "mapped into the atlas"],
    ["Proof surfaces", numberFormat(testCount), "tests and checks"],
    ["Runnable paths", numberFormat(runnableCount), "scripts and CLI entrypoints"],
    ["Idea signals", numberFormat(uniqueTagCount), "distinct tags across the repo"],
  ];

  elements.heroStats.innerHTML = chips
    .map(
      ([label, value, note]) => `
        <article class="metric-chip">
          <strong>${value}</strong>
          <span>${label}</span>
          <small>${note}</small>
        </article>
      `,
    )
    .join("");
}

function renderTracks() {
  if (!elements.trackRail) return;

  elements.trackRail.innerHTML = state.payload.tracks
    .map((track) => {
      const categoryLabels = track.categories
        .map((categoryKey) => state.payload.categories.find((item) => item.key === categoryKey)?.label || categoryKey)
        .join(" · ");

      return `
        <article class="track-card glass-panel" data-reveal>
          <p class="eyebrow">${numberFormat(track.file_count)} files</p>
          <h3>${track.title}</h3>
          <p>${track.description}</p>
          <div class="tag-row">
            ${track.categories
              .map((categoryKey) => {
                const label = state.payload.categories.find((item) => item.key === categoryKey)?.label || categoryKey;
                return `<span>${label}</span>`;
              })
              .join("")}
          </div>
          <p class="track-note">${categoryLabels}</p>
          <div class="card-footer">
            <span class="difficulty-pill">route</span>
            <button class="card-link card-button" type="button" data-track="${track.id}">
              Open route
            </button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderClusters() {
  if (!elements.clusterGrid) return;

  const maxCount = Math.max(...state.categoryMetrics.map((item) => item.file_count), 1);
  elements.clusterGrid.innerHTML = state.categoryMetrics
    .map(
      (category) => `
        <button
          class="cluster-card cluster-button glass-panel accent-${category.accent}"
          type="button"
          data-category="${category.key}"
          data-scroll="catalog"
          data-reveal
        >
          <div class="cluster-topline">
            <p class="eyebrow">${numberFormat(category.file_count)} files</p>
            <span class="depth-pill">Avg depth ${category.averageDepth}</span>
          </div>
          <h3>${category.label}</h3>
          <p>${category.description}</p>
          <div class="cluster-meta">
            <span>${category.topTag}</span>
            <span>${category.dominantDifficulty}</span>
            <span>${numberFormat(category.averageLines)} lines avg</span>
          </div>
          <div class="cluster-bar">
            <span style="width:${Math.max(14, (category.file_count / maxCount) * 100)}%"></span>
          </div>
        </button>
      `,
    )
    .join("");
}

function renderMethodGrid() {
  if (!elements.methodGrid) return;

  const totalAnchors = state.files.reduce((total, file) => total + file.anchors.length, 0);
  const commentLines = state.files.reduce((total, file) => total + file.stats.comment_lines, 0);
  const nonemptyLines = state.files.reduce((total, file) => total + file.stats.nonempty_lines, 0);
  const averageDepth = Math.round(
    state.files.reduce((total, file) => total + file.depth_index, 0) / Math.max(state.files.length, 1),
  );
  const commentShare = Math.round((commentLines / Math.max(nonemptyLines, 1)) * 100);

  const cards = [
    {
      label: "Average depth",
      value: averageDepth,
      note: "how dense the files are on average",
    },
    {
      label: "Visible anchors",
      value: numberFormat(totalAnchors),
      note: "top-level functions and classes surfaced in the drawer",
    },
    {
      label: "Comment share",
      value: `${commentShare}%`,
      note: "how much of the repo speaks in plain annotations",
    },
    {
      label: "Core function count",
      value: numberFormat(state.payload.stats.function_count),
      note: "named functions captured across the codebase",
    },
  ];

  elements.methodGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="method-card">
          <strong>${card.value}</strong>
          <span>${card.label}</span>
          <small>${card.note}</small>
        </article>
      `,
    )
    .join("");
}

function renderSignalNotes() {
  if (!elements.signalNotes) return;

  const notes = [
    {
      title: "Start small",
      copy:
        state.payload.tracks.find((track) => track.id === "foundations")?.description ||
        "Use the smaller files to get your footing before you move into the heavier modules.",
      category: "foundations",
    },
    {
      title: "Follow the proof",
      copy:
        state.payload.categories.find((category) => category.key === "tests")?.description ||
        "The tests show where behavior is pinned down and how the repo explains failure.",
      category: "tests",
    },
    {
      title: "Move into systems",
      copy:
        state.payload.tracks.find((track) => track.id === "systems")?.description ||
        "Core modules and scripts turn local ideas into reproducible workflows.",
      category: "core",
    },
  ];

  elements.signalNotes.innerHTML = notes
    .map(
      (note) => `
        <article class="signal-note-card">
          <h4>${note.title}</h4>
          <p>${note.copy}</p>
          <button class="card-link card-button" type="button" data-category="${note.category}" data-scroll="catalog">
            Open files
          </button>
        </article>
      `,
    )
    .join("");
}

function renderOrbitStrip() {
  if (!elements.orbitStrip) return;

  elements.orbitStrip.innerHTML = state.payload.featured
    .map(
      (file) => `
        <article class="orbit-card glass-panel accent-${categoryAccents[file.category_key] || "lavender"}" data-file-id="${file.id}" data-reveal>
          <div class="card-topline">
            <div>
              <p class="eyebrow">${file.category_label}</p>
              <p class="card-path">${file.path}</p>
            </div>
            <span class="depth-pill">Depth ${file.depth_index}</span>
          </div>
          <h3>${file.title}</h3>
          <p>${file.headline}</p>
          <div class="orbit-stats">
            <span>${numberFormat(file.stats.nonempty_lines)} lines</span>
            <span>${numberFormat(file.stats.function_count)} functions</span>
            <span>${file.difficulty}</span>
          </div>
          <div class="tag-row">
            ${file.tags.slice(0, 4).map((tag) => `<span>${tag}</span>`).join("")}
          </div>
          <div class="card-footer">
            <span class="difficulty-pill">${file.best_for}</span>
            <button class="card-link card-button" type="button" data-file-id="${file.id}">
              Open file
            </button>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderOrbitNodes() {
  if (!elements.orbitNodes) return;

  const orbitFiles = state.payload.featured.slice(0, 6);
  elements.orbitNodes.innerHTML = orbitFiles
    .map(
      (file) => `
        <button
          class="node accent-${categoryAccents[file.category_key] || "lavender"}"
          type="button"
          data-file-id="${file.id}"
          aria-label="Open ${escapeHtml(file.title)}"
        >
          <span class="node-inner">
            <span class="node-front">
              <span class="node-metric">${file.depth_index}</span>
              <span class="node-label">${escapeHtml(file.title)}</span>
              <span class="node-subtle">${escapeHtml(file.category_label)}</span>
            </span>
            <span class="node-back">
              <span class="node-subtle">${escapeHtml(file.difficulty)}</span>
              <span class="node-info">${escapeHtml(file.headline)}</span>
            </span>
          </span>
        </button>
      `,
    )
    .join("");
}

function renderFilterBar() {
  if (!elements.filterBar) return;

  const buttons = [
    { key: "all", label: "All files", count: state.files.length, accent: "lavender" },
    ...state.categoryMetrics.map((category) => ({
      key: category.key,
      label: category.label,
      count: category.file_count,
      accent: category.accent,
    })),
  ];

  elements.filterBar.innerHTML = buttons
    .map(
      (button) => `
        <button
          class="filter-button accent-${button.accent} ${button.key === state.activeCategory ? "active" : ""}"
          type="button"
          data-category="${button.key}"
        >
          ${button.label} <span class="count">${numberFormat(button.count)}</span>
        </button>
      `,
    )
    .join("");
}

function renderCatalogMeta() {
  if (!elements.catalogMeta) return;

  const categoryLabel =
    state.activeCategory === "all"
      ? "the whole repository"
      : getCategoryMetric(state.activeCategory)?.label || state.activeCategory;

  const note = state.query
    ? `Filtered by "${escapeHtml(state.query)}" inside ${categoryLabel}.`
    : `Showing ${categoryLabel}.`;

  elements.catalogMeta.innerHTML = `
    <span>${numberFormat(state.filtered.length)} of ${numberFormat(state.files.length)} files visible</span>
    <span>${note}</span>
  `;
}

function cardMarkup(file) {
  return `
    <article class="catalog-card glass-panel accent-${categoryAccents[file.category_key] || "lavender"}" data-file-id="${file.id}" data-reveal>
      <div class="card-topline">
        <div>
          <p class="eyebrow">${file.category_label}</p>
          <p class="card-path">${file.path}</p>
        </div>
        <span class="depth-pill">Depth ${file.depth_index}</span>
      </div>
      <h3>${file.title}</h3>
      <p>${file.headline}</p>
      <div class="catalog-stats">
        <span>${numberFormat(file.stats.nonempty_lines)} lines</span>
        <span>${numberFormat(file.stats.function_count)} functions</span>
        <span>${numberFormat(file.stats.class_count)} classes</span>
      </div>
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

function renderCatalogGuide() {
  if (!elements.guideTitle || !elements.guideNote || !elements.guideTags || !elements.guideList) return;

  const categoryLabel =
    state.activeCategory === "all"
      ? "Whole atlas"
      : getCategoryMetric(state.activeCategory)?.label || state.activeCategory;

  elements.guideTitle.textContent = categoryLabel;
  elements.guideNote.textContent = state.query
    ? `The view is narrowed by "${state.query}". Open a file below to move from search into detail.`
    : `Use the cards to open a file drawer, or shift categories to see how the teaching surface changes across the repository.`;

  const topTags = topValues(state.filtered.flatMap((file) => file.tags), 6).map(([tag]) => tag);
  elements.guideTags.innerHTML = topTags.length
    ? topTags.map((tag) => `<span>${tag}</span>`).join("")
    : "<span>No tags surfaced for this view yet.</span>";

  const quickFiles = state.filtered.slice(0, 5);
  elements.guideList.innerHTML = quickFiles.length
    ? quickFiles
        .map(
          (file) => `
            <button class="guide-button" type="button" data-file-id="${file.id}">
              <strong>${file.title}</strong>
              <span>${file.path}</span>
            </button>
          `,
        )
        .join("")
    : `
        <div class="empty-guide">
          <strong>No files matched this view.</strong>
          <span>Try clearing the search or switching categories.</span>
        </div>
      `;
}

function renderCatalog() {
  filterFiles();
  renderCatalogMeta();
  renderCatalogGuide();

  if (!elements.catalogGrid) return;

  if (!state.filtered.length) {
    elements.catalogGrid.innerHTML = `
      <article class="catalog-card glass-panel empty-state">
        <h3>No files matched this view</h3>
        <p>Try widening the search, changing the sort, or moving back to the full atlas.</p>
      </article>
    `;
    return;
  }

  elements.catalogGrid.innerHTML = state.filtered.map(cardMarkup).join("");
  revealVisible();
}

function renderSpotlight() {
  const featured = state.payload.featured;
  if (!featured.length) return;

  const file = featured[state.spotlightIndex % featured.length];

  if (elements.spotlightTitle) elements.spotlightTitle.textContent = file.title;
  if (elements.spotlightPath) elements.spotlightPath.textContent = file.path;
  if (elements.spotlightSummary) elements.spotlightSummary.textContent = file.headline;
  if (elements.spotlightWhy) elements.spotlightWhy.textContent = file.why_it_matters;
  if (elements.spotlightLink) {
    elements.spotlightLink.href = file.github_url;
    elements.spotlightLink.textContent = "Open on GitHub";
  }
  if (elements.spotlightTags) {
    elements.spotlightTags.innerHTML = file.tags
      .slice(0, 5)
      .map((tag) => `<span>${tag}</span>`)
      .join("");
  }
  if (elements.spotlightStats) {
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
  }

  if (elements.openSpotlight) {
    elements.openSpotlight.dataset.fileId = file.id;
  }
}

function renderGlyphLattice() {
  if (!elements.noteLattice) return;

  const glyphs = [
    "def",
    "class",
    "pytest",
    "seed",
    "schema",
    "plot",
    "fit",
    "audit",
    "->",
    "λ",
    "∑",
    "dict",
    "list",
    "grid",
    "loop",
    "check",
  ];
  const tags = topValues(state.payload.featured.flatMap((file) => file.tags), 8).map(([tag]) => tag.toLowerCase());
  const tokens = [...glyphs, ...tags].slice(0, 18);

  elements.noteLattice.innerHTML = tokens
    .map((token, index) => {
      const column = index % 3;
      const row = Math.floor(index / 3);
      const x = 14 + column * 26 + ((row % 2) * 4);
      const y = 12 + row * 12;
      const delay = (index * 0.45).toFixed(2);
      const duration = (6 + (index % 5) * 1.2).toFixed(2);

      return `
        <span
          class="lattice-glyph"
          style="--x:${x}%; --y:${y}%; --delay:${delay}s; --duration:${duration}s;"
        >${escapeHtml(token)}</span>
      `;
    })
    .join("");
}

function renderFooterField() {
  if (!elements.footerNoteField) return;

  const sequence = [
    "signal",
    "proof",
    "rerun",
    "compare",
    "inspect",
    "learn",
    "trace",
    "shape",
  ];
  const tagTokens = topValues(state.files.flatMap((file) => file.tags), 12).map(([tag]) => tag.toLowerCase());
  const difficultyTokens = topValues(state.files.map((file) => file.difficulty), 4).map(([tag]) => tag.toLowerCase());
  const tokens = [...sequence, ...tagTokens, ...difficultyTokens].slice(0, 18);

  elements.footerNoteField.innerHTML = tokens
    .map((token, index) => {
      const column = index % 6;
      const row = Math.floor(index / 6);
      const x = 10 + column * 16 + ((row % 2) * 2.5);
      const y = 18 + row * 24 + ((column % 2) * 2);
      const delay = (index * 0.38).toFixed(2);
      const duration = (7.5 + (index % 5) * 1.1).toFixed(2);

      return `
        <span
          class="footer-note"
          style="--x:${x}%; --y:${y}%; --delay:${delay}s; --duration:${duration}s;"
        >${escapeHtml(token)}</span>
      `;
    })
    .join("");
}

function renderDrawer(file) {
  if (!elements.drawerCategory) return;

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
  if (!file || !elements.drawer) return;

  renderDrawer(file);
  elements.drawer.classList.add("is-open");
  elements.drawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeDrawer() {
  if (!elements.drawer) return;
  elements.drawer.classList.remove("is-open");
  elements.drawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function renderGeneratedAt() {
  if (!elements.generatedAt) return;
  elements.generatedAt.textContent = `Catalog generated ${prettyDate(state.payload.generated_at)}`;
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
  { threshold: 0.12 },
);

function animateDepthField(featured) {
  const canvas = elements.canvas;
  if (!canvas || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const context = canvas.getContext("2d");
  if (!context) return;

  const palette = {
    core: "rgba(127, 228, 245, 0.84)",
    cli: "rgba(255, 178, 135, 0.82)",
    scripts: "rgba(243, 165, 207, 0.82)",
    tests: "rgba(255, 216, 170, 0.8)",
    environments: "rgba(152, 185, 255, 0.84)",
    foundations: "rgba(183, 156, 255, 0.84)",
    games: "rgba(127, 228, 245, 0.84)",
    applied: "rgba(243, 165, 207, 0.84)",
    labs: "rgba(255, 178, 135, 0.78)",
    "legacy-env": "rgba(152, 185, 255, 0.78)",
  };

  const deviceScale = Math.min(window.devicePixelRatio || 1, 1.6);
  let pointerX = 0;
  let pointerY = 0;
  let lastPaint = 0;

  const nodes = featured.slice(0, 12).map((file, index) => ({
    file,
    x: 0,
    y: 0,
    baseX: 0,
    baseY: 0,
    radius: 1.6 + (file.depth_index / 100) * 3.6,
    drift: 0.22 + (index % 5) * 0.08,
    phase: index * 0.9,
  }));

  function resize() {
    canvas.width = Math.round(window.innerWidth * deviceScale);
    canvas.height = Math.round(window.innerHeight * deviceScale);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    context.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);

    nodes.forEach((node, index) => {
      const row = Math.floor(index / 4);
      const col = index % 4;
      node.baseX = window.innerWidth * (0.18 + col * 0.19);
      node.baseY = window.innerHeight * (0.18 + row * 0.18);
    });
  }

  function draw(time) {
    if (time - lastPaint < 32) {
      window.requestAnimationFrame(draw);
      return;
    }
    lastPaint = time;

    const t = time * 0.00033;
    context.clearRect(0, 0, window.innerWidth, window.innerHeight);

    nodes.forEach((node, index) => {
      const driftX = (pointerX - window.innerWidth / 2) * (0.0012 + (index % 3) * 0.0003);
      const driftY = (pointerY - window.innerHeight / 2) * (0.0011 + (index % 3) * 0.0003);
      node.x = node.baseX + Math.cos(t + node.phase) * 34 * node.drift + driftX;
      node.y = node.baseY + Math.sin(t * 1.15 + node.phase) * 28 * node.drift + driftY;
    });

    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const left = nodes[i];
        const right = nodes[j];
        const distance = Math.hypot(left.x - right.x, left.y - right.y);
        if (distance > 260) continue;
        context.strokeStyle = `rgba(183, 156, 255, ${0.15 - distance / 2800})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(left.x, left.y);
        context.lineTo(right.x, right.y);
        context.stroke();
      }
    }

    nodes.forEach((node) => {
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
    });

    window.requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize, { passive: true });
  window.addEventListener(
    "pointermove",
    (event) => {
      pointerX = event.clientX;
      pointerY = event.clientY;
    },
    { passive: true },
  );

  resize();
  window.requestAnimationFrame(draw);
}

function autoRotateSpotlight() {
  window.setInterval(() => {
    if (!state.payload?.featured?.length) return;
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
  }, 7000);
}

function bindControls() {
  elements.search?.addEventListener("input", (event) => {
    state.query = event.target.value;
    renderCatalog();
  });

  elements.sortSelect?.addEventListener("change", (event) => {
    state.sortMode = event.target.value;
    renderCatalog();
  });

  elements.surpriseButton?.addEventListener("click", () => {
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
    applyCategory(state.payload.featured[state.spotlightIndex].category_key, false);
  });

  elements.nextFeatured?.addEventListener("click", () => {
    state.spotlightIndex = (state.spotlightIndex + 1) % state.payload.featured.length;
    renderSpotlight();
  });

  elements.drawerClose?.addEventListener("click", closeDrawer);
  elements.drawerBackdrop?.addEventListener("click", closeDrawer);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });

  document.addEventListener("click", (event) => {
    const fileNode = event.target.closest("[data-file-id]");
    if (fileNode) {
      openDrawer(fileNode.dataset.fileId);
      return;
    }

    const trackNode = event.target.closest("[data-track]");
    if (trackNode) {
      applyTrack(trackNode.dataset.track);
      return;
    }

    const categoryNode = event.target.closest("[data-category]");
    if (categoryNode) {
      const shouldScroll = categoryNode.dataset.scroll === "catalog";
      applyCategory(categoryNode.dataset.category, shouldScroll);
    }
  });
}

async function init() {
  try {
    state.payload = await loadCatalog();
    state.files = state.payload.files;
    buildCategoryMetrics();
    renderStats();
    renderTracks();
    renderClusters();
    renderMethodGrid();
    renderSignalNotes();
    renderOrbitStrip();
    renderOrbitNodes();
    renderFilterBar();
    renderGeneratedAt();
    renderCatalog();
    renderSpotlight();
    renderGlyphLattice();
    renderFooterField();
    bindControls();
    autoRotateSpotlight();
    animateDepthField(state.payload.featured);
    revealVisible();
  } catch (error) {
    if (elements.catalogMeta) {
      elements.catalogMeta.textContent = error.message;
    }
  }
}

init();
