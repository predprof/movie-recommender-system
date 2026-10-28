/**
 * week6/graph.js - HW6 graph / Personalized PageRank layer.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN, then the vendored d3 v7 build, then
 * `data.js`, then this file, then `script.js`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="vendor/d3.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="graph.js"></script>
 *   <script src="script.js"></script>
 *
 * `week6/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `genreNames`, `SPLIT_EXCLUSIONS` and `splitByTimestamp()`.
 * This file does not mutate them.
 *
 * IMPLEMENTED: training graph construction, equal outgoing-link probabilities,
 * global PageRank and PPR, including zero-degree mass redistribution.
 *
 * PROVIDED for you (scaffolding): the `REFERENCE_GRAPH` constant with the two
 * reference vectors and both rankings from readme.md section 4.1 (plus the
 * derived `REFERENCE_GRAPH.transition`, see below), the graph
 * accessors (`GRAPH`, `setGraph`, `getGraph`), the d3 graph view
 * (`renderGraphView`), the 4-node fixture teaching demo (`fixtureStep` /
 * `fixtureReset`), and the `#graph-summary-body` renderer.
 *
 * GRACEFUL DEGRADATION: the fixture demo and the graph view work before any
 * stub is implemented, so the page loads and renders with no console errors.
 *
 * SCOPE NOTE: d3 is used ONLY by the graph view. MovieLens propagation uses
 * TensorFlow.js tensor operations on `R` and its transpose (readme.md section
 * 3.5); the fixture teaching demo uses a small plain-JavaScript loop.
 *
 * @module week6/graph
 */

// ---------------------------------------------------------------------------
// Typedefs (documentation only - no runtime effect)
// ---------------------------------------------------------------------------

/**
 * A single rating record as produced by `data.js` / `splitByTimestamp()`.
 *
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating    1..5 (MovieLens scale).
 * @property {number} timestamp Unix seconds.
 */

/**
 * The bipartite user-movie graph returned by `buildGraph`.
 *
 * @typedef {Object} Graph
 * @property {number} nUsers        Number of distinct user ids observed in TRAIN.
 * @property {number} nMovies       Number of distinct movie ids observed in TRAIN.
 * @property {number[]} userIds     User ids, ascending; index = local user index.
 * @property {number[]} movieIds    Movie ids, ascending; index = local movie index.
 * @property {Map<number,number>} userIndexOf  userId -> local user index [0, nUsers).
 * @property {Map<number,number>} movieIndexOf movieId -> local movie index [0, nMovies).
 * @property {number} nodeCount     nUsers + nMovies (no padding node).
 * @property {function(number):number} nodeOfUser    userId -> global node index.
 * @property {function(number):number} movieOfNode   global node index -> movieId.
 * @property {function(number):boolean} isUserNode   global node index -> user?
 * @property {Float64Array} degree  Bipartite degree over node index.
 * @property {number[]} zeroDegreeUsers  User ids with degree 0.
 * @property {number[]} zeroDegreeMovies Movie ids with degree 0.
 * @property {number} positiveEdges Number of positive training edges (1 entries in R).
 * @property {number[][]} R         R[u][i] = 1 iff user u rated movie i >= 4.0
 *   in TRAINING, else 0. Shape [nUsers][nMovies].
 */

/**
 * The row-normalized transition returned by `rowNormalize`.
 *
 * @typedef {Object} RowNormalized
 * @property {number[][]} Q             Lazily materialized transition for
 *   inspection/tests; nonzero rows sum to 1, zero-degree rows are all zero.
 * @property {boolean[]} danglingMask   true for zero-degree node indices.
 * @property {Float64Array} inverseDegree Reciprocal degree, zero for dangling
 *   nodes; used with R/R-transpose without allocating the square Q on MovieLens.
 */

/**
 * A PageRank / PPR result.
 *
 * @typedef {Object} PageRankResult
 * @property {Float64Array} p     Node probabilities, node order, sums to ~1.
 * @property {number} iterations  Number of updates performed.
 * @property {number} residual    Final L1 difference between consecutive distributions.
 * @property {boolean} converged  true when residual < tol before the cap.
 */

/**
 * An explicit transition supplied to `pageRank` / `personalizedPageRank`
 * through `opts.transition`.
 *
 * Use this to propagate on a caller-provided graph instead of the global
 * `GRAPH` (the HW6 harness uses it to run the implementation on the 4-node
 * reference fixture, which has no MovieLens graph). When present it replaces
 * the global-graph lookup for that call.
 *
 * @typedef {Object} TransitionSpec
 * @property {number[][]} Q          Row-normalized transition; `Q[i][j]` is the
 *   probability of moving from node `i` to node `j`. Square (n x n). A
 *   positive-degree row sums to 1; a zero-degree row is all zero. Already
 *   normalized, so the implementation must NOT re-normalize it.
 * @property {Uint8Array} danglingMask Nonzero (1) exactly at the zero-degree
 *   node indices, indexed by position in the same node order as `Q`.
 * @property {string[]} [nodeLabels] Optional labels for display and tests only;
 *   the propagation never reads them. `REFERENCE_GRAPH.transition.nodeLabels`
 *   is `['A', 'B', 'C', 'D']`.
 */

/**
 * Options for `renderGraphView`.
 *
 * @typedef {Object} GraphViewOptions
 * @property {("fixture"|"ego")} [mode]   Which graph to draw.
 * @property {number|null} [seedUserId]   Seed user for the "ego" subgraph.
 * @property {("degree"|"score")} [colorBy] Node fill encoding.
 * @property {number} [maxNodes]          Cap on drawn nodes (default 120).
 * @property {function(number):number} [getScore] Node index -> score, used when
 *   colorBy === "score".
 */

// ---------------------------------------------------------------------------
// Reference graph (provided, read-only by convention)
//
// Exactly the readme.md section 4.1 fixture: node order A, B, C, D; directed
// edges A -> B, A -> C, B -> C, C -> A, D -> C; no dangling nodes. The two
// reference vectors and both rankings are reproduced verbatim from readme.md
// section 4.1.
// ---------------------------------------------------------------------------

/**
 * The 4-node reference graph from readme.md section 4.1, plus its reference
 * results. `nodes` assigns A, B to the left (user) column and C, D to the right
 * (movie) column so the fixture can reuse the same deterministic two-column
 * layout as the data view. Treat this object as read-only.
 */
const REFERENCE_GRAPH = {
  nodeOrder: ["A", "B", "C", "D"],
  nodes: [
    { index: 0, label: "A", kind: "user" },
    { index: 1, label: "B", kind: "user" },
    { index: 2, label: "C", kind: "movie" },
    { index: 3, label: "D", kind: "movie" }
  ],
  // [from, to] pairs in node index space (A = 0, B = 1, C = 2, D = 3).
  edges: [
    [0, 1], // A -> B
    [0, 2], // A -> C
    [1, 2], // B -> C
    [2, 0], // C -> A
    [3, 2]  // D -> C
  ],
  edgeLabels: ["A -> B", "A -> C", "B -> C", "C -> A", "D -> C"],
  hasDanglingNodes: false,
  reference: {
    // alpha = 0.85, uniform restart over all four nodes.
    uniform: {
      alpha: 0.85,
      restart: "uniform",
      p: [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000],
      ranking: [2, 0, 1, 3],
      rankingLabel: "C > A > B > D"
    },
    // alpha = 0.85, restart only at A.
    restartAtA: {
      alpha: 0.85,
      restart: "A",
      restartIndex: 0,
      p: [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000],
      ranking: [0, 2, 1, 3],
      rankingLabel: "A > C > B > D"
    }
  }
};

// PROVIDED: the fixture's row-normalized transition, DERIVED from
// `REFERENCE_GRAPH.edges` (never a hand-typed matrix). `Q` is what
// `pageRank` / `personalizedPageRank` must consume when the harness passes
// `opts.transition`; `danglingMask` is a Uint8Array, 1 at zero-degree node
// indices (all zeros here, since A/B/C/D all have an outgoing link).
//
// The load-time self-test below is deliberately NON-FATAL: it records state in
// `REFERENCE_GRAPH.transitionSelfTest` instead of throwing, so a malformed
// fixture can never break page load. It adds no global name.
(function () {
  const fixture = graphFixtureTransition(); // derives P from `edges`
  const n = REFERENCE_GRAPH.nodeOrder.length;
  const Q = fixture.P.map(function (row) { return row.slice(); });
  const danglingMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (fixture.out[i] === 0) danglingMask[i] = 1;
  }

  REFERENCE_GRAPH.transition = {
    Q: Q,
    danglingMask: danglingMask,
    nodeLabels: REFERENCE_GRAPH.nodeOrder.slice()
  };

  let maxRowSumError = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Q[i][j];
    maxRowSumError = Math.max(maxRowSumError, Math.abs(sum - 1));
  }
  let danglingCount = 0;
  for (let i = 0; i < n; i++) {
    if (danglingMask[i]) danglingCount += 1;
  }
  REFERENCE_GRAPH.transitionSelfTest = {
    rows: n,
    cols: Q.length ? Q[0].length : 0,
    maxRowSumError: maxRowSumError,
    rowsSumToOne: maxRowSumError < 1e-12,
    danglingCount: danglingCount,
    danglingMaskAllZero: danglingCount === 0
  };
})();

// ---------------------------------------------------------------------------
// Graph-view theme (private)
//
// Canvas is #FFFFFF; the categorical pair is Okabe-Ito blue/orange (red-green
// colour-blind safe), and scores use a single-hue sequential blue ramp.
// ---------------------------------------------------------------------------

/**
 * Fixed visual parameters for the graph view. Private to graph.js.
 */
const graphTheme = {
  width: 1000,
  height: 420,
  margin: { top: 34, right: 110, bottom: 26, left: 110 },
  userColor: "#0072B2",
  movieColor: "#E69F00",
  edgeColor: "#b9c0ca",
  nodeStroke: "#ffffff",
  zeroStroke: "#1c2128",
  labelColor: "#1c2128",
  noteColor: "#5b6472",
  minRadius: 6,
  maxRadius: 22,
  rampLight: [232, 243, 252],
  rampDark: [8, 48, 96],
  tooltipId: "graph-tooltip"
};

// ---------------------------------------------------------------------------
// Module state (public: GRAPH_VIEW_STATE)
// ---------------------------------------------------------------------------

/**
 * Small module-level state for the graph view. Mutated only by the rendering
 * and fixture-demo code in this file.
 */
const GRAPH_VIEW_STATE = {
  mode: "ego",
  seedUserId: null,
  colorBy: "degree",
  maxNodes: 120,
  lastRender: null,
  fixture: {
    iteration: 0,
    p: null,
    residual: null,
    restart: "uniform",
    alpha: 0.85
  }
};

// ---------------------------------------------------------------------------
// Graph accessors (public)
// ---------------------------------------------------------------------------

/**
 * The current graph, or null before `buildGraph` has been wired in.
 *
 * @type {Graph|null}
 */
let GRAPH = null;

// Preserve the current view and score accessor when the panel is resized.
const graphLastDrawings = new Map();
let graphResizeObserver = null;

/**
 * Replace the current graph and refresh the dependent UI.
 *
 * `script.js` should call this (rather than assigning `GRAPH` directly) so the
 * graph summary and the graph view re-render whenever the graph changes. A
 * `null` argument clears the graph and restores the empty states.
 *
 * @param {Graph|null} g
 * @returns {Graph|null} the stored graph.
 */
function setGraph(g) {
  GRAPH = g || null;
  try {
    graphRenderSummary(GRAPH);
  } catch (error) {
    graphSetLegendMessage("Graph summary failed: " + graphErrorMessage(error));
  }
  try {
    renderGraphView({ mode: "ego" });
  } catch (error) {
    graphSetLegendMessage("Graph view failed: " + graphErrorMessage(error));
  }
  return GRAPH;
}

/**
 * Read the current graph without changing any state.
 *
 * @returns {Graph|null}
 */
function getGraph() {
  return GRAPH;
}

// ---------------------------------------------------------------------------
// Graph construction, transition normalization and probability propagation
// ---------------------------------------------------------------------------

/**
 * Build the bipartite user-movie graph from the TRAINING split.
 *
 * Rules (readme.md section 3.1):
 *  - The node universe is the set of user ids and movie ids observed in
 *    `train` ONLY. Map ids to contiguous indices explicitly; never add a
 *    padding node (no index-0 "dummy" row).
 *  - Node order is users first, then movies:
 *      user u  -> global node userIndexOf.get(u) in [0, nUsers)
 *      movie m -> global node nUsers + movieIndexOf.get(m) in [nUsers, nUsers + nMovies)
 *    The graph view relies on this exact order (it treats node < nUsers as a
 *    user node and indexes degree by the global node index).
 *  - R[u][i] = 1 iff user u rated movie i with rating >= 4.0 in TRAINING, else
 *    0. A zero is "no positive edge", not a negative opinion.
 *  - degree is the bipartite degree over BOTH node kinds.
 *  - Repeated positive ratings for one pair produce one binary edge. Each
 *    edge is traversable in both directions, represented by R and R-transpose.
 *  - Keep nodes with no positive edges and report them as zero-degree nodes.
 *
 * Assumptions: `train` is an array of {userId, movieId, rating, timestamp}
 * records with numeric IDs and ratings; ratings >= 4.0 are positives. Missing
 * ratings remain unobserved; no negative examples or ratings are imputed.
 * The input is not modified. A non-array `train` returns an empty graph.
 * Mapping helpers return -1 for unknown users or invalid movie-node indices.
 *
 * @param {Rating[]} train
 * @returns {Graph}
 */
function buildGraph(train) {
  const records = Array.isArray(train) ? train : [];
  // Define the node universe before filtering positives, so users and movies
  // observed only in ratings below 4 still appear as zero-degree nodes.
  const userIds = Array.from(new Set(records.map(row => row.userId))).sort((a, b) => a - b);
  const movieIds = Array.from(new Set(records.map(row => row.movieId))).sort((a, b) => a - b);
  const nUsers = userIds.length;
  const nMovies = movieIds.length;
  const nodeCount = nUsers + nMovies;
  const userIndexOf = new Map(userIds.map((id, index) => [id, index]));
  const movieIndexOf = new Map(movieIds.map((id, index) => [id, index]));
  const R = Array.from({ length: nUsers }, () => new Array(nMovies).fill(0));
  const degree = new Float64Array(nodeCount);
  let positiveEdges = 0;

  for (const row of records) {
    if (!(row.rating >= 4.0)) continue;
    const u = userIndexOf.get(row.userId);
    const i = movieIndexOf.get(row.movieId);
    if (R[u][i] === 1) continue;
    R[u][i] = 1;
    degree[u]++;
    degree[nUsers + i]++;
    positiveEdges++;
  }

  return {
    nUsers,
    nMovies,
    userIds,
    movieIds,
    userIndexOf,
    movieIndexOf,
    nodeCount,
    nodeOfUser(userId) {
      const index = userIndexOf.get(Number(userId));
      return index === undefined ? -1 : index;
    },
    movieOfNode(node) {
      const index = Number(node) - nUsers;
      return Number.isInteger(index) && index >= 0 && index < nMovies ? movieIds[index] : -1;
    },
    isUserNode(node) {
      const index = Number(node);
      return Number.isInteger(index) && index >= 0 && index < nUsers;
    },
    degree,
    zeroDegreeUsers: userIds.filter((id, u) => degree[u] === 0),
    zeroDegreeMovies: movieIds.filter((id, i) => degree[nUsers + i] === 0),
    positiveEdges,
    R
  };
}

/**
 * Row-normalize the transition matrix and record the zero-degree nodes.
 *
 * For each positive-degree node, divide its row by its degree so the
 * row sums to 1; leave zero-degree rows all-zero and mark them in
 * `danglingMask`. The dangling mass must be handled in the update exactly as in
 * readme.md section 3.3:
 *
 *   p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * where d is the probability mass currently sitting on zero-degree nodes.
 *
 * Q is a lazy inspection view. The TensorFlow.js MovieLens path uses R,
 * R-transpose and inverseDegree directly, avoiding a full square transition.
 * Reading Q materializes its plain JavaScript rows for small-graph tests.
 *
 * Assumptions: `graph` comes from `buildGraph` (node order users then movies).
 * Failure cases: a null/missing graph returns { Q: [], danglingMask: [] }.
 *
 * @param {Graph} graph
 * @returns {RowNormalized}
 */
function rowNormalize(graph) {
  if (!graph) return { Q: [], danglingMask: [], inverseDegree: new Float64Array(0) };
  const { nUsers, nMovies, R } = graph;
  if (!Number.isInteger(nUsers) || nUsers < 0 || !Number.isInteger(nMovies) || nMovies < 0 ||
      !Array.isArray(R) || R.length !== nUsers) {
    throw new Error("rowNormalize requires a bipartite graph with a rectangular R matrix.");
  }
  const n = nUsers + nMovies;
  const degree = new Float64Array(n);
  for (let u = 0; u < nUsers; u++) {
    if (!Array.isArray(R[u]) || R[u].length !== nMovies) {
      throw new Error("rowNormalize: R dimensions do not match the graph.");
    }
    for (let i = 0; i < nMovies; i++) {
      const link = R[u][i];
      if (link !== 0 && link !== 1) throw new Error("rowNormalize requires binary positive edges.");
      degree[u] += link;
      degree[nUsers + i] += link;
    }
  }
  const danglingMask = Array.from(degree, value => value === 0);
  const inverseDegree = Float64Array.from(degree, value => value > 0 ? 1 / value : 0);
  let Q = null;
  return {
    danglingMask,
    inverseDegree,
    get Q() {
      if (Q === null) {
        Q = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let u = 0; u < nUsers; u++) {
          for (let i = 0; i < nMovies; i++) {
            if (R[u][i]) {
              Q[u][nUsers + i] = inverseDegree[u];
              Q[nUsers + i][u] = inverseDegree[nUsers + i];
            }
          }
        }
      }
      return Q;
    }
  };
}

/**
 * Global PageRank with a uniform restart distribution.
 *
 * Power iteration with a uniform restart vector v (readme.md sections 3.2-3.4),
 * using TensorFlow.js for the batched propagation:
 *
 *   p_0 = v ;  p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * Initialize p to v. Stop when the L1 difference between consecutive
 * distributions is below `tol`, or after `maxIter` iterations; an iteration cap
 * is NOT proof of convergence, so `converged` must be false when the cap is
 * hit without meeting tolerance. Do not force any node's score to equal the
 * restart probability.
 *
 * TRANSITION SOURCE:
 *  - `opts.transition` present (a {@link TransitionSpec}, e.g.
 *    `REFERENCE_GRAPH.transition`): propagate on THAT matrix and do not require
 *    the global `GRAPH`. `Q` is already row-normalized, so do not re-normalize
 *    it; read `danglingMask` by position (`danglingMask[i]` is 1 iff node `i` is
 *    zero-degree). This is how the HW6 harness runs your implementation on the
 *    4-node reference fixture.
 *  - `opts.transition` absent: behave exactly as for MovieLens - require the
 *    global `GRAPH` (from `buildGraph` + `rowNormalize`) and build the
 *    transition from its `R` / `degree`.
 *  - Neither present: throw a clear Error; never silently fall back to a
 *    uniform distribution.
 *
 * NODE-INDEX CONVENTION: node indices are global positions. For the MovieLens
 * graph they follow `buildGraph` (users `[0, nUsers)`, then movies
 * `[nUsers, nUsers + nMovies)`, no padding node). For a supplied
 * `opts.transition` they follow `Q`'s own order. `p[i]` is the probability of
 * node `i` in that same order.
 *
 * The zero-degree policy must be identical here and in `personalizedPageRank`.
 *
 * @param {number} alpha Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Defaults tol = 1e-6, maxIter = 500, transition absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}`: `p` is a
 *   Float64Array over node indices that sums to ~1; `iterations` is the number
 *   of updates performed; `residual` is the final L1 difference
 *   `sum_j |p_next[j] - p[j]|`; `converged` is true only when `residual < tol`
 *   within the iteration cap. Computation uses TensorFlow.js float32 tensors;
 *   the returned Float64Array is a container, not a double-precision solver.
 * @throws {Error} when neither `opts.transition` nor the global `GRAPH` exists.
 */
function pageRank(alpha, opts) {
  return graphPowerIteration(alpha, opts, null);
}

/**
 * Personalized PageRank with a one-hot restart at one user's node.
 *
 * Identical to `pageRank` except the restart vector v is one-hot at
 * one node. It accepts the same `opts.transition` override, the same stopping
 * rule, the same zero-degree policy and the same return shape
 * `{p, iterations, residual, converged}`.
 *
 * RESTART INDEX (node index, not a raw user id): the first argument is a node
 * INDEX and is used directly as the restart node. This is today's caller
 * convention; keep it.
 *  - default MovieLens path: the index is a global node index from `buildGraph`
 *    (a user node is `GRAPH.nodeOfUser(userId)` or equivalently any
 *    `index < GRAPH.nUsers`).
 *  - `opts.transition` path: the index must be in `[0, Q.length)` and refers to
 *    the same node order as `Q` (`REFERENCE_GRAPH.transition.nodeLabels` names
 *    the fixture's nodes for display).
 *
 * FAILURE CASES: an unknown / out-of-range restart index must throw a clear
 * Error naming the index rather than silently falling back to a uniform
 * restart; and when neither `opts.transition` nor the global `GRAPH` is
 * available, throw instead of guessing.
 *
 * @param {number} userId Node index of the restart node (see RESTART INDEX).
 * @param {number} alpha  Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Same options as `pageRank`; defaults tol = 1e-6, maxIter = 500, transition
 *   absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}` (see `pageRank`).
 * @throws {Error} when the restart index is unknown / out of range, or when
 *   neither `opts.transition` nor the global `GRAPH` exists.
 */
function personalizedPageRank(userId, alpha, opts) {
  return graphPowerIteration(alpha, opts, userId);
}

/**
 * Evaluate personal restarts together for a bounded batch of raw user IDs.
 *
 * Each probability column is an independent PPR run, initialized at its user.
 * R / R-transpose propagate the whole batch without a square adjacency matrix.
 * A column is frozen when its own L1 residual falls below tol, so convergence
 * metadata describes that user's stopping point rather than the slowest user.
 * All computations use float32, as in personalizedPageRank; p is returned in a
 * Float64Array container. opts.graph does not replace or mutate the global GRAPH.
 *
 * The optional async onBatch callback runs after the temporary batch tensors
 * have been disposed. Shared graph tensors live only until this call ends.
 * Synchronous GPU readback matches the single-user implementation. Timers yield
 * between groups of iterations and batches, including in background tabs.
 *
 * @param {number[]} userIds Raw training user IDs, in the requested result order.
 * @param {number} alpha Link-following probability in [0, 1].
 * @param {{graph?: Graph, batchSize?: number, tol?: number, maxIter?: number,
 *   onBatch?: function(Object[], number, number): (void|Promise<void>)}} [opts]
 *   batchSize defaults to 32; allowed integer range is 1..128. Defaults for tol
 *   and maxIter are 1e-6 and 500. An empty userIds array returns [].
 * @returns {Promise<Array<{userId: number, p: Float64Array, iterations: number,
 *   residual: number, converged: boolean}>>}
 */
async function personalizedPageRankBatch(userIds, alpha, opts) {
  if (opts !== undefined && (opts === null || typeof opts !== "object" || Array.isArray(opts))) {
    throw new TypeError("PPR batch options must be an object.");
  }
  const options = opts || {};
  const tol = options.tol === undefined ? 1e-6 : options.tol;
  const maxIter = options.maxIter === undefined ? 500 : options.maxIter;
  const batchSize = options.batchSize === undefined ? 32 : options.batchSize;
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new RangeError("alpha must be a finite link-following probability in [0, 1].");
  }
  if (!Number.isFinite(tol) || tol <= 0 || !Number.isInteger(maxIter) || maxIter < 1) {
    throw new RangeError("tol must be positive and maxIter must be a positive integer.");
  }
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 128) {
    throw new RangeError("PPR batchSize must be an integer from 1 to 128.");
  }
  if (options.onBatch !== undefined && typeof options.onBatch !== "function") {
    throw new TypeError("PPR batch onBatch must be a function.");
  }
  if (!Array.isArray(userIds)) throw new TypeError("PPR batch userIds must be an array of raw user IDs.");
  // Snapshot caller-owned arrays before the first await; a progress callback
  // cannot change which users are requested midway through the calculation.
  const ids = userIds.slice();
  if (!ids.length) return [];
  const graph = options.graph === undefined ? GRAPH : options.graph;
  if (!graph || typeof graph.nodeOfUser !== "function") {
    throw new Error("Build a training graph or supply opts.graph before running batched PPR.");
  }
  const normalized = rowNormalize(graph);
  const n = graph.nUsers + graph.nMovies;
  if (!n) throw new Error("Batched PPR requires at least one graph node.");
  const nodes = ids.map(userId => {
    const node = Number.isSafeInteger(userId) ? graph.nodeOfUser(userId) : -1;
    if (!Number.isInteger(node) || node < 0 || node >= graph.nUsers) {
      throw new RangeError("Unknown or invalid raw restart user ID: " + String(userId));
    }
    return node;
  });
  if (typeof tf === "undefined" || !tf.getBackend()) {
    throw new Error("TensorFlow.js must be ready before running batched PPR.");
  }

  let shared = null;
  const results = [];
  try {
    shared = tf.tidy(() => ({
      links: tf.tensor2d(graph.R, [graph.nUsers, graph.nMovies]),
      inverse: tf.tensor2d(Float32Array.from(normalized.inverseDegree), [n, 1]),
      dangling: tf.tensor2d(Float32Array.from(normalized.danglingMask, value => value ? 1 : 0), [n, 1])
    }));
    for (let start = 0; start < ids.length; start += batchSize) {
      const width = Math.min(batchSize, ids.length - start);
      const restartValues = new Float32Array(n * width);
      for (let col = 0; col < width; col++) restartValues[nodes[start + col] * width + col] = 1;
      const active = new Float32Array(width).fill(1);
      const residuals = new Float64Array(width).fill(Infinity);
      const iterations = new Uint32Array(width);
      const converged = new Array(width).fill(false);
      let remaining = width;
      let restart = null;
      let p = null;
      let batchResults;
      try {
        restart = tf.tensor2d(restartValues, [n, width]);
        p = restart.clone();
        for (let k = 0; k < maxIter && remaining > 0; k++) {
          const step = tf.tidy(() => {
            const weighted = p.mul(shared.inverse);
            const userMass = weighted.slice([0, 0], [graph.nUsers, width]);
            const movieMass = weighted.slice([graph.nUsers, 0], [graph.nMovies, width]);
            const incoming = tf.concat([
              tf.matMul(shared.links, movieMass),
              tf.matMul(shared.links, userMass, true, false)
            ], 0);
            // Sum over nodes independently for each restart column. A dangling
            // user's mass consequently returns to that same user, not a peer.
            const danglingMass = p.mul(shared.dangling).sum(0, true);
            const followed = incoming.add(restart.mul(danglingMass));
            // v + alpha * (followed - v) is the same restart equation. Correct
            // only Float32 mass drift before measuring consecutive residuals;
            // this avoids rounding cycles near the 1e-6 stopping threshold.
            const raw = restart.add(followed.sub(restart).mul(alpha));
            // A positive user seed alternates between the two partitions. Its
            // total user mass after t steps is known exactly, independently of
            // the edges: (1 + alpha * (-alpha)^t) / (1 + alpha). Enforce that
            // invariant to remove Float32 rounding cycles at high alpha.
            const userFraction = (1 + alpha * Math.pow(-alpha, k + 1)) / (1 + alpha);
            const targets = Float32Array.from(nodes.slice(start, start + width), node => graph.degree[node] ? userFraction : 1);
            const candidate = graphCorrectPartitionMass(raw, graph.nUsers, targets);
            const activeColumns = tf.tensor2d(active, [1, width]);
            const next = candidate.mul(activeColumns).add(p.mul(tf.onesLike(activeColumns).sub(activeColumns)));
            return { next, residual: next.sub(p).abs().sum(0) };
          });
          p.dispose();
          p = step.next;
          let differences;
          try {
            differences = step.residual.dataSync();
          } finally {
            step.residual.dispose();
          }
          for (let col = 0; col < width; col++) {
            if (!active[col]) continue;
            const residual = differences[col];
            if (!Number.isFinite(residual)) throw new Error("Batched PPR produced a non-finite residual.");
            residuals[col] = residual;
            iterations[col] = k + 1;
            if (residual < tol) {
              converged[col] = true;
              active[col] = 0;
              remaining--;
            }
          }
          if ((k + 1) % 16 === 0 && remaining > 0) await new Promise(resolve => setTimeout(resolve, 0));
        }
        const values = p.dataSync();
        batchResults = Array.from({ length: width }, (_, col) => {
          const probabilities = new Float64Array(n);
          for (let node = 0; node < n; node++) probabilities[node] = values[node * width + col];
          return {
            userId: ids[start + col], p: probabilities,
            iterations: iterations[col], residual: residuals[col], converged: converged[col]
          };
        });
      } finally {
        if (p) p.dispose();
        if (restart) restart.dispose();
      }
      results.push(...batchResults);
      if (options.onBatch) await options.onBatch(batchResults, results.length, ids.length);
      if (start + width < ids.length) await new Promise(resolve => setTimeout(resolve, 0));
    }
    return results;
  } finally {
    if (shared) for (const tensor of Object.values(shared)) tensor.dispose();
  }
}

/** Correct rounding drift using the analytically known bipartite mass totals.
 * Called inside tf.tidy; it changes no transition or restart probabilities.
 */
function graphCorrectPartitionMass(raw, nUsers, userTargets) {
  const width = raw.shape[1];
  const users = raw.slice([0, 0], [nUsers, width]);
  const movies = raw.slice([nUsers, 0], [raw.shape[0] - nUsers, width]);
  const target = tf.tensor2d(userTargets, [1, width]);
  return tf.concat([
    users.div(users.sum(0, true).maximum(1e-20)).mul(target),
    movies.div(movies.sum(0, true).maximum(1e-20)).mul(tf.onesLike(target).sub(target))
  ], 0);
}

/**
 * Shared synchronous TensorFlow.js iteration for global and personal restart.
 *
 * The browser's existing scoring and test interfaces are synchronous, so only
 * the scalar residual and final vector use dataSync(). All probability
 * propagation is batched tensor math. Per-step temporaries are scoped by tidy;
 * all surviving tensors are disposed even when a calculation fails.
 *
 * @param {number} alpha Link-following probability, 0..1.
 * @param {Object} [opts] Tolerance, iteration cap and optional explicit transition.
 * @param {number|null} restartNode Node index; null selects uniform restart.
 * @returns {PageRankResult}
 */
function graphPowerIteration(alpha, opts, restartNode) {
  const options = opts || {};
  const tol = options.tol === undefined ? 1e-6 : options.tol;
  const maxIter = options.maxIter === undefined ? 500 : options.maxIter;
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new RangeError("alpha must be a finite link-following probability in [0, 1].");
  }
  if (!Number.isFinite(tol) || tol <= 0 || !Number.isInteger(maxIter) || maxIter < 1) {
    throw new RangeError("tol must be positive and maxIter must be a positive integer.");
  }
  const explicit = options.transition !== undefined;
  let n;
  let normalized;
  let Q;
  if (explicit) {
    const transition = options.transition;
    Q = transition && transition.Q;
    n = Array.isArray(Q) ? Q.length : 0;
    const mask = transition && transition.danglingMask;
    if (!n || !mask || mask.length !== n || Q.some(row => !Array.isArray(row) || row.length !== n)) {
      throw new Error("transition requires a nonempty square Q and matching danglingMask.");
    }
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (const value of Q[i]) {
        if (!Number.isFinite(value) || value < 0) throw new Error("transition probabilities must be nonnegative and finite.");
        sum += value;
      }
      if (Math.abs(sum - (mask[i] ? 0 : 1)) > 1e-8) {
        throw new Error("transition rows must sum to 1, or 0 for dangling nodes.");
      }
    }
    normalized = { danglingMask: Array.from(mask, Boolean) };
  } else {
    if (!GRAPH) throw new Error("Build a training graph or supply opts.transition before running PageRank.");
    normalized = rowNormalize(GRAPH);
    n = GRAPH.nUsers + GRAPH.nMovies;
    if (!n) throw new Error("PageRank requires at least one graph node.");
  }
  if (restartNode !== null && (!Number.isInteger(restartNode) || restartNode < 0 || restartNode >= n ||
      (!explicit && restartNode >= GRAPH.nUsers))) {
    throw new RangeError("Unknown or out-of-range restart user node index: " + restartNode);
  }
  if (typeof tf === "undefined" || !tf.getBackend()) {
    throw new Error("TensorFlow.js must be ready before running PageRank.");
  }

  const restart = new Float32Array(n);
  if (restartNode === null) restart.fill(1 / n);
  else restart[restartNode] = 1;
  const tensors = tf.tidy(() => ({
    links: explicit ? tf.tensor2d(Q, [n, n]) : tf.tensor2d(GRAPH.R, [GRAPH.nUsers, GRAPH.nMovies]),
    inverse: explicit ? null : tf.tensor2d(Float32Array.from(normalized.inverseDegree), [n, 1]),
    dangling: tf.tensor2d(Float32Array.from(normalized.danglingMask, value => value ? 1 : 0), [n, 1]),
    restart: tf.tensor2d(restart, [n, 1])
  }));
  let p = null;
  let iterations = 0;
  let residual = Infinity;
  let converged = false;
  try {
    p = tensors.restart.clone();
    for (let k = 0; k < maxIter; k++) {
      const step = tf.tidy(() => {
        let incoming;
        if (explicit) {
          incoming = tf.matMul(tensors.links, p, true, false);
        } else {
          // Divide each source's mass equally over its outgoing edges. R sends
          // movie mass to users; R-transpose sends user mass to movies.
          const weighted = p.mul(tensors.inverse);
          const userMass = weighted.slice([0, 0], [GRAPH.nUsers, 1]);
          const movieMass = weighted.slice([GRAPH.nUsers, 0], [GRAPH.nMovies, 1]);
          incoming = tf.concat([
            tf.matMul(tensors.links, movieMass),
            tf.matMul(tensors.links, userMass, true, false)
          ], 0);
        }
        const danglingMass = p.mul(tensors.dangling).sum();
        const followed = incoming.add(tensors.restart.mul(danglingMass));
        const raw = tensors.restart.add(followed.sub(tensors.restart).mul(alpha));
        const next = !explicit && restartNode !== null && GRAPH.degree[restartNode] > 0
          ? graphCorrectPartitionMass(raw, GRAPH.nUsers,
            [(1 + alpha * Math.pow(-alpha, k + 1)) / (1 + alpha)])
          : raw.div(raw.sum());
        return { next, residual: next.sub(p).abs().sum() };
      });
      p.dispose();
      p = step.next;
      try {
        residual = step.residual.dataSync()[0];
      } finally {
        step.residual.dispose();
      }
      iterations = k + 1;
      if (!Number.isFinite(residual)) throw new Error("PageRank produced a non-finite residual.");
      if (residual < tol) {
        converged = true;
        break;
      }
    }
    return { p: Float64Array.from(p.dataSync()), iterations, residual, converged };
  } finally {
    if (p) p.dispose();
    for (const tensor of Object.values(tensors)) if (tensor) tensor.dispose();
  }
}

// ---------------------------------------------------------------------------
// Small helpers (private, all graph-prefixed to avoid collisions)
// ---------------------------------------------------------------------------

/**
 * Readable message from an unknown thrown value.
 *
 * @param {*} error
 * @returns {string}
 */
function graphErrorMessage(error) {
  return String(error && error.message ? error.message : error);
}

/**
 * Escape text before inserting it into HTML (the local copy, so this file does
 * not depend on any helper defined by script.js).
 *
 * @param {*} text
 * @returns {string}
 */
function graphEscape(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read the teaching fixture's independent alpha, falling back to 0.85.
 *
 * @returns {number}
 */
function graphReadAlpha() {
  const el = document.getElementById("fixture-alpha");
  const value = el ? Number(el.value) : Number.NaN;
  return Number.isFinite(value) ? value : 0.85;
}

/**
 * Map t in [0, 1] onto the single-hue sequential blue ramp.
 *
 * @param {number} t
 * @returns {string} an rgb(...) string.
 */
function graphRampColor(t) {
  const n = Number(t);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
  const lo = graphTheme.rampLight;
  const hi = graphTheme.rampDark;
  const mix = (a, b, s) => Math.round(a + (b - a) * s);
  const r = mix(lo[0], hi[0], clamped);
  const g = mix(lo[1], hi[1], clamped);
  const b = mix(lo[2], hi[2], clamped);
  return "rgb(" + r + ", " + g + ", " + b + ")";
}

/**
 * Position lookup with a safe fallback for a missing node.
 *
 * @param {Map<number,{x:number,y:number}>} positions
 * @param {number} node
 * @returns {{x:number,y:number}}
 */
function graphPos(positions, node) {
  const pt = positions.get(node);
  return pt || { x: 0, y: 0 };
}

// ---------------------------------------------------------------------------
// Fixture demo helpers (private)
// ---------------------------------------------------------------------------

/**
 * Out-degree of a fixture node (number of outgoing reference edges).
 *
 * @param {number} index
 * @returns {number}
 */
function graphFixtureOutDegree(index) {
  let d = 0;
  for (const edge of REFERENCE_GRAPH.edges) {
    if (edge[0] === index) d += 1;
  }
  return d;
}

/**
 * Build the row-stochastic transition for the 4-node reference fixture.
 *
 * @returns {{P: number[][], out: number[]}} P[from][to] and the out-degrees.
 */
function graphFixtureTransition() {
  const n = REFERENCE_GRAPH.nodes.length;
  const out = new Array(n).fill(0);
  for (const edge of REFERENCE_GRAPH.edges) out[edge[0]] += 1;

  const P = [];
  for (let i = 0; i < n; i++) P.push(new Array(n).fill(0));
  for (const edge of REFERENCE_GRAPH.edges) {
    const from = edge[0];
    const to = edge[1];
    if (out[from] > 0) P[from][to] = 1 / out[from];
  }
  return { P, out };
}

/**
 * Restart distribution for the fixture: uniform, or one-hot at a named node.
 *
 * @param {string} restart "uniform" or one of A/B/C/D.
 * @returns {number[]}
 */
function graphFixtureRestartVector(restart) {
  const n = REFERENCE_GRAPH.nodes.length;
  const v = new Array(n).fill(0);
  if (!restart || restart === "uniform") {
    v.fill(1 / n);
    return v;
  }
  const index = REFERENCE_GRAPH.nodeOrder.indexOf(restart);
  if (index < 0) {
    v.fill(1 / n);
    return v;
  }
  v[index] = 1;
  return v;
}

/**
 * One plain-JavaScript propagation step on the 4-node fixture.
 *
 * The fixture has no dangling nodes, so the d * v term from readme.md
 * section 3.3 is not needed here. Returns the new distribution and the L1
 * change.
 *
 * @param {number} alpha
 * @param {string} restart
 * @param {number[]} pPrev
 * @returns {{p: number[], residual: number}}
 */
function graphFixturePropagate(alpha, restart, pPrev) {
  const transition = graphFixtureTransition();
  const P = transition.P;
  const n = REFERENCE_GRAPH.nodes.length;
  const v = graphFixtureRestartVector(restart);
  const pNext = new Array(n).fill(0);

  // p_next[j] = alpha * sum_i p[i] * P[i][j] + (1 - alpha) * v[j]
  for (let i = 0; i < n; i++) {
    const mass = pPrev[i];
    if (!mass) continue;
    for (let j = 0; j < n; j++) {
      if (P[i][j] !== 0) pNext[j] += alpha * mass * P[i][j];
    }
  }
  for (let j = 0; j < n; j++) pNext[j] += (1 - alpha) * v[j];

  let residual = 0;
  for (let j = 0; j < n; j++) residual += Math.abs(pNext[j] - pPrev[j]);
  return { p: pNext, residual };
}

// ---------------------------------------------------------------------------
// View builders (private)
// ---------------------------------------------------------------------------

/**
 * Build the drawable view for the 4-node fixture.
 *
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildFixtureView() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || graphFixtureRestartVector(fixture.restart || "uniform");
  const nodes = REFERENCE_GRAPH.nodes.map((nd) => ({
    node: nd.index,
    key: "fixture:" + nd.label,
    label: nd.label,
    kind: nd.kind,
    id: nd.label,
    degree: graphFixtureOutDegree(nd.index),
    score: p[nd.index]
  }));
  const edges = REFERENCE_GRAPH.edges.map((edge) => ({
    source: edge[0],
    target: edge[1],
    directed: true
  }));
  return {
    kind: "fixture",
    nodes: nodes,
    edges: edges,
    omitted: 0,
    nodeCount: nodes.length,
    note: "4-node reference fixture (A, B, C, D) - no dangling nodes"
  };
}

/**
 * Neighbours of a node in the bipartite graph (users <-> movies), or [] when
 * the graph is missing. Uses the users-then-movies node order from `buildGraph`.
 *
 * @param {Graph} graph
 * @param {number} node
 * @returns {number[]}
 */
function graphNeighbors(graph, node) {
  const result = [];
  if (!graph || !graph.R) return result;
  if (graph.isUserNode(node)) {
    const row = graph.R[node];
    if (!row) return result;
    for (let i = 0; i < row.length; i++) {
      if (row[i]) result.push(graph.nUsers + i);
    }
  } else {
    const movieLocal = node - graph.nUsers;
    for (let u = 0; u < graph.nUsers; u++) {
      const row = graph.R[u];
      if (row && row[movieLocal]) result.push(u);
    }
  }
  return result;
}

/**
 * Build the capped ego subgraph for a seed user.
 *
 * Selection: the seed user's node plus its neighbours and their neighbours (a
 * 2-hop walk) in the bipartite graph, then apply the maxNodes cap in hop order,
 * with ascending node indices within each hop. Immediate movie neighbours
 * precede second-hop users, so the cap cannot hide every edge by selecting
 * only users. `omitted` counts reachable nodes left out by the cap.
 *
 * @param {number|null} seedUserId
 * @param {function(number):number} [getScore]
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildEgoView(seedUserId, getScore) {
  const graph = GRAPH;
  if (!graph || !graph.nodeCount) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: 0,
      note: "Build the training graph to show a user's movie neighbourhood."
    };
  }

  const seed = seedUserId === null || seedUserId === undefined ? null : Number(seedUserId);
  if (seed === null || !graph.userIndexOf || !graph.userIndexOf.has(seed)) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: graph.nodeCount,
      note: "Pick a seed user (click a user node) to draw its ego subgraph."
    };
  }

  const maxNodes = Math.max(1, Math.floor(GRAPH_VIEW_STATE.maxNodes) || 120);
  const startNode = graph.nodeOfUser(seed);

  // Unbounded 2-hop reachable set, then cap deterministically.
  const reachable = new Set([startNode]);
  const ordered = [startNode];
  let frontier = [startNode];
  for (let depth = 0; depth < 2; depth++) {
    const next = [];
    for (const node of frontier) {
      for (const nb of graphNeighbors(graph, node)) {
        if (!reachable.has(nb)) {
          reachable.add(nb);
          next.push(nb);
        }
      }
    }
    next.sort((a, b) => a - b);
    ordered.push(...next);
    frontier = next;
    if (reachable.size > 500000) break; // hard safety valve
  }

  const selected = new Set(ordered.slice(0, maxNodes));
  const omitted = reachable.size - selected.size;

  const nodes = [];
  const movieTitles = new Map(
    (typeof movies !== "undefined" ? movies : []).map(movie => [movie.id, movie.title])
  );
  for (const node of ordered) {
    if (!selected.has(node)) continue;
    const isUser = graph.isUserNode(node);
    const id = isUser ? graph.userIds[node] : graph.movieOfNode(node);
    let score = null;
    if (typeof getScore === "function") {
      const raw = Number(getScore(node));
      score = Number.isFinite(raw) ? raw : null;
    }
    nodes.push({
      node: node,
      key: "node:" + node,
      label: isUser ? "U" + id : "M" + id + " · " + (movieTitles.get(id) || "Movie " + id),
      kind: isUser ? "user" : "movie",
      id: id,
      degree: graph.degree ? graph.degree[node] : 0,
      score: score
    });
  }

  const edges = [];
  for (const a of nodes) {
    if (a.kind !== "user") continue;
    const uLocal = graph.userIndexOf.get(a.id);
    const row = graph.R ? graph.R[uLocal] : null;
    if (!row) continue;
    for (const b of nodes) {
      if (b.kind !== "movie") continue;
      const mLocal = graph.movieIndexOf.get(b.id);
      if (mLocal === undefined) continue;
      if (row[mLocal]) edges.push({ source: a.node, target: b.node, directed: false });
    }
  }

  return {
    kind: "ego",
    nodes: nodes,
    edges: edges,
    omitted: omitted,
    nodeCount: graph.nodeCount,
    note: "Ego subgraph of user " + seed + " - drawn " + nodes.length + " of " + graph.nodeCount + " nodes"
  };
}

/**
 * Deterministic two-column bipartite layout: users on the left, movies on the
 * right in ascending node order. MovieLens rows use measured label heights;
 * the small fixture retains an evenly spaced layout.
 *
 * @param {Object[]} nodes
 * @param {number} width
 * @param {number} height
 * @param {{top:number,right:number,bottom:number,left:number}} margin
 * @param {Map<number,number>|null} [rowHeights] Measured space for each node.
 * @returns {{positions: Map<number,{x:number,y:number}>, leftX:number, rightX:number}}
 */
function graphLayoutTwoColumn(nodes, width, height, margin, rowHeights) {
  const positions = new Map();
  const usableHeight = height - margin.top - margin.bottom;
  const leftX = rowHeights ? margin.left + graphTheme.maxRadius + 8 : width * 0.25;
  const rightX = rowHeights ? Math.max(leftX + 64, width * 0.43) : width * 0.75;

  const place = (list, x) => {
    const n = list.length;
    let nextY = margin.top;
    list.forEach((item, i) => {
      const rowHeight = rowHeights ? rowHeights.get(item.node) : 0;
      const y = rowHeights ? nextY + rowHeight / 2 :
        n <= 1 ? margin.top + usableHeight / 2 : margin.top + (usableHeight * i) / (n - 1);
      positions.set(item.node, { x: x, y: y });
      nextY += rowHeight;
    });
  };

  const users = nodes.filter((n) => n.kind === "user").sort((a, b) => a.node - b.node);
  const movies = nodes.filter((n) => n.kind === "movie").sort((a, b) => a.node - b.node);
  const others = nodes
    .filter((n) => n.kind !== "user" && n.kind !== "movie")
    .sort((a, b) => a.node - b.node);

  place(users, leftX);
  place(movies, rightX);
  place(others, leftX);

  return { positions: positions, leftX: leftX, rightX: rightX };
}

// ---------------------------------------------------------------------------
// d3 drawing (private)
// ---------------------------------------------------------------------------

/**
 * Draw one view into its main or fixture SVG canvas.
 *
 * Read-only with respect to GRAPH and any score arrays: it only reads nodes,
 * edges and score values. Zero-degree nodes are drawn as hollow rings with a
 * label so the dangling case is visible.
 *
 * @param {SVGElement} svgEl
 * @param {Object} view result of `graphBuildFixtureView` / `graphBuildEgoView`.
 * @param {{scoreAccessor: (function(number):number|null)|null, colorByScore: boolean}} opts
 * @returns {void}
 */
function graphDrawView(svgEl, view, opts) {
  graphHideTooltip();
  const scoreAccessor = opts && typeof opts.scoreAccessor === "function" ? opts.scoreAccessor : null;
  const colorByScore = !!(opts && opts.colorByScore);
  graphLastDrawings.set(svgEl.id, { svgEl, view, opts });
  // Match SVG units to CSS pixels instead of scaling text with a fixed viewBox.
  const width = svgEl.clientWidth || graphTheme.width;
  let height = 320;
  const margin = { top: view.kind === "fixture" ? 76 : 44, right: 16, bottom: 24, left: 12 };

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", "0 0 " + width + " " + height);
  svg.attr("height", height).style("height", height + "px");
  svg.attr("preserveAspectRatio", "xMidYMid meet");

  const defs = svg.append("defs");
  const markerId = svgEl.id + "-arrowhead";
  defs
    .append("marker")
    .attr("id", markerId)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 10)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", graphTheme.edgeColor);

  if (!view || view.nodes.length === 0) {
    svg
      .append("foreignObject")
      .attr("x", 12).attr("y", 40)
      .attr("width", Math.max(1, width - 24)).attr("height", height - 80)
      .append("xhtml:div")
      .attr("class", "graph-empty-label")
      .text(view && view.note ? view.note : "Nothing to draw yet.");
    return;
  }

  // Resolve scores and the shared scales.
  const scoreByNode = new Map();
  let maxScore = 0;
  for (const n of view.nodes) {
    let score = null;
    if (scoreAccessor) {
      const raw = Number(scoreAccessor(n.node));
      score = Number.isFinite(raw) ? raw : null;
    } else if (n.score !== null && n.score !== undefined) {
      const raw = Number(n.score);
      score = Number.isFinite(raw) ? raw : null;
    }
    scoreByNode.set(n.node, score);
    if (score !== null && score > maxScore) maxScore = score;
  }
  const maxDegree = view.nodes.reduce((m, n) => Math.max(m, Number(n.degree) || 0), 1);

  const normFor = (n) => {
    if (colorByScore) {
      const score = scoreByNode.get(n.node);
      return score !== null && maxScore > 0 ? Math.min(1, score / maxScore) : 0;
    }
    const deg = Math.max(0, Number(n.degree) || 0);
    return maxDegree > 0 ? Math.min(1, deg / maxDegree) : 0;
  };
  const radiusFor = (n) =>
    graphTheme.minRadius +
    (graphTheme.maxRadius - graphTheme.minRadius) * Math.sqrt(normFor(n));

  // Reserve the edge layer first so it remains underneath the nodes.
  const edgeLayer = svg.append("g").attr("class", "graph-edges");
  const rowHeights = new Map();
  const movieX = Math.max(margin.left + graphTheme.maxRadius + 8 + 64, width * 0.43);

  const nodeSel = svg
    .append("g")
    .attr("class", "graph-nodes")
    .selectAll("g.graph-node")
    .data(view.nodes, (d) => d.key)
    .join("g")
    .attr("class", "graph-node")
    .style("cursor", "pointer");

  nodeSel.each(function eachNode(d) {
    const group = d3.select(this);
    const degree = Math.max(0, Number(d.degree) || 0);
    const isZero = degree === 0;
    const isSeed = view.kind === "ego" && d.kind === "user" && d.id === GRAPH_VIEW_STATE.seedUserId;
    const radius = radiusFor(d);
    const score = scoreByNode.get(d.node);
    let rowHeight = 2 * graphTheme.maxRadius + 24;

    let fill = d.kind === "user" ? graphTheme.userColor : graphTheme.movieColor;
    if (colorByScore) fill = graphRampColor(normFor(d));

    group
      .append("circle")
      .attr("r", radius)
      .attr("fill", isZero ? "#ffffff" : fill)
      .attr("stroke", isZero ? graphTheme.zeroStroke : isSeed ? "#0f6e6e" : graphTheme.nodeStroke)
      .attr("stroke-width", isSeed ? 3 : isZero ? 2.5 : 1.5);

    group.append("title").text(d.label);
    if (view.kind === "ego" && d.kind === "movie") {
      const label = group.append("foreignObject")
        .attr("class", "graph-movie-label")
        .attr("x", radius + 10)
        .attr("width", Math.max(1, width - movieX - radius - 10 - margin.right))
        .attr("height", 1000)
        .attr("pointer-events", "none");
      const content = label.append("xhtml:div").attr("class", "graph-node-label").text(d.label);
      const labelHeight = Math.ceil(content.node().getBoundingClientRect().height) + 2;
      label.attr("height", labelHeight).attr("y", -labelHeight / 2);
      rowHeight = Math.max(rowHeight, labelHeight + 24);
    } else {
      group
        .append("text")
        .attr("x", 0)
        .attr("y", -(radius + 4))
        .attr("text-anchor", "middle")
        .attr("font-size", 14)
        .attr("fill", graphTheme.labelColor)
        .attr("pointer-events", "none")
        .text(d.label);
    }
    rowHeights.set(d.node, rowHeight);

    group.on("mouseenter", (event) => graphShowTooltip(event, d, score));
    group.on("mousemove", (event) => graphShowTooltip(event, d, score));
    group.on("mouseleave", graphHideTooltip);
    group.on("click", () => graphHandleNodeClick(d, view));
  });

  if (view.kind === "ego") {
    const columnHeight = kind => view.nodes.filter(node => node.kind === kind)
      .reduce((sum, node) => sum + rowHeights.get(node.node), 0);
    height = Math.max(height, margin.top + margin.bottom + Math.max(columnHeight("user"), columnHeight("movie")));
  }
  const layout = graphLayoutTwoColumn(view.nodes, width, height, margin, view.kind === "ego" ? rowHeights : null);
  const positions = layout.positions;
  svg.attr("viewBox", "0 0 " + width + " " + height)
    .attr("height", height).style("height", height + "px");
  nodeSel.attr("transform", d => {
    const pt = graphPos(positions, d.node);
    return "translate(" + pt.x + "," + pt.y + ")";
  });
  edgeLayer.selectAll("line").data(view.edges).join("line")
    .attr("x1", d => graphPos(positions, d.source).x)
    .attr("y1", d => graphPos(positions, d.source).y)
    .attr("x2", d => graphPos(positions, d.target).x)
    .attr("y2", d => graphPos(positions, d.target).y)
    .attr("stroke", graphTheme.edgeColor).attr("stroke-width", 1)
    .attr("marker-end", d => d.directed ? "url(#" + markerId + ")" : null);
  for (const [label, x] of [["users", layout.leftX], ["movies", layout.rightX]]) {
    svg.append("text").attr("x", x).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", graphTheme.noteColor).text(label);
  }
}

// ---------------------------------------------------------------------------
// Tooltip (private)
// ---------------------------------------------------------------------------

/**
 * Lazily create and return the single shared tooltip element.
 *
 * @returns {HTMLElement}
 */
function graphEnsureTooltip() {
  let el = document.getElementById(graphTheme.tooltipId);
  if (el) return el;
  el = document.createElement("div");
  el.id = graphTheme.tooltipId;
  el.setAttribute("role", "tooltip");
  el.style.position = "fixed";
  el.style.pointerEvents = "none";
  el.style.background = "#1c2128";
  el.style.color = "#ffffff";
  el.style.padding = "0.35rem 0.5rem";
  el.style.borderRadius = "6px";
  el.style.fontSize = "0.78rem";
  el.style.lineHeight = "1.35";
  el.style.zIndex = "10";
  el.style.display = "none";
  document.body.appendChild(el);
  return el;
}

/**
 * Show the hover tooltip for a node: id, kind, degree and current score.
 *
 * @param {MouseEvent} event
 * @param {{id:(number|string), kind:string, node:number, degree:number}} d
 * @param {number|null} score
 * @returns {void}
 */
function graphShowTooltip(event, d, score) {
  const el = graphEnsureTooltip();
  const kind = d.kind === "user" ? "user" : "movie";
  const scoreText = score === null || score === undefined || !Number.isFinite(Number(score))
    ? "-"
    : Number(score).toFixed(6);
  el.textContent =
    d.label + " · " + kind + " " + d.id + " | node " + d.node + " | degree " + d.degree + " | score " + scoreText;
  el.style.display = "block";
  const bounds = el.getBoundingClientRect();
  const left = Math.max(8, Math.min((event ? event.clientX : 0) + 12, window.innerWidth - bounds.width - 8));
  const top = Math.max(8, Math.min((event ? event.clientY : 0) + 12, window.innerHeight - bounds.height - 8));
  el.style.left = left + "px";
  el.style.top = top + "px";
}

/**
 * Hide the hover tooltip.
 *
 * @returns {void}
 */
function graphHideTooltip() {
  const el = document.getElementById(graphTheme.tooltipId);
  if (el) el.style.display = "none";
}

/**
 * Handle a click on a node: in the fixture mode it moves the restart node; in
 * the ego mode it selects a seed user. Never mutates GRAPH or any score array.
 *
 * @param {{label:string, kind:string, id:(number|string)}} d
 * @param {{kind:string}} view
 * @returns {void}
 */
function graphHandleNodeClick(d, view) {
  graphHideTooltip();
  if (view.kind === "fixture") {
    GRAPH_VIEW_STATE.fixture.restart = d.label;
    fixtureReset();
    return;
  }
  if (view.kind === "ego" && d.kind === "user") {
    if (typeof hw6SelectGraphUser === "function") {
      hw6SelectGraphUser(Number(d.id));
    } else if (GRAPH_VIEW_STATE.seedUserId !== Number(d.id)) {
      renderGraphView({ mode: "ego", seedUserId: Number(d.id), colorBy: "degree" });
    }
  }
}

// ---------------------------------------------------------------------------
// Legend and controls (private)
// ---------------------------------------------------------------------------

/**
 * A coloured square swatch for the legend.
 *
 * @param {string} color
 * @returns {string} HTML
 */
function graphSwatch(color) {
  return '<span class="legend-swatch" style="background:' + color + '"></span>';
}

/**
 * A hollow-ring swatch for the zero-degree legend entry.
 *
 * @returns {string} HTML
 */
function graphRingSwatch() {
  return (
    '<span class="legend-swatch" style="background:#ffffff;border:2px solid ' +
    graphTheme.zeroStroke +
    ';border-radius:50%"></span>'
  );
}

/**
 * A gradient swatch showing the single-hue sequential ramp.
 *
 * @returns {string} HTML
 */
function graphRampSwatch() {
  return (
    '<span class="legend-swatch" style="background:linear-gradient(90deg, ' +
    graphRampColor(0) +
    ", " +
    graphRampColor(1) +
    ')"></span>'
  );
}

/**
 * One legend row: swatch (may be empty) + text.
 *
 * @param {string} swatchHtml
 * @param {string} text
 * @returns {string} HTML
 */
function graphLegendItem(swatchHtml, text) {
  return (
    '<span class="legend-item">' + swatchHtml + "<span>" + graphEscape(text) + "</span></span>"
  );
}

/**
 * Replace the legend with a plain message (used for errors / unavailable d3).
 *
 * @param {string} message
 * @returns {void}
 */
function graphSetLegendMessage(message, mode) {
  const target = document.getElementById(mode === "fixture" ? "fixture-graph-legend" : "graph-legend");
  if (!target) return;
  target.innerHTML = '<p class="empty-state">' + graphEscape(message) + "</p>";
}

/**
 * Write one line per colour/size meaning into #graph-legend.
 *
 * @param {Object} view
 * @param {{colorByScore: boolean}} opts
 * @returns {void}
 */
function graphRenderLegend(view, opts) {
  const target = document.getElementById(view && view.kind === "fixture" ? "fixture-graph-legend" : "graph-legend");
  if (!target) return;
  const colorByScore = !!(opts && opts.colorByScore);
  const items = [];

  if (view && view.kind === "fixture") {
    items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (A, B)"));
    items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (C, D)"));
    items.push(graphLegendItem(graphRampSwatch(), "fill = current probability mass (single-hue ramp)"));
    items.push(graphLegendItem(graphRingSwatch(), "hollow ring = zero-degree node"));
    items.push(graphLegendItem("", "arrow = link direction; this fixture has no dangling nodes"));
    items.push(graphLegendItem("", "radius = outgoing degree; click a node to restart there"));
    target.innerHTML = items.join("");
    return;
  }

  if (!view || view.kind === "empty") {
    const note = view && view.note ? view.note : "No graph to draw yet.";
    target.innerHTML = '<p class="empty-state">' + graphEscape(note) + "</p>";
    return;
  }

  items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (left column)"));
  items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (right column)"));
  items.push(graphLegendItem("", "outlined user = restart seed"));
  if (colorByScore) {
    items.push(graphLegendItem(graphRampSwatch(), "fill = score (single-hue sequential ramp)"));
  } else {
    items.push(graphLegendItem("", "fill = node kind; switch colour-by to score to use the ramp"));
  }
  items.push(graphLegendItem(graphRingSwatch(), "hollow ring with label = zero-degree node"));
  items.push(
    graphLegendItem(
      "",
      colorByScore ? "radius scales with score" : "radius scales with degree"
    )
  );
  if (view.omitted > 0) {
    items.push(
      graphLegendItem(
        "",
        view.omitted + " node(s) omitted by the " + GRAPH_VIEW_STATE.maxNodes + "-node cap"
      )
    );
  }
  target.innerHTML = items.join("");
}

/**
 * Fill #graph-controls with the main graph's colour and seed controls.
 *
 * @returns {void}
 */
function graphRenderControls() {
  const target = document.getElementById("graph-controls");
  if (!target) return;
  const state = GRAPH_VIEW_STATE;
  const colorOptions = ["degree", "score"]
    .map(c => '<option value="' + c + '"' + (state.colorBy === c ? " selected" : "") + ">" + c + "</option>")
    .join("");
  const seedText = state.seedUserId === null ? "- (click a user node)" : String(state.seedUserId);
  target.innerHTML =
    '<label class="legend-item">Colour by <select id="graph-colorby-select">' + colorOptions + "</select></label>" +
    '<span class="hint">Seed user: ' + graphEscape(seedText) + "</span>";
  document.getElementById("graph-colorby-select").addEventListener("change", event => {
    if (typeof hw6SafeRenderGraphView === "function") {
      hw6SafeRenderGraphView({ colorBy: event.target.value });
    } else {
      renderGraphView({ mode: "ego", colorBy: event.target.value });
    }
  });
}

// ---------------------------------------------------------------------------
// Graph summary (private)
// ---------------------------------------------------------------------------

/**
 * Numeric value of a count-or-list field (arrays report their length).
 *
 * @param {*} value
 * @returns {number}
 */
function graphCount(value) {
  if (Array.isArray(value)) return value.length;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Render the #graph-summary-body panel: nodes, edges and zero-degree counts.
 *
 * @param {Graph|null} graph
 * @returns {void}
 */
function graphRenderSummary(graph) {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press "Build graph" to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>";
    return;
  }
  const nUsers = graphCount(graph.nUsers);
  const nMovies = graphCount(graph.nMovies);
  const nodeCount = graphCount(graph.nodeCount) || nUsers + nMovies;
  const positiveEdges = graphCount(graph.positiveEdges);
  const zeroUsers = graphCount(graph.zeroDegreeUsers);
  const zeroMovies = graphCount(graph.zeroDegreeMovies);

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>User nodes</dt><dd>" + nUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes</dt><dd>" + nMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Nodes (no padding node)</dt><dd>" + nodeCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training edges</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>";
}

// ---------------------------------------------------------------------------
// Graph view (public: renderGraphView)
// ---------------------------------------------------------------------------

/**
 * Render the main graph or teaching fixture into its own canvas and legend.
 *
 * Uses `opts.mode` to select the canvas, builds either the deterministic 4-node
 * fixture view or the capped ego subgraph of the seed user, and draws it with
 * d3 v7 in a two-column layout. Main settings and fixture progress are stored
 * independently. The view never modifies GRAPH or any score array.
 *
 * Degrades gracefully: with no d3, no graph, or no seed user it writes a plain
 * message instead of throwing, and an unimplemented `buildGraph` stub simply
 * leaves GRAPH null (the fixture demo still draws).
 *
 * @param {GraphViewOptions} [opts]
 * @returns {(Object|null)} a small description of the last render, or null.
 */
function renderGraphView(opts) {
  const o = opts || {};
  const mode = o.mode === "fixture" ? "fixture" : "ego";
  if (o.seedUserId !== undefined) {
    GRAPH_VIEW_STATE.seedUserId = o.seedUserId === null ? null : Number(o.seedUserId);
  }
  if (o.colorBy === "degree" || o.colorBy === "score") GRAPH_VIEW_STATE.colorBy = o.colorBy;
  if (o.maxNodes !== undefined && o.maxNodes !== null && Number.isFinite(Number(o.maxNodes))) {
    GRAPH_VIEW_STATE.maxNodes = Math.max(1, Math.floor(Number(o.maxNodes)));
  }

  const svgEl = document.getElementById(mode === "fixture" ? "fixture-graph-view" : "graph-view");
  if (!svgEl) return null;
  if (typeof d3 === "undefined") {
    graphSetLegendMessage("d3 is not available, so the graph view cannot be drawn. Check that vendor/d3.min.js loaded.", mode);
    return null;
  }

  const external = typeof o.getScore === "function" ? o.getScore : null;
  let view;
  let scoreAccessor = null;
  let colorByScore = false;

  if (mode === "fixture") {
    view = graphBuildFixtureView();
    scoreAccessor = (node) => {
      const p = GRAPH_VIEW_STATE.fixture.p;
      return p ? p[node] : 0;
    };
    colorByScore = true;
  } else {
    view = graphBuildEgoView(GRAPH_VIEW_STATE.seedUserId, external);
    if (external && GRAPH_VIEW_STATE.colorBy === "score") {
      scoreAccessor = external;
      colorByScore = true;
    } else if (!external) {
      // Do not leave the selector saying "score" when no scores were supplied.
      GRAPH_VIEW_STATE.colorBy = "degree";
    }
  }

  try {
    graphDrawView(svgEl, view, { scoreAccessor: scoreAccessor, colorByScore: colorByScore });
  } catch (error) {
    try {
      d3.select(svgEl).selectAll("*").remove();
    } catch (ignored) {
      /* the SVG may not be an element we can clear */
    }
    graphSetLegendMessage("Graph view could not be drawn: " + graphErrorMessage(error), mode);
  }

  graphRenderLegend(view, { colorByScore: colorByScore });
  if (mode === "ego") graphRenderControls();

  const description = {
    mode: mode,
    seedUserId: GRAPH_VIEW_STATE.seedUserId,
    colorByScore: colorByScore,
    nodeCount: view.nodes.length,
    edgeCount: view.edges.length,
    omitted: view.omitted || 0,
    note: view.note || ""
  };
  if (mode === "fixture") GRAPH_VIEW_STATE.fixture.lastRender = description;
  else GRAPH_VIEW_STATE.lastRender = description;
  return description;
}

// ---------------------------------------------------------------------------
// Fixture teaching demo (public: fixtureStep / fixtureReset)
//
// THIS DEMO EXISTS ONLY FOR THE 4-NODE FIXTURE. It is a tiny plain-JavaScript
// propagation so the animation works before you write any stub, and it is NOT a
// template for the MovieLens implementation. The MovieLens propagation must use
// TensorFlow.js tensor operations on R and R-transpose (readme.md section 3.5),
// as required by pageRank / personalizedPageRank.
// ---------------------------------------------------------------------------

/**
 * Update the fixture's own calculation panel with its iteration and residual.
 *
 * @returns {void}
 */
function graphRenderConvergence() {
  const target = document.getElementById("fixture-convergence-results");
  if (!target) return;
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || [];
  const values = p
    .map((value, i) => REFERENCE_GRAPH.nodeOrder[i] + " " + Number(value).toFixed(10))
    .join(", ");
  const residual = fixture.residual === null ? "-" : Number(fixture.residual).toExponential(3);

  target.innerHTML =
    "<p><strong>Fixture demo (4-node reference graph)</strong> - iteration " +
    fixture.iteration +
    ", alpha " +
    Number(fixture.alpha !== undefined ? fixture.alpha : graphReadAlpha()).toFixed(2) +
    ", restart " +
    graphEscape(fixture.restart || "uniform") +
    ".</p>" +
    '<dl class="summary-list">' +
    "<dt>L1 residual (last step)</dt><dd>" + graphEscape(residual) + "</dd>" +
    "<dt>Current p</dt><dd>" + graphEscape(values) + "</dd>" +
    "</dl>" +
    '<p class="hint">This plain-JavaScript demo exists only for the 4-node fixture and is not a ' +
    "template for the MovieLens implementation, which must use TensorFlow.js tensor operations " +
    "on R and R-transpose (readme.md section 3.5).</p>";
}

/**
 * Advance the 4-node fixture demo by one propagation step and redraw it.
 *
 * @returns {void}
 */
function fixtureStep() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const alpha = graphReadAlpha();
  if (!fixture.p) {
    fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
    fixture.iteration = 0;
  }
  const result = graphFixturePropagate(alpha, fixture.restart || "uniform", fixture.p);
  fixture.p = result.p;
  fixture.iteration += 1;
  fixture.residual = result.residual;
  fixture.alpha = alpha;
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

/**
 * Reset the fixture demo to iteration 0 (the restart distribution) and redraw.
 *
 * @returns {void}
 */
function fixtureReset() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  fixture.alpha = graphReadAlpha();
  fixture.iteration = 0;
  fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
  fixture.residual = null;
  const restartSelect = document.getElementById("fixture-restart");
  if (restartSelect) restartSelect.value = fixture.restart;
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

// ---------------------------------------------------------------------------
// Bootstrap (private)
// ---------------------------------------------------------------------------

/** Redraw visible canvases after a tab or width change, preserving their scores. */
function graphRedrawVisibleViews() {
  for (const drawing of Array.from(graphLastDrawings.values())) {
    if (drawing.svgEl.clientWidth > 0) graphDrawView(drawing.svgEl, drawing.view, drawing.opts);
  }
}

/**
 * Wire the fixture step/reset buttons and draw the fixture demo once.
 *
 * @returns {void}
 */
function graphInitGraphView() {
  const stepButton = document.getElementById("step-btn");
  if (stepButton) stepButton.addEventListener("click", fixtureStep);
  const resetButton = document.getElementById("reset-steps");
  if (resetButton) resetButton.addEventListener("click", fixtureReset);
  const alpha = document.getElementById("fixture-alpha");
  if (alpha) alpha.addEventListener("input", () => {
    const readout = document.getElementById("fixture-alpha-value");
    if (readout) readout.textContent = graphReadAlpha().toFixed(2);
    fixtureReset();
  });
  const restart = document.getElementById("fixture-restart");
  if (restart) restart.addEventListener("change", () => {
    GRAPH_VIEW_STATE.fixture.restart = restart.value;
    fixtureReset();
  });
  fixtureReset();
  renderGraphView({ mode: "ego" });
  if (typeof ResizeObserver !== "undefined") {
    const canvases = [document.getElementById("graph-view"), document.getElementById("fixture-graph-view")].filter(Boolean);
    const widths = new Map(canvases.map(svg => [svg, svg.clientWidth]));
    let frameQueued = false;
    graphResizeObserver = new ResizeObserver(() => {
      let changed = false;
      for (const svg of canvases) {
        if (svg.clientWidth !== widths.get(svg)) changed = true;
        widths.set(svg, svg.clientWidth);
      }
      if (!changed || frameQueued) return;
      frameQueued = true;
      window.requestAnimationFrame(() => {
        frameQueued = false;
        graphRedrawVisibleViews();
      });
    });
    canvases.forEach(svg => graphResizeObserver.observe(svg.parentElement));
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", graphInitGraphView);
} else {
  graphInitGraphView();
}
