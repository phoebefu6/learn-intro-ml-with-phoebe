/* ============================================================================
   ml-live.js - the live machine-learning playground for learn-intro-ml-with-phoebe.
   Three tools, zero dependencies, theme-aware (reads the site CSS vars):

     1. split - the train/test split. Watch points get dealt into a TRAIN pile
        (the model learns here) and a held-out TEST pile (the honest scorecard).
        Drag the test size. The whole point: you never grade on rows you trained on.

     2. fit   - THE centrepiece. Over- and under-fitting on Lumen order value vs
        prior-30-day spend. Slide a decision tree's max_depth from 1 to 12 and
        watch the step-function bend: shallow = underfit (high train AND test
        error), deep = overfit (train error -> 0, test error climbs). The
        U-shaped test-error curve and the train/test GAP are drawn live.

     3. threshold - the accuracy paradox + precision/recall. Lumen converts at
        ~3.2%, so "always predict no" scores 96.8% accuracy and catches zero
        buyers. Slide the classification threshold and watch the confusion matrix
        and accuracy / precision / recall / F1 move against that useless baseline.

   Drop <div class="ml-live" data-tool="split|fit|threshold"></div> on any page.
   Defaults seeded from the Lumen canon (materials/lumen-canon.md).
   by Phoebe Fu
   ==========================================================================*/
(function () {
  "use strict";

  /* ---------- theme helpers ---------- */
  function cssVar(name, fallback) {
    try {
      var v = getComputedStyle(document.documentElement).getPropertyValue(name);
      return (v && v.trim()) || fallback;
    } catch (e) { return fallback; }
  }
  function palette() {
    return {
      blue:  cssVar("--indigo", "#2C86BD"),
      deep:  cssVar("--indigo-deep", "#1B5F8C"),
      soft:  cssVar("--indigo-soft", "#A5D8EF"),
      tint:  cssVar("--indigo-50", "#EAF6FC"),
      orange:cssVar("--amber", "#F7931E"),
      ink:   cssVar("--ink", "#12222B"),
      muted: cssVar("--muted", "#52707E"),
      hair:  cssVar("--hairline", "#DCEDF4")
    };
  }

  /* ---------- tiny seeded RNG (mulberry32) so demos are reproducible ---------- */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { // Box-Muller
    var u = 1 - r(), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function svgEl(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function fmt(x, d) { return x.toFixed(d == null ? 2 : d); }

  /* =========================================================================
     TOOL 1 - train/test split
     ========================================================================= */
  function buildSplit(root) {
    var P = palette();
    var W = 560, H = 260, N = 120;
    var wrap = el("div", "mllive-wrap");
    var head = el("div", "mllive-head",
      "<strong>Train / test split</strong><span class='mllive-sub'>The model learns on the blue pile. It is graded only on the held-out orange pile.</span>");
    wrap.appendChild(head);

    var ctrls = el("div", "mllive-ctrls");
    var lab = el("label", "mllive-lab", "Test size: <b>20%</b>");
    var slider = el("input");
    slider.type = "range"; slider.min = "10"; slider.max = "50"; slider.step = "5"; slider.value = "20";
    ctrls.appendChild(lab); ctrls.appendChild(slider);
    wrap.appendChild(ctrls);

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "mllive-svg", width: "100%" });
    wrap.appendChild(svg);

    var note = el("div", "mllive-note");
    wrap.appendChild(note);
    root.appendChild(wrap);

    // fixed point cloud
    var r = rng(42), pts = [];
    for (var i = 0; i < N; i++) {
      pts.push({ x: 30 + r() * (W - 60), y: 30 + r() * (H - 60), u: r() });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var testFrac = (+slider.value) / 100;
      lab.innerHTML = "Test size: <b>" + slider.value + "%</b>";
      var nTest = 0;
      pts.forEach(function (p) {
        var isTest = p.u < testFrac;
        if (isTest) nTest++;
        svg.appendChild(svgEl("circle", {
          cx: p.x, cy: p.y, r: 6,
          fill: isTest ? P.orange : P.blue,
          "fill-opacity": isTest ? 0.9 : 0.55,
          stroke: isTest ? P.orange : P.deep, "stroke-width": isTest ? 1.5 : 1
        }));
      });
      var nTrain = N - nTest;
      note.innerHTML = "<span class='mllive-pill mllive-blue'>" + nTrain + " train rows</span>" +
        "<span class='mllive-pill mllive-orange'>" + nTest + " test rows</span>" +
        "<span class='mllive-hint'>More test = a steadier grade but less to learn from. 20-25% is the usual pick.</span>";
    }
    slider.addEventListener("input", draw);
    draw();
  }

  /* =========================================================================
     TOOL 2 - over/underfitting with a 1-D regression tree
     ========================================================================= */
  // recursive 1-D CART on (x,y): split to minimise SSE, stop at max_depth / min leaf
  function fitTree(xs, ys, depth, maxDepth, minLeaf) {
    var n = xs.length;
    var mean = 0; for (var i = 0; i < n; i++) mean += ys[i]; mean /= n;
    if (depth >= maxDepth || n < 2 * minLeaf) return { leaf: true, val: mean };
    // candidate splits = midpoints of sorted unique x
    var idx = xs.map(function (_, k) { return k; }).sort(function (a, b) { return xs[a] - xs[b]; });
    var bestSSE = Infinity, bestT = null, bestL = null, bestR = null;
    for (var s = minLeaf; s <= n - minLeaf; s++) {
      var t = (xs[idx[s - 1]] + xs[idx[s]]) / 2;
      if (t === xs[idx[s - 1]]) continue;
      var lSum = 0, lN = 0, rSum = 0, rN = 0, j;
      for (j = 0; j < n; j++) { if (xs[j] <= t) { lSum += ys[j]; lN++; } else { rSum += ys[j]; rN++; } }
      if (lN < minLeaf || rN < minLeaf) continue;
      var lM = lSum / lN, rM = rSum / rN, sse = 0;
      for (j = 0; j < n; j++) { var m = xs[j] <= t ? lM : rM; sse += (ys[j] - m) * (ys[j] - m); }
      if (sse < bestSSE) { bestSSE = sse; bestT = t; }
    }
    if (bestT == null) return { leaf: true, val: mean };
    var lx = [], ly = [], rx = [], ry = [];
    for (var k = 0; k < n; k++) { if (xs[k] <= bestT) { lx.push(xs[k]); ly.push(ys[k]); } else { rx.push(xs[k]); ry.push(ys[k]); } }
    return {
      leaf: false, t: bestT,
      L: fitTree(lx, ly, depth + 1, maxDepth, minLeaf),
      R: fitTree(rx, ry, depth + 1, maxDepth, minLeaf)
    };
  }
  function predict(tree, x) {
    while (!tree.leaf) tree = x <= tree.t ? tree.L : tree.R;
    return tree.val;
  }
  function rmse(tree, xs, ys) {
    var s = 0; for (var i = 0; i < xs.length; i++) { var e = ys[i] - predict(tree, xs[i]); s += e * e; }
    return Math.sqrt(s / xs.length);
  }

  function buildFit(root) {
    var P = palette();
    var W = 560, H = 300, pad = 40;
    var wrap = el("div", "mllive-wrap");
    wrap.appendChild(el("div", "mllive-head",
      "<strong>Under-fit &rarr; sweet spot &rarr; over-fit</strong><span class='mllive-sub'>A decision tree predicting Lumen order value from prior-30-day spend. Slide the depth.</span>"));

    var ctrls = el("div", "mllive-ctrls");
    var lab = el("label", "mllive-lab", "Tree max_depth: <b>3</b>");
    var slider = el("input");
    slider.type = "range"; slider.min = "1"; slider.max = "12"; slider.step = "1"; slider.value = "3";
    ctrls.appendChild(lab); ctrls.appendChild(slider);
    wrap.appendChild(ctrls);

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "mllive-svg", width: "100%" });
    wrap.appendChild(svg);
    var scoreRow = el("div", "mllive-note");
    wrap.appendChild(scoreRow);
    var uwrap = el("div", "mllive-ucurve");
    uwrap.appendChild(el("div", "mllive-uhead", "Test error by depth (the U you are hunting for the bottom of)"));
    var usvg = svgEl("svg", { viewBox: "0 0 " + W + " 120", class: "mllive-svg", width: "100%" });
    uwrap.appendChild(usvg);
    wrap.appendChild(uwrap);
    root.appendChild(wrap);

    // data: order_value ~ 16 + 0.26*spend + two sine waves + noise; spend in [0,200].
    // The high-frequency wave is real structure a too-shallow tree misses (underfit) and a
    // too-deep tree drowns in noise (overfit) - so test error bottoms around depth 3-4.
    var r = rng(7), all = [];
    for (var i = 0; i < 100; i++) {
      var x = r() * 200;
      var signal = 16 + 0.26 * x + 16 * Math.sin(x / 30) + 7 * Math.sin(x / 10);
      var y = signal + gauss(r) * 11;
      all.push({ x: x, y: Math.max(5, y) });
    }
    // split
    var trX = [], trY = [], teX = [], teY = [];
    all.forEach(function (p, k) { if (rng(100 + k)() < 0.32) { teX.push(p.x); teY.push(p.y); } else { trX.push(p.x); trY.push(p.y); } });

    var xMin = 0, xMax = 200, yMin = 0, yMax = 110;
    function sx(x) { return pad + (x - xMin) / (xMax - xMin) * (W - 2 * pad); }
    function sy(y) { return H - pad - (y - yMin) / (yMax - yMin) * (H - 2 * pad); }

    // precompute test rmse across depths for the U-curve
    var uCurve = [];
    for (var d = 1; d <= 12; d++) {
      var tr = fitTree(trX, trY, 0, d, 1);
      uCurve.push({ d: d, tr: rmse(tr, trX, trY), te: rmse(tr, teX, teY) });
    }

    function drawU(curD) {
      while (usvg.firstChild) usvg.removeChild(usvg.firstChild);
      var maxE = 0; uCurve.forEach(function (c) { maxE = Math.max(maxE, c.te, c.tr); });
      function ux(d) { return 30 + (d - 1) / 11 * (W - 50); }
      function uy(e) { return 100 - e / maxE * 80; }
      ["tr", "te"].forEach(function (key) {
        var col = key === "te" ? P.orange : P.soft, path = "";
        uCurve.forEach(function (c, k) { path += (k ? "L" : "M") + ux(c.d) + " " + uy(c[key]) + " "; });
        usvg.appendChild(svgEl("path", { d: path, fill: "none", stroke: col, "stroke-width": key === "te" ? 2.5 : 1.6 }));
      });
      // marker for current depth
      usvg.appendChild(svgEl("line", { x1: ux(curD), y1: 12, x2: ux(curD), y2: 104, stroke: P.ink, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.5 }));
      var lg = svgEl("text", { x: W - 8, y: 16, "text-anchor": "end", "font-size": 10, fill: P.muted });
      lg.textContent = "orange = test   ·   light = train";
      usvg.appendChild(lg);
    }

    function draw() {
      var d = +slider.value;
      lab.innerHTML = "Tree max_depth: <b>" + d + "</b>";
      var tree = fitTree(trX, trY, 0, d, 1);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      // axes
      svg.appendChild(svgEl("line", { x1: pad, y1: H - pad, x2: W - pad, y2: H - pad, stroke: P.hair, "stroke-width": 1 }));
      svg.appendChild(svgEl("line", { x1: pad, y1: pad, x2: pad, y2: H - pad, stroke: P.hair, "stroke-width": 1 }));
      var xl = svgEl("text", { x: W / 2, y: H - 6, "text-anchor": "middle", "font-size": 11, fill: P.muted }); xl.textContent = "prior 30-day spend ($)"; svg.appendChild(xl);
      var yl = svgEl("text", { x: 12, y: H / 2, "text-anchor": "middle", "font-size": 11, fill: P.muted, transform: "rotate(-90 12 " + (H / 2) + ")" }); yl.textContent = "order value ($)"; svg.appendChild(yl);
      // train points
      trX.forEach(function (x, i) { svg.appendChild(svgEl("circle", { cx: sx(x), cy: sy(trY[i]), r: 3.4, fill: P.blue, "fill-opacity": 0.5 })); });
      // test points
      teX.forEach(function (x, i) { svg.appendChild(svgEl("circle", { cx: sx(x), cy: sy(teY[i]), r: 3.8, fill: P.orange, "fill-opacity": 0.9 })); });
      // prediction step function
      var path = "", step = 1.2;
      for (var gx = xMin; gx <= xMax; gx += step) { var gy = predict(tree, gx); path += (gx === xMin ? "M" : "L") + sx(gx) + " " + sy(gy) + " "; }
      svg.appendChild(svgEl("path", { d: path, fill: "none", stroke: P.deep, "stroke-width": 2.4 }));

      var trE = rmse(tree, trX, trY), teE = rmse(tree, teX, teY), gap = teE - trE;
      var diag = d <= 2 ? "Underfit - too simple, high error on both" :
        (gap > 6 ? "Overfit - memorising the training dots" : "Near the sweet spot - lowest test error");
      var diagCls = d <= 2 ? "mllive-warn" : (gap > 6 ? "mllive-warn" : "mllive-ok");
      scoreRow.innerHTML =
        "<span class='mllive-pill mllive-blue'>train RMSE $" + fmt(trE, 1) + "</span>" +
        "<span class='mllive-pill mllive-orange'>test RMSE $" + fmt(teE, 1) + "</span>" +
        "<span class='mllive-pill'>gap $" + fmt(gap, 1) + "</span>" +
        "<span class='mllive-diag " + diagCls + "'>" + diag + "</span>";
      drawU(d);
    }
    slider.addEventListener("input", draw);
    draw();
  }

  /* =========================================================================
     TOOL 3 - classification threshold, confusion matrix, accuracy paradox
     ========================================================================= */
  function buildThreshold(root) {
    var P = palette();
    var wrap = el("div", "mllive-wrap");
    wrap.appendChild(el("div", "mllive-head",
      "<strong>The accuracy paradox</strong><span class='mllive-sub'>Lumen converts at 3.2%. \"Always predict no\" scores 96.8% and catches zero buyers. Slide the threshold.</span>"));

    var ctrls = el("div", "mllive-ctrls");
    var lab = el("label", "mllive-lab", "Decision threshold: <b>0.50</b>");
    var slider = el("input");
    slider.type = "range"; slider.min = "2"; slider.max = "98"; slider.step = "1"; slider.value = "50";
    ctrls.appendChild(lab); ctrls.appendChild(slider);
    wrap.appendChild(ctrls);

    var grid = el("div", "mllive-cm");
    wrap.appendChild(grid);
    var note = el("div", "mllive-note");
    wrap.appendChild(note);
    root.appendChild(wrap);

    // simulate 3000 sessions: 3.2% positive. Model scores: positives ~ Beta-ish high, negatives low,
    // tuned so ROC-AUC ~ 0.78.
    var r = rng(21), N = 3000, base = 0.032, data = [];
    function sig(z) { return 1 / (1 + Math.exp(-z)); }
    for (var i = 0; i < N; i++) {
      var pos = r() < base;
      // latent score: positives shifted up, negatives well below 0.5, heavy overlap -> AUC ~0.77
      var z = (pos ? 0.0 : -1.15) + gauss(r) * 1.05;
      data.push({ y: pos ? 1 : 0, s: sig(z) });
    }
    var nPos = data.filter(function (d) { return d.y === 1; }).length;
    var nNeg = N - nPos;

    function cell(v, big) { return "<div class='mllive-cmcell " + (big || "") + "'>" + v + "</div>"; }

    function draw() {
      var th = (+slider.value) / 100;
      lab.innerHTML = "Decision threshold: <b>" + fmt(th, 2) + "</b>";
      var TP = 0, FP = 0, TN = 0, FN = 0;
      data.forEach(function (d) {
        var pred = d.s >= th ? 1 : 0;
        if (pred === 1 && d.y === 1) TP++;
        else if (pred === 1 && d.y === 0) FP++;
        else if (pred === 0 && d.y === 0) TN++;
        else FN++;
      });
      var acc = (TP + TN) / N;
      var prec = TP + FP ? TP / (TP + FP) : 0;
      var rec = TP + FN ? TP / (TP + FN) : 0;
      var f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;

      grid.innerHTML =
        "<div class='mllive-cmhdr'></div><div class='mllive-cmhdr'>Predicted BUY</div><div class='mllive-cmhdr'>Predicted no</div>" +
        "<div class='mllive-cmhdr'>Actually bought</div>" + cell("<b>" + TP + "</b><span>true positive</span>", "mllive-tp") + cell(FN + "<span>false negative</span>", "mllive-fn") +
        "<div class='mllive-cmhdr'>Actually didn't</div>" + cell(FP + "<span>false positive</span>", "mllive-fp") + cell("<b>" + TN + "</b><span>true negative</span>", "mllive-tn");

      note.innerHTML =
        "<span class='mllive-pill'>accuracy " + fmt(acc * 100, 1) + "%</span>" +
        "<span class='mllive-pill mllive-blue'>precision " + fmt(prec * 100, 1) + "%</span>" +
        "<span class='mllive-pill mllive-orange'>recall " + fmt(rec * 100, 1) + "%</span>" +
        "<span class='mllive-pill'>F1 " + fmt(f1 * 100, 1) + "%</span>" +
        "<div class='mllive-hint'>The \"always predict no\" baseline: accuracy <b>" + fmt(nNeg / N * 100, 1) + "%</b>, recall <b>0%</b> - it never finds a buyer. Accuracy alone hides that. Lower the threshold to buy recall with precision.</div>";
    }
    slider.addEventListener("input", draw);
    draw();
  }

  /* ---------- scoped, theme-aware styles (injected once) ---------- */
  function injectStyles() {
    if (document.getElementById("mllive-css")) return;
    var s = document.createElement("style");
    s.id = "mllive-css";
    s.textContent =
      ".ml-live{border:1px solid var(--hairline);border-radius:16px;padding:1.4rem 1.5rem;margin:1.6rem 0;background:linear-gradient(180deg,var(--indigo-50),#fff);box-shadow:0 10px 30px rgba(44,134,189,.09)}" +
      ".mllive-head{display:flex;flex-direction:column;gap:.25rem;margin-bottom:.9rem}" +
      ".mllive-head strong{font-size:1.02rem;color:var(--ink)}" +
      ".mllive-sub{font-size:.86rem;color:var(--muted);line-height:1.5}" +
      ".mllive-ctrls{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin:.4rem 0 .9rem}" +
      ".mllive-lab{font-size:.86rem;font-weight:700;color:var(--ink);white-space:nowrap}" +
      ".mllive-lab b{color:var(--indigo-deep)}" +
      ".ml-live input[type=range]{flex:1;min-width:180px;accent-color:var(--amber)}" +
      ".mllive-svg{background:#fff;border:1px solid var(--hairline);border-radius:12px;display:block}" +
      ".mllive-note{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:.9rem}" +
      ".mllive-pill{background:#fff;border:1px solid var(--hairline);color:var(--ink);font-weight:700;font-size:.8rem;padding:.28rem .7rem;border-radius:999px}" +
      ".mllive-pill.mllive-blue{background:var(--indigo-50);border-color:var(--indigo-soft);color:var(--indigo-deep)}" +
      ".mllive-pill.mllive-orange{background:var(--amber-50);border-color:var(--amber);color:var(--amber-ink)}" +
      ".mllive-hint{flex-basis:100%;font-size:.82rem;color:var(--muted);line-height:1.55;margin-top:.3rem}" +
      ".mllive-hint b{color:var(--ink)}" +
      ".mllive-diag{font-weight:800;font-size:.82rem;padding:.28rem .7rem;border-radius:999px}" +
      ".mllive-diag.mllive-warn{background:var(--amber-50);color:var(--amber-ink)}" +
      ".mllive-diag.mllive-ok{background:var(--indigo-50);color:var(--indigo-deep)}" +
      ".mllive-ucurve{margin-top:1rem}" +
      ".mllive-uhead{font-size:.8rem;font-weight:700;color:var(--muted);margin-bottom:.35rem}" +
      ".mllive-cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:.4rem;margin:.4rem 0 .2rem}" +
      ".mllive-cmhdr{font-size:.76rem;font-weight:800;color:var(--muted);display:flex;align-items:center;justify-content:center;text-align:center;padding:.3rem}" +
      ".mllive-cmcell{background:#fff;border:1px solid var(--hairline);border-radius:10px;padding:.7rem;text-align:center;font-weight:800;font-size:1.3rem;color:var(--ink);display:flex;flex-direction:column;gap:.1rem}" +
      ".mllive-cmcell span{font-size:.68rem;font-weight:600;color:var(--muted)}" +
      ".mllive-cmcell.mllive-tp{background:var(--indigo-50);border-color:var(--indigo-soft)}" +
      ".mllive-cmcell.mllive-tn{background:var(--indigo-50);border-color:var(--indigo-soft)}" +
      ".mllive-cmcell.mllive-fp{background:var(--amber-50);border-color:var(--amber)}" +
      ".mllive-cmcell.mllive-fn{background:var(--amber-50);border-color:var(--amber)}";
    document.head.appendChild(s);
  }

  /* ---------- boot ---------- */
  function boot() {
    injectStyles();
    var nodes = document.querySelectorAll(".ml-live");
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.getAttribute("data-mllive-done")) continue;
      n.setAttribute("data-mllive-done", "1");
      var tool = n.getAttribute("data-tool") || "fit";
      if (tool === "split") buildSplit(n);
      else if (tool === "threshold") buildThreshold(n);
      else buildFit(n);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
