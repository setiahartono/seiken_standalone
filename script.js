import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = null;
let aspect = 0.75;   // videoHeight / videoWidth, refreshed every frame
let lastSeen = 0;    // timestamp of the last frame with a detected hand
let stageAR = 4 / 3; // stage aspect ratio (video width / height)
let running = false; // camera + detection loop active
let mediaStream = null;
let rafId = 0;

const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const mainEl = document.getElementById("main");
const stageEl = document.getElementById("container");
const startOverlay = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");
const startMsg = document.getElementById("startMsg");
const stopBtn = document.getElementById("stopBtn");
const camStateEl = document.getElementById("camState");
const statusEl = document.getElementById("status");
const debugEl = document.getElementById("debug");
const instructEl = document.getElementById("instruct");
const ringEl = document.getElementById("ring");
const ringLabel = document.getElementById("ringLabel");
const hudStepEls = ["hud1", "hud2", "hud3"].map((id) => document.getElementById(id));
const stepImg = document.getElementById("stepImg");
const stepCaption = document.getElementById("stepCaption");
let flashUntil = 0; // success-flash end timestamp
let lastHint = "";

const IDLE_HINT = "Camera is off — press Start camera to train.";

// The MediaPipe model is loaded lazily and cached. Opening the camera is a
// separate, explicit user action (see startCamera below).
let modelPromise = null;
function ensureModel() {
  if (handLandmarker) return Promise.resolve(handLandmarker);
  if (!modelPromise) {
    modelPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
      });
      return handLandmarker;
    })();
    // Allow a retry after a failed download instead of caching the rejection.
    modelPromise.catch(() => { modelPromise = null; handLandmarker = null; });
  }
  return modelPromise;
}

function setCamState(text, on) {
  if (!camStateEl) return;
  camStateEl.textContent = text;
  camStateEl.className = on ? "pill on" : "pill";
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

// Fit the stage into the main section while keeping the video aspect ratio,
// so the 80/20 split stays responsive without stretching the picture.
function fitStage() {
  if (!mainEl || !stageEl) return;
  const box = mainEl.getBoundingClientRect();
  const ar = stageAR > 0 ? stageAR : 4 / 3;
  let w = box.width, h = w / ar;
  if (h > box.height) { h = box.height; w = h * ar; }
  stageEl.style.width = Math.max(0, Math.floor(w)) + "px";
  stageEl.style.height = Math.max(0, Math.floor(h)) + "px";
}

async function init() {
  // No camera here: the model is prepared, the camera waits for the button.
  updateHud(0, IDLE_HINT, training.step);
  setCamState("Off", false);
  fitStage();
  setStatus("Loading hand model…");
  try {
    await ensureModel();
    setStatus("Model ready. Press Start camera to train.");
    startBtn.disabled = false;
    startBtn.textContent = "Start camera to train";
  } catch (err) {
    console.error(err);
    setStatus("Hand model failed to load.");
    startMsg.textContent = "The hand-tracking model could not be downloaded. Check your connection, then retry.";
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
  }
}

// Explicit camera start (button). Nothing is captured before this runs.
async function startCamera() {
  if (running) return;
  startBtn.disabled = true;
  startBtn.textContent = "Starting camera…";
  startMsg.textContent = "Allow camera access in the browser prompt.";
  setStatus("Starting camera…");

  try {
    await ensureModel();
  } catch (err) {
    console.error(err);
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
    startMsg.textContent = "The hand-tracking model could not be downloaded. Check your connection, then retry.";
    setStatus("Hand model failed to load.");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = mediaStream;
    await video.play();
    if (video.videoWidth && video.videoHeight) {
      stageAR = video.videoWidth / video.videoHeight;
      aspect = video.videoHeight / video.videoWidth;
    }
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    fitStage();

    running = true;
    lastSeen = performance.now();
    resetTraining();
    for (const k in smoothState) delete smoothState[k];
    flashUntil = 0;
    if (debugEl) debugEl.innerText = "";

    startBtn.disabled = false;
    startBtn.textContent = "Start camera to train";
    startOverlay.classList.add("hidden");
    stopBtn.hidden = false;
    setCamState("On", true);
    setStatus("Camera running. Show your hand to the stage.");
    lastHint = "";
    updateHud(0, "Show your hand to start!", training.step);

    predictWebcam();
  } catch (err) {
    console.error(err);
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    video.srcObject = null;
    running = false;
    startBtn.disabled = false;
    startBtn.textContent = "Start camera to train";
    startMsg.textContent = err && err.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow camera access in the address bar, then try again."
      : "Could not open the camera. Close other apps that use it, then try again.";
    setCamState("Off", false);
    setStatus("Camera unavailable.");
  }
}

function stopCamera() {
  running = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resetTraining();
  for (const k in smoothState) delete smoothState[k];
  flashUntil = 0;
  if (debugEl) debugEl.innerText = "";
  startOverlay.classList.remove("hidden");
  stopBtn.hidden = true;
  startBtn.disabled = false;
  startBtn.textContent = "Start camera to train";
  startMsg.textContent = "Camera off. Press start when you are ready.";
  setCamState("Off", false);
  setStatus("Camera off.");
  lastHint = "";
  updateHud(0, IDLE_HINT, training.step);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("resize", fitStage);
window.addEventListener("orientationchange", fitStage);
if (window.ResizeObserver && mainEl) new ResizeObserver(fitStage).observe(mainEl);

// Mathematical utility: Distance between 3D points
function dist(p1, p2) {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
}

async function predictWebcam() {
  if (!running || !handLandmarker) return;
  // Intrinsic video size drives the drawing resolution (set once, not per frame).
  if (video.videoWidth && canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const now = performance.now();

  let results = null;
  try {
    results = await handLandmarker.detectForVideo(video, now);
  } catch (err) {
    console.error(err);
  }
  if (!running) return; // camera stopped while detecting

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results && results.landmarks && results.landmarks.length > 0) {
    lastSeen = now;
    const lm = results.landmarks[0];
    aspect = video.videoWidth ? video.videoHeight / video.videoWidth : 0.75;
    const result = evaluateSeiken(lm, now);
    drawLandmarks(lm, result, now);
    updateDebug(result);
  } else {
    if (instructEl) instructEl.textContent = "Show your hand to start!";
    lastHint = "Show your hand to start!";
    if (debugEl) debugEl.innerText = "";
    // A dropped frame must not count toward the hold; drop stale smoothing too.
    training.holdStart = null;
    for (const k in smoothState) delete smoothState[k];
    // After a short grace period absent, restart the whole progression.
    if (now - lastSeen > 800) resetTraining();
    updateHud(0, lastHint, training.step);
  }

  rafId = requestAnimationFrame(predictWebcam);
}

// ---------- Hand topology (MediaPipe landmarks) ----------
// 0 wrist | 4 thumb tip | 5 index MCP, 6 PIP, 7 DIP, 8 tip
// 9 middle MCP, 10 PIP, 11 DIP, 12 tip | 13 ring | 17 pinky MCP, 20 pinky tip
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

// Tunable thresholds. Distances are expressed relative to the palm width
// (landmark 5 <-> 17) so the trainer behaves the same near or far from the camera.
const THRESH = {
  extendMinAngle: 150,    // joint angle at PIP/DIP above this => finger is straight
  curlMaxAngle: 115,      // joint angle at PIP/DIP below this => finger is curled
  thumbGapMin: 0.30,      // fraction middle(0) -> ring(1): middle-finger side of the crease
  thumbGapMax: 0.58,      // past this the thumb has slid off toward the ring finger
  thumbGapIdeal: 0.44,    // ideal tip spot: just toward the middle finger's centre, not its edge
  creaseTolerance: 0.35,  // max offset from the middle-ring crease (x palm width) -> locked tight
  outsideDepth: 0.10,     // thumb may sit at most this much deeper than the finger ridge
  contactDepth: 0.25,     // max the thumb may float IN FRONT of the ridge (press it down)
  trappedDepth: 0.40,     // deeper than this => thumb is buried INSIDE the fist
  holdMs: 2000            // continuous hold time required to clear each step
};

const COLORS = {
  point: "#FF0000",
  bone: "rgba(255,255,255,0.35)",
  crease: "#29B6F6",
  arrow: "#4FC3F7",
  active: "#4FC3F7",
  good: "#4CAF50",
  warn: "#FFC107",
  bad: "#F44336"
};

// Exponential smoothing to tame MediaPipe jitter on the noisy scalars.
const smoothState = {};
function smooth(key, value, alpha = 0.5) {
  const prev = smoothState[key];
  smoothState[key] = prev === undefined ? value : prev + alpha * (value - prev);
  return smoothState[key];
}

// ---------- Small 3D vector helpers ----------
function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; }
function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }; }
function mag(v) { return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Angle (degrees) at joint b formed by a-b-c. ~180 = straight, small = bent.
function angleAt(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const denom = mag(v1) * mag(v2) || 1e-6;
  const cos = Math.max(-1, Math.min(1, dot(v1, v2) / denom));
  return Math.acos(cos) * (180 / Math.PI);
}

// Tightest of the PIP and DIP joint angles for a finger.
function fingerBend(lm, mcp, pip, dip, tip) {
  return Math.min(
    angleAt(lm[mcp], lm[pip], lm[dip]),
    angleAt(lm[pip], lm[dip], lm[tip])
  );
}

// ---------- Target pose templates (synthesized, no assets) ----------
function rotDir(d, deg) {
  const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r);
  return { x: d.x * c - d.y * s, y: d.x * s + d.y * c };
}

// Curl a finger into the palm: three 85-degree joints (~95-degree bends).
function curledChain(mcp, L = 0.055, stepDeg = 85) {
  const pip = { x: mcp.x, y: mcp.y - L };
  const d2 = rotDir({ x: 0, y: -1 }, -stepDeg);
  const dip = { x: pip.x + L * d2.x, y: pip.y + L * d2.y };
  const d3 = rotDir(d2, -stepDeg);
  const tip = { x: dip.x + L * d3.x, y: dip.y + L * d3.y };
  return { pip, dip, tip };
}

function blankHand() {
  return new Array(21).fill(null).map(() => ({ x: 0.5, y: 0.5 }));
}

function buildTemplates() {
  // OPEN hand: four straight fingers.
  const open = blankHand();
  open[0] = { x: 0.5, y: 0.85 };
  open[1] = { x: 0.40, y: 0.75 }; open[2] = { x: 0.36, y: 0.72 };
  open[3] = { x: 0.34, y: 0.69 }; open[4] = { x: 0.33, y: 0.66 };
  open[5] = { x: 0.44, y: 0.62 }; open[6] = { x: 0.44, y: 0.55 };
  open[7] = { x: 0.44, y: 0.48 }; open[8] = { x: 0.44, y: 0.41 };
  open[9] = { x: 0.49, y: 0.60 }; open[10] = { x: 0.49, y: 0.53 };
  open[11] = { x: 0.49, y: 0.46 }; open[12] = { x: 0.49, y: 0.39 };
  open[13] = { x: 0.54, y: 0.62 }; open[14] = { x: 0.54, y: 0.55 };
  open[15] = { x: 0.54, y: 0.48 }; open[16] = { x: 0.54, y: 0.41 };
  open[17] = { x: 0.585, y: 0.66 }; open[18] = { x: 0.585, y: 0.60 };
  open[19] = { x: 0.585, y: 0.54 }; open[20] = { x: 0.585, y: 0.48 };

  // FIST: same MCPs, each finger curled into the palm.
  const fist = blankHand();
  fist[0] = { x: 0.5, y: 0.85 };
  fist[1] = { x: 0.42, y: 0.74 }; fist[2] = { x: 0.42, y: 0.71 };
  fist[3] = { x: 0.43, y: 0.66 }; fist[4] = { x: 0.44, y: 0.60 };
  const mcps = [
    [5, 0.44, 0.62, 6, 7, 8], [9, 0.49, 0.60, 10, 11, 12],
    [13, 0.54, 0.62, 14, 15, 16], [17, 0.585, 0.66, 18, 19, 20]
  ];
  for (const [mi, mx, my, pi, di, ti] of mcps) {
    fist[mi] = { x: mx, y: my };
    const c = curledChain({ x: mx, y: my });
    fist[pi] = c.pip; fist[di] = c.dip; fist[ti] = c.tip;
  }

  // SEIKEN: fist with the thumb tip at the ideal middle->ring spot.
  const seiken = fist.map((p) => ({ x: p.x, y: p.y }));
  const A = mid(seiken[10], seiken[11]);
  const B = mid(seiken[14], seiken[15]);
  seiken[4] = {
    x: A.x + THRESH.thumbGapIdeal * (B.x - A.x),
    y: A.y + THRESH.thumbGapIdeal * (B.y - A.y)
  };

  return { 1: toLocal(open), 2: toLocal(fist), 3: toLocal(seiken) };
}

// Register template points to the live hand via a palm-local frame
// (origin = wrist, x-axis = index MCP -> pinky MCP, scale = palm width).
function handFrame(lm) {
  const w = Math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y) || 1e-6;
  const e1 = { x: (lm[5].x - lm[17].x) / w, y: (lm[5].y - lm[17].y) / w };
  return { o: lm[0], e1, e2: { x: -e1.y, y: e1.x }, w };
}
function toLocal(lm) {
  const f = handFrame(lm);
  return lm.map((p) => {
    const dx = p.x - f.o.x, dy = p.y - f.o.y;
    return {
      x: (dx * f.e1.x + dy * f.e1.y) / f.w,
      y: (dx * f.e2.x + dy * f.e2.y) / f.w
    };
  });
}
function fromLocal(local, lm) {
  const f = handFrame(lm);
  return local.map((p) => ({
    x: f.o.x + f.w * (p.x * f.e1.x + p.y * f.e2.x),
    y: f.o.y + f.w * (p.x * f.e1.y + p.y * f.e2.y)
  }));
}

const POSES = buildTemplates();

// ---------- Gradual hold-to-advance progression ----------
// Step 1: extend fingers  ->  Step 2: curl fingers  ->  Step 3: perfect Seiken.
// Each step must be held continuously for THRESH.holdMs before advancing.
const training = {
  step: 1,
  holdStart: null,
  done: { open: false, curl: false, seiken: false }
};

function resetTraining() {
  training.step = 1;
  training.holdStart = null;
  training.done.open = false;
  training.done.curl = false;
  training.done.seiken = false;
}

// Returns hold progress in [0, 1]; resets the timer whenever the pose breaks.
function advanceHold(conditionMet, now) {
  if (!conditionMet) {
    training.holdStart = null;
    return 0;
  }
  if (training.holdStart === null) training.holdStart = now;
  return Math.min(1, (now - training.holdStart) / THRESH.holdMs);
}

// Sidebar step picture: one placeholder illustration per step (1..3).
const STEP_PREVIEWS = {
  1: { src: "assets/step-1.jpeg", alt: "Step 1 of 3: extend all four fingers straight", caption: "Step 1 of 3 — Extend" },
  2: { src: "assets/step-2.jpeg", alt: "Step 2 of 3: curl the four fingers into a fist", caption: "Step 2 of 3 — Curl" },
  3: { src: "assets/step-3.jpeg", alt: "Step 3 of 3: thumb tip on the middle-ring crease", caption: "Step 3 of 3 — Seiken" }
};
let previewStep = 0; // step whose picture is currently shown

function updateStepPreview(step) {
  const n = Math.min(Math.max(step || 1, 1), 3);
  if (n === previewStep) return; // avoid re-assigning src every frame
  previewStep = n;
  const preview = STEP_PREVIEWS[n];
  if (stepImg) {
    stepImg.src = preview.src;
    stepImg.alt = preview.alt;
  }
  if (stepCaption) stepCaption.textContent = preview.caption;
}

// Update the DOM HUD in place (no per-frame innerHTML rebuild).
function updateHud(progress, hint, step) {
  updateStepPreview(step);
  if (instructEl && hint !== lastHint) {
    instructEl.textContent = hint;
    lastHint = hint;
  }
  for (let n = 1; n <= 3; n++) {
    const el = hudStepEls[n - 1];
    if (!el) continue;
    const done = (n === 1 && training.done.open) || (n === 2 && training.done.curl) || (n === 3 && training.done.seiken);
    const cls = done ? "step done" : (n === step ? "step active" : "step");
    if (el.className !== cls) el.className = cls;
  }
  if (ringEl) {
    const p = Math.max(0, Math.min(1, progress || 0));
    const deg = Math.round(p * 360);
    const color = step >= 4 ? "#4CAF50" : "#FFC107";
    ringEl.style.background = "conic-gradient(" + color + " " + deg + "deg, #2a2a2a " + deg + "deg)";
    if (ringLabel) {
      const holdS = THRESH.holdMs / 1000;
      ringLabel.textContent = step >= 4 ? "✓" : (holdS * (1 - p)).toFixed(1) + "s";
    }
  }
}

function evaluateSeiken(lm, now) {
  // Aspect-corrected frame: y is rescaled to the same unit as x (and z),
  // so distances/angles are isotropic regardless of the video aspect ratio.
  const P = lm.map((p) => ({ x: p.x, y: p.y * aspect, z: p.z }));

  const wrist = P[0];
  const palmWidth = mag(sub(P[5], P[17])) || 1e-3;

  // ----- Finger state: extended (step 1) and curled (step 2) -----
  const fingerDefs = [
    ["index", 5, 6, 7, 8],
    ["middle", 9, 10, 11, 12],
    ["ring", 13, 14, 15, 16],
    ["pinky", 17, 18, 19, 20]
  ];
  const curled = {}, extended = {};
  let allCurled = true, allExtended = true;
  for (const [name, mcp, pip, dip, tip] of fingerDefs) {
    const bend = fingerBend(P, mcp, pip, dip, tip);
    // Curled: joint sharply bent and tip folded back toward the wrist.
    // Extended: joint nearly straight and tip reaching away from the wrist.
    const tipCloserToWrist = mag(sub(P[tip], wrist)) < mag(sub(P[pip], wrist));
    curled[name] = bend <= THRESH.curlMaxAngle && tipCloserToWrist;
    extended[name] = bend >= THRESH.extendMinAngle && !tipCloserToWrist;
    if (!curled[name]) allCurled = false;
    if (!extended[name]) allExtended = false;
  }

  // ----- Thumb: right at the edge of the middle finger, barely touching the ring -----
  // The crease runs from the middle finger (PIP-DIP) to the ring finger (PIP-DIP).
  const A = mid(P[10], P[11]); // middle finger contact zone
  const B = mid(P[14], P[15]); // ring finger contact zone
  const abx = B.x - A.x, aby = B.y - A.y;
  const abLenSq = abx * abx + aby * aby || 1e-6;
  const thumb = P[4];
  // Fraction along middle -> ring: 0 = middle center, 0.5 = touching edges, 1 = ring center.
  const t = smooth("t", ((thumb.x - A.x) * abx + (thumb.y - A.y) * aby) / abLenSq);
  const projX = A.x + t * abx, projY = A.y + t * aby;
  // Perpendicular distance from the crease (how close the thumb hugs the fingers).
  const creaseDist = smooth("creaseDist", Math.hypot(thumb.x - projX, thumb.y - projY));

  // "Slightly toward the middle finger's centre, not the edge" => t just
  // under 0.5 (0 = middle centre, 0.5 = boundary, 1 = ring centre). The
  // narrow window + tight crease tolerance keeps the thumb LOCKED in place.
  const thumbEdge =
    t >= THRESH.thumbGapMin &&
    t <= THRESH.thumbGapMax &&
    creaseDist <= THRESH.creaseTolerance * palmWidth;

  // ----- Thumb lies OUTSIDE the finger curls (near the camera) -----
  // A thumb tucked inside the fist sits deeper than the curled middle/ring
  // ridge (larger z = farther from camera in MediaPipe). "Locked tight" also
  // means it must rest ON the ridge, not float in front of it.
  const refZ = Math.min(P[10].z, P[14].z);
  const depthDelta = smooth("depthDelta", thumb.z - refZ);
  const outside = depthDelta <= THRESH.outsideDepth * palmWidth;
  const contact = depthDelta >= -THRESH.contactDepth * palmWidth;
  const trapped = depthDelta >= THRESH.trappedDepth * palmWidth;

  const thumbOk = allCurled && thumbEdge && outside && contact && !trapped;

  // ----- Gradual, hold-to-advance steps (>= THRESH.holdMs each) -----
  let progress = 0;
  if (training.step === 1) {
    progress = advanceHold(allExtended, now);
    if (progress >= 1) {
      training.done.open = true;
      training.step = 2;
      training.holdStart = null;
      flashUntil = now + 600;
      progress = 0;
    }
  } else if (training.step === 2) {
    progress = advanceHold(allCurled, now);
    if (progress >= 1) {
      training.done.curl = true;
      training.step = 3;
      training.holdStart = null;
      flashUntil = now + 600;
      progress = 0;
    }
  } else if (training.step === 3) {
    progress = advanceHold(thumbOk, now);
    if (progress >= 1) {
      training.done.seiken = true;
      training.step = 4;
      training.holdStart = null;
      flashUntil = now + 600;
      progress = 1;
    }
  } else {
    progress = 1; // finished: all three steps held
  }

  // ----- Feedback: checklist + hold bar + current instruction -----
  let hint;
  if (training.step === 1) {
    const missing = Object.keys(extended).filter((k) => !extended[k]);
    hint = allExtended
      ? "Hold the extended hand straight..."
      : "Step 1: Extend " + (missing.length ? missing.join(" & ") : "all") + " fingers straight and flat.";
  } else if (training.step === 2) {
    const missing = Object.keys(curled).filter((k) => !curled[k]);
    hint = allCurled
      ? "Hold the fist..."
      : "Step 2: Curl " + (missing.length ? missing.join(" & ") : "all") + " fingers tightly into a fist.";
  } else if (training.step === 3) {
    if (trapped) hint = "Step 3: Thumb is INSIDE the fist. Keep it on the OUTSIDE of the curls.";
    else if (!thumbEdge) hint = "Step 3: Put the thumb tip just toward the middle of the middle finger (not on the edge), pressed against it.";
    else if (!outside) hint = "Step 3: Rest the thumb ON the outside of the fingers (closer to the camera).";
    else if (!contact) hint = "Step 3: Press the thumb tight onto the finger — no gap, lock it down.";
    else hint = "Hold the perfect Seiken... thumb locked tight.";
  } else {
    hint = "Perfect Seiken! All steps held. Strike with the index & middle knuckles.";
  }

  updateHud(progress, hint, training.step);

  return {
    curled, extended, allCurled, allExtended,
    t, creaseDist, thumbEdge,
    depthDelta, outside, contact, trapped, palmWidth, thumbOk,
    step: training.step, done: training.done, progress,
    points: { A, B }, tIdeal: THRESH.thumbGapIdeal
  };
}

// ---------- Visual aids ----------
const FINGER_BONES = [
  [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]
];
const THUMB_BONES = [[0, 1], [1, 2], [2, 3], [3, 4]];
const FINGER_TIPS = { index: 8, middle: 12, ring: 16, pinky: 20 };
const FINGER_PIPS = { index: 6, middle: 10, ring: 14, pinky: 18 };

function toPx(p, W, H) { return { x: p.x * W, y: p.y * H }; }

function drawArrow(from, to, color, width) {
  const ang = Math.atan2(to.y - from.y, to.x - from.x);
  const len = Math.hypot(to.x - from.x, to.y - from.y) || 1;
  const head = Math.min(12, len * 0.6);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(to.x, to.y);
  ctx.lineTo(to.x - head * Math.cos(ang - Math.PI / 7), to.y - head * Math.sin(ang - Math.PI / 7));
  ctx.lineTo(to.x - head * Math.cos(ang + Math.PI / 7), to.y - head * Math.sin(ang + Math.PI / 7));
  ctx.closePath();
  ctx.fill();
}

function drawGhost(landmarks, step, satisfied, W, H) {
  const local = POSES[step];
  if (!local) return;
  const pts = fromLocal(local, landmarks);
  const alpha = satisfied ? 0.10 : 0.34;
  ctx.strokeStyle = "rgba(255,255,255," + alpha + ")";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pts[a].x * W, pts[a].y * H);
    ctx.lineTo(pts[b].x * W, pts[b].y * H);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255," + alpha + ")";
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x * W, p.y * H, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawStatusRing(at, ok, radius) {
  ctx.strokeStyle = ok ? COLORS.good : COLORS.bad;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(at.x, at.y, radius, 0, 2 * Math.PI);
  ctx.stroke();
}

// Per-step motion cues: outward stretch (1), curl inward (2), thumb -> target (3).
function drawStepAids(landmarks, result, W, H) {
  if (result.step === 1) {
    for (const name of Object.keys(FINGER_TIPS)) {
      const tip = toPx(landmarks[FINGER_TIPS[name]], W, H);
      const pip = toPx(landmarks[FINGER_PIPS[name]], W, H);
      if (!result.extended[name]) {
        const dx = tip.x - pip.x, dy = tip.y - pip.y, l = Math.hypot(dx, dy) || 1;
        drawArrow(tip, { x: tip.x + (dx / l) * 30, y: tip.y + (dy / l) * 30 }, COLORS.arrow, 3);
      }
      drawStatusRing(tip, result.extended[name], 8);
    }
  } else if (result.step === 2) {
    for (const name of Object.keys(FINGER_TIPS)) {
      const tip = toPx(landmarks[FINGER_TIPS[name]], W, H);
      const pip = toPx(landmarks[FINGER_PIPS[name]], W, H);
      if (!result.curled[name]) {
        const dx = pip.x - tip.x, dy = pip.y - tip.y, l = Math.hypot(dx, dy) || 1;
        drawArrow(tip, { x: tip.x + (dx / l) * 26, y: tip.y + (dy / l) * 26 }, COLORS.arrow, 3);
      }
      drawStatusRing(tip, result.curled[name], 8);
    }
  } else if (result.step === 3) {
    const thumb = toPx(landmarks[4], W, H);
    const A = result.points.A, B = result.points.B;
    const target = {
      x: (A.x + result.tIdeal * (B.x - A.x)) * W,
      y: ((A.y + result.tIdeal * (B.y - A.y)) / aspect) * H
    };
    if (!result.thumbEdge || !result.contact) {
      const between = result.t >= THRESH.thumbGapMin && result.t <= THRESH.thumbGapMax;
      drawArrow(thumb, target, between ? COLORS.warn : COLORS.bad, 3);
    }
    drawStatusRing(thumb, result.thumbOk, 10);
  }
}

function drawFlash(now, W, H) {
  if (now >= flashUntil) return;
  const k = 1 - (flashUntil - now) / 600;
  ctx.strokeStyle = "rgba(76,175,80," + (1 - k).toFixed(2) + ")";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 40 + 150 * k, 0, 2 * Math.PI);
  ctx.stroke();
}

function drawLandmarks(landmarks, result, now) {
  const W = canvas.width, H = canvas.height;
  const step = result ? Math.min(result.step, 3) : 0;
  const finished = !!(result && result.step >= 4);

  // Base skeleton (dim).
  ctx.strokeStyle = COLORS.bone;
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * W, landmarks[a].y * H);
    ctx.lineTo(landmarks[b].x * W, landmarks[b].y * H);
    ctx.stroke();
  }
  ctx.fillStyle = COLORS.point;
  for (const pt of landmarks) {
    ctx.beginPath();
    ctx.arc(pt.x * W, pt.y * H, 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  if (!result) return;

  // Highlight the part of the hand the current step is about.
  let activeBones = [], activeJoints = [];
  if (step === 1 || step === 2) {
    activeBones = FINGER_BONES;
    activeJoints = step === 1 ? [8, 12, 16, 20] : [6, 10, 14, 18];
  } else if (step === 3) {
    activeBones = THUMB_BONES;
    activeJoints = [1, 2, 3, 4];
  }
  ctx.strokeStyle = COLORS.active;
  ctx.lineWidth = 3.5;
  for (const [a, b] of activeBones) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * W, landmarks[a].y * H);
    ctx.lineTo(landmarks[b].x * W, landmarks[b].y * H);
    ctx.stroke();
  }
  ctx.fillStyle = COLORS.active;
  for (const j of activeJoints) {
    ctx.beginPath();
    ctx.arc(landmarks[j].x * W, landmarks[j].y * H, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Ghost target shape, fading once the step's shape matches.
  if (!finished) {
    const satisfied = step === 1 ? result.allExtended : step === 2 ? result.allCurled : result.thumbOk;
    drawGhost(landmarks, step, satisfied, W, H);
  }

  // Middle -> ring crease + target spot (only relevant for step 3).
  if (step === 3 || finished) {
    const A = result.points.A, B = result.points.B;
    const ax = A.x * W, ay = (A.y / aspect) * H;
    const bx = B.x * W, by = (B.y / aspect) * H;
    ctx.strokeStyle = COLORS.crease;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    const ti = result.tIdeal != null ? result.tIdeal : THRESH.thumbGapIdeal;
    ctx.strokeStyle = result.thumbEdge ? COLORS.good : COLORS.warn;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(ax + (bx - ax) * ti, ay + (by - ay) * ti, 9, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Motion arrows + per-finger ticks while the step is unmet.
  if (!finished) drawStepAids(landmarks, result, W, H);

  // Thumb tip: green only for a fully valid Seiken thumb.
  const tip = landmarks[4];
  ctx.fillStyle = step === 3 ? (result.thumbOk ? COLORS.good : COLORS.bad) : "rgba(255,255,255,0.55)";
  ctx.beginPath();
  ctx.arc(tip.x * W, tip.y * H, 7, 0, 2 * Math.PI);
  ctx.fill();

  drawFlash(now, W, H);
}

function updateDebug(r) {
  if (!debugEl) return;
  const curledCount = Object.keys(r.curled).filter((k) => r.curled[k]).length;
  const extendedCount = Object.keys(r.extended).filter((k) => r.extended[k]).length;
  debugEl.innerText =
    "step " + (r.step >= 4 ? "done 3/3" : Math.min(r.step, 3) + "/3") +
    " | edge t=" + r.t.toFixed(2) +
    " | crease=" + (r.creaseDist / r.palmWidth).toFixed(2) + "W" +
    " | depth=" + (r.depthDelta / r.palmWidth).toFixed(2) + "W" +
    " | lock=" + (r.thumbOk ? "Y" : "N") +
    " | curl=" + curledCount + "/4 ext=" + extendedCount + "/4";
}

init();
