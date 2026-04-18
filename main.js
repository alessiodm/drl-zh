/* =========================================================
   Deep Reinforcement Learning: Zero to Hero — main.js
   Hero particles · live gridworld · companion sim · misc
   ========================================================= */

/* ---------------------------------------------------------
   Utility
   --------------------------------------------------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOutCubic = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const rand = (a, b) => a + Math.random() * (b - a);

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------------------------------------------------------
   1. Hero particle system — morphing poses
   Poses: robot → neural network → lunar lander → galaxy → loop
   --------------------------------------------------------- */
(function heroParticles() {
    const canvas = $('#heroCanvas');
    if (!canvas || prefersReducedMotion) return;

    const ctx = canvas.getContext('2d');
    const morphLabel = $('#morphLabel');

    const PARTICLE_COUNT = 500;
    const POSE_DURATION = 6500;   // ms to hold a pose
    const TRANSITION_DURATION = 2200;
    const AMBER = 'rgba(255, 196, 61, ';
    const CYAN  = 'rgba(34, 211, 238, ';

    let W = 0, H = 0, DPR = 1;
    let particles = [];
    const poses = {};
    const poseOrder = ['robot', 'nn', 'lander', 'galaxy'];
    let poseIndex = 0;
    let lastPoseSwitch = performance.now();
    let inTransition = false;
    let transitionStart = 0;

    function resize() {
        DPR = Math.min(window.devicePixelRatio || 1, 2);
        W = canvas.clientWidth;
        H = canvas.clientHeight;
        canvas.width = W * DPR;
        canvas.height = H * DPR;
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
        regeneratePoses();
    }

    /* ----- Pose generation: each produces N (x,y) targets in canvas coords ----- */

    function poseRobot(n) {
        // Realistic humanoid robot — Atlas/Optimus silhouette: helmet head
        // with a visor (no face), trapezoidal torso, visible joints at
        // shoulders/elbows/wrists/hips/knees/ankles, solid feet. Built from
        // rotated-rectangle outlines and annular rings so the silhouette
        // reads mechanical, not cartoonish.
        const cx = W * 0.5;
        const cy = H * 0.50;
        const scale = Math.min(W, H) * 0.85;
        const pts = [];
        const toAbs = (x, y) => [cx + x * scale, cy + y * scale];

        // Sample points on a rotated rectangle outline (segment with thickness).
        function sampleSegOutline(count, ax, ay, bx, by, thickness) {
            const dx = bx - ax, dy = by - ay;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const ux = dx / len, uy = dy / len;
            const nx = -uy, ny = ux;
            const halfT = thickness / 2;
            const perim = 2 * len + 2 * thickness;
            for (let i = 0; i < count; i++) {
                let d = Math.random() * perim;
                let lineT, offN;
                if (d < len) { lineT = d / len; offN = halfT; }
                else if (d < len + thickness) { lineT = 1; offN = halfT - (d - len); }
                else if (d < 2 * len + thickness) { lineT = 1 - (d - len - thickness) / len; offN = -halfT; }
                else { lineT = 0; offN = -halfT + (d - 2 * len - thickness); }
                pts.push([ax + dx * lineT + nx * offN, ay + dy * lineT + ny * offN]);
            }
        }
        function sampleRing(count, px, py, r, thickness) {
            for (let i = 0; i < count; i++) {
                const rr = r + (Math.random() - 0.5) * thickness;
                const ang = Math.random() * Math.PI * 2;
                pts.push([px + Math.cos(ang) * rr, py + Math.sin(ang) * rr]);
            }
        }
        function sampleFilledRect(count, left, right, top, bot) {
            for (let i = 0; i < count; i++) {
                pts.push([rand(left, right), rand(top, bot)]);
            }
        }
        // Rounded rectangle outline (for helmet head)
        function sampleRoundRectOutline(count, left, right, top, bot, r) {
            const w = right - left, h = bot - top;
            const sH = Math.max(0, w - 2 * r);
            const sV = Math.max(0, h - 2 * r);
            const arc = (Math.PI * r) / 2;
            const perim = 2 * sH + 2 * sV + 4 * arc;
            for (let i = 0; i < count; i++) {
                let d = Math.random() * perim;
                if (d < sH) { pts.push([left + r + d, top]); continue; }
                d -= sH;
                if (d < arc) { const a = -Math.PI / 2 + d / r; pts.push([right - r + Math.cos(a) * r, top + r + Math.sin(a) * r]); continue; }
                d -= arc;
                if (d < sV) { pts.push([right, top + r + d]); continue; }
                d -= sV;
                if (d < arc) { const a = d / r; pts.push([right - r + Math.cos(a) * r, bot - r + Math.sin(a) * r]); continue; }
                d -= arc;
                if (d < sH) { pts.push([right - r - d, bot]); continue; }
                d -= sH;
                if (d < arc) { const a = Math.PI / 2 + d / r; pts.push([left + r + Math.cos(a) * r, bot - r + Math.sin(a) * r]); continue; }
                d -= arc;
                if (d < sV) { pts.push([left, bot - r - d]); continue; }
                d -= sV;
                const a = Math.PI + d / r;
                pts.push([left + r + Math.cos(a) * r, top + r + Math.sin(a) * r]);
            }
        }

        // ---- Landmarks (normalized; scaled via toAbs) ----
        // Head (helmet) bounding box
        const headL = -0.055, headR = 0.055;
        const headT = -0.40,  headB = -0.26;
        // Visor line (horizontal strip across upper half of head)
        const visorY = -0.34;
        // Neck
        const neckTop = toAbs( 0,    -0.26);
        const neckBot = toAbs( 0,    -0.22);
        // Torso (trapezoid)
        const shL = toAbs(-0.13, -0.22);  // left shoulder (torso top-left)
        const shR = toAbs( 0.13, -0.22);
        const waL = toAbs(-0.09,  0.02);  // left waist (torso bottom-left)
        const waR = toAbs( 0.09,  0.02);
        // Arms
        const elbowL = toAbs(-0.17, -0.05);
        const elbowR = toAbs( 0.17, -0.05);
        const wristL = toAbs(-0.165, 0.12);
        const wristR = toAbs( 0.165, 0.12);
        // Hips
        const hipL = toAbs(-0.065, 0.03);
        const hipR = toAbs( 0.065, 0.03);
        // Legs
        const kneeL  = toAbs(-0.065, 0.20);
        const kneeR  = toAbs( 0.065, 0.20);
        const ankleL = toAbs(-0.065, 0.38);
        const ankleR = toAbs( 0.065, 0.38);

        // ---- Budget (normalized automatically by sum) ----
        const b = {
            head:        0.09,
            visor:       0.03,
            neck:        0.015,
            torsoShoulder: 0.04,
            torsoWaist:    0.03,
            torsoLeftD:    0.05,
            torsoRightD:   0.05,
            sternum:       0.03,
            shL_ring: 0.025, shR_ring: 0.025,
            upperArmL: 0.055, upperArmR: 0.055,
            elbowL_ring: 0.02, elbowR_ring: 0.02,
            forearmL: 0.055, forearmR: 0.055,
            wristL_ring: 0.018, wristR_ring: 0.018,
            hipL_ring: 0.015, hipR_ring: 0.015,
            upperLegL: 0.065, upperLegR: 0.065,
            kneeL_ring: 0.025, kneeR_ring: 0.025,
            lowerLegL: 0.055, lowerLegR: 0.055,
            ankleL_ring: 0.012, ankleR_ring: 0.012,
            footL: 0.02, footR: 0.02,
        };
        const tot = Object.values(b).reduce((s, v) => s + v, 0);
        const c = {}; let used = 0;
        for (const k in b) { c[k] = Math.floor((n * b[k]) / tot); used += c[k]; }
        c.torsoLeftD += (n - used);

        // ---- HEAD (helmet outline + horizontal visor strip) ----
        sampleRoundRectOutline(c.head,
            cx + headL * scale, cx + headR * scale,
            cy + headT * scale, cy + headB * scale,
            0.025 * scale);
        // Visor: a thin horizontal band across the helmet (sampled as filled rect)
        sampleFilledRect(c.visor,
            cx + (headL + 0.005) * scale, cx + (headR - 0.005) * scale,
            cy + (visorY - 0.008) * scale, cy + (visorY + 0.008) * scale);

        // ---- NECK ----
        sampleSegOutline(c.neck, neckTop[0], neckTop[1], neckBot[0], neckBot[1], 0.03 * scale);

        // ---- TORSO (trapezoid outline + sternum line) ----
        sampleSegOutline(c.torsoShoulder, shL[0], shL[1], shR[0], shR[1], 3);
        sampleSegOutline(c.torsoWaist,    waL[0], waL[1], waR[0], waR[1], 3);
        sampleSegOutline(c.torsoLeftD,    shL[0], shL[1], waL[0], waL[1], 3);
        sampleSegOutline(c.torsoRightD,   shR[0], shR[1], waR[0], waR[1], 3);
        // Sternum / chest plate divider
        sampleSegOutline(c.sternum,
            cx, (shL[1] + waL[1]) / 2 - 0.08 * scale,
            cx, (shL[1] + waL[1]) / 2 + 0.08 * scale, 3);

        // ---- ARMS ----
        sampleRing(c.shL_ring, shL[0], shL[1], 0.035 * scale, 0.016 * scale);
        sampleRing(c.shR_ring, shR[0], shR[1], 0.035 * scale, 0.016 * scale);
        sampleSegOutline(c.upperArmL, shL[0], shL[1], elbowL[0], elbowL[1], 0.042 * scale);
        sampleSegOutline(c.upperArmR, shR[0], shR[1], elbowR[0], elbowR[1], 0.042 * scale);
        sampleRing(c.elbowL_ring, elbowL[0], elbowL[1], 0.028 * scale, 0.014 * scale);
        sampleRing(c.elbowR_ring, elbowR[0], elbowR[1], 0.028 * scale, 0.014 * scale);
        sampleSegOutline(c.forearmL, elbowL[0], elbowL[1], wristL[0], wristL[1], 0.036 * scale);
        sampleSegOutline(c.forearmR, elbowR[0], elbowR[1], wristR[0], wristR[1], 0.036 * scale);
        sampleRing(c.wristL_ring, wristL[0], wristL[1], 0.022 * scale, 0.012 * scale);
        sampleRing(c.wristR_ring, wristR[0], wristR[1], 0.022 * scale, 0.012 * scale);

        // ---- HIPS / LEGS ----
        sampleRing(c.hipL_ring, hipL[0], hipL[1], 0.03 * scale, 0.014 * scale);
        sampleRing(c.hipR_ring, hipR[0], hipR[1], 0.03 * scale, 0.014 * scale);
        sampleSegOutline(c.upperLegL, hipL[0], hipL[1], kneeL[0], kneeL[1], 0.05 * scale);
        sampleSegOutline(c.upperLegR, hipR[0], hipR[1], kneeR[0], kneeR[1], 0.05 * scale);
        sampleRing(c.kneeL_ring, kneeL[0], kneeL[1], 0.032 * scale, 0.015 * scale);
        sampleRing(c.kneeR_ring, kneeR[0], kneeR[1], 0.032 * scale, 0.015 * scale);
        sampleSegOutline(c.lowerLegL, kneeL[0], kneeL[1], ankleL[0], ankleL[1], 0.042 * scale);
        sampleSegOutline(c.lowerLegR, kneeR[0], kneeR[1], ankleR[0], ankleR[1], 0.042 * scale);
        sampleRing(c.ankleL_ring, ankleL[0], ankleL[1], 0.022 * scale, 0.012 * scale);
        sampleRing(c.ankleR_ring, ankleR[0], ankleR[1], 0.022 * scale, 0.012 * scale);

        // ---- FEET (filled small rectangles extending slightly forward) ----
        sampleFilledRect(c.footL,
            ankleL[0] - 0.04 * scale, ankleL[0] + 0.06 * scale,
            ankleL[1] + 0.005 * scale, ankleL[1] + 0.025 * scale);
        sampleFilledRect(c.footR,
            ankleR[0] - 0.04 * scale, ankleR[0] + 0.06 * scale,
            ankleR[1] + 0.005 * scale, ankleR[1] + 0.025 * scale);

        while (pts.length < n) pts.push([cx, cy]);
        return pts.slice(0, n);
    }

    function poseNN(n) {
        // Lens-shaped feed-forward net with curved synaptic edges.
        // Narrow input, wide hidden, narrower output — reads more "neural" than a grid.
        const layers = [3, 7, 9, 7, 3];
        const cx = W * 0.5;
        const cy = H * 0.5;
        const fieldW = Math.min(W * 0.78, 1000);
        const layerSpan = Math.min(H * 0.66, 520);

        // Build node positions. Y positions sit on a gentle arc per layer
        // (slight bow outward at the middle) so columns don't read perfectly straight.
        const nodePositions = [];
        for (let li = 0; li < layers.length; li++) {
            const count = layers[li];
            const lx = cx - fieldW / 2 + (fieldW * li) / (layers.length - 1);
            for (let ni = 0; ni < count; ni++) {
                const t = count === 1 ? 0 : (ni - (count - 1) / 2) / Math.max(1, (count - 1) / 2);
                // t in [-1, 1]
                const y = cy + (t * layerSpan) / 2;
                // Slight horizontal bow: outer nodes pulled slightly toward center x
                const bow = Math.sin(t * Math.PI / 2) * 0.02 * fieldW;
                nodePositions.push([lx + bow * Math.sign(li - (layers.length - 1) / 2) * 0, y, li]);
            }
        }

        const pts = [];

        // 18% cluster at nodes (soft halos, not hard dots)
        const nodeBudget = Math.floor(n * 0.18);
        for (let i = 0; i < nodeBudget; i++) {
            const node = nodePositions[i % nodePositions.length];
            const r = Math.sqrt(Math.random()) * 7;
            const a = Math.random() * Math.PI * 2;
            pts.push([node[0] + Math.cos(a) * r, node[1] + Math.sin(a) * r]);
        }

        // 82% along curved edges between adjacent layers (quadratic bezier)
        const edgePairs = [];
        for (let li = 0; li < layers.length - 1; li++) {
            const leftNodes = nodePositions.filter((p) => p[2] === li);
            const rightNodes = nodePositions.filter((p) => p[2] === li + 1);
            for (const a of leftNodes) for (const b of rightNodes) edgePairs.push([a, b]);
        }

        const remaining = n - nodeBudget;
        for (let i = 0; i < remaining; i++) {
            const [a, b] = edgePairs[Math.floor(Math.random() * edgePairs.length)];
            const t = Math.random();

            // Curve: offset control point perpendicular to segment.
            // Sign alternates in a pseudo-random, stable way so edges arc both ways.
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const nx = -dy / len;
            const ny = dx / len;

            // Stable sign based on edge identity, plus varying magnitude.
            const hash = (a[0] * 13 + a[1] * 7 + b[0] * 5 + b[1] * 3) | 0;
            const sign = (hash % 2 === 0) ? 1 : -1;
            const curveMag = 10 + (Math.abs(hash) % 18);
            const offset = sign * curveMag;

            const mx = (a[0] + b[0]) / 2 + nx * offset;
            const my = (a[1] + b[1]) / 2 + ny * offset;

            // Quadratic bezier point
            const u = 1 - t;
            const px = u * u * a[0] + 2 * u * t * mx + t * t * b[0];
            const py = u * u * a[1] + 2 * u * t * my + t * t * b[1];

            pts.push([px + rand(-0.8, 0.8), py + rand(-0.8, 0.8)]);
        }

        while (pts.length < n) {
            const node = nodePositions[pts.length % nodePositions.length];
            pts.push([node[0], node[1]]);
        }
        return pts.slice(0, n);
    }

    function poseLander(n) {
        // Lunar lander: diamond body, two legs, surface dust line below.
        const cx = W * 0.5;
        const cy = H * 0.46;
        const scale = Math.min(W, H) * 0.36;

        const pts = [];
        // 55% body (diamond outline + fill sparse)
        const body = Math.floor(n * 0.45);
        for (let i = 0; i < body; i++) {
            const tri = Math.random();
            let rx, ry;
            if (tri < 0.5) {
                const t = Math.random();
                rx = lerp(-0.25, 0.25, t);
                ry = lerp(-0.28, 0.0, Math.abs(t - 0.5) * 2);
            } else {
                const t = Math.random();
                rx = lerp(-0.30, 0.30, t);
                ry = lerp(0.0, 0.18, 1 - Math.abs(t - 0.5) * 2);
            }
            pts.push([cx + rx * scale, cy + ry * scale]);
        }
        // 20% legs
        const legs = Math.floor(n * 0.15);
        for (let i = 0; i < legs; i++) {
            const side = Math.random() < 0.5 ? -1 : 1;
            const t = Math.random();
            const x = side * lerp(0.18, 0.40, t);
            const y = lerp(0.0, 0.30, t);
            pts.push([cx + x * scale, cy + y * scale]);
        }
        // 15% thruster plume
        const plume = Math.floor(n * 0.12);
        for (let i = 0; i < plume; i++) {
            const t = Math.random();
            const x = rand(-0.08, 0.08);
            const y = 0.18 + t * 0.18 + rand(-0.02, 0.02);
            pts.push([cx + x * scale, cy + y * scale]);
        }
        // 10% surface dust scattered horizontally
        const surfaceY = cy + 0.36 * scale;
        while (pts.length < n) {
            const x = rand(-W * 0.5, W * 0.5);
            const y = surfaceY + rand(-4, 20);
            pts.push([cx + x, y]);
        }
        return pts.slice(0, n);
    }

    function poseGalaxy(n) {
        const cx = W * 0.5;
        const cy = H * 0.5;
        const maxR = Math.min(W, H) * 0.42;
        const arms = 3;
        const pts = [];
        for (let i = 0; i < n; i++) {
            // Pick an arm
            const arm = i % arms;
            // Radial distribution favoring outer ring
            const r = maxR * Math.pow(Math.random(), 0.65);
            // Log-spiral angle
            const baseAngle = (arm / arms) * Math.PI * 2;
            const swirl = 2.2 * Math.log(1 + r / (maxR * 0.2));
            const spread = rand(-0.25, 0.25);
            const angle = baseAngle + swirl + spread;
            pts.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
        }
        return pts.slice(0, n);
    }

    function regeneratePoses() {
        poses.robot = poseRobot(PARTICLE_COUNT);
        poses.nn = poseNN(PARTICLE_COUNT);
        poses.lander = poseLander(PARTICLE_COUNT);
        poses.galaxy = poseGalaxy(PARTICLE_COUNT);

        // Initialize or reassign particle targets
        if (particles.length === 0) {
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const start = poses[poseOrder[0]][i];
                particles.push({
                    x: start[0] + rand(-30, 30),
                    y: start[1] + rand(-30, 30),
                    px: 0, py: 0,
                    tx: start[0], ty: start[1],
                    startX: start[0], startY: start[1],
                    phase: Math.random() * Math.PI * 2,
                    size: rand(0.8, 1.8),
                    colorMix: Math.random(),  // 0 = amber, 1 = cyan
                });
            }
        } else {
            // On resize, re-target without jarring jumps
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const t = poses[poseOrder[poseIndex]][i];
                particles[i].tx = t[0];
                particles[i].ty = t[1];
                particles[i].startX = t[0];
                particles[i].startY = t[1];
            }
        }
    }

    function advancePose() {
        poseIndex = (poseIndex + 1) % poseOrder.length;
        const nextPose = poses[poseOrder[poseIndex]];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles[i].startX = particles[i].x;
            particles[i].startY = particles[i].y;
            particles[i].tx = nextPose[i][0];
            particles[i].ty = nextPose[i][1];
        }
        if (morphLabel) {
            morphLabel.style.color = 'var(--amber)';
            morphLabel.textContent = poseOrder[poseIndex];
            setTimeout(() => { morphLabel.style.color = ''; }, 600);
        }
        inTransition = true;
        transitionStart = performance.now();
        lastPoseSwitch = performance.now();
    }

    // Spatial grid — reused across frames, sized on resize
    const EDGE_DIST = 46;
    const EDGE_DIST_SQ = EDGE_DIST * EDGE_DIST;
    let gridCols = 0, gridRows = 0;
    let grid = null;
    // Edge buffers — per-frame bucketed line segments for batched strokes.
    // Each bucket holds flat [x1,y1,x2,y2,...] numbers. One strokeStyle/stroke() per bucket.
    const EDGE_BUCKET_COUNT = 3;
    const EDGE_BUCKET_ALPHAS = [0.04, 0.09, 0.15];
    const edgeBuckets = [];
    for (let i = 0; i < EDGE_BUCKET_COUNT; i++) edgeBuckets.push([]);

    function ensureGrid() {
        const cols = Math.max(1, Math.ceil(W / EDGE_DIST) + 1);
        const rows = Math.max(1, Math.ceil(H / EDGE_DIST) + 1);
        if (cols !== gridCols || rows !== gridRows || !grid) {
            gridCols = cols; gridRows = rows;
            grid = new Array(cols * rows);
            for (let i = 0; i < grid.length; i++) grid[i] = [];
        }
    }

    // Pause hero when it scrolls off-screen.
    let heroRunning = true;
    const heroIO = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
            const wasRunning = heroRunning;
            heroRunning = e.isIntersecting;
            if (heroRunning && !wasRunning) {
                // Re-anchor timing so pose doesn't snap-switch on resume
                lastPoseSwitch = performance.now();
                requestAnimationFrame(frame);
            }
        });
    }, { threshold: 0 });
    heroIO.observe(canvas);

    function frame(now) {
        if (!heroRunning) return;
        ctx.clearRect(0, 0, W, H);

        // Transition progress
        let tProg = 1;
        if (inTransition) {
            tProg = clamp((now - transitionStart) / TRANSITION_DURATION, 0, 1);
            if (tProg >= 1) inTransition = false;
        }
        const eased = easeInOutCubic(tProg);

        // Schedule next pose
        if (!inTransition && now - lastPoseSwitch > POSE_DURATION) {
            advancePose();
        }

        // --- Update particle positions ---
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const p = particles[i];
            const bx = lerp(p.startX, p.tx, eased);
            const by = lerp(p.startY, p.ty, eased);
            const idleAmp = inTransition ? 0 : 2.5;
            const driftX = Math.sin(now * 0.0006 + p.phase) * idleAmp;
            const driftY = Math.cos(now * 0.0005 + p.phase * 1.3) * idleAmp;
            p.x = bx + driftX;
            p.y = by + driftY;
        }

        // --- Edge pass: bucket nearby-particle pairs, draw each bucket with a single stroke() ---
        ensureGrid();
        for (let i = 0; i < grid.length; i++) grid[i].length = 0;
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const p = particles[i];
            const gx = clamp(Math.floor(p.x / EDGE_DIST), 0, gridCols - 1);
            const gy = clamp(Math.floor(p.y / EDGE_DIST), 0, gridRows - 1);
            grid[gy * gridCols + gx].push(i);
        }
        for (let k = 0; k < EDGE_BUCKET_COUNT; k++) edgeBuckets[k].length = 0;
        for (let gy = 0; gy < gridRows; gy++) {
            for (let gx = 0; gx < gridCols; gx++) {
                const cell = grid[gy * gridCols + gx];
                if (!cell.length) continue;
                // Own cell + 4 of 8 neighbors (right, down-left, down, down-right) to avoid doubles
                for (let n = 0; n < 5; n++) {
                    let ox, oy;
                    if (n === 0) { ox = 0; oy = 0; }
                    else if (n === 1) { ox = 1; oy = 0; }
                    else if (n === 2) { ox = -1; oy = 1; }
                    else if (n === 3) { ox = 0; oy = 1; }
                    else { ox = 1; oy = 1; }
                    const nx = gx + ox, ny = gy + oy;
                    if (nx < 0 || nx >= gridCols || ny < 0 || ny >= gridRows) continue;
                    const other = grid[ny * gridCols + nx];
                    if (!other.length) continue;
                    const sameCell = (n === 0);
                    for (let a = 0; a < cell.length; a++) {
                        const pa = particles[cell[a]];
                        for (let b = (sameCell ? a + 1 : 0); b < other.length; b++) {
                            const pb = particles[other[b]];
                            const dx = pb.x - pa.x;
                            const dy = pb.y - pa.y;
                            const distSq = dx * dx + dy * dy;
                            if (distSq > EDGE_DIST_SQ) continue;
                            // Bucket: 0 (faint, far) → 2 (brightest, closest)
                            // distSq in [0, EDGE_DIST_SQ] → bucket index via thresholds
                            let k;
                            if (distSq < EDGE_DIST_SQ * 0.25) k = 2;
                            else if (distSq < EDGE_DIST_SQ * 0.6) k = 1;
                            else k = 0;
                            const buf = edgeBuckets[k];
                            buf.push(pa.x, pa.y, pb.x, pb.y);
                        }
                    }
                }
            }
        }

        // Draw each bucket as a single path + stroke
        const edgeAlphaMul = inTransition ? 0.85 : 1.0;
        ctx.lineWidth = 0.6;
        for (let k = 0; k < EDGE_BUCKET_COUNT; k++) {
            const buf = edgeBuckets[k];
            if (!buf.length) continue;
            ctx.strokeStyle = AMBER + (EDGE_BUCKET_ALPHAS[k] * edgeAlphaMul) + ')';
            ctx.beginPath();
            for (let i = 0; i < buf.length; i += 4) {
                ctx.moveTo(buf[i], buf[i + 1]);
                ctx.lineTo(buf[i + 2], buf[i + 3]);
            }
            ctx.stroke();
        }

        // --- Particle pass (drawn on top of edges) ---
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const p = particles[i];
            const cyanProb = inTransition ? 0.18 + (1 - Math.abs(eased - 0.5) * 2) * 0.25 : 0.12;
            const useCyan = p.colorMix < cyanProb;
            const alpha = inTransition ? 0.55 + Math.sin(tProg * Math.PI) * 0.35 : 0.75;
            ctx.fillStyle = (useCyan ? CYAN : AMBER) + alpha + ')';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        }

        requestAnimationFrame(frame);
    }

    window.addEventListener('resize', resize);
    resize();
    // Kick off first transition after short initial hold
    setTimeout(advancePose, 2500);
    requestAnimationFrame(frame);
})();

/* ---------------------------------------------------------
   2. Live gridworld — tabular Q-learning, runs in-browser
   --------------------------------------------------------- */
(function gridworld() {
    const canvas = $('#gridworld');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const elEpisode = $('#gwEpisode');
    const elSteps = $('#gwSteps');
    const elEpsilon = $('#gwEpsilon');
    const elReward = $('#gwReward');
    const rewardCanvas = $('#rewardCurve');
    const rewardCtx = rewardCanvas.getContext('2d');
    const btnReset = $('#gwReset');
    const btnSpeed = $('#gwSpeed');

    const ROWS = 7, COLS = 7;
    const ACTIONS = 4; // 0=up,1=right,2=down,3=left
    const DX = [0, 1, 0, -1];
    const DY = [-1, 0, 1, 0];
    const ALPHA = 0.2;
    const GAMMA = 0.95;
    const EPS_START = 1.0;
    const EPS_END = 0.05;
    const EPS_DECAY_EPISODES = 80;
    const STEP_PENALTY = -0.04;
    const GOAL_REWARD = 1.0;
    const WALL_REWARD = -0.5;
    const MAX_STEPS_PER_EP = 80;

    // Walls (set of "r,c")
    const walls = new Set(['2,1', '2,2', '2,3', '4,3', '4,4', '4,5', '1,5']);
    const start = { r: 0, c: 0 };
    const goal  = { r: ROWS - 1, c: COLS - 1 };

    let Q, agent, episode, stepsInEpisode, totalSteps, eps, running, reward;
    let rewardHistory = [];      // per-episode rewards
    let avgReward = 0;
    let speedMultiplier = 1;
    let stepsPerFrame = 3;

    function isWall(r, c) { return walls.has(`${r},${c}`); }

    function reset() {
        Q = Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]);
        agent = { ...start };
        episode = 0;
        stepsInEpisode = 0;
        totalSteps = 0;
        eps = EPS_START;
        reward = 0;
        rewardHistory = [];
        avgReward = 0;
    }

    function idx(r, c) { return r * COLS + c; }

    function stepAgent() {
        // ε-greedy action
        const s = idx(agent.r, agent.c);
        let a;
        if (Math.random() < eps) {
            a = Math.floor(Math.random() * ACTIONS);
        } else {
            const qs = Q[s];
            let best = 0;
            for (let i = 1; i < ACTIONS; i++) if (qs[i] > qs[best]) best = i;
            a = best;
        }

        let nr = agent.r + DY[a];
        let nc = agent.c + DX[a];
        let r = STEP_PENALTY;
        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || isWall(nr, nc)) {
            nr = agent.r; nc = agent.c;
            r = WALL_REWARD;
        }
        let done = false;
        if (nr === goal.r && nc === goal.c) {
            r = GOAL_REWARD;
            done = true;
        }

        const sNext = idx(nr, nc);
        const maxNext = done ? 0 : Math.max(...Q[sNext]);
        Q[s][a] += ALPHA * (r + GAMMA * maxNext - Q[s][a]);

        agent.r = nr;
        agent.c = nc;
        reward += r;
        stepsInEpisode++;
        totalSteps++;

        if (done || stepsInEpisode >= MAX_STEPS_PER_EP) {
            rewardHistory.push(reward);
            if (rewardHistory.length > 400) rewardHistory.shift();
            const recent = rewardHistory.slice(-100);
            avgReward = recent.reduce((s, v) => s + v, 0) / recent.length;
            episode++;
            stepsInEpisode = 0;
            reward = 0;
            agent = { ...start };
            eps = EPS_END + (EPS_START - EPS_END) * Math.max(0, 1 - episode / EPS_DECAY_EPISODES);
        }
    }

    function draw() {
        const w = canvas.width, h = canvas.height;
        const cellW = w / COLS;
        const cellH = h / ROWS;

        ctx.clearRect(0, 0, w, h);

        // Cells with Q-value shading
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const x = c * cellW;
                const y = r * cellH;
                if (isWall(r, c)) {
                    ctx.fillStyle = '#1c2030';
                    ctx.fillRect(x, y, cellW, cellH);
                    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
                    ctx.strokeRect(x, y, cellW, cellH);
                    continue;
                }
                // Max Q for this cell → shade intensity
                const qMax = Math.max(...Q[idx(r, c)]);
                const intensity = clamp(qMax / 1.0, -0.3, 1);
                let bg;
                if (intensity >= 0) {
                    const t = clamp(intensity, 0, 1);
                    // amber shade
                    bg = `rgba(255, 196, 61, ${t * 0.25})`;
                } else {
                    bg = 'rgba(248, 113, 113, 0.08)';
                }
                ctx.fillStyle = '#0a0c12';
                ctx.fillRect(x, y, cellW, cellH);
                ctx.fillStyle = bg;
                ctx.fillRect(x, y, cellW, cellH);
                ctx.strokeStyle = 'rgba(255,255,255,0.04)';
                ctx.strokeRect(x, y, cellW, cellH);

                // Draw best-action arrow (faint) once Q has signal
                if (qMax > 0.05) {
                    const qs = Q[idx(r, c)];
                    let best = 0;
                    for (let i = 1; i < ACTIONS; i++) if (qs[i] > qs[best]) best = i;
                    drawArrow(x + cellW / 2, y + cellH / 2, best, cellW * 0.25);
                }
            }
        }

        // Goal
        const gx = goal.c * cellW, gy = goal.r * cellH;
        ctx.fillStyle = 'rgba(255, 196, 61, 0.9)';
        ctx.shadowColor = 'rgba(255, 196, 61, 0.6)';
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(gx + cellW / 2, gy + cellH / 2, Math.min(cellW, cellH) * 0.22, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Agent
        const ax = agent.c * cellW, ay = agent.r * cellH;
        ctx.fillStyle = '#22d3ee';
        ctx.shadowColor = 'rgba(34, 211, 238, 0.8)';
        ctx.shadowBlur = 14;
        ctx.beginPath();
        ctx.arc(ax + cellW / 2, ay + cellH / 2, Math.min(cellW, cellH) * 0.27, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    function drawArrow(cx, cy, action, len) {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate((action * Math.PI) / 2);  // 0=up → angle 0 pointing up
        // Up arrow: we draw pointing "up" = -y
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, len * 0.8);
        ctx.lineTo(0, -len * 0.5);
        ctx.moveTo(-len * 0.25, -len * 0.2);
        ctx.lineTo(0, -len * 0.5);
        ctx.lineTo(len * 0.25, -len * 0.2);
        ctx.stroke();
        ctx.restore();
    }

    function drawRewardCurve() {
        const w = rewardCanvas.width, h = rewardCanvas.height;
        rewardCtx.clearRect(0, 0, w, h);
        // Axis baseline
        rewardCtx.strokeStyle = 'rgba(255,255,255,0.07)';
        rewardCtx.lineWidth = 1;
        rewardCtx.beginPath();
        rewardCtx.moveTo(0, h - 10);
        rewardCtx.lineTo(w, h - 10);
        rewardCtx.stroke();

        if (rewardHistory.length < 2) return;
        const minR = -4, maxR = 1;
        const scaleY = (r) => h - 10 - ((r - minR) / (maxR - minR)) * (h - 20);

        // Raw dots
        rewardCtx.fillStyle = 'rgba(34, 211, 238, 0.25)';
        for (let i = 0; i < rewardHistory.length; i++) {
            const x = (i / (rewardHistory.length - 1)) * w;
            const y = scaleY(rewardHistory[i]);
            rewardCtx.beginPath();
            rewardCtx.arc(x, y, 1.3, 0, Math.PI * 2);
            rewardCtx.fill();
        }

        // Moving average line
        const window = 10;
        rewardCtx.strokeStyle = '#ffc43d';
        rewardCtx.lineWidth = 2;
        rewardCtx.beginPath();
        let first = true;
        for (let i = 0; i < rewardHistory.length; i++) {
            const from = Math.max(0, i - window);
            const slice = rewardHistory.slice(from, i + 1);
            const avg = slice.reduce((s, v) => s + v, 0) / slice.length;
            const x = (i / (rewardHistory.length - 1)) * w;
            const y = scaleY(avg);
            if (first) { rewardCtx.moveTo(x, y); first = false; }
            else rewardCtx.lineTo(x, y);
        }
        rewardCtx.stroke();
    }

    function updateStats() {
        elEpisode.textContent = episode;
        elSteps.textContent = totalSteps;
        elEpsilon.textContent = eps.toFixed(2);
        elReward.textContent = rewardHistory.length >= 5 ? avgReward.toFixed(2) : '—';
    }

    let rafId = null;
    function loop() {
        if (!running) return;
        for (let i = 0; i < stepsPerFrame; i++) stepAgent();
        draw();
        if (totalSteps % 6 === 0) {
            updateStats();
            drawRewardCurve();
        }
        rafId = requestAnimationFrame(loop);
    }

    btnReset.addEventListener('click', () => {
        reset();
        updateStats();
        draw();
        drawRewardCurve();
    });
    btnSpeed.addEventListener('click', () => {
        const cycle = [1, 2, 4, 8];
        const next = cycle[(cycle.indexOf(speedMultiplier) + 1) % cycle.length];
        speedMultiplier = next;
        stepsPerFrame = 3 * next;
        btnSpeed.textContent = `Speed ×${next}`;
    });

    // Pause off-screen, resume on-screen
    const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
            if (e.isIntersecting && !running) {
                running = true;
                if (!rafId) loop();
            } else if (!e.isIntersecting && running) {
                running = false;
                if (rafId) cancelAnimationFrame(rafId);
                rafId = null;
            }
        });
    }, { threshold: 0.25 });
    io.observe(canvas);

    reset();
    updateStats();
    draw();
    drawRewardCurve();
})();

/* ---------------------------------------------------------
   3. Companion simulator — scripted, choreographed demo
   --------------------------------------------------------- */
(function companionSim() {
    const sim = $('#companionSim');
    if (!sim) return;

    const btn = $('#simPlay');
    const btnLabel = $('#simPlayLabel');
    const messages = $('#compMessages');
    const signals = $$('.signal', sim);
    const cell = sim.querySelector('.vsc-cell:not(.muted)');
    const cursor = $('#simCursor');
    const voiceBtn = $('#voiceToggle');
    const voiceLabel = voiceBtn.querySelector('.voice-label');
    const waveform = $('#waveform');
    const waveCtx = waveform.getContext('2d');

    let playing = false;
    let voiceOn = false;
    let waveActive = false;
    let waveRaf = null;

    function setSignal(name) {
        signals.forEach((s) => {
            s.classList.toggle('active', s.dataset.sig === name);
        });
    }

    function clearMessages(keepSys = true) {
        messages.querySelectorAll('.msg').forEach((m, i) => {
            if (keepSys && i === 0 && m.classList.contains('msg-sys')) return;
            m.remove();
        });
    }

    function addMessage(kind, html) {
        const el = document.createElement('div');
        el.className = `msg msg-${kind}`;
        el.innerHTML = html;
        messages.appendChild(el);
        messages.scrollTop = messages.scrollHeight;
        return el;
    }

    function typeInto(el, text, speed = 24) {
        return new Promise((resolve) => {
            el.innerHTML = '';
            let i = 0;
            const timer = setInterval(() => {
                i++;
                el.textContent = text.slice(0, i);
                messages.scrollTop = messages.scrollHeight;
                if (i >= text.length) {
                    clearInterval(timer);
                    resolve();
                }
            }, speed);
        });
    }

    function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

    function startWaveform() {
        waveActive = true;
        sim.classList.add('voice-active');
        const w = waveform.width;
        const h = waveform.height;
        const bars = 48;
        const barW = w / bars;
        function frame() {
            if (!waveActive) return;
            waveCtx.clearRect(0, 0, w, h);
            for (let i = 0; i < bars; i++) {
                const t = performance.now() * 0.003;
                const amp = Math.abs(Math.sin(t + i * 0.4) * Math.sin(t * 0.7 + i * 0.2));
                const barH = 2 + amp * (h - 4);
                waveCtx.fillStyle = `rgba(34, 211, 238, ${0.3 + amp * 0.55})`;
                waveCtx.fillRect(i * barW + 1, (h - barH) / 2, barW - 2, barH);
            }
            waveRaf = requestAnimationFrame(frame);
        }
        frame();
    }
    function stopWaveform() {
        waveActive = false;
        if (waveRaf) cancelAnimationFrame(waveRaf);
        waveRaf = null;
        waveCtx.clearRect(0, 0, waveform.width, waveform.height);
    }

    voiceBtn.addEventListener('click', () => {
        voiceOn = !voiceOn;
        voiceBtn.classList.toggle('on', voiceOn);
        voiceLabel.textContent = voiceOn ? 'Voice mode: on' : 'Voice mode: off';
        if (voiceOn) startWaveform();
        else stopWaveform();
    });

    async function playDemo() {
        if (playing) return;
        playing = true;
        btn.disabled = true;
        btnLabel.textContent = 'Playing…';

        clearMessages(true);
        setSignal('reading');
        cell.classList.add('active-cell');
        if (cursor) cursor.style.opacity = '1';

        await wait(900);
        addMessage('sys', 'cursor moved to <code>03_DQN.ipynb · cell 3 · line 5</code>');
        await wait(1100);
        setSignal('idle');
        await wait(1300);
        setSignal('stuck');
        addMessage('sys', 'signal fired: <code>stuck</code> · 3 runs, same error');
        await wait(700);

        const ai1 = addMessage('ai', '');
        await typeInto(ai1, "Noticed you've tried a single nn.Linear a few times — let's zoom out. What's the Q-network's job in DQN: regression or classification?", 22);

        await wait(1400);
        const you1 = addMessage('you', '');
        await typeInto(you1, "Regression — each output is a Q-value for an action.", 28);

        await wait(900);
        setSignal('flow');
        if (!voiceOn) {
            voiceOn = true;
            voiceBtn.classList.add('on');
            voiceLabel.textContent = 'Voice mode: on';
            startWaveform();
            addMessage('sys', 'voice mode enabled · Kokoro TTS');
            await wait(600);
        }

        const ai2 = addMessage('ai', '');
        await typeInto(ai2,
            "Right. So the output layer is a Linear with n_actions units and no activation. Two hidden Linear+ReLU layers of ~128 units will do. Want to sketch it together?",
            22);

        await wait(1800);
        const you2 = addMessage('you', '');
        await typeInto(you2, "Got it — I'll write nn.Sequential(Linear, ReLU, Linear, ReLU, Linear).", 28);

        await wait(800);
        addMessage('sys', '✓ drift check passed · implementation aligned with TODO');
        setSignal('flow');

        await wait(1600);
        playing = false;
        btn.disabled = false;
        btnLabel.textContent = 'Replay demo';
    }

    btn.addEventListener('click', playDemo);

    // Subtle idle state so the panel isn't dead before first click
    setSignal('idle');
})();

/* ---------------------------------------------------------
   4. Prerequisites self-check
   --------------------------------------------------------- */
(function prereqCheck() {
    const checks = $$('.prereq-check');
    const verdict = $('#prereqVerdict');
    if (!checks.length || !verdict) return;

    const iconEl = verdict.querySelector('.verdict-icon');
    const textEl = verdict.querySelector('.verdict-text');

    function update() {
        const count = checks.filter((c) => c.checked).length;
        verdict.classList.remove('ready', 'almost');
        if (count === 0) {
            iconEl.textContent = '◦';
            textEl.textContent = 'Tick what applies to you.';
        } else if (count === 1) {
            iconEl.textContent = '◔';
            textEl.textContent = 'Doable — the course introduces a lot, you\'ll fill gaps as you go.';
            verdict.classList.add('almost');
        } else if (count === 2) {
            iconEl.textContent = '◑';
            textEl.textContent = 'You\'re in great shape. Dive in.';
            verdict.classList.add('almost');
        } else {
            iconEl.textContent = '●';
            textEl.textContent = 'You\'re ready. Go from zero to hero.';
            verdict.classList.add('ready');
        }
    }
    checks.forEach((c) => c.addEventListener('change', update));
    update();
})();

/* ---------------------------------------------------------
   5. Code block copy-to-clipboard
   --------------------------------------------------------- */
(function copyButton() {
    const btn = $('#copyCmd');
    const payload = $('#codePayload');
    if (!btn || !payload) return;
    btn.addEventListener('click', async () => {
        const text = payload.innerText;
        try {
            await navigator.clipboard.writeText(text);
            btn.textContent = 'Copied';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.textContent = 'Copy';
                btn.classList.remove('copied');
            }, 1600);
        } catch (e) {
            btn.textContent = 'Copy failed';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
        }
    });
})();

/* ---------------------------------------------------------
   6. Scroll-reveal for sections
   --------------------------------------------------------- */
(function scrollReveal() {
    const targets = $$('.section-header, .chapter-card, .feat, .prereq, .code-block, .companion-sim, .gridworld-wrap, .video-frame, .quote');
    targets.forEach((el) => el.classList.add('reveal'));
    const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
            if (e.isIntersecting) {
                e.target.classList.add('shown');
                io.unobserve(e.target);
            }
        });
    }, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });
    targets.forEach((el) => io.observe(el));
})();
