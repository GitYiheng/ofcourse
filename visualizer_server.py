import json
import time

from flask import Flask, jsonify, render_template_string, request
from flask_sock import Sock


app = Flask(__name__)
sock = Sock(app)
current_state = {}


@app.route("/api/state", methods=["POST"])
def receive_state():
    global current_state
    current_state = request.get_json(silent=True) or {}
    return jsonify({"ok": True})


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(current_state)


@sock.route("/ws")
def websocket(ws):
    last_payload = None
    while True:
        if current_state:
            payload = json.dumps(current_state)
            if payload != last_payload:
                ws.send(payload)
                last_payload = payload
        time.sleep(0.1)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OFCOURSE Visualizer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef3f8;
      --panel: #ffffff;
      --line: #cad6e0;
      --ink: #17202a;
      --muted: #617080;
      --blue: #1f6feb;
      --teal: #12866f;
      --amber: #b7791f;
      --red: #c2413b;
      --shadow: 0 12px 30px rgba(28, 43, 58, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--amber);
    }

    .connected .dot { background: var(--teal); }
    .offline .dot { background: var(--red); }

    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
    }

    .canvas-wrap {
      min-height: 0;
      padding: 18px 24px 24px;
    }

    canvas {
      width: 100%;
      height: calc(100vh - 98px);
      min-height: 520px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fdfefe;
      box-shadow: var(--shadow);
    }

    aside {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
      border-left: 1px solid var(--line);
      background: var(--panel);
    }

    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 18px;
      border-bottom: 1px solid var(--line);
    }

    .metric {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }

    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 22px;
    }

    .panel {
      min-height: 0;
      overflow: auto;
      padding: 18px;
    }

    .agent {
      margin-bottom: 18px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }

    .agent h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }

    .unit {
      margin: 10px 0;
      font-size: 13px;
    }

    .unit-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
    }

    .bar {
      height: 8px;
      margin-top: 6px;
      overflow: hidden;
      border-radius: 4px;
      background: #e6edf3;
    }

    .bar-fill {
      height: 100%;
      min-width: 2px;
      background: var(--blue);
    }

    details {
      border-top: 1px solid var(--line);
      padding: 14px 18px;
    }

    summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }

    pre {
      max-height: 260px;
      overflow: auto;
      margin: 12px 0 0;
      padding: 12px;
      border-radius: 8px;
      background: #18212b;
      color: #d9e6f2;
      font-size: 11px;
      line-height: 1.45;
    }

    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { min-height: auto; border-left: 0; border-top: 1px solid var(--line); }
      canvas { height: 60vh; min-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <main>
      <header>
        <h1>OFCOURSE Real-Time Visualization</h1>
        <div id="status" class="status offline"><span class="dot"></span><span>Waiting for data</span></div>
      </header>
      <div class="canvas-wrap">
        <canvas id="network"></canvas>
      </div>
    </main>
    <aside>
      <section id="metrics" class="metrics"></section>
      <section id="agents" class="panel"></section>
      <details>
        <summary>Raw state</summary>
        <pre id="raw">{}</pre>
      </details>
    </aside>
  </div>

  <script>
    const canvas = document.getElementById("network");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const metricsEl = document.getElementById("metrics");
    const agentsEl = document.getElementById("agents");
    const rawEl = document.getElementById("raw");
    let state = null;
    let ws = null;
    let lastSeen = 0;

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
      ws.onopen = () => setStatus("Connected", "connected");
      ws.onclose = () => {
        setStatus("Disconnected", "offline");
        setTimeout(connect, 1500);
      };
      ws.onmessage = (event) => {
        state = JSON.parse(event.data);
        lastSeen = Date.now();
        setStatus(`Step ${state.step ?? 0}`, "connected");
        render();
      };
    }

    function setStatus(text, cls) {
      statusEl.className = `status ${cls}`;
      statusEl.lastElementChild.textContent = text;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    function render() {
      resizeCanvas();
      drawNetwork();
      drawPanel();
      rawEl.textContent = JSON.stringify(state || {}, null, 2);
    }

    function drawNetwork() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      drawGrid(rect.width, rect.height);

      if (!state || !state.agents || state.agents.length === 0) {
        ctx.fillStyle = "#617080";
        ctx.font = "15px Inter, Segoe UI, sans-serif";
        ctx.fillText("Start visualizer_server.py, then run main.py with --visualize.", 28, 36);
        return;
      }

      const agents = state.agents;
      const laneHeight = Math.max(72, (rect.height - 80) / agents.length);
      agents.forEach((agent, agentIndex) => {
        const y = 56 + agentIndex * laneHeight;
        const units = agent.units || [];
        const xStart = 72;
        const xEnd = rect.width - 72;
        const spacing = units.length > 1 ? (xEnd - xStart) / (units.length - 1) : 0;

        ctx.fillStyle = "#17202a";
        ctx.font = "600 13px Inter, Segoe UI, sans-serif";
        ctx.fillText(`Agent ${agent.id}`, 18, y + 4);

        ctx.strokeStyle = "#cad6e0";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xStart, y);
        ctx.lineTo(xEnd, y);
        ctx.stroke();

        units.forEach((unit, unitIndex) => {
          const x = xStart + spacing * unitIndex;
          drawUnitBuffers(unit, x, y + 10, spacing);

          ctx.fillStyle = "#17202a";
          ctx.font = "11px Inter, Segoe UI, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(`U${unit.id}`, x, y - 18);
          ctx.fillStyle = "#617080";
          ctx.fillText(`${unitOrderCount(unit)}`, x, y + 42);
          ctx.textAlign = "start";
        });

        drawMovingOrders(units, xStart, spacing, y);
      });
    }

    function drawGrid(width, height) {
      ctx.strokeStyle = "#edf2f7";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 36) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 36) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    function drawMovingOrders(units, xStart, spacing, y) {
      const orders = [];
      units.forEach((unit, unitIndex) => {
        (unit.containers || []).forEach((container) => {
          for (let i = 0; i < Math.min(container.occupied || 0, 4); i += 1) {
            orders.push({unitIndex, offset: i});
          }
        });
      });
      orders.slice(0, 28).forEach((order, index) => {
        const pulse = ((state.step || 0) * 0.08 + index * 0.19) % 1;
        const x = xStart + spacing * order.unitIndex + (pulse - 0.5) * Math.min(44, spacing * 0.5);
        const yy = y + 76 + (order.offset * 8);
        drawIsoBox(x - 5, yy - 4, 10, 8, 4, {
          front: "#c7d9ec",
          top: "#edf5ff",
          side: "#94abc2",
          stroke: "#5c7895",
        });
      });
    }

    function drawUnitBuffers(unit, centerX, y, spacing) {
      const containers = unit.containers || [];
      const buffers = containers.filter((container) => container.type === "buffer");
      const inventories = containers.filter((container) => container.type === "inventory");
      const visible = buffers.slice(0, 2);
      const rowGapX = -8;
      const rowGapY = 16;
      const cellW = Math.max(8, Math.min(15, spacing / 7 || 12));
      const cellH = 11;
      const depth = 5;
      drawPlatform(centerX - 52, y - 13, 108, visible.length > 1 ? 44 : 28, 4);

      if (inventories.length > 0) {
        const inventoryLoad = inventories.reduce((sum, item) => sum + (item.occupied || 0), 0);
        drawIsoBox(centerX - 48, y - 9, 22, 15, 7, {
          front: inventoryLoad > 0 ? "#c8dfc6" : "#e5f0e3",
          top: "#f4faf2",
          side: "#9ab89b",
          stroke: "#6f8b70",
        });
      }

      visible.forEach((buffer, rowIndex) => {
        const capacity = Math.max(1, Math.min(buffer.capacity || 5, 6));
        const occupied = Math.max(0, Math.min(buffer.occupied || 0, capacity));
        const startX = centerX - ((capacity * cellW) + ((capacity - 1) * 2)) / 2 + 7 + rowIndex * rowGapX;
        const rowY = y - 1 + rowIndex * rowGapY;

        for (let i = 0; i < capacity; i += 1) {
          const filled = i < occupied;
          const active = filled && i === occupied - 1;
          const x = startX + i * (cellW + 2);
          drawIsoBox(x, rowY, cellW, cellH, depth, {
            front: active ? "#f3d5aa" : filled ? "#cfe0f3" : "#edf3f8",
            top: active ? "#fff0d9" : filled ? "#edf5ff" : "#ffffff",
            side: active ? "#c49a6d" : filled ? "#9fb5cc" : "#c8d2dc",
            stroke: active ? "#9a764e" : "#6d8298",
          });
        }
      });
    }

    function drawPlatform(x, y, width, surfaceDepth, thickness) {
      ctx.save();
      ctx.lineJoin = "round";
      ctx.shadowColor = "rgba(55, 65, 75, 0.24)";
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 4;
      ctx.shadowOffsetY = 5;

      const backLeft = {x: x + surfaceDepth, y};
      const backRight = {x: x + width + surfaceDepth, y};
      const frontRight = {x: x + width, y: y + surfaceDepth};
      const frontLeft = {x, y: y + surfaceDepth};

      ctx.beginPath();
      ctx.moveTo(backLeft.x, backLeft.y);
      ctx.lineTo(backRight.x, backRight.y);
      ctx.lineTo(frontRight.x, frontRight.y);
      ctx.lineTo(frontLeft.x, frontLeft.y);
      ctx.closePath();
      ctx.fillStyle = "#dddddd";
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.strokeStyle = "#b8b8b8";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(frontLeft.x, frontLeft.y);
      ctx.lineTo(frontRight.x, frontRight.y);
      ctx.lineTo(frontRight.x, frontRight.y + thickness);
      ctx.lineTo(frontLeft.x, frontLeft.y + thickness);
      ctx.closePath();
      ctx.fillStyle = "#c7c7c7";
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(frontRight.x, frontRight.y);
      ctx.lineTo(backRight.x, backRight.y);
      ctx.lineTo(backRight.x, backRight.y + thickness);
      ctx.lineTo(frontRight.x, frontRight.y + thickness);
      ctx.closePath();
      ctx.fillStyle = "#cfcfcf";
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(frontLeft.x + 8, frontLeft.y - 7);
      ctx.lineTo(frontRight.x - 8, frontRight.y - 7);
      ctx.strokeStyle = "#c1c1c1";
      ctx.stroke();
      ctx.restore();
    }

    function drawIsoBox(x, y, width, height, depth, colors) {
      ctx.save();
      ctx.lineJoin = "round";
      ctx.lineWidth = 1.2;

      const frontY = y + depth;

      ctx.beginPath();
      ctx.moveTo(x, frontY);
      ctx.lineTo(x + depth, y);
      ctx.lineTo(x + width + depth, y);
      ctx.lineTo(x + width, frontY);
      ctx.closePath();
      ctx.fillStyle = colors.top;
      ctx.strokeStyle = colors.stroke;
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x + width, frontY);
      ctx.lineTo(x + width + depth, y);
      ctx.lineTo(x + width + depth, y + height);
      ctx.lineTo(x + width, frontY + height);
      ctx.closePath();
      ctx.fillStyle = colors.side;
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.rect(x, frontY, width, height);
      ctx.fillStyle = colors.front;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function tint(hex, amount) {
      return mixColor(hex, "#ffffff", amount);
    }

    function shade(hex, amount) {
      return mixColor(hex, "#000000", 1 - amount);
    }

    function mixColor(hex, target, amount) {
      const from = parseHex(hex);
      const to = parseHex(target);
      const channel = (a, b) => Math.round(a + (b - a) * amount);
      return `rgb(${channel(from.r, to.r)}, ${channel(from.g, to.g)}, ${channel(from.b, to.b)})`;
    }

    function parseHex(hex) {
      const normalized = hex.replace("#", "");
      return {
        r: parseInt(normalized.slice(0, 2), 16),
        g: parseInt(normalized.slice(2, 4), 16),
        b: parseInt(normalized.slice(4, 6), 16),
      };
    }

    function unitLoad(unit) {
      const containers = unit.containers || [];
      const finite = containers.filter((container) => container.capacity > 0);
      if (finite.length === 0) {
        return Math.min(1, unitOrderCount(unit) / 12);
      }
      const occupied = finite.reduce((sum, container) => sum + (container.occupied || 0), 0);
      const capacity = finite.reduce((sum, container) => sum + (container.capacity || 0), 0);
      return capacity > 0 ? occupied / capacity : 0;
    }

    function unitOrderCount(unit) {
      return (unit.containers || []).reduce((sum, container) => sum + (container.occupied || 0), 0);
    }

    function drawPanel() {
      const metrics = state?.metrics || {};
      metricsEl.innerHTML = [
        metricCard("Step", state?.step ?? 0),
        metricCard("Agents", metrics.agent_count ?? 0),
        metricCard("Orders", metrics.order_count ?? 0),
        metricCard("Reward", formatNumber(metrics.total_reward)),
      ].join("");

      agentsEl.innerHTML = (state?.agents || []).map((agent) => {
        const units = (agent.units || []).map((unit) => {
          const occupied = unitOrderCount(unit);
          const capacity = (unit.containers || [])
            .filter((container) => container.capacity > 0)
            .reduce((sum, container) => sum + container.capacity, 0);
          const pct = capacity > 0 ? Math.min(100, occupied / capacity * 100) : Math.min(100, occupied * 8);
          return `
            <div class="unit">
              <div class="unit-title"><span>${unit.label}</span><span>${occupied}${capacity ? ` / ${capacity}` : ""}</span></div>
              <div class="bar"><div class="bar-fill" style="width: ${pct}%"></div></div>
            </div>
          `;
        }).join("");
        return `<article class="agent"><h2>Agent ${agent.id}</h2>${units}</article>`;
      }).join("");
    }

    function metricCard(label, value) {
      return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
    }

    function formatNumber(value) {
      if (value === null || value === undefined) return "-";
      return Number(value).toFixed(1);
    }

    window.addEventListener("resize", render);
    setInterval(() => {
      if (lastSeen && Date.now() - lastSeen > 3000) {
        setStatus(`Step ${state?.step ?? 0} stale`, "offline");
      }
    }, 1000);
    connect();
    render();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
