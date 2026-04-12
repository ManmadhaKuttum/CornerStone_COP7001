let ws = null;
let wsConnected = false;
const seen = {
    partialAsr: false,
    finalAsr: false,
    partialTrans: false,
    finalTrans: false,
};

function resetSeenFlags() {
    seen.partialAsr = false;
    seen.finalAsr = false;
    seen.partialTrans = false;
    seen.finalTrans = false;
}

async function restartSpeech() {
    const btn = document.getElementById("reset-btn");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "Resetting...";
    try {
        const res = await fetch("/api/reset-session", { method: "POST" });
        if (!res.ok) throw new Error("reset failed");
        resetSeenFlags();
        document.getElementById("runtime-val").textContent = "0s";
        document.getElementById("partial-asr").textContent = "Waiting for speech...";
        document.getElementById("final-asr").textContent = "";
        document.getElementById("partial-trans").textContent = "Waiting for ASR output...";
        document.getElementById("final-trans").textContent = "";
        document.getElementById("energyFill").style.width = "0%";
        document.getElementById("energy-val").textContent = "0.00000";
        document.getElementById("frames-val").textContent = "0";
        document.getElementById("words-asr").textContent = "0";
        document.getElementById("words-trans").textContent = "0";
        document.getElementById("lat-asr").textContent = "—";
        document.getElementById("lat-asr").className = "lat-value";
        document.getElementById("lat-trans").textContent = "—";
        document.getElementById("lat-trans").className = "lat-value";
        document.getElementById("lat-tts").textContent = "—";
        document.getElementById("lat-tts").className = "lat-value";
        document.getElementById("lat-e2e").textContent = "—";
        document.getElementById("lat-e2e").className = "lat-value";
        document.getElementById("speaking-indicator").textContent = "🔇 Silence";
        document.getElementById("speaking-indicator").className = "speaking-indicator";
        document.getElementById("playback-status").textContent = "Playback: idle";
        document.getElementById("playback-status").className = "playback-status";
        setDemoItem("demo-partial-asr", false);
        setDemoItem("demo-final-asr", false);
        setDemoItem("demo-partial-trans", false);
        setDemoItem("demo-final-trans", false);
        setDemoItem("demo-e2e", false, "E2E Latency");
    } catch (err) {
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = "Reset Session";
    }
}

function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
        wsConnected = true;
        document.getElementById("conn-status").className = "status-pill online";
        document.getElementById("conn-status").textContent = "🟢 Connected";
        setDemoItem("demo-link", true);
    };

    ws.onclose = () => {
        wsConnected = false;
        document.getElementById("conn-status").className = "status-pill offline";
        document.getElementById("conn-status").textContent = "🔴 Disconnected";
        setDemoItem("demo-link", false);
        setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
        const d = JSON.parse(event.data);
        update(d);
    };
}

function setDot(id, status) {
    const badge = document.getElementById(id);
    if (!badge) return;
    const dot = badge.querySelector(".dot");
    dot.className = "dot";
    if (status === "ready" || status === "active" || status === "idle") {
        dot.classList.add("active");
    } else if (status === "loading" || status === "speaking") {
        dot.classList.add("processing");
    } else if (status === "unavailable" || status === "error") {
        dot.classList.add("bad");
    }
}

function colorLatency(el, ms) {
    el.className = "lat-value";
    if (ms > 3000) el.classList.add("bad");
    else if (ms > 2000) el.classList.add("warn");
}

function setDemoItem(id, ok, labelText) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("pass", "fail", "warn");
    const label = labelText || el.dataset.label || "";
    if (ok === true) {
        el.classList.add("pass");
        el.textContent = "✓ " + label;
    } else if (ok === false) {
        el.classList.add("fail");
        el.textContent = "✗ " + label;
    } else {
        el.classList.add("warn");
        el.textContent = "• " + label;
    }
}

function update(d) {
    // Runtime
    document.getElementById("runtime-val").textContent = d.runtime + "s";

    // Liveness
    setDot("badge-mic", d.mic_status || (d.frames > 0 ? "ready" : "loading"));
    setDot("badge-asr", d.asr_status);
    setDot("badge-trans", d.trans_status);
    setDot("badge-tts", d.tts_status === "speaking" ? "speaking" : d.tts_status.includes("ready") ? "ready" : "loading");

    // Speaking indicator
    const spk = document.getElementById("speaking-indicator");
    if (d.speech_active) {
        spk.textContent = "🎙️ Speaking...";
        spk.className = "speaking-indicator active";
    } else {
        spk.textContent = "🔇 Silence";
        spk.className = "speaking-indicator";
    }

    // Playback status
    const pb = document.getElementById("playback-status");
    if (pb) {
        const tts = (d.tts_status || "").toLowerCase();
        pb.className = "playback-status";
        if (tts.includes("speaking")) {
            pb.textContent = "Playback: speaking";
            pb.classList.add("active");
        } else if (tts.includes("ready") || tts.includes("idle")) {
            pb.textContent = "Playback: idle";
        } else if (tts.includes("loading")) {
            pb.textContent = "Playback: loading";
            pb.classList.add("warn");
        } else if (tts.includes("unavailable")) {
            pb.textContent = "Playback: unavailable";
            pb.classList.add("bad");
        } else {
            pb.textContent = "Playback: unknown";
        }
    }

    // Demo checklist
    if ((d.partial_transcript || "").trim().length > 0) seen.partialAsr = true;
    if ((d.final_transcript || "").trim().length > 0) seen.finalAsr = true;
    if ((d.partial_translation || "").trim().length > 0) seen.partialTrans = true;
    if ((d.final_translation || "").trim().length > 0) seen.finalTrans = true;

    setDemoItem("demo-link", wsConnected);
    setDemoItem("demo-offline", d.offline_mode === true);
    setDemoItem("demo-mic", d.frames > 0);
    setDemoItem("demo-asr", (d.asr_status || "").includes("ready"));
    setDemoItem("demo-trans", (d.trans_status || "").includes("ready"));
    setDemoItem("demo-tts", /(ready|idle|speaking)/.test((d.tts_status || "").toLowerCase()));
    setDemoItem("demo-partial-asr", seen.partialAsr);
    setDemoItem("demo-final-asr", seen.finalAsr);
    setDemoItem("demo-partial-trans", seen.partialTrans);
    setDemoItem("demo-final-trans", seen.finalTrans);
    const e2eOk = (d.e2e_latency_ms || 0) > 0;
    const e2eLabel = e2eOk ? `E2E ${d.e2e_latency_ms}ms` : "E2E Latency";
    setDemoItem("demo-e2e", e2eOk, e2eLabel);

    // Energy meter
    const pct = Math.min(100, d.energy * 800);
    document.getElementById("energyFill").style.width = pct + "%";
    document.getElementById("energy-val").textContent = d.energy.toFixed(5);
    document.getElementById("frames-val").textContent = d.frames.toLocaleString();
    document.getElementById("words-asr").textContent = d.total_words_asr;
    document.getElementById("words-trans").textContent = d.total_words_trans;

    // Latency cards
    const asrEl = document.getElementById("lat-asr");
    const trEl = document.getElementById("lat-trans");
    const ttsEl = document.getElementById("lat-tts");
    const e2eEl = document.getElementById("lat-e2e");

    if (d.asr_latency_ms) {
        asrEl.textContent = d.asr_latency_ms + " ms";
        colorLatency(asrEl, d.asr_latency_ms);
    }
    if (d.trans_latency_ms) {
        trEl.textContent = d.trans_latency_ms + " ms";
        colorLatency(trEl, d.trans_latency_ms);
    }
    if (d.tts_latency_ms) {
        ttsEl.textContent = d.tts_latency_ms + " ms";
    }
    if (d.e2e_latency_ms) {
        e2eEl.textContent = d.e2e_latency_ms + " ms";
        colorLatency(e2eEl, d.e2e_latency_ms);
    }

    if (d.trans_backend) {
        document.getElementById("trans-backend").textContent = d.trans_backend;
    }

    // Transcripts
    if (d.partial_transcript !== undefined)
        document.getElementById("partial-asr").textContent = d.partial_transcript || "Listening...";
    if (d.final_transcript !== undefined)
        document.getElementById("final-asr").textContent = d.final_transcript;
    if (d.partial_translation !== undefined)
        document.getElementById("partial-trans").textContent = d.partial_translation || "Waiting...";
    if (d.final_translation !== undefined)
        document.getElementById("final-trans").textContent = d.final_translation;
}

connect();

const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
    resetBtn.addEventListener("click", restartSpeech);
}
