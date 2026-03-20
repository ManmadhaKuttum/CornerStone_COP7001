let ws;

function connect() {
    ws = new WebSocket("ws://" + location.host + "/ws");

    ws.onopen = function () {
        let conn = document.getElementById("conn-status");
        conn.innerText = "🟢 Connected";
        conn.className = "status-pill online";
    };

    ws.onclose = function () {
        let conn = document.getElementById("conn-status");
        conn.innerText = "🔴 Disconnected";
        conn.className = "status-pill offline";
        setTimeout(connect, 2000);  // auto-reconnect
    };

    ws.onerror = function () { ws.close(); };

    ws.onmessage = function (event) {
        let data = JSON.parse(event.data);

        // ── Audio Meter ───────────────────────────────────────────────────
        let energyPercent = Math.min(data.energy * 2000, 100);
        document.getElementById("energyFill").style.width = energyPercent + "%";
        document.getElementById("energy-val").innerText = data.energy.toFixed(5);
        document.getElementById("frames-val").innerText = data.frames.toLocaleString();

        // ── Runtime ───────────────────────────────────────────────────────
        if (document.getElementById("runtime-val")) {
            document.getElementById("runtime-val").innerText = formatRuntime(data.runtime);
        }

        // ── Speaking Indicator (fix: use speech_active not is_speaking) ───
        let speakInd = document.getElementById("speaking-indicator");
        if (data.speech_active) {
            speakInd.innerText = "🗣️ Speaking...";
            speakInd.className = "speaking-indicator active";
        } else {
            speakInd.innerText = "🔇 Silence";
            speakInd.className = "speaking-indicator";
        }

        // ── Liveness Dots ─────────────────────────────────────────────────
        document.querySelector("#badge-mic .dot").className = "dot active";

        let asrDot = document.querySelector("#badge-asr .dot");
        if (data.asr_status === "processing") asrDot.className = "dot processing";
        else if (data.asr_status === "idle")  asrDot.className = "dot active";
        else                                   asrDot.className = "dot";

        let transDot = document.querySelector("#badge-trans .dot");
        if (data.trans_status === "processing")    transDot.className = "dot processing";
        else if (data.trans_status === "idle")      transDot.className = "dot active";
        else if (data.trans_status === "unavailable") transDot.className = "dot"; // grey
        else                                          transDot.className = "dot";

        let ttsDot = document.querySelector("#badge-tts .dot");
        ttsDot.className = data.tts_status === "playing" ? "dot active" : "dot";

        // ── Latency ───────────────────────────────────────────────────────
        setLatency("lat-asr",   data.asr_latency_ms);
        setLatency("lat-trans", data.trans_latency_ms);
        setLatency("lat-e2e",   data.e2e_latency_ms);

        // ── Translation backend badge ─────────────────────────────────────
        let backendEl = document.getElementById("trans-backend");
        if (backendEl && data.trans_backend && data.trans_backend !== "loading") {
            backendEl.innerText = "⚙ " + data.trans_backend;
        }

        // ── Transcription ─────────────────────────────────────────────────
        document.getElementById("partial-asr").innerText =
            data.partial_transcript || "Waiting for speech...";
        if (data.final_transcript) {
            let el = document.getElementById("final-asr");
            el.innerText = data.final_transcript;
            el.scrollTop = el.scrollHeight;
        }

        // ── Translation ───────────────────────────────────────────────────
        document.getElementById("partial-trans").innerText =
            data.partial_translation || "Waiting for ASR output...";
        if (data.final_translation) {
            let el = document.getElementById("final-trans");
            el.innerText = data.final_translation;
            el.scrollTop = el.scrollHeight;
        }

        // ── TTS / Playback ────────────────────────────────────────────────
        let playbackStatus = document.getElementById("playback-status");
        if (data.tts_status === "playing") {
            playbackStatus.innerText = "🔊 Playing Translated Audio...";
            playbackStatus.className = "playback-active";
        } else {
            playbackStatus.innerText = "🔇 Idle — Waiting for finalized translation";
            playbackStatus.className = "playback-idle";
        }

        // ── Word counts ───────────────────────────────────────────────────
        if (document.getElementById("words-asr"))
            document.getElementById("words-asr").innerText = data.total_words_asr || 0;
        if (document.getElementById("words-trans"))
            document.getElementById("words-trans").innerText = data.total_words_trans || 0;
    };
}

function setLatency(id, val) {
    let el = document.getElementById(id);
    if (!el) return;
    if (!val || val === 0) { el.innerText = "—"; el.style.color = ""; return; }
    el.innerText = Math.round(val);
    if (val < 1500)       el.style.color = "var(--accent-green)";
    else if (val < 3000)  el.style.color = "var(--accent-warning)";
    else                  el.style.color = "#ef4444";
}

function formatRuntime(s) {
    let h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    return h > 0 ? `${h}h ${m}m ${sec}s` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

connect();