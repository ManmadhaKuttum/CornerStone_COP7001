let ws = null;

function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
        document.getElementById("conn-status").className = "status-pill online";
        document.getElementById("conn-status").textContent = "🟢 Connected";
    };

    ws.onclose = () => {
        document.getElementById("conn-status").className = "status-pill offline";
        document.getElementById("conn-status").textContent = "🔴 Disconnected";
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
    }
}

function colorLatency(el, ms) {
    el.className = "lat-value";
    if (ms > 3000) el.classList.add("bad");
    else if (ms > 2000) el.classList.add("warn");
}

function update(d) {
    // Runtime
    document.getElementById("runtime-val").textContent = d.runtime + "s";

    // Liveness
    setDot("badge-mic", d.frames > 0 ? "ready" : "loading");
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