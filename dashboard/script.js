let ws = new WebSocket("ws://" + location.host + "/ws");

ws.onmessage = function(event) {
    let data = JSON.parse(event.data);

    let energyPercent = Math.min(data.energy * 2000, 100);
    document.getElementById("energyFill").style.width = energyPercent + "%";

    document.getElementById("frames").innerText = data.frames;
    document.getElementById("runtime").innerText = data.runtime;
    document.getElementById("alive").innerText = data.alive ? "🟢 Running" : "🔴 Stopped";
    document.getElementById("speech").innerText = data.speech_active ? "🗣 Active" : "🔇 Silence";
    document.getElementById("latency").innerText = data.asr_latency_ms;
    document.getElementById("partial").innerText = data.partial_transcript || "";
    document.getElementById("final").innerText = data.final_transcript || "";
};