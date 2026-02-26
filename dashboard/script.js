let ws = new WebSocket("ws://" + location.host + "/ws");

ws.onmessage = function(event) {
    let data = JSON.parse(event.data);

    let energyPercent = Math.min(data.energy * 2000, 100);

    document.getElementById("energyFill").style.width =
        energyPercent + "%";

    document.getElementById("frames").innerText = data.frames;
    document.getElementById("runtime").innerText = data.runtime;
    document.getElementById("alive").innerText =
        data.alive ? "🟢 Running" : "🔴 Stopped";
};