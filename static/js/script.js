document.addEventListener('DOMContentLoaded', function() {
    const cpuUsageElement = document.getElementById('cpu-usage');
    const memoryUsageElement = document.getElementById('memory-usage');
    const uptimeElement = document.getElementById('uptime');
    const logContentElement = document.getElementById('log-content');

    const socket = io();

    function fetchMachineStats() {
        fetch('/machine-stats')
            .then(response => response.json())
            .then(data => {
                cpuUsageElement.textContent = data.cpu_usage + '%';
                memoryUsageElement.textContent = data.memory_info + '%';
                uptimeElement.textContent = data.uptime;
            });
    }

    function fetchLogs() {
        fetch('/logs')
            .then(response => response.text())
            .then(logs => {
                logContentElement.textContent = logs;
            });
    }

    // Fetch stats immediately when the page loads
    fetchMachineStats();

    // Fetch logs on page load
    fetchLogs();

    // Update stats every 10 seconds
    setInterval(fetchMachineStats, 10000);

    socket.on('log', function(data) {
        logContentElement.textContent += `Epoch ${data.epoch + 1}: Loss=${data.loss}, Accuracy=${data.accuracy}, Val Loss=${data.val_loss}, Val Accuracy=${data.val_accuracy}\n`;
    });
});
