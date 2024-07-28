document.addEventListener('DOMContentLoaded', function() {
    console.log("JavaScript file loaded");

    function fetchMachineStats() {
        fetch('/machine-stats')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log("Data fetched:", data);
                document.getElementById('cpu-usage').innerText = `CPU Usage: ${data.cpu_usage}%`;
                document.getElementById('memory-info').innerText = `Memory Usage: ${data.memory_info}%`;
                document.getElementById('uptime').innerText = `Uptime: ${data.uptime}`;
                document.getElementById('cores').innerText = `Cores: ${data.cores}`;
            })
            .catch(error => {
                console.error('Error fetching machine stats:', error);
            });
    }

    fetchMachineStats();
    setInterval(fetchMachineStats, 1000); // Refresh every 1 second
});
