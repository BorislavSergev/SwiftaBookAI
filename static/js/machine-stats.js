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
                document.getElementById('cpu-usage').innerText = `CPU: ${data.CPU}`;
                document.getElementById('memory-info').innerText = `Memory: ${data.RAM}`;
                document.getElementById('uptime').innerText = `System: ${data.System} ${data.Release}`;
                document.getElementById('cores').innerText = `GPU: ${data.GPU}`;
            })
            .catch(error => {
                console.error('Error fetching machine stats:', error);
                document.getElementById('cpu-usage').innerText = 'Error loading data';
                document.getElementById('memory-info').innerText = 'Error loading data';
                document.getElementById('uptime').innerText = 'Error loading data';
                document.getElementById('cores').innerText = 'Error loading data';
            });
    }

    fetchMachineStats();
    setInterval(fetchMachineStats, 1000); // Refresh every 5 seconds
});
