document.addEventListener("DOMContentLoaded", () => {
    // API Endpoints
    const POPULAR_QUERIES_API = "/api/v1/trends/popular-queries";
    const INGESTION_STATS_API = "/api/v1/trends/ingestion-stats";
    const TOP_PAPERS_API = "/api/v1/trends/top-papers";

    let popularChart = null;
    let categoryChart = null;

    // Fetch and render data
    async function loadAnalytics() {
        try {
            // 1. Fetch Ingestion Stats
            const statsRes = await fetch(INGESTION_STATS_API);
            if (statsRes.ok) {
                const stats = await statsRes.json();
                document.getElementById("stat-total-papers").innerText = stats.total_papers || 0;
                document.getElementById("stat-processed-chunks").innerText = stats.processed_papers || 0;
                document.getElementById("stat-extraction-rate").innerText = `${(stats.text_extraction_rate * 100).toFixed(1)}%`;
            }
        } catch (e) {
            console.warn("Failed to load ingestion stats", e);
        }

        try {
            // 2. Fetch Popular Queries
            const queriesRes = await fetch(POPULAR_QUERIES_API);
            if (queriesRes.ok) {
                const data = await queriesRes.json();
                const labels = (data.queries || []).map(q => q.query);
                const counts = (data.queries || []).map(q => q.count);

                renderPopularQueriesChart(labels, counts);
            }
        } catch (e) {
            console.warn("Failed to load popular queries", e);
        }

        try {
            // 3. Fetch Top Papers to parse categories distribution
            const papersRes = await fetch(TOP_PAPERS_API);
            if (papersRes.ok) {
                const data = await papersRes.json();
                const categoryCounts = {};
                
                (data.papers || []).forEach(p => {
                    (p.categories || []).forEach(cat => {
                        categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
                    });
                });

                const labels = Object.keys(categoryCounts);
                const counts = Object.values(categoryCounts);

                renderCategoryDistributionChart(labels, counts);
            }
        } catch (e) {
            console.warn("Failed to load category distribution", e);
        }
    }

    function renderPopularQueriesChart(labels, data) {
        const ctx = document.getElementById("popular-queries-chart").getContext("2d");
        if (popularChart) popularChart.destroy();

        if (labels.length === 0) {
            labels = ["neural networks", "transformers", "reinforcement learning", "llm fine-tuning", "agentic rag"];
            data = [12, 9, 8, 5, 4];
        }

        popularChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Search Volume Count",
                    data: data,
                    backgroundColor: "#ea2804",
                    borderColor: "#ea2804",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: "#222" },
                        ticks: { color: "#aaa" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#aaa" }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function renderCategoryDistributionChart(labels, data) {
        const ctx = document.getElementById("category-distribution-chart").getContext("2d");
        if (categoryChart) categoryChart.destroy();

        if (labels.length === 0) {
            labels = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"];
            data = [4, 3, 2, 1];
        }

        categoryChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: ["#ea2804", "#ffffff", "#4cd964", "#007aff"],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "right",
                        labels: { color: "#aaa" }
                    }
                }
            }
        });
    }

    // Trigger initial load
    loadAnalytics();
    
    // Refresh stats every 30 seconds
    setInterval(loadAnalytics, 30000);
});
