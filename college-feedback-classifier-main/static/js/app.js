/* ──────────────────────────────────────────────
   Global Variables & State
────────────────────────────────────────────── */
let dashboardLoaded = false;
let datasetLoaded = false;
let batchCategoryChartInstance = null;

const COLORS = {
    Academics: '#3b82f6',
    Facilities: '#10b981',
    Administration: '#f59e0b',
    positive: '#10b981',
    negative: '#ef4444',
    neutral: '#64748b'
};

/* ──────────────────────────────────────────────
   DOM Elements
────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    // Input Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const inputAreas = document.querySelectorAll('.input-area');

    // Analysis
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');

    // ──────────────────────────────────────────────
    // Navigation Logic
    // ──────────────────────────────────────────────
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update nav state
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Update views
            const targetView = item.dataset.tab;
            views.forEach(v => v.classList.remove('active-view'));
            document.getElementById(targetView).classList.add('active-view');

            // Update headers
            if(targetView === 'classifier') {
                pageTitle.textContent = 'Feedback Classifier';
                pageSubtitle.textContent = 'AI-powered categorization and sentiment analysis.';
            } else if (targetView === 'dashboard') {
                pageTitle.textContent = 'Dashboard Analytics';
                pageSubtitle.textContent = 'Insights drawn from the student feedback dataset.';
                if(!dashboardLoaded) loadDashboard();
            } else if (targetView === 'dataset') {
                pageTitle.textContent = 'Dataset Explorer';
                pageSubtitle.textContent = 'Browse and filter the training data.';
                if(!datasetLoaded) loadDataset(1);
            }
        });
    });

    // ──────────────────────────────────────────────
    // Input Tabs Logic (Single vs Batch)
    // ──────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            inputAreas.forEach(area => area.classList.remove('active'));
            document.getElementById(`${btn.dataset.target}-input-area`).classList.add('active');
        });
    });

    // ──────────────────────────────────────────────
    // Analysis Logic
    // ──────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        const mode = document.querySelector('.tab-btn.active').dataset.target;
        
        // UI Loading state
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        
        document.getElementById('results-placeholder').classList.add('hidden');
        document.getElementById('single-result-container').classList.add('hidden');
        document.getElementById('batch-result-container').classList.add('hidden');

        try {
            if (mode === 'single') {
                const text = document.getElementById('feedback-input').value;
                if(!text) throw new Error("Please enter some text.");
                await processSingle(text);
            } else {
                const text = document.getElementById('batch-feedback-input').value;
                if(!text) throw new Error("Please enter some text.");
                const lines = text.split('\n').filter(l => l.trim() !== '');
                if(lines.length === 0) throw new Error("Please enter valid text lines.");
                await processBatch(lines);
            }
        } catch (err) {
            alert(err.message || "An error occurred during analysis.");
            document.getElementById('results-placeholder').classList.remove('hidden');
        } finally {
            // Restore UI
            analyzeBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    async function processSingle(text) {
        const res = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        if(data.error) throw new Error(data.error);

        // Update UI
        const catEl = document.getElementById('res-category');
        catEl.textContent = data.category;
        catEl.className = `value-badge cat-${data.category.toLowerCase()}`;
        
        document.getElementById('res-cat-conf-fill').style.width = `${data.category_confidence * 100}%`;
        document.getElementById('res-cat-conf').textContent = `${(data.category_confidence * 100).toFixed(1)}% confidence`;

        const sentEl = document.getElementById('res-sentiment');
        sentEl.textContent = data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1);
        sentEl.className = `value-badge sent-${data.sentiment}`;

        document.getElementById('res-sent-conf-fill').style.width = `${data.sentiment_confidence * 100}%`;
        document.getElementById('res-sent-conf').textContent = `${(data.sentiment_confidence * 100).toFixed(1)}% confidence`;

        document.getElementById('single-result-container').classList.remove('hidden');
    }

    async function processBatch(texts) {
        const res = await fetch('/api/batch-classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts })
        });
        const data = await res.json();
        if(data.error) throw new Error(data.error);

        document.getElementById('batch-count').textContent = data.summary.total;
        
        // Render Chart
        const ctx = document.getElementById('batchCategoryChart').getContext('2d');
        if(batchCategoryChartInstance) batchCategoryChartInstance.destroy();
        
        const labels = Object.keys(data.summary.category_distribution);
        const values = Object.values(data.summary.category_distribution);
        const bgColors = labels.map(l => COLORS[l]);

        batchCategoryChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: bgColors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#e2e8f0' } }
                }
            }
        });

        document.getElementById('batch-result-container').classList.remove('hidden');
    }

    // ──────────────────────────────────────────────
    // Dashboard Logic
    // ──────────────────────────────────────────────
    async function loadDashboard() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();

            // Update top cards
            document.getElementById('dash-total').textContent = data.total_samples;
            document.getElementById('dash-accuracy').textContent = `${(data.model_accuracy * 100).toFixed(1)}%`;
            document.getElementById('dash-train').textContent = data.train_samples;

            // Render Charts
            renderChart('mainCategoryChart', 'doughnut', data.category_distribution);
            renderChart('mainSentimentChart', 'bar', data.sentiment_distribution);

            dashboardLoaded = true;
        } catch (err) {
            console.error("Failed to load dashboard stats", err);
        }
    }

    function renderChart(canvasId, type, dataObj) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const labels = Object.keys(dataObj);
        const values = Object.values(dataObj);
        const bgColors = labels.map(l => COLORS[l] || COLORS[l.toLowerCase()]);

        new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: values,
                    backgroundColor: bgColors,
                    borderWidth: type === 'bar' ? 1 : 0,
                    borderRadius: type === 'bar' ? 6 : 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: type === 'doughnut', position: 'right', labels: { color: '#e2e8f0' } }
                },
                scales: type === 'bar' ? {
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                } : {}
            }
        });
    }

    // ──────────────────────────────────────────────
    // Dataset Explorer Logic
    // ──────────────────────────────────────────────
    const catFilter = document.getElementById('category-filter');
    catFilter.addEventListener('change', () => {
        loadDataset(1, catFilter.value);
    });

    async function loadDataset(page = 1, category = 'all') {
        try {
            const res = await fetch(`/api/dataset?page=${page}&category=${category}`);
            const data = await res.json();

            const tbody = document.querySelector('#dataset-table tbody');
            tbody.innerHTML = '';

            data.entries.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.text}</td>
                    <td><span style="color: ${COLORS[row.category]}">${row.category}</span></td>
                    <td><span class="value-badge sent-${row.sentiment}" style="padding: 0.25rem 0.75rem; font-size: 0.75rem; margin:0;">${row.sentiment}</span></td>
                `;
                tbody.appendChild(tr);
            });

            // Pagination logic
            const pag = document.getElementById('pagination');
            pag.innerHTML = '';
            
            if (data.total_pages > 1) {
                // Prev
                const prev = document.createElement('button');
                prev.className = 'page-btn';
                prev.textContent = '«';
                prev.disabled = page === 1;
                prev.onclick = () => loadDataset(page - 1, category);
                pag.appendChild(prev);

                // Current
                const curr = document.createElement('button');
                curr.className = 'page-btn active';
                curr.textContent = page;
                pag.appendChild(curr);

                // Next
                const next = document.createElement('button');
                next.className = 'page-btn';
                next.textContent = '»';
                next.disabled = page === data.total_pages;
                next.onclick = () => loadDataset(page + 1, category);
                pag.appendChild(next);
            }

            datasetLoaded = true;
        } catch (err) {
            console.error("Failed to load dataset", err);
        }
    }
});
