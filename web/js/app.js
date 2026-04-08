document.addEventListener('DOMContentLoaded', () => {
    
    // -- Navigation Handling --
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-container');
    const viewTitle = document.getElementById('view-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetView = item.getAttribute('data-view');
            
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            viewTitle.textContent = targetView.charAt(0).toUpperCase() + targetView.slice(1) + ' Workspace';

            views.forEach(v => {
                if (v.id === `view-${targetView}`) {
                    v.classList.remove('hidden');
                } else {
                    v.classList.add('hidden');
                }
            });
        });
    });

    // -- Retrieval Engine --
    const retrieveForm = document.getElementById('retrieve-form');
    const resultsBody = document.getElementById('results-body');

    retrieveForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = document.getElementById('query-input').value;
        const useHyde = document.getElementById('use-hyde').checked;
        const topK = parseInt(document.getElementById('top-k').value, 10);
        
        const submitBtn = retrieveForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> Searching...';
        lucide.createIcons();

        try {
            const response = await fetch('/v1/retrieve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK, use_hyde: useHyde })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            renderResults(data.results);
        } catch (error) {
            console.error("Retrieval failed", error);
            resultsBody.innerHTML = `<tr><td colspan="4" style="color:var(--danger)">Error retrieving context: ${error.message}</td></tr>`;
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Query';
        }
    });

    function renderResults(results) {
        if (!results || results.length === 0) {
            resultsBody.innerHTML = '<tr class="empty-state"><td colspan="4">No results found for that query.</td></tr>';
            return;
        }

        resultsBody.innerHTML = results.map(r => `
            <tr>
                <td><span class="pill">${r.score.toFixed(4)}</span></td>
                <td>
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:8px">ID: ${r.id}</div>
                    <div class="content-preview">${escapeHtml(r.content)}</div>
                </td>
                <td><span class="utility-val">${r.utility.toFixed(4)}</span></td>
                <td class="feedback-actions">
                    <button class="btn btn-sm btn-outline feedback-btn" data-id="${r.id}" data-val="1.0" title="Helpful (+1.0)"><i data-lucide="thumbs-up"></i></button>
                    <button class="btn btn-sm btn-outline feedback-btn" data-id="${r.id}" data-val="-1.0" title="Not Helpful (-1.0)"><i data-lucide="thumbs-down"></i></button>
                </td>
            </tr>
        `).join('');

        lucide.createIcons();
        attachFeedbackListeners();
    }

    function attachFeedbackListeners() {
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const button = e.currentTarget;
                const docId = button.getAttribute('data-id');
                const val = parseFloat(button.getAttribute('data-val'));
                
                // Optimistic UI update
                button.innerHTML = '<i data-lucide="check"></i>';
                button.disabled = true;
                lucide.createIcons();
                
                try {
                    await fetch('/v1/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ document_id: docId, reward: val })
                    });
                } catch (error) {
                    console.error("Feedback failed", error);
                    button.innerHTML = '<i data-lucide="x"></i>';
                    lucide.createIcons();
                }
            });
        });
    }

    // -- Ingestion Engine --
    const ingestForm = document.getElementById('ingest-form');
    const jobStatus = document.getElementById('job-status');
    const jobText = document.getElementById('job-text');
    const jobIndicator = jobStatus.querySelector('.status-indicator');

    ingestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const path = document.getElementById('path-input').value;
        const submitBtn = document.getElementById('ingest-btn');
        
        submitBtn.disabled = true;
        jobIndicator.className = 'status-indicator working';
        jobText.textContent = `Ingesting chunks from ${path}...`;

        try {
            const response = await fetch('/v1/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            jobIndicator.className = 'status-indicator online';
            jobText.textContent = `Success: ${data.message} (${data.chunks_indexed} chunks indexed).`;
        } catch (error) {
            console.error("Ingestion failed", error);
            jobIndicator.className = 'status-indicator error';
            jobText.textContent = `Error: ${error.message}`;
        } finally {
            submitBtn.disabled = false;
        }
    });

    // Utilities
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
