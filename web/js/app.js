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

            const titles = {
                'retrieval': 'Dynamic Explore',
                'ingestion': 'Data Pipeline',
                'config': 'System State'
            };

            viewTitle.textContent = titles[targetView] || 'Workspace';

            views.forEach(v => {
                if (v.id === `view-${targetView}`) {
                    v.classList.remove('hidden');
                    if (targetView === 'config') {
                        loadCollections();
                    }
                } else {
                    v.classList.add('hidden');
                }
            });
        });
    });

    // -- Custom Number Input Logic --
    const topkInput = document.getElementById('top-k');
    const decBtn = document.getElementById('dec-top-k');
    const incBtn = document.getElementById('inc-top-k');

    if (decBtn && incBtn && topkInput) {
        decBtn.addEventListener('click', () => {
            const val = parseInt(topkInput.value, 10);
            if (val > 1) topkInput.value = val - 1;
        });
        incBtn.addEventListener('click', () => {
            const val = parseInt(topkInput.value, 10);
            if (val < 25) topkInput.value = val + 1;
        });
    }

    // -- Retrieval Engine --
    const retrieveForm = document.getElementById('retrieve-form');
    const resultsBody = document.getElementById('results-body');

    if (retrieveForm) {
        retrieveForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = document.getElementById('query-input').value.trim();
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
    }

    function renderResults(results) {
        if (!results || results.length === 0) {
            resultsBody.innerHTML = '<tr class="empty-state"><td colspan="4">No results found for that query.</td></tr>';
            return;
        }

        resultsBody.innerHTML = results.map(r => `
            <tr>
                <td><span class="pill">${r.score.toFixed(4)}</span></td>
                <td>
                    <div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:8px;font-family:monospace">ID: ${r.id}</div>
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
    const jobIndicator = jobStatus?.querySelector('.status-indicator');

    if (ingestForm) {
        ingestForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const path = document.getElementById('path-input').value.trim();
            const submitBtn = document.getElementById('ingest-btn');

            submitBtn.disabled = true;
            if (jobIndicator) jobIndicator.className = 'status-indicator working';
            if (jobText) jobText.textContent = `Ingesting chunks from ${path}...`;

            try {
                const response = await fetch('/v1/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });

                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

                const data = await response.json();
                if (jobIndicator) jobIndicator.className = 'status-indicator online';
                if (jobText) jobText.textContent = `Success: ${data.message} (${data.chunks_indexed} chunks indexed).`;
            } catch (error) {
                console.error("Ingestion failed", error);
                if (jobIndicator) jobIndicator.className = 'status-indicator error';
                if (jobText) jobText.textContent = `Error: ${error.message}`;
            } finally {
                submitBtn.disabled = false;
            }
        });
    }

    // -- Configuration & Management --
    const collectionsBody = document.getElementById('collections-body');
    const clearCacheBtn = document.getElementById('clear-cache-btn');
    const refreshColsBtn = document.getElementById('refresh-cols-btn');

    async function loadCollections() {
        if (!collectionsBody) return;
        collectionsBody.innerHTML = '<tr><td colspan="2" class="loading-state">Loading collections...</td></tr>';

        try {
            const response = await fetch('/v1/collections');
            if (!response.ok) throw new Error('Failed to fetch');
            const data = await response.json();
            
            // Artificial delay to prevent "flicker" on super fast connections
            setTimeout(() => {
                renderCollections(data.collections);
            }, 300);
        } catch (error) {
            console.error("Failed to load collections", error);
            collectionsBody.innerHTML = '<tr><td colspan="2" class="loading-state" style="color:var(--danger)">Failed to load collections</td></tr>';
        }
    }

    function renderCollections(collections) {
        if (!collectionsBody) return;
        collectionsBody.innerHTML = '';

        if (!collections || collections.length === 0) {
            collectionsBody.innerHTML = '<tr><td colspan="2" class="loading-state">No collections found</td></tr>';
            return;
        }

        const html = collections.map(name => `
            <tr class="fade-in">
                <td style="font-family:monospace">${name}</td>
                <td>
                    <button class="btn btn-sm btn-outline btn-danger delete-col-btn" data-name="${name}">
                        <i data-lucide="trash-2"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        collectionsBody.innerHTML = html;
        lucide.createIcons();
        attachCollectionListeners();
    }

    function attachCollectionListeners() {
        document.querySelectorAll('.delete-col-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const name = e.currentTarget.getAttribute('data-name');

                const confirmed = await showModal({
                    title: 'Delete Collection',
                    message: `Are you sure you want to permanently delete the collection '${name}'? This action cannot be undone and will clear the ingestion cache.`,
                    confirmText: 'Delete Collection',
                    type: 'danger'
                });

                if (!confirmed) return;

                e.currentTarget.disabled = true;
                e.currentTarget.innerHTML = '<i data-lucide="loader" class="spin"></i>';
                lucide.createIcons();

                try {
                    const response = await fetch(`/v1/collections/${name}`, { method: 'DELETE' });
                    const data = await response.json();

                    await showModal({
                        title: 'Success',
                        message: data.message,
                        confirmText: 'Dismiss',
                        type: 'info',
                        hideCancel: true
                    });

                    loadCollections();
                } catch (error) {
                    console.error("Delete failed", error);
                    await showModal({
                        title: 'Error',
                        message: `Failed to delete collection: ${error.message}`,
                        confirmText: 'Dismiss',
                        type: 'info',
                        hideCancel: true
                    });

                    e.currentTarget.disabled = false;
                    e.currentTarget.innerHTML = '<i data-lucide="trash-2"></i>';
                    lucide.createIcons();
                }
            });
        });
    }

    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', async () => {
            const confirmed = await showModal({
                title: 'Clear Cache',
                message: "Are you sure you want to clear the ingestion cache? This will force a full re-index of all files on the next ingestion cycle.",
                confirmText: 'Clear Cache',
                type: 'danger'
            });

            if (!confirmed) return;

            clearCacheBtn.disabled = true;
            const originalContent = clearCacheBtn.innerHTML;
            clearCacheBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> Clearing...';
            lucide.createIcons();

            try {
                const response = await fetch('/v1/cache/clear', { method: 'POST' });
                const data = await response.json();

                await showModal({
                    title: 'Cache Cleared',
                    message: data.message,
                    confirmText: 'Got it',
                    type: 'info',
                    hideCancel: true
                });
            } catch (error) {
                console.error("Clear cache failed", error);
                await showModal({
                    title: 'Error',
                    message: `Failed to clear cache: ${error.message}`,
                    confirmText: 'Dismiss',
                    type: 'info',
                    hideCancel: true
                });
            } finally {
                clearCacheBtn.disabled = false;
                clearCacheBtn.innerHTML = originalContent;
                lucide.createIcons();
            }
        });
    }

    if (refreshColsBtn) {
        refreshColsBtn.addEventListener('click', () => {
            const icon = refreshColsBtn.querySelector('i');
            if (icon) icon.classList.add('spin');
            loadCollections();
            if (icon) setTimeout(() => icon.classList.remove('spin'), 600);
        });
    }

    // -- Modal System --
    function showModal(config) {
        const modalContainer = document.getElementById('modal-container');
        const titleEl = document.getElementById('modal-title');
        const messageEl = document.getElementById('modal-message');
        const iconEl = document.getElementById('modal-icon');
        const confirmBtn = document.getElementById('modal-confirm-btn');
        const cancelBtn = document.getElementById('modal-cancel-btn');

        if (!modalContainer) return Promise.resolve(false);

        titleEl.textContent = config.title || 'Confirm';
        messageEl.textContent = config.message || '';
        confirmBtn.textContent = config.confirmText || 'Confirm';

        if (config.type === 'danger') {
            iconEl.setAttribute('data-lucide', 'alert-triangle');
            iconEl.className = 'modal-icon-warning';
            confirmBtn.className = 'btn btn-primary btn-danger';
        } else {
            iconEl.setAttribute('data-lucide', 'info');
            iconEl.className = 'modal-icon-info';
            confirmBtn.className = 'btn btn-primary';
        }

        if (config.hideCancel) {
            cancelBtn.classList.add('hidden');
        } else {
            cancelBtn.classList.remove('hidden');
        }

        lucide.createIcons();
        modalContainer.classList.remove('hidden');

        return new Promise((resolve) => {
            const cleanup = (value) => {
                modalContainer.classList.add('hidden');
                confirmBtn.removeEventListener('click', onConfirm);
                cancelBtn.removeEventListener('click', onCancel);
                window.removeEventListener('keydown', onKeyDown);
                resolve(value);
            };

            const onConfirm = () => cleanup(true);
            const onCancel = () => cleanup(false);
            const onKeyDown = (e) => {
                if (e.key === 'Escape') onCancel();
                if (e.key === 'Enter' && !modalContainer.classList.contains('hidden')) {
                    e.preventDefault();
                    onConfirm();
                }
            };

            confirmBtn.addEventListener('click', onConfirm);
            cancelBtn.addEventListener('click', onCancel);
            window.addEventListener('keydown', onKeyDown);

            modalContainer.onclick = (e) => {
                if (e.target === modalContainer) onCancel();
            };
        });
    }

    // Utilities
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Initial Load
    lucide.createIcons();
});
