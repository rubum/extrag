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
                'dashboard': 'System Dashboard',
                'retrieval': 'Dynamic Explore',
                'ingestion': 'Data Pipeline',
                'config': 'System State',
                'documentation': 'Knowledge Base'
            };

            viewTitle.textContent = titles[targetView] || 'Workspace';

            views.forEach(v => {
                if (v.id === `view-${targetView}`) {
                    v.classList.remove('hidden');
                    if (targetView === 'config') loadCollections();
                    if (targetView === 'documentation') initDocTabs();
                    if (targetView === 'dashboard') startTelemetryPolling();
                } else {
                    v.classList.add('hidden');
                }
            });

            if (targetView !== 'dashboard') {
                stopTelemetryPolling();
            }
        });
    });

    // -- Telemetry Dashboard Logic --
    let telemetryInterval = null;

    function formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024, sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async function fetchTelemetry() {
        try {
            // Add native cache-busting parameter and no-store to prevent aggressive browser caching
            const res = await fetch(`/v1/telemetry?_t=${Date.now()}`, { cache: 'no-store' });
            if (!res.ok) throw new Error('Network error');
            const data = await res.json();


            document.getElementById('dash-chunks').textContent = formatNumber(data.total_chunks);
            document.getElementById('dash-bytes').textContent = formatBytes(data.total_bytes);
            document.getElementById('dash-queries').textContent = formatNumber(data.total_retrievals);
            document.getElementById('dash-prompt').textContent = formatNumber(data.total_prompt);
            document.getElementById('dash-completion').textContent = formatNumber(data.total_completion);
            document.getElementById('dash-errors').textContent = formatNumber(data.total_errors);

            // Render live activity feed
            const activityLog = document.getElementById('activity-log');
            if (activityLog && data.events) {
                const logsHtml = data.events.map(ev => `
                    <div class="log-entry ${ev.level}">
                        <span class="log-time">${ev.timestamp}</span>
                        <span class="log-level">${ev.level}</span>
                        <span class="log-message">${escapeHtml(ev.message)}</span>
                    </div>
                `).join('');

                // Only update if changed to avoid jumpy scrolling
                if (activityLog.dataset.lastCount !== data.events.length.toString() || activityLog.dataset.lastMsg !== data.events[data.events.length - 1]?.message) {
                    activityLog.innerHTML = logsHtml;
                    activityLog.scrollTop = activityLog.scrollHeight;
                    activityLog.dataset.lastCount = data.events.length;
                    activityLog.dataset.lastMsg = data.events[data.events.length - 1]?.message;
                }
            }

        } catch (error) {
            console.error('Failed to fetch telemetry:', error);
        }
    }

    function startTelemetryPolling() {
        if (telemetryInterval) return;
        fetchTelemetry(); // initial fetch
        telemetryInterval = setInterval(fetchTelemetry, 2000);
    }

    function stopTelemetryPolling() {
        if (telemetryInterval) {
            clearInterval(telemetryInterval);
            telemetryInterval = null;
        }
    }

    // Trigger initial load if spawned on dashboard
    if (document.querySelector('.nav-item.active').getAttribute('data-view') === 'dashboard') {
        startTelemetryPolling();
    }

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

    const toggleChunksBtn = document.getElementById('toggle-chunks-btn');
    if (toggleChunksBtn) {
        toggleChunksBtn.addEventListener('click', () => {
            const chunksPanel = document.getElementById('chunks-panel');
            if (chunksPanel.classList.contains('hidden')) {
                chunksPanel.classList.remove('hidden');
                toggleChunksBtn.innerHTML = '<i data-lucide="chevron-up"></i> Hide Context Sources';
            } else {
                chunksPanel.classList.add('hidden');
                toggleChunksBtn.innerHTML = '<i data-lucide="chevron-down"></i> View Context Sources';
            }
            lucide.createIcons();
        });
    }

    if (retrieveForm) {
        // Auto-scaling logic for all textareas in forms
        document.querySelectorAll('.search-form textarea').forEach(textarea => {
            textarea.addEventListener('input', () => {
                textarea.style.height = 'auto';
                textarea.style.height = (textarea.scrollHeight) + 'px';
            });

            // Handle Enter key for submission (Shift+Enter for new line)
            textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    textarea.closest('form').requestSubmit();
                }
            });
        });

        // Add HyDE toggle logic
        const hydeToggle = document.getElementById('hyde-toggle');
        if (hydeToggle) {
            hydeToggle.addEventListener('click', () => {
                const panel = document.getElementById('hyde-panel');
                if (panel) panel.classList.toggle('collapsed');
            });
        }

        retrieveForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = document.getElementById('query-input').value.trim();
            const useHyde = document.getElementById('use-hyde').checked;
            const topK = parseInt(document.getElementById('top-k').value, 10);

            const submitBtn = retrieveForm.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> Querying...';
            lucide.createIcons();

            // Reset UI state for new query
            const hydePanel = document.getElementById('hyde-panel');
            const genPanel = document.getElementById('generation-panel');
            const chunksPanel = document.getElementById('chunks-panel');
            const genContent = document.getElementById('generation-content');

            if (hydePanel) {
                hydePanel.classList.add('hidden');
                hydePanel.classList.remove('collapsed'); // Reset to expanded for new query
            }
            if (genPanel) {
                genPanel.classList.remove('hidden');
                genContent.innerHTML = `<div class="status-row ink"></i> 1. Inspecting retrieved context chunks...</div>`;
                lucide.createIcons();
            }
            if (chunksPanel) chunksPanel.classList.add('hidden');
            resultsBody.innerHTML = '';

            try {
                const response = await fetch('/v1/retrieve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: topK, use_hyde: useHyde })
                });

                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                // Clear status row and start a collapsible log group
                if (genContent) {
                    genContent.innerHTML = `
                        <div class="panel-trigger" id="logs-toggle" style="margin-top: 0; padding-bottom: var(--sp-2);">
                            <span class="section-header" style="margin-bottom: 0; border-bottom: none;">System Trace</span>
                            <i data-lucide="chevron-down" class="chevron"></i>
                        </div>
                        <div id="logs-wrapper" class="collapsible-wrapper">
                            <div class="system-logs" id="retrieval-logs"></div>
                        </div>
                    `;
                    lucide.createIcons();

                    // Add toggle listener for the logs specifically
                    const logsToggle = document.getElementById('logs-toggle');
                    if (logsToggle) {
                        logsToggle.addEventListener('click', () => {
                            const wrapper = document.getElementById('logs-wrapper');
                            const chevron = logsToggle.querySelector('.chevron');
                            
                            wrapper.classList.toggle('collapsed');
                            if (chevron) {
                                chevron.classList.toggle('collapsed');
                            }
                        });
                    }
                }
                const logContainer = document.getElementById('retrieval-logs');

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const str = line.trim();
                        if (!str || !str.startsWith('data: ')) continue;

                        try {
                            const event = JSON.parse(str.slice(6));
                            if (event.type === 'log') {
                                const logLine = document.createElement('div');
                                logLine.className = 'log-entry';
                                logLine.innerHTML = `<span class="log-time">${new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span> <span class="log-msg">${escapeHtml(event.message)}</span>`;
                                logContainer.appendChild(logLine);
                                logContainer.scrollTop = logContainer.scrollHeight;
                            } else if (event.type === 'error') {
                                throw new Error(event.message);
                            } else if (event.type === 'result') {
                                const data = event.data;
                                console.log("Final retrieval data received:", data);

                                // Present HyDE thought trace if present
                                if (hydePanel && data.hyde_doc) {
                                    hydePanel.classList.remove('hidden');
                                    document.getElementById('hyde-content').textContent = data.hyde_doc;
                                }

                                // Present generative synthesized response
                                if (genPanel && data.generation && data.generation.trim().length > 0) {
                                    genPanel.classList.remove('hidden');
                                    if (typeof marked !== 'undefined') {
                                        // Keep logs at top, then response? Or replace? 
                                        // User said "instead of", but logs are cool. 
                                        // I'll put a divider.
                                        const hr = document.createElement('hr');
                                        hr.className = 'content-divider';
                                        genContent.appendChild(hr);

                                        const resDiv = document.createElement('div');
                                        resDiv.className = 'generation-text fade-in';
                                        resDiv.innerHTML = marked.parse(data.generation);
                                        genContent.appendChild(resDiv);
                                    } else {
                                        const resDiv = document.createElement('div');
                                        resDiv.className = 'generation-text';
                                        resDiv.textContent = data.generation;
                                        genContent.appendChild(resDiv);
                                    }
                                }

                                renderResults(data.results);

                                // Show chunks panel if no results or if specific logic requires it
                                if (!data.results || data.results.length === 0) {
                                    if (chunksPanel) chunksPanel.classList.remove('hidden');
                                }
                            }
                        } catch (parseErr) {
                            console.warn("Failed to parse SSE event", str, parseErr);
                        }
                    }
                }

                const toggleChunksBtn = document.getElementById('toggle-chunks-btn');
                if (toggleChunksBtn) {
                    toggleChunksBtn.innerHTML = '<i data-lucide="chevron-down"></i> View Context Sources';
                    lucide.createIcons();
                }

            } catch (error) {
                console.error("Retrieval failed", error);

                const genPanel = document.getElementById('generation-panel');
                const hydePanel = document.getElementById('hyde-panel');
                if (genPanel) genPanel.classList.add('hidden');
                if (hydePanel) hydePanel.classList.add('hidden');
                document.getElementById('chunks-panel').classList.remove('hidden');

                resultsBody.innerHTML = `<tr><td colspan="4" style="color:var(--danger)">Error retrieving context: ${error.message}</td></tr>`;
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i data-lucide="play"></i> Run Query';
                lucide.createIcons();
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
                <td>
                    <div class="result-id">Chunk ID: ${r.id}</div>
                    <div class="content-preview">${escapeHtml(r.content)}</div>
                </td>
                <td>
                    <div class="result-meta">
                        <div class="metric-item">
                            <span class="metric-label">Semantic Score</span>
                            <span class="pill">${r.score.toFixed(4)}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">RL Utility (Q)</span>
                            <span class="pill">${r.utility.toFixed(4)}</span>
                        </div>
                        <div class="feedback-group">
                            <button class="btn btn-sm btn-outline feedback-btn" data-id="${r.id}" data-val="1.0" title="Helpful (+1.0)"><i data-lucide="thumbs-up"></i></button>
                            <button class="btn btn-sm btn-outline feedback-btn" data-id="${r.id}" data-val="-1.0" title="Not Helpful (-1.0)"><i data-lucide="thumbs-down"></i></button>
                        </div>
                    </div>
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
                submitBtn.innerHTML = '<i data-lucide="zap"></i> Run Pipeline';
                lucide.createIcons();
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

    // -- Documentation Logic --
    function initDocTabs() {
        const docLinks = document.querySelectorAll('.doc-nav a');
        const codeTabs = document.querySelectorAll('.code-tabs');

        // Language Switching Logic
        codeTabs.forEach(container => {
            const btns = container.querySelectorAll('.tab-btn');
            const panes = container.querySelectorAll('.tab-pane');

            btns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const lang = btn.getAttribute('data-lang');

                    // Update all tab groups to the same language for consistency
                    document.querySelectorAll(`.tab-btn[data-lang="${lang}"]`).forEach(b => {
                        const parent = b.closest('.code-tabs');
                        parent.querySelectorAll('.tab-btn').forEach(tb => tb.classList.remove('active'));
                        parent.querySelectorAll('.tab-pane').forEach(tp => tp.classList.remove('active'));

                        b.classList.add('active');
                        parent.querySelector(`.tab-pane[data-lang="${lang}"]`).classList.add('active');
                    });
                });
            });
        });

        // Smooth Scroll for Doc Nav
        docLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href.startsWith('#')) {
                    e.preventDefault();
                    const targetEl = document.querySelector(href);
                    if (targetEl) {
                        targetEl.scrollIntoView({ behavior: 'smooth' });
                    }
                }
            });
        });
    }

    // Copy to clipboard logic
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const tabsContainer = btn.closest('.code-tabs');
            const activePane = tabsContainer.querySelector('.tab-pane.active');
            if (!activePane) return;

            const text = activePane.innerText;

            try {
                await navigator.clipboard.writeText(text);
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i data-lucide="check"></i>';
                btn.classList.add('copied');
                if (window.lucide) lucide.createIcons();

                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.classList.remove('copied');
                    if (window.lucide) lucide.createIcons();
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code: ', err);
            }
        });
    });

    // ScrollSpy for Documentation
    const mainContent = document.querySelector('.main-content');
    const docSections = document.querySelectorAll('.doc-section');
    const docSideLinks = document.querySelectorAll('.doc-nav a');

    if (mainContent && docSections.length > 0) {
        const observerOptions = {
            root: mainContent,
            threshold: 0,
            rootMargin: '-10% 0px -80% 0px' // High sensitivity to items near the top
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    docSideLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, observerOptions);

        docSections.forEach(section => observer.observe(section));
    }

    // Initial Load
    lucide.createIcons();
});
