/* DAG Viewer — Cytoscape initialization, node click, run selector.
   Expects globals: nodes, edges, artifacts, runs (set by template). */

const RUN_COLORS = ['#6366f1', '#f97316', '#06b6d4', '#84cc16', '#ec4899'];

const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes: nodes, edges: edges },
    style: [
        {
            selector: 'node[type="artifact"]',
            style: {
                'shape': 'round-rectangle',
                'width': 'label',
                'height': 30,
                'padding': '12px',
                'background-color': '#9ca3af',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '11px',
                'font-weight': 'bold',
                'color': 'white'
            }
        },
        {
            selector: 'node[type="artifact"][status="accepted"]',
            style: { 'background-color': '#10b981' }
        },
        {
            selector: 'node[type="artifact"][status="rejected"]',
            style: { 'background-color': '#ef4444' }
        },
        {
            selector: 'node[type="artifact"][status="pending"]',
            style: { 'background-color': '#f59e0b' }
        },
        {
            selector: 'node[type="artifact"][status="superseded"]',
            style: { 'background-color': '#6b7280' }
        },
        {
            selector: 'node[type="artifact"][has_escalation_feedback]',
            style: {
                'border-width': 3,
                'border-color': '#ec4899',
                'border-style': 'dashed'
            }
        },
        {
            selector: 'edge[type="causal"]',
            style: {
                'width': 3,
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'line-color': 'data(color)',
                'target-arrow-color': 'data(color)'
            }
        },
        {
            selector: 'edge[type="retry"]',
            style: {
                'width': 2,
                'line-color': '#9ca3af',
                'line-style': 'dashed',
                'target-arrow-color': '#9ca3af',
                'target-arrow-shape': 'chevron',
                'curve-style': 'bezier'
            }
        },
        {
            selector: 'edge[type="escalation_retry"]',
            style: {
                'width': 2,
                'line-color': '#ec4899',
                'line-style': 'dashed',
                'target-arrow-color': '#ec4899',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
            }
        },
        {
            selector: 'node:selected',
            style: {
                'border-width': 4,
                'border-color': '#3b82f6'
            }
        }
    ],
    layout: {
        name: 'dagre',
        rankDir: 'LR',
        nodeSep: 50,
        rankSep: 100,
        padding: 30
    }
});

// ── Node click → sidebar ──────────────────────────────

cy.on('tap', 'node[type="artifact"]', function(evt) {
    const artifactId = evt.target.data('artifact_id');
    showArtifactDetail(artifactId);
});

function showArtifactDetail(artifactId) {
    const artifact = artifacts[artifactId];
    if (!artifact) return;

    const panel = document.getElementById('detail-panel');

    // Dependency artifacts
    let depsHtml = '';
    if (artifact.context.dependency_artifacts && artifact.context.dependency_artifacts.length > 0) {
        depsHtml = artifact.context.dependency_artifacts.map(dep =>
            `<div class="dep-link"><strong>${escapeHtml(dep.action_pair_id)}</strong><span class="dep-id">${dep.artifact_id.substring(0,8)}...</span></div>`
        ).join('');
    } else {
        depsHtml = '<p class="no-selection">No dependencies</p>';
    }

    // Feedback history
    let feedbackHtml = '';
    if (artifact.context.feedback_history && artifact.context.feedback_history.length > 0) {
        feedbackHtml = artifact.context.feedback_history.map((entry, i) =>
            `<div class="feedback-entry"><div class="entry-number">Attempt ${i + 1}</div><div class="entry-content">${escapeHtml(entry.feedback)}</div></div>`
        ).join('');
    } else {
        feedbackHtml = '<p class="no-selection">No previous feedback</p>';
    }

    // Escalation feedback
    let escalationHtml = '';
    if (artifact.context.escalation_feedback && artifact.context.escalation_feedback.length > 0) {
        escalationHtml = artifact.context.escalation_feedback.map((fb, i) =>
            `<div class="escalation-feedback"><div class="label">Escalation ${i + 1}</div><div>${escapeHtml(fb)}</div></div>`
        ).join('');
    } else {
        escalationHtml = '<p class="no-selection">No escalation feedback</p>';
    }

    // Guard result
    let guardHtml = '';
    if (artifact.guard_result) {
        const gr = artifact.guard_result;
        guardHtml = `<div class="guard-result ${gr.passed ? 'passed' : 'failed'}">
            <div class="guard-name">${gr.guard_name || 'Guard'}: ${gr.passed ? 'PASSED' : 'FAILED'}</div>
            ${gr.feedback ? `<div class="guard-feedback">${escapeHtml(gr.feedback)}</div>` : ''}
        </div>`;
        if (gr.sub_results && gr.sub_results.length > 0) {
            guardHtml += gr.sub_results.map(sr =>
                `<div class="guard-result ${sr.passed ? 'passed' : 'failed'}" style="margin-left:1rem;">
                    <div class="guard-name">${sr.guard_name}: ${sr.passed ? 'PASSED' : 'FAILED'}</div>
                    ${sr.feedback ? `<div class="guard-feedback">${escapeHtml(sr.feedback)}</div>` : ''}
                </div>`
            ).join('');
        }
    } else {
        guardHtml = '<p class="no-selection">No guard result (pending)</p>';
    }

    // Content language detection
    let langClass = '';
    if (artifact.content) {
        const trimmed = artifact.content.trimStart();
        if (trimmed.startsWith('{') || trimmed.startsWith('[')) langClass = 'language-json';
        else if (trimmed.startsWith('diff ') || trimmed.startsWith('---') || trimmed.startsWith('@@')) langClass = 'language-diff';
        else langClass = 'language-python';
    }

    panel.innerHTML = `
        <div class="detail-section">
            <div class="meta-grid">
                <div class="meta-item"><label>Step</label><span>${artifact.action_pair_id}</span></div>
                <div class="meta-item"><label>Attempt</label><span>#${artifact.attempt_number}</span></div>
                <div class="meta-item"><label>Status</label><span style="text-transform:uppercase;">${artifact.status}</span></div>
                <div class="meta-item"><label>Created</label><span>${new Date(artifact.created_at).toLocaleString()}</span></div>
            </div>
            <div style="margin-top:0.75rem;padding:0.5rem 0.75rem;background:#f9fafb;border-radius:6px;font-size:0.78rem;color:#6b7280;word-break:break-all;">
                <strong>ID:</strong> ${artifact.artifact_id}
                ${artifact.previous_attempt_id ? '<br><strong>Previous:</strong> ' + artifact.previous_attempt_id : ''}
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible expanded">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Content</h4><span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">
                    <pre class="code-block"><code class="${langClass}">${escapeHtml(artifact.content)}</code></pre>
                </div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible expanded">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Guard Result</h4><span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">${guardHtml}</div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Dependencies</h4>
                    <span class="badge">${artifact.context.dependency_artifacts ? artifact.context.dependency_artifacts.length : 0}</span>
                    <span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">${depsHtml}</div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Specification</h4><span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">
                    <pre class="code-block" style="max-height:250px;">${escapeHtml(artifact.context.specification || '')}</pre>
                </div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Constraints</h4><span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">
                    <pre class="code-block" style="max-height:250px;">${escapeHtml(artifact.context.constraints || '')}</pre>
                </div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Feedback History</h4>
                    <span class="badge">${artifact.context.feedback_history ? artifact.context.feedback_history.length : 0} entries</span>
                    <span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">${feedbackHtml}</div>
            </div>
        </div>
        <div class="detail-section">
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible(this)">
                    <h4>Escalation Feedback</h4>
                    <span class="badge">${artifact.context.escalation_feedback ? artifact.context.escalation_feedback.length : 0} entries</span>
                    <span class="toggle">&#9660;</span>
                </div>
                <div class="collapsible-content">${escalationHtml}</div>
            </div>
        </div>
    `;

    // Syntax highlighting
    panel.querySelectorAll('pre code[class]').forEach(block => {
        hljs.highlightElement(block);
    });
}

// ── Run selector ──────────────────────────────────────

(function initRunSelector() {
    if (runs.length <= 1) return;

    const container = document.getElementById('run-selector');
    const btnContainer = document.getElementById('run-buttons');
    container.style.display = 'flex';

    const allBtn = document.createElement('button');
    allBtn.className = 'run-btn active';
    allBtn.textContent = 'All';
    allBtn.addEventListener('click', () => selectRun(-1));
    btnContainer.appendChild(allBtn);

    runs.forEach((run, idx) => {
        const color = RUN_COLORS[idx] || RUN_COLORS[0];
        const btn = document.createElement('button');
        btn.className = 'run-btn';
        btn.textContent = run.label;
        btn.style.borderLeft = `4px solid ${color}`;
        btn.addEventListener('click', () => selectRun(idx));
        btnContainer.appendChild(btn);
    });

    // Replace single causal legend with per-run colours
    const legendCausal = document.getElementById('legend-causal');
    if (legendCausal && runs.length > 1) {
        let html = '';
        runs.forEach((run, idx) => {
            const color = RUN_COLORS[idx] || RUN_COLORS[0];
            html += `<div class="legend-item"><div class="legend-line" style="background:${color}"></div><span>${run.label}</span></div>`;
        });
        legendCausal.outerHTML = html;
    }

    function selectRun(runIdx) {
        btnContainer.querySelectorAll('.run-btn').forEach((b, i) => {
            b.classList.toggle('active', i === runIdx + 1);
        });

        if (runIdx === -1) {
            cy.batch(() => { cy.elements().style('opacity', 1); });
            return;
        }

        const run = runs[runIdx];
        const nodeSet = new Set(run.node_ids);
        const edgeSet = new Set(run.edge_ids);

        cy.batch(() => {
            cy.nodes().forEach(n => { n.style('opacity', nodeSet.has(n.id()) ? 1 : 0.12); });
            cy.edges().forEach(e => { e.style('opacity', edgeSet.has(e.id()) ? 1 : 0.06); });
        });
    }
})();

// ── Utilities ─────────────────────────────────────────

function toggleCollapsible(header) {
    header.parentElement.classList.toggle('expanded');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
