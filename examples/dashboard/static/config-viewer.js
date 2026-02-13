/* Config Viewer — Cytoscape initialization, node click, prompt sidebar.
   Expects globals: nodes, edges, prompts (set by template). */

const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes: nodes, edges: edges },
    style: [
        {
            selector: 'node',
            style: {
                'shape': 'round-rectangle',
                'width': 'label',
                'height': 40,
                'padding': '14px',
                'background-color': '#6366f1',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '11px',
                'font-weight': 'bold',
                'color': 'white'
            }
        },
        {
            selector: 'node[is_composite]',
            style: { 'background-color': '#7c3aed' }
        },
        {
            selector: 'node[has_escalation]',
            style: {
                'border-width': 3,
                'border-color': '#ec4899',
                'border-style': 'dashed'
            }
        },
        {
            selector: 'edge[type="requires"]',
            style: {
                'width': 3,
                'line-color': '#6366f1',
                'target-arrow-color': '#6366f1',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
            }
        },
        {
            selector: 'edge[type="escalation"]',
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
            selector: 'edge[type="guard_escalation"]',
            style: {
                'width': 2,
                'line-color': '#ec4899',
                'line-style': 'dotted',
                'target-arrow-color': '#ec4899',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'label': 'data(guard_label)',
                'font-size': '9px',
                'color': '#9d174d',
                'text-background-color': 'white',
                'text-background-opacity': 0.9,
                'text-background-padding': '2px',
                'text-rotation': 'autorotate'
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
        nodeSep: 60,
        rankSep: 120,
        padding: 30
    }
});

// ── Node click → sidebar ──────────────────────────────

cy.on('tap', 'node', function(evt) {
    showNodeDetail(evt.target.data());
});

function showNodeDetail(d) {
    const panel = document.getElementById('detail-panel');
    const p = prompts[d.id] || {};

    // Guards list
    let guardsHtml = '';
    if (d.is_composite && d.guards && d.guards.length > 0) {
        guardsHtml = `<div class="meta-item" style="grid-column:1/-1;">
            <label>Guards (composite)</label>
            <div class="guards-list">
                ${d.guards.map(g => `<span class="guard-tag">${escapeHtml(g)}</span>`).join('')}
            </div>
        </div>`;
    }

    // Guard config
    let guardConfigHtml = '';
    if (d.guard_config) {
        guardConfigHtml = `<div class="collapsible">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <h4>Guard Config</h4><span class="toggle">&#9660;</span>
            </div>
            <div class="collapsible-content">
                <pre class="code-block">${escapeHtml(formatJson(d.guard_config))}</pre>
            </div>
        </div>`;
    }

    // Prompt sections
    let promptSections = '';
    const fields = [
        ['role', 'Prompt: Role', false],
        ['task', 'Prompt: Task', true],
        ['constraints', 'Prompt: Constraints', false],
        ['feedback_wrapper', 'Prompt: Feedback Wrapper', false],
        ['escalation_feedback_wrapper', 'Prompt: Escalation Feedback Wrapper', false],
    ];
    for (const [key, title, expandByDefault] of fields) {
        const val = p[key];
        if (!val) continue;
        const expanded = expandByDefault ? ' expanded' : '';
        promptSections += `<div class="collapsible${expanded}">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <h4>${title}</h4><span class="toggle">&#9660;</span>
            </div>
            <div class="collapsible-content">
                <pre class="code-block">${highlightPlaceholders(escapeHtml(val))}</pre>
            </div>
        </div>`;
    }

    if (!promptSections && !Object.keys(p).length) {
        promptSections = '<p class="no-selection">No prompt data available</p>';
    }

    panel.innerHTML = `
        <div class="detail-section">
            <div class="meta-grid">
                <div class="meta-item"><label>Generator</label><span>${escapeHtml(d.generator)}</span></div>
                <div class="meta-item"><label>Guard</label><span>${escapeHtml(d.guard)}</span></div>
                <div class="meta-item" style="grid-column:1/-1;"><label>Description</label><span>${escapeHtml(d.description)}</span></div>
                <div class="meta-item"><label>r_patience</label><span>${d.r_patience || 'default'}</span></div>
                <div class="meta-item"><label>e_max</label><span>${d.e_max || 'default'}</span></div>
                ${guardsHtml}
            </div>
        </div>
        ${guardConfigHtml}
        ${promptSections}
    `;
}

// ── Utilities ─────────────────────────────────────────

function toggleCollapsible(header) {
    header.parentElement.classList.toggle('expanded');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function highlightPlaceholders(escapedHtml) {
    return escapedHtml.replace(
        /\{([^}]+)\}/g,
        '<span class="placeholder-token">{$1}</span>'
    );
}

function formatJson(str) {
    try { return JSON.stringify(JSON.parse(str), null, 2); }
    catch { return str; }
}
