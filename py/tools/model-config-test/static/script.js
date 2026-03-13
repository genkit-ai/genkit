/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const API_BASE = '/api';

// State
let currentState = {
    scenarios: [],
    models: [],
    selectedScenarioId: null,
    selectedModel: null,
    stats: { pass: 0, fail: 0 }
};

// DOM Elements
const els = {
    sampleSelect: document.getElementById('sample-select'),
    currentScenarioTitle: document.getElementById('current-scenario-title'),
    modelCount: document.getElementById('model-count'),
    modelSelect: document.getElementById('model-select'),
    paramConfig: document.getElementById('param-config'),
    systemPrompt: document.getElementById('system-prompt'),
    userPrompt: document.getElementById('user-prompt'),
    runBtn: document.getElementById('run-btn'),
    resultsList: document.getElementById('results-list'),
    statsBar: document.getElementById('stats-bar'),
    passCount: document.getElementById('pass-count'),
    failCount: document.getElementById('fail-count')
};

// Init
async function init() {
    try {
        const scenarios = await fetchAPI('/scenarios');
        currentState.scenarios = scenarios;
        renderScenariosDropdown();

        // Don't auto-select - let user choose
        // Leave model dropdown empty until sample is selected
        els.modelSelect.innerHTML = '<option value="">Select a sample first</option>';
        els.modelSelect.disabled = true;
    } catch (e) {
        console.error('Failed to init:', e);
        if (els.sampleSelect) els.sampleSelect.innerHTML = `<option>Error loading samples</option>`;
    }
}

// Actions
async function selectScenario(id) {
    currentState.selectedScenarioId = id;
    const scenario = currentState.scenarios.find(s => s.id === id);

    // Update UI title
    els.currentScenarioTitle.textContent = scenario.name;
    els.systemPrompt.value = scenario.system_prompt;
    els.userPrompt.value = scenario.user_prompt;

    await loadModels(id);
}

async function loadModels(scenarioId) {
    els.modelSelect.innerHTML = '<option>Loading models...</option>';
    els.modelSelect.disabled = true;

    try {
        // Pass sample parameter to get sample-specific models
        const models = await fetchAPI(`/models?sample=${scenarioId}`);

        // Debug logging
        console.log('Received models:', models);
        console.log('First model:', models[0]);

        currentState.models = models;

        els.modelCount.textContent = `${models.length} Models`;

        // Render Select with numbered display names
        els.modelSelect.innerHTML = models.length ?
            models.map(m => {
                // Fallback if display_name or name is undefined
                const displayName = m.display_name || m.name || 'Unknown Model';
                const modelName = m.name || '';
                console.log('Rendering model:', modelName, displayName);
                return `<option value="${modelName}">${displayName}</option>`;
            }).join('') :
            '<option value="">No models found</option>';

        els.modelSelect.disabled = false;

        // Trigger select change to load params
        if (models.length > 0) {
            handleModelChange(models[0].name);
        } else {
            if (els.paramConfig) {
                els.paramConfig.innerHTML = '';
            }
        }
    } catch (e) {
        console.error('Failed to load models:', e);
        els.modelSelect.innerHTML = '<option>Error loading models</option>';
    }
}

function handleModelChange(modelName) {
    const model = currentState.models.find(m => m.name === modelName);
    currentState.selectedModel = model;
}

async function runTest() {
    if (!currentState.selectedModel) return;

    els.runBtn.disabled = true;
    els.runBtn.textContent = 'Running...';

    // Clear previous results (including comprehensive test summary)
    els.resultsList.innerHTML = '';

    // Send empty config as we removed the inputs
    const config = {};

    try {
        const result = await fetchAPI('/run', 'POST', {
            model: currentState.selectedModel.name,
            config: config,
            scenario_id: currentState.selectedScenarioId,
            user_prompt: els.userPrompt.value,
            system_prompt: els.systemPrompt.value
        });

        addResultCard(result);
        updateStats(result.success);

    } catch (e) {
        console.error(e);
        addResultCard({ success: false, error: e.message, timing: 0 });
        updateStats(false);
    } finally {
        els.runBtn.disabled = false;
        els.runBtn.textContent = 'Run Test';
    }
}

async function runComprehensiveTest() {
    if (!currentState.selectedModel || !currentState.selectedScenarioId) return;

    const comprehensiveBtn = document.getElementById('run-comprehensive-btn');
    comprehensiveBtn.disabled = true;
    comprehensiveBtn.textContent = 'Running comprehensive test...';

    // Clear previous results
    els.resultsList.innerHTML = '<div class="empty-state">Running comprehensive test...</div>';

    try {
        const result = await fetchAPI('/run-comprehensive', 'POST', {
            sample: currentState.selectedScenarioId,
            model: currentState.selectedModel.name,
            user_prompt: els.userPrompt.value,
            system_prompt: els.systemPrompt.value
        });

        // Clear loading message
        els.resultsList.innerHTML = '';

        // Add summary card
        const summaryCard = document.createElement('div');
        summaryCard.className = 'result-card summary';
        summaryCard.innerHTML = `
            <h4>Comprehensive Test Summary</h4>
            <div class="summary-stats">
                <div class="stat-item">Total Tests: <strong>${result.total_tests}</strong></div>
                <div class="stat-item success">Passed: <strong>${result.passed}</strong></div>
                <div class="stat-item error">Failed: <strong>${result.failed}</strong></div>
                <div class="stat-item">Success Rate: <strong>${((result.passed / result.total_tests) * 100).toFixed(1)}%</strong></div>
            </div>
        `;
        els.resultsList.appendChild(summaryCard);

        // Add individual test results
        result.results.forEach((testResult, idx) => {
            const card = document.createElement('div');
            card.className = `result-card ${testResult.success ? 'success' : 'error'}`;

            const configStr = JSON.stringify(testResult.config);
            const configDisplay = configStr === '{}' ? 'Default (all parameters at default)' : configStr;

            card.innerHTML = `
                <div class="result-meta">
                    <span class="test-number">Test ${idx + 1}</span>
                    <span class="config-label">Config: ${escapeHtml(configDisplay)}</span>
                    <span class="timing">${testResult.timing}s</span>
                </div>
                <div class="result-content">
                    ${testResult.success ? escapeHtml(testResult.response || 'No response') : escapeHtml(testResult.error || 'Unknown error')}
                </div>
            `;

            els.resultsList.appendChild(card);
        });

        // Update global stats
        currentState.stats.pass += result.passed;
        currentState.stats.fail += result.failed;
        els.passCount.textContent = currentState.stats.pass;
        els.failCount.textContent = currentState.stats.fail;
        els.statsBar.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        els.resultsList.innerHTML = `<div class="result-card error">
            <div class="result-content">Error running comprehensive test: ${escapeHtml(e.message)}</div>
        </div>`;
    } finally {
        comprehensiveBtn.disabled = false;
        comprehensiveBtn.textContent = 'Run Comprehensive Test';
    }
}

// Helpers
async function fetchAPI(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    if (body) options.body = JSON.stringify(body);

    const res = await fetch(API_BASE + endpoint, options);
    if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
    return await res.json();
}

function renderScenariosDropdown() {
    const options = ['<option value="">-- Select a Sample --</option>'];
    options.push(...currentState.scenarios.map(s =>
        `<option value="${s.id}">${s.name}</option>`
    ));
    els.sampleSelect.innerHTML = options.join('');
}

function addResultCard(result) {
    const card = document.createElement('div');
    card.className = `result-card ${result.success ? 'success' : 'error'}`;

    card.innerHTML = `
        <div class="result-meta">
            <span class="model-name">${currentState.selectedModel.name}</span>
            <span class="timing">${result.timing}s</span>
        </div>
        <div class="result-content">${result.success ? escapeHtml(result.response) : result.error}</div>
    `;

    els.resultsList.prepend(card);

    // Clear empty state
    const empty = els.resultsList.querySelector('.empty-state');
    if (empty) empty.remove();
}

function updateStats(success) {
    if (success) currentState.stats.pass++;
    else currentState.stats.fail++;

    els.passCount.textContent = currentState.stats.pass;
    els.failCount.textContent = currentState.stats.fail;
    els.statsBar.classList.remove('hidden');
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Event Listeners
els.sampleSelect.addEventListener('change', (e) => selectScenario(e.target.value));
els.modelSelect.addEventListener('change', (e) => handleModelChange(e.target.value));
els.runBtn.addEventListener('click', runTest);
document.getElementById('run-comprehensive-btn').addEventListener('click', runComprehensiveTest);

// Start
init();
