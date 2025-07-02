let currentChart = null;

document.addEventListener('DOMContentLoaded', function() {
    loadOverview();
    loadInsights();
    
    // Set up graph form
    document.getElementById('graphForm').addEventListener('submit', handleGraphSubmit);
});

async function loadOverview() {
    try {
        const response = await fetch(`/api/overview/${FILE_ID}`);
        if (!response.ok) throw new Error('Failed to load overview');
        
        const data = await response.json();
        displayOverview(data);
        populateColumnSelects(data.columns);
        
        // Show main content and hide loading
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('mainContent').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading overview:', error);
        showError('Failed to load dataset overview: ' + error.message);
    }
}

function displayOverview(data) {
    const container = document.getElementById('overviewContent');
    
    const html = `
        <div class="row g-3">
            <div class="col-md-3">
                <div class="bg-primary bg-opacity-10 rounded p-3 text-center">
                    <i class="fas fa-table fa-2x text-primary mb-2"></i>
                    <h4 class="mb-1">${data.shape.rows.toLocaleString()}</h4>
                    <small class="text-muted">Rows</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="bg-success bg-opacity-10 rounded p-3 text-center">
                    <i class="fas fa-columns fa-2x text-success mb-2"></i>
                    <h4 class="mb-1">${data.shape.columns}</h4>
                    <small class="text-muted">Columns</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="bg-info bg-opacity-10 rounded p-3 text-center">
                    <i class="fas fa-hashtag fa-2x text-info mb-2"></i>
                    <h4 class="mb-1">${data.insights.numeric_columns}</h4>
                    <small class="text-muted">Numeric</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="bg-warning bg-opacity-10 rounded p-3 text-center">
                    <i class="fas fa-font fa-2x text-warning mb-2"></i>
                    <h4 class="mb-1">${data.insights.categorical_columns}</h4>
                    <small class="text-muted">Categorical</small>
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h6 class="fw-semibold mb-3">Column Information</h6>
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead class="table-dark">
                        <tr>
                            <th>Column</th>
                            <th>Type</th>
                            <th>Unique Values</th>
                            <th>Missing</th>
                            <th>Sample Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.columns.map(col => `
                            <tr>
                                <td><strong>${col.name}</strong></td>
                                <td><span class="badge bg-secondary">${col.type}</span></td>
                                <td>${col.unique_count.toLocaleString()}</td>
                                <td>${col.missing_count > 0 ? `<span class="text-warning">${col.missing_count}</span>` : '0'}</td>
                                <td class="small text-muted">${col.sample_values.slice(0, 3).join(', ')}${col.sample_values.length > 3 ? '...' : ''}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function populateColumnSelects(columns) {
    const xSelect = document.getElementById('xColumn');
    const ySelect = document.getElementById('yColumn');
    
    // Clear existing options
    xSelect.innerHTML = '<option value="">Select X variable...</option>';
    ySelect.innerHTML = '<option value="">Select Y variable...</option>';
    
    // Add columns as options
    columns.forEach(col => {
        const option1 = new Option(col.name, col.name);
        const option2 = new Option(col.name, col.name);
        xSelect.add(option1);
        ySelect.add(option2);
    });
}

async function handleGraphSubmit(e) {
    e.preventDefault();
    
    const xColumn = document.getElementById('xColumn').value;
    const yColumn = document.getElementById('yColumn').value;
    
    if (!xColumn || !yColumn) {
        showError('Please select both X and Y variables');
        return;
    }
    
    try {
        // Load graph data
        const graphResponse = await fetch(`/api/graph/${FILE_ID}?x=${encodeURIComponent(xColumn)}&y=${encodeURIComponent(yColumn)}`);
        if (!graphResponse.ok) throw new Error('Failed to load graph data');
        
        const graphData = await graphResponse.json();
        displayChart(graphData);
        displayGraphStats(graphData);
        
        // Load equation if both variables are numeric
        if (graphData.x_type.includes('int') || graphData.x_type.includes('float')) {
            if (graphData.y_type.includes('int') || graphData.y_type.includes('float')) {
                loadEquation(xColumn, yColumn);
            }
        }
        
    } catch (error) {
        console.error('Error generating graph:', error);
        showError('Failed to generate graph: ' + error.message);
    }
}

function displayChart(data) {
    const canvas = document.getElementById('dataChart');
    const container = document.getElementById('chartContainer');
    
    // Show canvas and hide placeholder
    canvas.style.display = 'block';
    container.style.display = 'none';
    
    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    // Prepare chart data
    const chartData = {
        labels: data.x_values,
        datasets: [{
            label: `${data.y_label} vs ${data.x_label}`,
            data: data.chart_type === 'scatter' ? 
                data.x_values.map((x, i) => ({x: x, y: data.y_values[i]})) :
                data.y_values,
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2,
            pointRadius: 3
        }]
    };
    
    const config = {
        type: data.chart_type === 'scatter' ? 'scatter' : 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${data.y_label} vs ${data.x_label}`
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: data.x_label
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: data.y_label
                    }
                }
            }
        }
    };
    
    currentChart = new Chart(ctx, config);
}

function displayGraphStats(data) {
    const container = document.getElementById('graphStats');
    const content = document.getElementById('statsContent');
    
    let html = `
        <div class="small">
            <div class="d-flex justify-content-between mb-1">
                <span>Data Points:</span>
                <strong>${data.data_points.toLocaleString()}</strong>
            </div>
    `;
    
    if (data.correlation !== null) {
        const corrClass = Math.abs(data.correlation) > 0.7 ? 'text-success' : 
                         Math.abs(data.correlation) > 0.3 ? 'text-warning' : 'text-muted';
        html += `
            <div class="d-flex justify-content-between mb-1">
                <span>Correlation:</span>
                <strong class="${corrClass}">${data.correlation.toFixed(3)}</strong>
            </div>
        `;
    }
    
    if (data.x_stats) {
        html += `
            <hr class="my-2">
            <div class="fw-semibold">${data.x_label} Statistics</div>
            <div class="d-flex justify-content-between">
                <span>Range:</span>
                <span>${data.x_stats.min.toFixed(2)} - ${data.x_stats.max.toFixed(2)}</span>
            </div>
            <div class="d-flex justify-content-between">
                <span>Mean:</span>
                <span>${data.x_stats.mean.toFixed(2)}</span>
            </div>
        `;
    }
    
    if (data.y_stats) {
        html += `
            <hr class="my-2">
            <div class="fw-semibold">${data.y_label} Statistics</div>
            <div class="d-flex justify-content-between">
                <span>Range:</span>
                <span>${data.y_stats.min.toFixed(2)} - ${data.y_stats.max.toFixed(2)}</span>
            </div>
            <div class="d-flex justify-content-between">
                <span>Mean:</span>
                <span>${data.y_stats.mean.toFixed(2)}</span>
            </div>
        `;
    }
    
    html += '</div>';
    content.innerHTML = html;
    container.style.display = 'block';
}

async function loadEquation(xColumn, yColumn) {
    try {
        const response = await fetch(`/api/equation/${FILE_ID}?x=${encodeURIComponent(xColumn)}&y=${encodeURIComponent(yColumn)}`);
        if (!response.ok) throw new Error('Failed to load equation');
        
        const data = await response.json();
        displayEquation(data);
        
    } catch (error) {
        console.error('Error loading equation:', error);
        const container = document.getElementById('equationContent');
        container.innerHTML = `
            <div class="text-center text-muted py-3">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <p>Could not generate equation for these variables</p>
                <small>${error.message}</small>
            </div>
        `;
    }
}

function displayEquation(data) {
    const container = document.getElementById('equationContent');
    const bestEq = data.best_equation;
    
    const html = `
        <div class="text-center mb-3">
            <h5 class="text-primary">Best Fit Equation</h5>
            <div class="bg-light rounded p-3 mb-3">
                <code class="fs-5">${bestEq.formula}</code>
            </div>
            <div class="row g-2">
                <div class="col-6">
                    <small class="text-muted">Type</small>
                    <div class="fw-semibold text-capitalize">${bestEq.type}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">R² Score</small>
                    <div class="fw-semibold ${bestEq.r_squared > 0.8 ? 'text-success' : bestEq.r_squared > 0.5 ? 'text-warning' : 'text-muted'}">
                        ${(bestEq.r_squared * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>
        
        ${data.all_equations.length > 1 ? `
            <details class="small">
                <summary class="fw-semibold">All Equations (${data.all_equations.length})</summary>
                <div class="mt-2">
                    ${data.all_equations.map(eq => `
                        <div class="border rounded p-2 mb-2">
                            <code>${eq.formula}</code>
                            <div class="d-flex justify-content-between mt-1">
                                <span class="text-capitalize">${eq.type}</span>
                                <span>R²: ${(eq.r_squared * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </details>
        ` : ''}
    `;
    
    container.innerHTML = html;
}

async function loadInsights() {
    try {
        const response = await fetch(`/api/insights/${FILE_ID}`);
        if (!response.ok) throw new Error('Failed to load insights');
        
        const data = await response.json();
        displayInsights(data);
        
    } catch (error) {
        console.error('Error loading insights:', error);
        const container = document.getElementById('insightsContent');
        container.innerHTML = `
            <div class="text-center text-muted py-3">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <p>Could not generate AI insights</p>
                <small>${error.message}</small>
            </div>
        `;
    }
}

function displayInsights(data) {
    const container = document.getElementById('insightsContent');
    const insights = data.ai_insights;
    
    const html = `
        <div class="mb-3">
            <h6 class="fw-semibold text-primary">Summary</h6>
            <p class="small">${insights.summary}</p>
        </div>
        
        <div class="mb-3">
            <h6 class="fw-semibold text-success">Key Findings</h6>
            <ul class="small mb-0">
                ${insights.key_findings.map(finding => `<li>${finding}</li>`).join('')}
            </ul>
        </div>
        
        <div class="mb-3">
            <h6 class="fw-semibold text-info">Data Quality</h6>
            <p class="small">${insights.data_quality}</p>
        </div>
        
        <details class="small">
            <summary class="fw-semibold text-warning">Recommendations</summary>
            <ul class="mt-2 mb-0">
                ${insights.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </details>
        
        ${insights.correlations_analysis ? `
            <details class="small mt-2">
                <summary class="fw-semibold">Correlation Analysis</summary>
                <p class="mt-2 mb-0">${insights.correlations_analysis}</p>
            </details>
        ` : ''}
    `;
    
    container.innerHTML = html;
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    const modal = new bootstrap.Modal(document.getElementById('errorModal'));
    modal.show();
}
