// ==========================================================================
// Code Snippets Database
// ==========================================================================
const codeSnippets = {
    python: `import httpx

# 1. Ask a question using streaming hybrid RAG
url = "http://localhost:8000/api/v1/stream"
payload = {
    "query": "What are transformer models?",
    "top_k": 3,
    "use_hybrid": True,
    "model": "llama3.2:1b"
}

with httpx.stream("POST", url, json=payload) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            print(line[6:]) # Process streaming chunks`,
    js: `const fetch = require('node-fetch');

// 1. Execute agentic RAG workflow with LangGraph
async function askAgentic() {
  const response = await fetch('http://localhost:8000/api/v1/ask-agentic', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: "How do convolutional neural networks work?",
      model: "llama3.2:1b"
    })
  });
  
  const result = await response.json();
  console.log("Answer:", result.answer);
  console.log("Reasoning Steps:", result.reasoning_steps);
}`,
    curl: `curl -X POST "http://localhost:8000/api/v1/stream" \\
     -H "Content-Type: application/json" \\
     -d '{
       "query": "What is reinforcement learning?",
       "top_k": 4,
       "use_hybrid": true,
       "model": "llama3.2:1b"
     }'`
};

let currentTab = 'python';

function switchCodeTab(tab) {
    currentTab = tab;
    // Update active tab button classes
    document.querySelectorAll('.code-tab').forEach(btn => {
        if (btn.textContent.toLowerCase() === tab || (tab === 'js' && btn.textContent.toLowerCase().includes('node'))) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    // Update code text
    const codeWell = document.getElementById('code-well');
    if (codeWell) {
        codeWell.textContent = codeSnippets[tab];
    }
}

function copyCodeContent() {
    const codeWell = document.getElementById('code-well');
    if (codeWell) {
        navigator.clipboard.writeText(codeWell.textContent).then(() => {
            const btn = document.querySelector('.btn-copy-code');
            const originalSvg = btn.innerHTML;
            btn.innerHTML = '✓';
            setTimeout(() => {
                btn.innerHTML = originalSvg;
            }, 1500);
        });
    }
}

// ==========================================================================
// Category Quick Select
// ==========================================================================
function selectCategory(category) {
    const categoryInput = document.getElementById('category-tags');
    if (categoryInput) {
        let current = categoryInput.value.trim();
        if (current) {
            let parts = current.split(',').map(p => p.trim());
            if (!parts.includes(category)) {
                parts.push(category);
                categoryInput.value = parts.join(', ');
            }
        } else {
            categoryInput.value = category;
        }
    }
}

// ==========================================================================
// Paper Registry Listing
// ==========================================================================
const mockPapers = [
    {
        arxiv_id: "1706.03762",
        title: "Attention Is All You Need",
        authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        categories: ["cs.CL", "cs.LG"],
        pdf_url: "https://arxiv.org/pdf/1706.03762.pdf",
        pdf_processed: true
    },
    {
        arxiv_id: "1810.04805",
        title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        abstract: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations.",
        categories: ["cs.CL", "cs.LG"],
        pdf_url: "https://arxiv.org/pdf/1810.04805.pdf",
        pdf_processed: true
    },
    {
        arxiv_id: "2005.14165",
        title: "Language Models are Few-Shot Learners (GPT-3)",
        authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah"],
        abstract: "We train GPT-3, an autoregressive language model with 175 billion parameters, and test its performance in the few-shot setting. We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
        categories: ["cs.CL", "cs.LG", "cs.AI"],
        pdf_url: "https://arxiv.org/pdf/2005.14165.pdf",
        pdf_processed: true
    },
    {
        arxiv_id: "2303.08774",
        title: "GPT-4 Technical Report",
        authors: ["OpenAI"],
        abstract: "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. GPT-4 exhibits human-level performance on various professional and academic benchmarks.",
        categories: ["cs.CL", "cs.AI", "cs.LG"],
        pdf_url: "https://arxiv.org/pdf/2303.08774.pdf",
        pdf_processed: true
    }
];

function generateThumbnailColor(arxivId) {
    // Generate a harmonious pastel background color based on the arXiv ID digits
    const cleanId = arxivId.replace(/[^0-9]/g, '');
    const num = cleanId ? parseInt(cleanId.slice(0, 4)) : 1234;
    const hue = num % 360;
    return `hsl(${hue}, 25%, 90%)`;
}

function loadPapers() {
    const grid = document.getElementById('papers-grid');
    if (!grid) return;

    fetch('/api/v1/papers')
        .then(response => {
            if (!response.ok) throw new Error('API down');
            return response.json();
        })
        .then(papers => {
            grid.innerHTML = '';
            if (!papers || papers.length === 0) {
                // If API succeeds but DB is empty, render beautiful samples
                renderPapersList(mockPapers, true);
            } else {
                renderPapersList(papers, false);
            }
        })
        .catch(err => {
            console.log("Retrieving live papers failed, loading default repository specimens:", err);
            // Render mocks if backend fails or database connection is down
            grid.innerHTML = '';
            renderPapersList(mockPapers, true);
        });
}

function renderPapersList(papers, isMock) {
    const grid = document.getElementById('papers-grid');
    if (!grid) return;
    
    papers.forEach(paper => {
        const authors = Array.isArray(paper.authors) ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '') : (paper.authors || 'Unknown');
        const categories = Array.isArray(paper.categories) ? paper.categories : [paper.categories];
        const tagsHtml = categories.map(cat => `<span class="badge-tag">${cat}</span>`).join(' ');
        const isProcessed = paper.pdf_processed;
        const color = generateThumbnailColor(paper.arxiv_id);
        
        const cardHtml = `
            <div class="model-card">
                <div class="card-thumbnail" style="background-color: ${color}">
                    <div class="card-image-placeholder">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <polyline points="10 9 9 9 8 9"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="card-author">${authors}</div>
                <h4 class="card-title">${paper.title}</h4>
                <p class="card-desc">${paper.abstract}</p>
                <div class="card-footer">
                    <span class="badge-status" style="background-color: ${isProcessed ? '#2b9a66' : '#8d8d8d'}">
                        ${isProcessed ? 'Ready' : 'Ingested'}
                    </span>
                    <div class="card-tags">
                        ${tagsHtml}
                    </div>
                    <a href="${paper.pdf_url}" target="_blank" class="nav-icon-btn" title="View on arXiv" style="margin-left: 8px;">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                    </a>
                </div>
            </div>
        `;
        grid.innerHTML += cardHtml;
    });

    if (isMock) {
        const warningBanner = document.createElement('div');
        warningBanner.className = 'loading-placeholder';
        warningBanner.style.padding = '16px';
        warningBanner.style.color = 'var(--color-charcoal)';
        warningBanner.innerHTML = `<p style="font-size: 14px; text-align: center;">💡 Displaying mock papers library. To load real documents, start the Airflow data ingestion pipeline or configure your postgres database.</p>`;
        grid.parentNode.insertBefore(warningBanner, grid.nextSibling);
    }
}

// ==========================================================================
// Search & RAG Execution Logic
// ==========================================================================
let currentTraceId = null;

async function executeRAG() {
    const queryInput = document.getElementById('query-text');
    const query = queryInput ? queryInput.value.trim() : '';
    if (!query) return;

    const runBtn = document.getElementById('run-query-btn');
    const outputContainer = document.getElementById('output-container');
    const timelineContainer = document.getElementById('timeline-container');
    const timelineSteps = document.getElementById('timeline-steps');
    const citationsContainer = document.getElementById('citations-container');
    const citationsList = document.getElementById('citations-list');
    const statusText = document.getElementById('terminal-status-text');
    const statusDot = document.querySelector('.status-dot');
    const latencyLabel = document.getElementById('terminal-latency');
    const modeLabel = document.getElementById('terminal-mode');
    const feedbackPanel = document.getElementById('feedback-panel');
    const traceIdDisplay = document.getElementById('trace-id-display');
    const feedbackStatus = document.getElementById('feedback-status');

    // Reset feedback panel state
    feedbackPanel.style.display = 'none';
    traceIdDisplay.textContent = '';
    feedbackStatus.textContent = '';
    currentTraceId = null;

    // Read parameters
    const mode = document.querySelector('input[name="search-mode"]:checked').value;
    const model = document.getElementById('model-select').value;
    const topK = parseInt(document.getElementById('top-k-slider').value);
    const categoryFilterStr = document.getElementById('category-tags').value.trim();
    const categories = categoryFilterStr ? categoryFilterStr.split(',').map(c => c.trim()) : null;

    // Update terminal header status
    statusText.textContent = 'Executing';
    statusDot.className = 'status-dot loading';
    latencyLabel.textContent = 'running...';
    modeLabel.textContent = mode;

    // Clear output contents
    outputContainer.innerHTML = '<p class="placeholder-text">Initializing connection to LLM engine...</p>';
    timelineContainer.style.display = 'none';
    timelineSteps.innerHTML = '';
    citationsContainer.style.display = 'none';
    citationsList.innerHTML = '';

    const startTime = Date.now();

    if (mode === 'agentic') {
        // Agentic RAG Mode (LangGraph POST request)
        try {
            const payload = {
                query: query,
                top_k: topK,
                use_hybrid: true,
                model: model,
                categories: categories
            };

            const response = await fetch('/api/v1/ask-agentic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const elapsed = Date.now() - startTime;
            latencyLabel.textContent = `${elapsed} ms`;

            if (!response.ok) {
                const errJson = await response.json().catch(() => ({detail: "Server Error"}));
                throw new Error(errJson.detail || `Server returned ${response.status}`);
            }

            const data = await response.json();
            
            // Render Answer
            outputContainer.innerHTML = `<p>${formatMarkdown(data.answer)}</p>`;

            // Render Timeline Reasoning Steps
            if (data.reasoning_steps && data.reasoning_steps.length > 0) {
                timelineContainer.style.display = 'block';
                data.reasoning_steps.forEach(step => {
                    const li = document.createElement('li');
                    li.textContent = step;
                    timelineSteps.appendChild(li);
                });
            }

            // Render Citations
            if (data.sources && data.sources.length > 0) {
                citationsContainer.style.display = 'block';
                data.sources.forEach(src => {
                    const li = document.createElement('li');
                    const fileName = src.split('/').pop();
                    li.innerHTML = `<a href="${src}" target="_blank">${fileName}</a>`;
                    citationsList.appendChild(li);
                });
            }

            // Display trace and feedback if trace_id is present
            if (data.trace_id) {
                currentTraceId = data.trace_id;
                feedbackPanel.style.display = 'flex';
                traceIdDisplay.textContent = `Trace ID: ${data.trace_id}`;
            }

            statusText.textContent = 'Completed';
            statusDot.className = 'status-dot success';

        } catch (err) {
            outputContainer.innerHTML = `<p style="color: var(--color-primary);">Error running agentic query: ${err.message}. Make sure Docker Compose services (PostgreSQL, OpenSearch, Ollama) are fully running.</p>`;
            statusText.textContent = 'Failed';
            statusDot.className = 'status-dot warning';
            latencyLabel.textContent = '-- ms';
        }
    } else {
        // Standard / Hybrid search (Streaming endpoint `/api/v1/stream`)
        try {
            const payload = {
                query: query,
                top_k: topK,
                use_hybrid: mode === 'hybrid',
                model: model,
                categories: categories
            };

            const response = await fetch('/api/v1/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API stream error: status ${response.status}`);
            }

            outputContainer.innerHTML = '';
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let currentText = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6).trim();
                        if (!dataStr) continue;

                        try {
                            const parsed = JSON.parse(dataStr);

                            if (parsed.error) {
                                throw new Error(parsed.error);
                            }

                            // Handle Metadata chunk
                            if (parsed.sources) {
                                if (parsed.sources.length > 0) {
                                    citationsContainer.style.display = 'block';
                                    parsed.sources.forEach(src => {
                                        const li = document.createElement('li');
                                        const fileName = src.split('/').pop();
                                        li.innerHTML = `<a href="${src}" target="_blank">${fileName}</a>`;
                                        citationsList.appendChild(li);
                                    });
                                }
                                continue;
                            }

                            // Handle streaming tokens
                            if (parsed.chunk) {
                                currentText += parsed.chunk;
                                outputContainer.innerHTML = `<p>${formatMarkdown(currentText)}</p>`;
                            }

                            // Handle completion
                            if (parsed.done) {
                                if (parsed.answer && parsed.answer !== currentText) {
                                    outputContainer.innerHTML = `<p>${formatMarkdown(parsed.answer)}</p>`;
                                }
                                statusText.textContent = 'Completed';
                                statusDot.className = 'status-dot success';
                                const elapsed = Date.now() - startTime;
                                latencyLabel.textContent = `${elapsed} ms`;
                            }
                        } catch (e) {
                            // Suppress decode warnings
                        }
                    }
                }
            }
        } catch (err) {
            outputContainer.innerHTML = `<p style="color: var(--color-primary);">Connection error: ${err.message}. Ensure backend RAG API server is started.</p>`;
            statusText.textContent = 'Failed';
            statusDot.className = 'status-dot warning';
            latencyLabel.textContent = '-- ms';
        }
    }
}

// Simple Helper to convert Markdown italics/bold/newlines into clean HTML
function formatMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: var(--font-code);">$1</code>');
}

// ==========================================================================
// Langfuse Trace Feedback Ratings
// ==========================================================================
function submitFeedback(score) {
    if (!currentTraceId) return;

    const feedbackStatus = document.getElementById('feedback-status');
    feedbackStatus.textContent = 'Submitting...';

    fetch('/api/v1/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            trace_id: currentTraceId,
            score: score,
            comment: score > 0 ? "Thumbs up from Web UI" : "Thumbs down from Web UI"
        })
    })
    .then(res => {
        if (!res.ok) throw new Error();
        return res.json();
    })
    .then(() => {
        feedbackStatus.textContent = '✓ Recorded';
        feedbackStatus.style.color = 'var(--color-badge-success)';
    })
    .catch(() => {
        feedbackStatus.textContent = 'Failed';
        feedbackStatus.style.color = 'var(--color-primary)';
    });
}

// ==========================================================================
// Initialization & Listeners
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // 1. Set initial code well content
    switchCodeTab('python');

    // 2. Load indexed papers grid
    loadPapers();

    // 3. Register Execute Button listener
    const runBtn = document.getElementById('run-query-btn');
    if (runBtn) {
        runBtn.addEventListener('click', executeRAG);
    }

    // 4. Register slider listener
    const slider = document.getElementById('top-k-slider');
    const sliderVal = document.getElementById('top-k-value');
    if (slider && sliderVal) {
        slider.addEventListener('input', (e) => {
            sliderVal.textContent = `${e.target.value} chunks`;
        });
    }

    // 5. Register feedback ratings
    const upBtn = document.getElementById('feedback-up');
    const downBtn = document.getElementById('feedback-down');
    if (upBtn) {
        upBtn.addEventListener('click', () => submitFeedback(1.0));
    }
    if (downBtn) {
        downBtn.addEventListener('click', () => submitFeedback(-1.0));
    }
});
