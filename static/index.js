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
    document.querySelectorAll('.code-tab').forEach(btn => {
        if (btn.textContent.toLowerCase() === tab || (tab === 'js' && btn.textContent.toLowerCase().includes('node'))) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
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
        authors: ["Ashish Vaswani", "Noam Shazeer"],
        abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        categories: ["cs.CL", "cs.LG"],
        pdf_url: "https://arxiv.org/pdf/1706.03762.pdf",
        pdf_processed: true
    },
    {
        arxiv_id: "1810.04805",
        title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors: ["Jacob Devlin", "Ming-Wei Chang"],
        abstract: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        categories: ["cs.CL", "cs.LG"],
        pdf_url: "https://arxiv.org/pdf/1810.04805.pdf",
        pdf_processed: true
    }
];

function generateThumbnailColor(arxivId) {
    const cleanId = arxivId.replace(/[^0-9]/g, '');
    const num = cleanId ? parseInt(cleanId.slice(0, 4)) : 1234;
    const hue = num % 360;
    return `hsl(${hue}, 25%, 90%)`;
}

function loadPapers() {
    const grid = document.getElementById('papers-grid');
    if (!grid) return;

    fetch('/api/v1/papers', {
        headers: {
            'X-API-Key': 'dev-test-key-999',
            'X-Tenant-ID': 'default'
        }
    })
        .then(response => {
            if (!response.ok) throw new Error('API down');
            return response.json();
        })
        .then(data => {
            grid.innerHTML = '';
            const papersList = data.papers || [];
            if (!papersList || papersList.length === 0) {
                renderPapersList(mockPapers, true);
            } else {
                renderPapersList(papersList, false);
            }
        })
        .catch(err => {
            console.log("Retrieving live papers failed, loading default repository specimens:", err);
            grid.innerHTML = '';
            renderPapersList(mockPapers, true);
        });
}

function renderPapersList(papers, isMock) {
    const grid = document.getElementById('papers-grid');
    if (!grid) return;
    grid.innerHTML = '';
    
    papers.forEach(paper => {
        const authors = Array.isArray(paper.authors) ? paper.authors.slice(0, 2).join(', ') : (paper.authors || 'Unknown');
        const categories = Array.isArray(paper.categories) ? paper.categories : [paper.categories];
        const tagsHtml = categories.map(cat => `<span style="background: rgba(255,255,255,0.06); color: #888; font-size: 10px; padding: 1px 4px; border-radius: 3px; font-family: var(--font-code);">${cat}</span>`).join(' ');
        
        const isUpload = !paper.pdf_url || paper.pdf_url === '#' || paper.arxiv_id.startsWith('upload_') || paper.pdf_url.includes('upload_');
        const pdfLinkHtml = isUpload
            ? `<span style="opacity: 0.3; color: #888; font-size: 11px;">(Upload)</span>`
            : `<a href="${paper.pdf_url}" target="_blank" style="color: var(--color-primary); font-size: 11px; text-decoration: underline;">arXiv PDF</a>`;

        const row = document.createElement('div');
        row.style.background = '#181818';
        row.style.border = '1px solid #282828';
        row.style.borderRadius = '6px';
        row.style.padding = '10px 12px';
        row.style.display = 'flex';
        row.style.flexDirection = 'column';
        row.style.gap = '4px';
        row.style.transition = 'border-color 0.2s';
        row.style.cursor = 'pointer';
        row.onmouseenter = () => { row.style.borderColor = '#444'; };
        row.onmouseleave = () => { row.style.borderColor = '#282828'; };
        row.onclick = () => {
            const queryArea = document.getElementById('query-text');
            if (queryArea) {
                queryArea.value = `Explain the methodology and results of the paper titled "${paper.title}".`;
                queryArea.style.height = '';
                queryArea.style.height = queryArea.scrollHeight + 'px';
                queryArea.focus();
            }
        };

        row.innerHTML = `
            <div style="font-weight: 600; font-size: 13px; color: #fff; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;" title="${paper.title}">
                ${paper.title}
            </div>
            <div style="font-size: 11px; color: #888; display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span>by ${authors}</span>
                ${pdfLinkHtml}
            </div>
            <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-top: 2px;">
                ${tagsHtml}
            </div>
        `;
        grid.appendChild(row);
    });

    if (isMock) {
        const warningBanner = document.createElement('div');
        warningBanner.style.padding = '10px 12px';
        warningBanner.style.color = '#555';
        warningBanner.style.fontSize = '11px';
        warningBanner.style.textAlign = 'center';
        warningBanner.style.lineHeight = '1.4';
        warningBanner.innerHTML = `<p>💡 Showing mock repository. Sync OpenSearch database to see live indexed papers.</p>`;
        grid.appendChild(warningBanner);
    }
}

function renderCitationItem(src) {
    const li = document.createElement('li');
    li.style.display = 'inline-block';
    if (src.includes('upload_') || src === '#' || !src.startsWith('http')) {
        const label = src.includes('upload_') ? src.split('/').pop() : 'Local Uploaded PDF';
        li.innerHTML = `<span style="background-color: #333; color: #ccc; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: default;">Local: ${label}</span>`;
    } else if (src.includes('arxiv.org')) {
        const fileName = src.split('/').pop();
        const arxivId = fileName.replace('.pdf', '');
        li.innerHTML = `<a href="${src}" target="_blank" style="background-color: rgba(234,40,4,0.15); color: var(--color-primary); border: 1px solid rgba(234,40,4,0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; display: inline-flex; align-items: center; gap: 4px; font-weight: 600;"><span>arXiv</span>${arxivId}</a>`;
    } else {
        try {
            const domain = new URL(src).hostname.replace('www.', '');
            li.innerHTML = `<a href="${src}" target="_blank" style="background-color: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; display: inline-flex; align-items: center; gap: 4px; font-weight: 600;"><span>Web</span>${domain}</a>`;
        } catch (e) {
            li.innerHTML = `<a href="${src}" target="_blank" style="background-color: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; display: inline-flex; align-items: center; gap: 4px; font-weight: 600;"><span>Web</span>Link</a>`;
        }
    }
    return li;
}

// ==========================================================================
// Ingestion Pipeline Handler
// ==========================================================================
function triggerPaperIngest() {
    const input = document.getElementById('ingest-arxiv-id');
    const msg = document.getElementById('ingest-status-msg');
    if (!input || !msg) return;

    const arxivId = input.value.trim();
    if (!arxivId) {
        msg.textContent = 'Enter a valid arXiv ID';
        msg.style.color = 'var(--color-primary)';
        return;
    }

    msg.textContent = 'Ingesting paper...';
    msg.style.color = '#aaa';

    fetch(`/api/v1/papers/ingest?arxiv_id=${encodeURIComponent(arxivId)}`, {
        method: 'POST',
        headers: {
            'X-API-Key': 'dev-test-key-999',
            'X-Tenant-ID': 'default'
        }
    })
    .then(res => {
        if (!res.ok) throw new Error('Ingest service error');
        return res.json();
    })
    .then(data => {
        msg.textContent = '✓ Ingested successfully!';
        msg.style.color = 'var(--color-badge-success)';
        input.value = '';
        setTimeout(() => { msg.textContent = ''; }, 3000);
        loadPapers();
    })
    .catch(err => {
        console.error(err);
        msg.textContent = 'Failed to ingest paper';
        msg.style.color = 'var(--color-primary)';
    });
}

// ==========================================================================
// Search & RAG Execution Logic (ChatGPT-style Chat)
// ==========================================================================
let currentTraceId = null;

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderStreamedResponse(text, answerBodyElement) {
    let thinkingText = '';
    let finalAnswerText = '';
    
    const thinkStart = text.indexOf('<think>');
    const thinkEnd = text.indexOf('</think>');
    
    if (thinkStart !== -1) {
        if (thinkEnd !== -1) {
            thinkingText = text.substring(thinkStart + 7, thinkEnd).trim();
            finalAnswerText = text.substring(thinkEnd + 8).trim();
        } else {
            thinkingText = text.substring(thinkStart + 7).trim();
        }
    } else {
        finalAnswerText = text;
    }
    
    let html = '';
    if (thinkingText) {
        html += `
            <div class="thinking-container" style="background: rgba(255, 255, 255, 0.02); border-left: 3px solid var(--color-primary); padding: 12px 16px; margin-bottom: 14px; border-radius: 0 8px 8px 0; font-size: 13.5px; color: #a0a0a0; box-shadow: inset 2px 0 0 rgba(0,0,0,0.5);">
                <div style="font-weight: bold; color: var(--color-primary); margin-bottom: 6px; display: flex; align-items: center; gap: 6px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                    <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2.5" fill="none" class="pulse-icon" style="animation: brain-pulse 1.6s infinite; display: inline-block; vertical-align: middle;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/></svg>
                    Thinking Process
                </div>
                <div style="font-style: italic; line-height: 1.5; white-space: pre-wrap; font-family: var(--font-body);">${escapeHtml(thinkingText)}</div>
            </div>
        `;
    }
    if (finalAnswerText) {
        html += `<div class="final-response">${formatMarkdown(finalAnswerText)}</div>`;
    }
    answerBodyElement.innerHTML = html;
}


async function executeRAG() {
    const queryInput = document.getElementById('query-text');
    const query = queryInput ? queryInput.value.trim() : '';
    if (!query) return;

    const chatHistory = document.getElementById('chat-history');
    const model = document.getElementById('model-select').value;
    const topK = parseInt(document.getElementById('top-k-slider').value);
    const categoryFilterStr = document.getElementById('category-tags').value.trim();
    const categories = categoryFilterStr ? categoryFilterStr.split(',').map(c => c.trim()) : null;
    const statusText = document.getElementById('terminal-status-text');
    const statusDot = document.getElementById('status-dot-indicator');
    const latencyLabel = document.getElementById('terminal-latency');
    const feedbackPanel = document.getElementById('feedback-panel');
    const feedbackStatus = document.getElementById('feedback-status');

    // Reset feedback
    feedbackPanel.style.display = 'none';
    feedbackStatus.textContent = '';
    currentTraceId = null;

    // Reset Input Box height
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // 1. Append User Message
    const userMsg = document.createElement('div');
    userMsg.className = 'msg-bubble user';
    userMsg.style.display = 'flex';
    userMsg.style.gap = '16px';
    userMsg.style.padding = '18px';
    userMsg.style.background = '#111';
    userMsg.style.border = '1px solid #1a1a1a';
    userMsg.style.borderRadius = '8px';
    userMsg.style.maxWidth = '90%';
    userMsg.style.marginLeft = 'auto';
    userMsg.innerHTML = `
        <div style="flex: 1; text-align: right;">
            <p style="font-size: 14.5px; line-height: 1.6; color: #fff; margin: 0;">${escapeHtml(query)}</p>
        </div>
        <div style="width: 32px; height: 32px; background: #333; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; font-size: 12px; flex-shrink: 0; font-family: var(--font-display);">U</div>
    `;
    chatHistory.appendChild(userMsg);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Update status to loading
    statusText.textContent = 'Thinking';
    statusDot.className = 'status-dot loading';
    statusDot.style.background = '#ffc000';
    statusDot.style.boxShadow = '0 0 8px #ffc000';
    latencyLabel.textContent = 'running...';

    // 2. Append Agent Message Bubble
    const agentMsg = document.createElement('div');
    agentMsg.className = 'msg-bubble agent';
    agentMsg.style.display = 'flex';
    agentMsg.style.gap = '16px';
    agentMsg.style.padding = '18px';
    agentMsg.style.background = '#0a0a0a';
    agentMsg.style.border = '1px solid #181818';
    agentMsg.style.borderRadius = '8px';
    agentMsg.style.maxWidth = '90%';
    
    agentMsg.innerHTML = `
        <div style="width: 32px; height: 32px; background: var(--color-primary); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; font-size: 12px; flex-shrink: 0; font-family: var(--font-display);">AI</div>
        <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
            <div class="agent-activity" style="font-size: 12.5px; color: #888; font-style: italic; display: flex; align-items: center; gap: 8px;">
                <span class="spinner" style="border: 2px solid rgba(255,255,255,0.1); border-top-color: var(--color-primary); border-radius: 50%; width: 14px; height: 14px; display: inline-block; animation: spin 1s linear infinite;"></span>
                <span class="activity-text">Contacting agent router...</span>
            </div>
            <details class="thought-process-details" style="display: none; background: #0e0e0e; border: 1px solid #222; border-radius: 6px; padding: 10px 14px;">
                <summary style="font-size: 12px; cursor: pointer; user-select: none; color: #888; font-weight: 600; outline: none;">🧠 Thought Process</summary>
                <ul class="thought-steps-list" style="margin-top: 10px; margin-left: 18px; padding: 0; font-size: 12px; color: #aaa; display: flex; flex-direction: column; gap: 6px;"></ul>
            </details>
            <div class="answer-body" style="font-size: 14.5px; line-height: 1.6; color: #e0e0e0; min-height: 18px;"></div>
            <div class="visual-gallery" style="display: none; margin-top: 12px;"></div>
            <div class="citations-footer" style="display: none; border-top: 1px solid #1c1c1c; padding-top: 10px; flex-direction: column; gap: 6px;">
                <div style="font-size: 11.5px; font-weight: 600; color: #646464; text-transform: uppercase;">Retained Sources:</div>
                <ul class="citations-ul" style="display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 0;"></ul>
            </div>
        </div>
    `;
    chatHistory.appendChild(agentMsg);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    const activityText = agentMsg.querySelector('.activity-text');
    const thoughtDetails = agentMsg.querySelector('.thought-process-details');
    const thoughtList = agentMsg.querySelector('.thought-steps-list');
    const answerBody = agentMsg.querySelector('.answer-body');
    const visualGallery = agentMsg.querySelector('.visual-gallery');
    const citationsFooter = agentMsg.querySelector('.citations-footer');
    const citationsUl = agentMsg.querySelector('.citations-ul');
    const activityDiv = agentMsg.querySelector('.agent-activity');

    const startTime = Date.now();

    try {
        const payload = {
            query: query,
            top_k: topK,
            use_hybrid: true,
            model: model,
            categories: categories,
            search_mode: "auto"
        };

        const response = await fetch('/api/v1/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'dev-test-key-999',
                'X-Tenant-ID': 'default'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API stream error: status ${response.status}`);
        }

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

                        // 1. Process steps (Thought process)
                        if (parsed.step) {
                            activityText.textContent = parsed.step;
                            thoughtDetails.style.display = 'block';
                            const li = document.createElement('li');
                            li.textContent = parsed.step;
                            thoughtList.appendChild(li);
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }

                        // 2. Process search_mode details
                        if (parsed.search_mode) {
                            const badge = document.getElementById('routing-badge');
                            if (badge) {
                                badge.textContent = `⚡ Routed: ${parsed.search_mode.toUpperCase()}`;
                            }
                        }

                        // 3. Process visual results (ColPali vision)
                        if (parsed.visual_results && parsed.visual_results.length > 0) {
                            activityDiv.style.display = 'none';
                            let html = `
                                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top:10px;">
                            `;
                            parsed.visual_results.forEach(hit => {
                                html += `
                                    <div style="background:#141414; border:1px solid #222; border-radius:6px; overflow:hidden; display:flex; flex-direction:column; padding:8px;">
                                        <div style="position:relative; width:100%; height:160px; background:#000; border-radius:4px; overflow:hidden; display:flex; align-items:center; justify-content:center;">
                                            <img src="${hit.image_path}" style="max-width:100%; max-height:100%; object-fit:contain; cursor:pointer;" onclick="window.open('${hit.image_path}', '_blank')" title="Click to view layout page" />
                                        </div>
                                        <div style="margin-top:8px; font-size:11px; display:flex; flex-direction:column; gap:2px; color:#aaa;">
                                            <span style="font-weight:bold; color:var(--color-primary);">arXiv ID: ${hit.arxiv_id}</span>
                                            <span>Page: <strong style="color:white;">${hit.page_number}</strong></span>
                                            <span>Score: <strong style="color:#4caf50;">${hit.score.toFixed(4)}</strong></span>
                                        </div>
                                    </div>
                                `;
                            });
                            html += `</div>`;
                            visualGallery.innerHTML = html;
                            visualGallery.style.display = 'block';
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }

                        // 4. Process citations sources
                        if (parsed.sources && parsed.sources.length > 0) {
                            citationsFooter.style.display = 'flex';
                            parsed.sources.forEach(src => {
                                citationsUl.appendChild(renderCitationItem(src));
                            });
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }

                        // 5. Process streaming chunk
                        if (parsed.chunk) {
                            activityDiv.style.display = 'none';
                            currentText += parsed.chunk;
                            renderStreamedResponse(currentText, answerBody);
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }

                        // 6. Process completion
                        if (parsed.done) {
                            activityDiv.style.display = 'none';
                            if (parsed.answer && parsed.answer !== currentText) {
                                renderStreamedResponse(parsed.answer, answerBody);
                            } else {
                                renderStreamedResponse(currentText, answerBody);
                            }
                            statusText.textContent = 'Standby';
                            statusDot.className = 'status-dot success';
                            statusDot.style.background = '#2b9a66';
                            statusDot.style.boxShadow = '0 0 8px #2b9a66';
                            const elapsed = Date.now() - startTime;
                            latencyLabel.textContent = `${elapsed} ms`;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }

                        // 7. Process Langfuse trace feedback key
                        if (parsed.trace_id) {
                            currentTraceId = parsed.trace_id;
                            feedbackPanel.style.display = 'flex';
                        }

                    } catch (e) {
                        // ignore malformed chunks
                    }
                }
            }
        }
    } catch (err) {
        activityDiv.style.display = 'none';
        answerBody.innerHTML = `<p style="color: var(--color-primary); margin:0;">Connection error: ${err.message}. Ensure backend RAG API server is started.</p>`;
        statusText.textContent = 'Failed';
        statusDot.className = 'status-dot warning';
        statusDot.style.background = '#ea2804';
        statusDot.style.boxShadow = '0 0 8px #ea2804';
        latencyLabel.textContent = '-- ms';
        chatHistory.scrollTop = chatHistory.scrollHeight;
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
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'dev-test-key-999',
            'X-Tenant-ID': 'default'
        },
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
    })
    .catch(() => {
        feedbackStatus.textContent = 'Failed';
    });
}

// ==========================================================================
// Initialization & Listeners
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Inject brain-pulse animation style
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes brain-pulse {
            0% { opacity: 0.4; transform: scale(0.96); }
            50% { opacity: 1; transform: scale(1.06); }
            100% { opacity: 0.4; transform: scale(0.96); }
        }
    `;
    document.head.appendChild(style);

    switchCodeTab('python');
    loadPapers();

    const runBtn = document.getElementById('run-query-btn');
    if (runBtn) {
        runBtn.addEventListener('click', executeRAG);
    }

    const queryInput = document.getElementById('query-text');
    if (queryInput) {
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                executeRAG();
            }
        });
    }

    const ingestBtn = document.getElementById('btn-ingest-paper');
    if (ingestBtn) {
        ingestBtn.addEventListener('click', triggerPaperIngest);
    }

    const ingestInput = document.getElementById('ingest-arxiv-id');
    if (ingestInput) {
        ingestInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                triggerPaperIngest();
            }
        });
    }

    const slider = document.getElementById('top-k-slider');
    const sliderVal = document.getElementById('top-k-value');
    if (slider && sliderVal) {
        slider.addEventListener('input', (e) => {
            sliderVal.textContent = e.target.value;
        });
    }

    const upBtn = document.getElementById('feedback-up');
    const downBtn = document.getElementById('feedback-down');
    if (upBtn) {
        upBtn.addEventListener('click', () => submitFeedback(1.0));
    }
    if (downBtn) {
        downBtn.addEventListener('click', () => submitFeedback(-1.0));
    }
});
