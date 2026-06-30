import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Search, 
  Play, 
  Loader2, 
  AlertCircle, 
  ExternalLink,
  BookOpen,
  Download,
  ChevronDown,
  ChevronUp,
  BrainCircuit
} from 'lucide-react';

interface Paper {
  paper_id: string;
  title: string;
  authors: string;
  year: number | string;
  url: string;
  full_text_path: string;
}

interface Chunk {
  chunk_id: string;
  chunk_text: string;
  chunk_index: number;
  page_number: number;
  section_title: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  thinking?: string;
  showThinking?: boolean;
  citations?: Record<string, any>;
  verification?: Record<string, any>;
  agent_logs?: string[];
  isLoading?: boolean;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'explorer'>('chat');
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeAgent, setActiveAgent] = useState<string>('Idle');
  const [currentLogs, setCurrentLogs] = useState<string[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  
  // Explorer states
  const [selectedPaperId, setSelectedPaperId] = useState<string>('');
  const [selectedPaperPage, setSelectedPaperPage] = useState<number>(1);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  
  // Citation card state
  const [activeCitation, setActiveCitation] = useState<any>(null);
  const [activeChunkId, setActiveChunkId] = useState<string>('');
  const [models, setModels] = useState<string[]>(["llama3.2:1b", "gemma4:latest"]);
  const [selectedModel, setSelectedModel] = useState<string>("llama3.2:1b");
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch papers at load
  useEffect(() => {
    fetchPapers();
    fetchModels();
    
    // Register global showCitation function for HTML badges in assistant output
    (window as any).showCitation = (id: string) => {
      // Find citation in the last assistant message
      setMessages(prev => {
        const lastMsg = [...prev].reverse().find(m => m.role === 'assistant');
        if (lastMsg && lastMsg.citations && lastMsg.citations[id]) {
          const cit = lastMsg.citations[id];
          setActiveCitation(cit);
          setActiveChunkId(cit.chunk_id || '');
          setSelectedPaperId(cit.paper_id);
          setSelectedPaperPage(cit.page || 1);
          setActiveTab('explorer');
        }
        return prev;
      });
    };

    return () => {
      delete (window as any).showCitation;
    };
  }, []);

  // Scroll active chunk into view when highlighted
  useEffect(() => {
    if (activeChunkId && chunks.length > 0) {
      const timer = setTimeout(() => {
        const element = document.getElementById(`chunk-${activeChunkId}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 250);
      return () => clearTimeout(timer);
    }
  }, [activeChunkId, chunks]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentLogs]);

  useEffect(() => {
    if (selectedPaperId) {
      fetchChunks(selectedPaperId);
    }
  }, [selectedPaperId]);

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8502/api/models');
      const data = await res.json();
      if (data.models && data.models.length > 0) {
        setModels(data.models);
        const defaultModel = data.models.includes("llama3.2:1b") ? "llama3.2:1b" : data.models[0];
        setSelectedModel(defaultModel);
      }
    } catch (err) {
      console.error("Error fetching models:", err);
    }
  };

  const fetchPapers = async () => {
    try {
      const res = await fetch('http://localhost:8502/api/papers');
      const data = await res.json();
      setPapers(data);
      if (data.length > 0 && !selectedPaperId) {
        setSelectedPaperId(data[0].paper_id);
      }
    } catch (err) {
      console.error("Error fetching papers:", err);
    }
  };

  const fetchChunks = async (paperId: string) => {
    try {
      const res = await fetch(`http://localhost:8502/api/chunks/${paperId}`);
      const data = await res.json();
      setChunks(data);
      setSelectedPaperPage(1);
    } catch (err) {
      console.error("Error fetching chunks:", err);
    }
  };

  const handleIngest = async () => {
    setActiveAgent('download_and_index');
    try {
      await fetch('http://localhost:8502/api/query?query=__ingest_trigger__');
    } catch (err) {
      console.error("Ingestion error:", err);
    } finally {
      setActiveAgent('Idle');
      fetchPapers(); // refresh count
    }
  };

  const handleExportOKF = () => {
    window.open('http://localhost:8502/api/okf/export', '_blank');
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage: Message = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    
    const assistantIndex = messages.length + 1;
    setCurrentLogs(["Supervisor: Initializing LangGraph state machine..."]);
    setActiveAgent('check_knowledge');
    
    // Add temporary loading message
    setMessages(prev => [...prev, { role: 'assistant', content: '', isLoading: true }]);
    
    const eventSource = new EventSource(`http://localhost:8502/api/query?query=${encodeURIComponent(query)}&model=${encodeURIComponent(selectedModel)}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.error) {
        setMessages(prev => {
          const updated = [...prev];
          updated[assistantIndex] = {
            role: 'assistant',
            content: `Failed to process request: ${data.error}`,
            isLoading: false
          };
          return updated;
        });
        eventSource.close();
        setActiveAgent('Idle');
        return;
      }

      if (data.node) {
        setActiveAgent(data.node);
      }
      
      if (data.agent_logs) {
        setCurrentLogs(data.agent_logs);
      }

      // If final state generated
      if (data.answer) {
        setMessages(prev => {
          const updated = [...prev];
          updated[assistantIndex] = {
            role: 'assistant',
            content: data.answer,
            thinking: data.thinking || '',
            showThinking: false,
            citations: data.citations || {},
            verification: data.verification || {},
            agent_logs: data.agent_logs || [],
            isLoading: false
          };
          return updated;
        });
        
        eventSource.close();
        setActiveAgent('Idle');
        fetchPapers(); // refresh paper count
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      setMessages(prev => {
        const updated = [...prev];
        if (updated[assistantIndex]?.isLoading) {
          updated[assistantIndex] = {
            role: 'assistant',
            content: "Connection to local LLM failed or timed out.",
            isLoading: false
          };
        }
        return updated;
      });
      eventSource.close();
      setActiveAgent('Idle');
    };

    setQuery('');
  };

  const getAgentStatusClass = (agentNode: string) => {
    if (activeAgent === agentNode) return 'status-working';
    if (activeAgent === 'Idle') return 'status-idle';
    return 'status-standby';
  };

  const getAgentStatusText = (agentNode: string) => {
    if (activeAgent === agentNode) return 'WORKING';
    if (activeAgent === 'Idle') return 'IDLE';
    return 'STANDBY';
  };

  return (
    <div className="flex w-full h-screen overflow-hidden bg-[#0B0E17] text-[#ECEFF4] font-sans">
      
      {/* 1. Sidebar Panel */}
      <aside className="w-80 h-full border-r border-[rgba(255,90,0,0.15)] bg-[rgba(11,14,23,0.95)] backdrop-blur-md flex flex-col justify-between p-6 overflow-y-auto">
        <div className="flex flex-col gap-6">
          <div>
            <h1 className="text-2xl font-black tracking-tight bg-gradient-to-r from-[#FF6B00] to-[#E04E00] bg-clip-text text-transparent">
              🌌 Antigravity
            </h1>
            <p className="text-xs text-[#718096] italic mt-0.5">Local Agentic RAG Platform</p>
          </div>
          
          <hr className="border-t border-[rgba(255,255,255,0.06)]" />
          
          {/* Agent Status Dashboard */}
          <div>
            <h3 className="text-sm font-semibold tracking-wide text-[#FF5A00] mb-3 flex items-center gap-2">
              🤖 Agent Status Dashboard
            </h3>
            
            <div className="flex flex-col gap-2 bg-[#161D2B] p-4 rounded-xl border border-gray-800">
              {[
                { name: "Supervisor", node: "check_knowledge" },
                { name: "Web Search", node: "web_search" },
                { name: "PDF Ingestion", node: "download_and_index" },
                { name: "Retrieval", node: "retrieve" },
                { name: "Generation", node: "generate" },
                { name: "Verification", node: "verify" }
              ].map(agent => (
                <div key={agent.name} className="flex justify-between items-center text-sm font-medium">
                  <span className="text-[#ECEFF4]">{agent.name}</span>
                  <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold tracking-wider ${getAgentStatusClass(agent.node)}`}>
                    {getAgentStatusText(agent.node)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          <hr className="border-t border-[rgba(255,255,255,0.06)]" />
          
          {/* Database Info & Stats */}
          <div>
            <h3 className="text-sm font-semibold tracking-wide text-[#FF5A00] mb-2">
              📊 System Statistics
            </h3>
            <div className="bg-[#161D2B] p-4 rounded-xl border border-gray-800 flex flex-col gap-3">
              <div>
                <p className="text-xs text-[#718096]">Ingested Papers</p>
                <p className="text-2xl font-bold text-[#FF6B00] mt-0.5">{papers.length}</p>
              </div>
              {papers.length > 0 && (
                <div>
                  <p className="text-xs text-[#718096]">Latest Paper</p>
                  <p className="text-xs font-semibold text-[#E2E8F0] line-clamp-2 mt-0.5">
                    {papers[0].title}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons at bottom */}
        <div className="flex flex-col gap-3">
          <button 
            onClick={handleIngest}
            disabled={activeAgent !== 'Idle'}
            className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-[#FF6B00] to-[#E04E00] text-white font-semibold rounded-xl shadow-lg shadow-[rgba(255,107,0,0.25)] hover:shadow-[rgba(255,107,0,0.45)] transform hover:-translate-y-0.5 transition active:translate-y-0 disabled:opacity-50 disabled:pointer-events-none text-sm"
          >
            <Play size={15} />
            <span>Trigger Scheduled Ingestion</span>
          </button>
          
          <button 
            onClick={handleExportOKF}
            className="w-full flex items-center justify-center gap-2 py-3 bg-[#161D2B] border border-gray-800 hover:border-[#FF5A00] text-[#E2E8F0] hover:text-[#FF5A00] font-semibold rounded-xl transition text-sm cursor-pointer"
          >
            <Download size={15} />
            <span>Export OKF Bundle</span>
          </button>
        </div>
      </aside>

      {/* 2. Main Content Area */}
      <main className="flex-1 h-full flex flex-col overflow-hidden">
        
        {/* Navigation Tabs Header */}
        <header className="h-16 border-b border-gray-800 bg-[#0E121E] flex items-center justify-between px-8">
          <div className="flex gap-4">
            <button 
              onClick={() => setActiveTab('chat')}
              className={`h-16 flex items-center gap-2 px-4 border-b-2 font-semibold text-sm transition ${
                activeTab === 'chat' ? 'border-[#FF5A00] text-[#FF5A00]' : 'border-transparent text-[#718096] hover:text-[#FF5A00]'
              }`}
            >
              <MessageSquare size={16} />
              <span>Assistant Chat</span>
            </button>
            <button 
              onClick={() => setActiveTab('explorer')}
              className={`h-16 flex items-center gap-2 px-4 border-b-2 font-semibold text-sm transition ${
                activeTab === 'explorer' ? 'border-[#FF5A00] text-[#FF5A00]' : 'border-transparent text-[#718096] hover:text-[#FF5A00]'
              }`}
            >
              <Search size={16} />
              <span>Deep Source Explorer</span>
            </button>
          </div>
        </header>

        {/* Tab Components */}
        <div className="flex-1 overflow-hidden relative">
          
          {/* A. Chat Interface */}
          {activeTab === 'chat' && (
            <div className="w-full h-full flex flex-col justify-between">
              
              {/* Message scroll list */}
              <div className="flex-1 overflow-y-auto p-8 space-y-6">
                {messages.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-4">
                    <BookOpen size={48} className="text-[#FF5A00]" />
                    <h2 className="text-xl font-bold">Ask Research Questions</h2>
                    <p className="text-sm text-[#718096]">
                      Submit natural language questions. Antigravity will index the web dynamically, cross-check claims, and output subscript citations.
                    </p>
                  </div>
                )}
                
                {messages.map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`p-5 rounded-2xl max-w-3xl shadow-lg border backdrop-blur-sm ${
                      msg.role === 'user' 
                        ? 'bg-gradient-to-br from-[rgba(255,107,0,0.08)] to-[rgba(255,107,0,0.02)] border-[rgba(255,107,0,0.15)] text-[#ECEFF4]' 
                        : 'bg-[#141A26] border-gray-800 text-[#E2E8F0]'
                    }`}>
                      {msg.role === 'assistant' && (
                        <div className="text-xs font-bold text-[#FF5A00] mb-2 tracking-wider">
                          🌌 ANTIGRAVITY ASSISTANT
                        </div>
                      )}
                      
                      {msg.isLoading ? (
                        <div className="flex items-center gap-2 text-sm text-[#718096]">
                          <Loader2 size={16} className="animate-spin" />
                          <span>Generating tokens...</span>
                        </div>
                      ) : (
                        <div className="flex flex-col gap-3">
                          {msg.thinking && (
                            <div className="mb-1 border border-gray-800 rounded-xl bg-[#0D121E]/80 overflow-hidden transition">
                              <button
                                onClick={() => {
                                  setMessages(prev => {
                                    const updated = [...prev];
                                    updated[idx].showThinking = !updated[idx].showThinking;
                                    return updated;
                                  });
                                }}
                                className="w-full flex items-center justify-between px-4 py-2.5 bg-[#121824]/60 hover:bg-[#121824] transition text-xs font-semibold text-[#FF5A00] tracking-wide"
                              >
                                <span className="flex items-center gap-1.5">
                                  <BrainCircuit size={13} className="text-[#FF5A00]" />
                                  <span>View LLM Thinking Process</span>
                                </span>
                                {msg.showThinking ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                              </button>
                              
                              {msg.showThinking && (
                                <div className="p-4 border-t border-gray-800/60 text-xs text-[#A0AEC0] italic leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto bg-[#0E1320]/40">
                                  {msg.thinking}
                                </div>
                              )}
                            </div>
                          )}
                          
                          <div 
                            className="prose prose-invert text-[14.5px] leading-relaxed"
                            dangerouslySetInnerHTML={{ __html: msg.content }}
                          />
                        </div>
                      )}

                      {/* Fact Checking Flags */}
                      {msg.verification && Object.keys(msg.verification).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-800 flex flex-wrap gap-2">
                          {Object.entries(msg.verification).map(([claim, val]: any) => (
                            val.status === 'refuted' && (
                              <div key={claim} className="flex items-center gap-1.5 text-xs text-red-500 font-semibold bg-red-950/40 border border-red-900/60 px-2.5 py-1 rounded-md">
                                <AlertCircle size={12} />
                                <span>Refuted Statement Flagged: "{claim.substring(0, 30)}..."</span>
                              </div>
                            )
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Streaming Agent thinking panel */}
                {activeAgent !== 'Idle' && (
                  <div className="flex justify-start">
                    <div className="p-5 rounded-2xl bg-[#141A26] border-gray-800 w-full max-w-3xl">
                      <div className="text-xs font-bold text-[#FF5A00] mb-3 tracking-wider flex items-center gap-2">
                        <Loader2 size={14} className="animate-spin" />
                        <span>🧠 LOG ENGINE (SSE STREAM)</span>
                      </div>
                      <div className="bg-[#0B0E17] rounded-xl p-4 border border-gray-900 font-mono text-[11.5px] leading-relaxed text-[#ECEFF4] max-h-48 overflow-y-auto space-y-1">
                        {currentLogs.map((log, i) => (
                          <div key={i}>{log}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={chatEndRef} />
              </div>

              {/* Floating Popover citation details card */}
              {activeCitation && (
                <div className="absolute bottom-24 left-8 right-8 bg-[#1E2536]/95 backdrop-blur-md text-[#ECEFF4] border border-[rgba(255,90,0,0.25)] rounded-xl p-4 shadow-2xl flex flex-col gap-2 max-h-40 overflow-y-auto z-50">
                  <div className="flex justify-between items-center text-sm font-bold text-[#FF6B00]">
                    <span>{activeCitation.paper_title} ({activeCitation.year})</span>
                    <button 
                      onClick={() => setActiveCitation(null)}
                      className="text-red-500 hover:text-red-400 font-bold text-lg"
                    >
                      ×
                    </button>
                  </div>
                  <div className="text-[11px] italic text-[#A0AEC0]">By: {activeCitation.authors}</div>
                  <div className="text-xs bg-[#0D111C]/60 border-l-2 border-[#FF6B00] p-2 rounded-r-md text-[#E2E8F0] whitespace-pre-wrap">
                    {activeCitation.excerpt}
                  </div>
                </div>
              )}

              {/* Chat Input panel */}
              <form onSubmit={handleSearchSubmit} className="p-8 bg-[#0B0E17] border-t border-gray-800 flex gap-4 items-center">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={activeAgent !== 'Idle'}
                  className="bg-[#121824] border border-gray-800 text-[#ECEFF4] rounded-xl px-3 py-3 text-sm focus:outline-none focus:border-[#FF5A00] transition disabled:opacity-50 font-semibold"
                >
                  {models.map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
                <input 
                  type="text" 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a research question..."
                  disabled={activeAgent !== 'Idle'}
                  className="flex-1 bg-[#121824] border border-gray-800 text-[#ECEFF4] rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-[#FF5A00] transition disabled:opacity-50"
                />
                <button 
                  type="submit"
                  disabled={activeAgent !== 'Idle' || !query.trim()}
                  className="px-5 py-3 bg-gradient-to-r from-[#FF6B00] to-[#E04E00] text-white rounded-xl shadow-md hover:shadow-lg transition font-semibold text-sm flex items-center gap-1.5 disabled:opacity-50"
                >
                  <MessageSquare size={16} />
                  <span>Send</span>
                </button>
              </form>
            </div>
          )}

          {/* B. Deep Source Explorer */}
          {activeTab === 'explorer' && (
            <div className="w-full h-full flex overflow-hidden">
              
              {/* Left Column: Papers list and Chunks accordion */}
              <div className="w-1/2 h-full border-r border-gray-800 flex flex-col p-6 overflow-y-auto gap-4">
                <div>
                  <label className="text-xs font-bold uppercase tracking-wider text-[#FF5A00]">Select Indexed Document</label>
                  <select 
                    value={selectedPaperId}
                    onChange={(e) => setSelectedPaperId(e.target.value)}
                    className="w-full mt-2 bg-[#121824] border border-gray-800 text-[#ECEFF4] p-3 rounded-lg text-sm focus:outline-none focus:border-[#FF5A00]"
                  >
                    {papers.map(p => (
                      <option key={p.paper_id} value={p.paper_id}>{p.title}</option>
                    ))}
                  </select>
                </div>
                
                <hr className="border-t border-gray-800" />
                
                <div className="flex-1 space-y-3">
                  <h4 className="text-sm font-bold text-[#E2E8F0]">Extracted Text Chunks</h4>
                  {chunks.length === 0 ? (
                    <p className="text-xs text-[#718096]">No text chunks found for this paper.</p>
                  ) : (
                    <div className="space-y-2">
                      {chunks.map((chunk) => (
                        <div 
                          key={chunk.chunk_id}
                          id={`chunk-${chunk.chunk_id}`}
                          onClick={() => {
                            setSelectedPaperPage(chunk.page_number);
                            setActiveChunkId(chunk.chunk_id);
                          }}
                          className={`p-3 rounded-lg border text-left cursor-pointer transition ${
                            chunk.chunk_id === activeChunkId
                              ? 'bg-[rgba(255,107,0,0.08)] border-[#FF5A00] shadow-md shadow-[rgba(255,107,0,0.05)]'
                              : selectedPaperPage === chunk.page_number
                              ? 'bg-gradient-to-r from-[rgba(255,107,0,0.06)] to-transparent border-[rgba(255,107,0,0.25)]'
                              : 'bg-[#121824]/40 border-gray-900 hover:border-gray-700'
                          }`}
                        >
                          <div className="flex justify-between items-center text-xs font-semibold text-[#FF5A00] mb-1">
                            <span>{chunk.section_title || `Section chunk ${chunk.chunk_index}`}</span>
                            <span className="bg-gray-800 px-2 py-0.5 rounded text-[10px] text-gray-400">
                              Page {chunk.page_number}
                            </span>
                          </div>
                          <p className="text-xs text-[#A0AEC0] line-clamp-3 leading-relaxed">{chunk.chunk_text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: PDF Viewer Frame */}
              <div className="w-1/2 h-full flex flex-col p-6 bg-[#0E121E]">
                {selectedPaperId ? (
                  <div className="w-full h-full flex flex-col justify-between gap-4">
                    <div className="flex justify-between items-center text-sm font-semibold">
                      <span className="text-[#ECEFF4]">PDF Viewer (Page {selectedPaperPage})</span>
                      <a 
                        href={`http://localhost:8502/pdf/${selectedPaperId}?page=${selectedPaperPage}`}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1 text-xs text-[#FF6B00] hover:underline"
                      >
                        <ExternalLink size={12} />
                        <span>Open PDF in new tab</span>
                      </a>
                    </div>
                    <div className="flex-1 w-full bg-white rounded-lg overflow-hidden relative border border-gray-800">
                      <iframe 
                        src={`http://localhost:8502/pdf/${selectedPaperId}#page=${selectedPaperPage}`}
                        title="PDF Viewer"
                        className="w-full h-full border-none"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-center text-[#718096]">
                    Select a paper on the left to load the PDF view.
                  </div>
                )}
              </div>

            </div>
          )}

        </div>
      </main>

    </div>
  );
}
