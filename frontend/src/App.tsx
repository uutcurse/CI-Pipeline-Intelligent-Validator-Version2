import React, { useState, useEffect } from 'react';
import { 
  checkHealth, 
  getModelInfo, 
  predictWorkflow
} from './services/api';

import type {
  PredictionResponse, 
  ModelInfoResponse 
} from './services/api';

import './App.css';

const App: React.FC = () => {
  const [yamlInput, setYamlInput] = useState<string>('name: CI\n\non:\n  push:\n\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'Connected' | 'Offline' | 'Checking'>('Checking');
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [showModelInfo, setShowModelInfo] = useState(false);

  useEffect(() => {
    const init = async () => {
      try {
        const health = await checkHealth();
        if (health.status === 'UP') {
          setBackendStatus('Connected');
          const info = await getModelInfo();
          setModelInfo(info);
        } else {
          setBackendStatus('Offline');
        }
      } catch (err) {
        setBackendStatus('Offline');
      }
    };
    init();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 1024 * 1024) {
      setError("File exceeds 1MB limit.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (evt) => {
      const content = evt.target?.result as string;
      setYamlInput(content);
      setError(null);
    };
    reader.readAsText(file);
  };

  const handleAnalyze = async () => {
    if (!yamlInput.trim()) {
      setError("Input is empty.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await predictWorkflow(yamlInput);
      setResult(res);
    } catch (err: any) {
      const msg = err.response?.data?.error?.message || "An unexpected error occurred.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (label: string) => {
    switch(label) {
      case 'LOW': return 'var(--color-success)';
      case 'MEDIUM': return 'var(--color-warning)';
      case 'HIGH': return 'var(--color-error)';
      default: return 'var(--text-main)';
    }
  };

  return (
    <>
      <div className="wallpaper-layer"></div>
      <div className="wallpaper-overlay"></div>
      
      <div style={{ position: 'relative', zIndex: 2, padding: '2rem', maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        
        {/* Header */}
        <header className="glass-panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.5rem', marginBottom: '0.25rem', letterSpacing: '0.5px' }}>RunSure</h2>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>CI Workflow Risk Validator</span>
              <span style={{ fontSize: '0.75rem', background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px', color: 'var(--accent-primary)' }}>Static ML Risk Analysis</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', fontSize: '0.9rem' }}>
            <span className={`status-dot status-${backendStatus.toLowerCase()}`}></span>
            Backend {backendStatus}
          </div>
        </header>

        {/* Hero */}
        <section style={{ textAlign: 'center', padding: '1rem 0' }}>
          <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', background: 'linear-gradient(90deg, #F5F7FA, #AEB8C7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Analyze CI Workflow Risk</h1>
          <p style={{ fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto', lineHeight: '1.6' }}>Evaluate GitHub Actions workflows before execution using hybrid textual and structural machine learning.</p>
        </section>

        {/* Main Analyzer Card */}
        <section className="glass-panel">
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Workflow Analyzer</h3>
            <p style={{ fontSize: '0.9rem' }}>Paste GitHub Actions YAML or upload a .yml/.yaml workflow.</p>
          </div>
          
          <textarea
            data-testid="yaml-input"
            className="textarea-modern"
            value={yamlInput}
            onChange={(e) => setYamlInput(e.target.value)}
            style={{ height: '350px' }}
            spellCheck="false"
          />
          
          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <button className="glass-button-primary" data-testid="analyze-button" onClick={handleAnalyze} disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Workflow'}
            </button>
            
            <label className="glass-button-secondary" style={{ display: 'inline-block' }}>
              Upload YAML
              <input type="file" accept=".yml,.yaml" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
            
            <button className="glass-button-secondary" onClick={() => { setYamlInput(''); setResult(null); setError(null); }}>
              Clear
            </button>
          </div>
          
          {error && (
            <div data-testid="error-message" className="animate-fade-in" style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(251, 113, 133, 0.1)', borderLeft: '4px solid var(--color-error)', color: '#Fecdd3', borderRadius: '0 4px 4px 0' }}>
              <strong>Error: </strong> {error}
            </div>
          )}
        </section>

        {/* Results */}
        {result && (
          <section data-testid="prediction-result" className="glass-panel animate-fade-in" style={{ borderTop: `4px solid ${getRiskColor(result.prediction.predicted_label)}` }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}>Risk Assessment</h3>
            
            <div style={{ display: 'flex', gap: '3rem', flexWrap: 'wrap' }}>
              {/* Prediction */}
              <div style={{ flex: '1 1 300px' }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Predicted Risk Level</div>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: getRiskColor(result.prediction.predicted_label), marginBottom: '2rem' }}>
                  {result.prediction.predicted_label}
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {['LOW', 'MEDIUM', 'HIGH'].map(cls => {
                    const prob = result.prediction.probabilities[cls as keyof typeof result.prediction.probabilities];
                    const pct = (prob * 100).toFixed(1);
                    return (
                      <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{ width: '60px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{cls}</div>
                        <div style={{ flex: 1, background: 'rgba(0,0,0,0.3)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{ width: `${pct}%`, height: '100%', background: getRiskColor(cls), borderRadius: '4px' }} />
                        </div>
                        <div style={{ width: '50px', textAlign: 'right', fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>{pct}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>
              
              {/* Workflow Metadata */}
              <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Workflow Summary</div>
                
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <div style={{ flex: 1, background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--accent-primary)' }}>{result.workflow.job_count}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Jobs</div>
                  </div>
                  <div style={{ flex: 1, background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--accent-primary)' }}>{result.workflow.step_count}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Steps</div>
                  </div>
                </div>
                
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Inference Latency</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>{result.inference_duration_ms.toFixed(1)} ms</span>
                </div>
              </div>
            </div>
            
            {/* Why this risk? Explanation */}
            {result.explanation && result.explanation.signals.length > 0 && (
              <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border)' }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  Why this risk?
                </div>
                <h4 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--text-main)' }}>
                  {result.explanation.title}
                </h4>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                  {result.explanation.signals.map((signal, idx) => (
                    <div key={idx} style={{ 
                      background: 'rgba(0,0,0,0.2)', 
                      border: '1px solid var(--border)', 
                      borderRadius: '8px', 
                      padding: '1rem',
                      borderLeft: `3px solid ${getRiskColor(result.prediction.predicted_label)}`
                    }}>
                      <div style={{ fontWeight: '600', marginBottom: '0.25rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', backgroundColor: getRiskColor(result.prediction.predicted_label) }}></span>
                        {signal.name}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                        {signal.detail}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
        
        {/* Inference Explanation */}
        <section className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center' }}>
          <h4 style={{ fontSize: '1rem', marginBottom: '1.5rem', color: 'var(--text-main)' }}>How RunSure analyzes this workflow</h4>
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--border)' }}>YAML</span>
            <span>&rarr;</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--border)' }}>Text Representation</span>
              <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--border)' }}>Structural Representation</span>
            </div>
            <span>&rarr;</span>
            <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--accent-primary)', color: 'var(--accent-primary)' }}>Hybrid Model</span>
            <span>&rarr;</span>
            <span style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--border)', color: 'var(--text-main)' }}>Risk Prediction</span>
          </div>
        </section>

        {/* Model Info */}
        {modelInfo && (
          <section className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => setShowModelInfo(!showModelInfo)}>
              <h4 style={{ fontSize: '1rem', margin: 0 }}>Model Information</h4>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{showModelInfo ? 'Hide' : 'Show'}</span>
            </div>
            
            {showModelInfo && (
              <div className="animate-fade-in" style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', fontSize: '0.85rem' }}>
                <div>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Model ID</div>
                  <div style={{ fontFamily: 'var(--font-mono)' }}>{modelInfo.model_id}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Type</div>
                  <div style={{ fontFamily: 'var(--font-mono)' }}>{modelInfo.model_type}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Feature Count</div>
                  <div style={{ fontFamily: 'var(--font-mono)' }}>{modelInfo.feature_count.toLocaleString()} dimensions</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Status</div>
                  <div style={{ fontFamily: 'var(--font-mono)' }}>{modelInfo.frozen ? 'Frozen (Production)' : 'Training'}</div>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Footer */}
        <footer style={{ textAlign: 'center', padding: '1rem 0 2rem 0', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
          <div>RunSure — Static CI/CD Workflow Risk Classification</div>
          <div style={{ marginTop: '0.25rem', opacity: 0.6 }}>Research Prototype</div>
        </footer>
        
      </div>
    </>
  );
};

export default App;

