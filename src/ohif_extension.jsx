/**
 * NeuroScope AI - OHIF Extension
 * AI Analysis Panel + Grad-CAM Overlay + Treatment Sidebar
 *
 * Usage: drop into OHIF v3 extensions folder
 * Calls NeuroScope API at http://localhost:8000
 */

import React, { useState, useCallback, useRef } from 'react';

const API_BASE = process.env.REACT_APP_NEUROSCOPE_API || 'http://localhost:8000';

const CANCER_TYPES = ['brain', 'lung', 'breast', 'liver', 'skin', 'spine'];

const VERDICT_COLORS = {
  CANCER_FLAGGED  : '#D4537E',
  NORMAL          : '#1D9E75',
  REVIEW_REQUIRED : '#EF9F27',
};

const PRIORITY_LABELS = {
  1: '🚨 CRITICAL',
  2: '⚠️  URGENT',
  3: '📋 ROUTINE',
  4: '✅ NORMAL',
};


// ── Utility: convert canvas to base64 ─────────────────────────────────────────
function getViewportImageBase64(viewportRef) {
  try {
    const canvas = viewportRef?.current?.querySelector('canvas');
    if (!canvas) return null;
    return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
  } catch {
    return null;
  }
}


// ── Agent Progress Bar ────────────────────────────────────────────────────────
function AgentProgress({ agents }) {
  const AGENT_LABELS = {
    normalization : 'Scanner Normalization',
    triage        : 'Triage',
    detection     : 'Detection Ensemble',
    segmentation  : 'Segmentation',
    classification: 'Classification',
    clinical      : 'Clinical Intelligence',
    report        : 'Report Generation',
    monitoring    : 'QA & Ethics',
    complete      : 'Complete',
  };

  return (
    <div style={{ margin: '8px 0' }}>
      {Object.entries(AGENT_LABELS).map(([key, label]) => {
        const status = agents[key];
        const color  = status === 'done'    ? '#1D9E75'
                     : status === 'running' ? '#EF9F27'
                     : '#444';
        const icon   = status === 'done'    ? '✓'
                     : status === 'running' ? '⟳'
                     : '○';
        return (
          <div key={key} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '2px 0', fontSize: 11, color,
          }}>
            <span style={{ fontWeight: 'bold', minWidth: 14 }}>{icon}</span>
            <span>{label}</span>
          </div>
        );
      })}
    </div>
  );
}


// ── Result Card ───────────────────────────────────────────────────────────────
function ResultCard({ result }) {
  const verdictColor = VERDICT_COLORS[result.verdict] || '#888';
  const [showReport, setShowReport] = useState(false);

  return (
    <div style={{ marginTop: 8 }}>
      {/* Verdict banner */}
      <div style={{
        background: verdictColor, borderRadius: 6,
        padding: '8px 12px', marginBottom: 8,
      }}>
        <div style={{ color: 'white', fontWeight: 'bold', fontSize: 13 }}>
          {result.verdict.replace('_', ' ')}
        </div>
        <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11 }}>
          {PRIORITY_LABELS[result.priority]} &nbsp;|&nbsp;
          P(cancer): {(result.cancer_prob * 100).toFixed(1)}%
        </div>
      </div>

      {/* Classification */}
      <div style={{ background: '#1a1a1a', borderRadius: 4, padding: 8, marginBottom: 6, fontSize: 11 }}>
        <div><b style={{ color: '#7F77DD' }}>Tumor type:</b> {result.tumor_type || '—'}</div>
        <div><b style={{ color: '#7F77DD' }}>Confidence:</b> {((result.confidence || 0) * 100).toFixed(1)}%</div>
        {result.who_grade && <div><b style={{ color: '#7F77DD' }}>WHO Grade:</b> {result.who_grade}</div>}
        <div style={{ color: '#666', fontSize: 10 }}>Latency: {result.latency_ms}ms</div>
      </div>

      {/* Treatment recs */}
      {result.treatment_recs?.length > 0 && (
        <div style={{ background: '#1a1a1a', borderRadius: 4, padding: 8, marginBottom: 6 }}>
          <div style={{ color: '#EF9F27', fontWeight: 'bold', fontSize: 11, marginBottom: 4 }}>
            Treatment Recommendations
          </div>
          {result.treatment_recs.slice(0, 4).map((rec, i) => (
            <div key={i} style={{ fontSize: 10, color: '#ccc', padding: '2px 0' }}>
              {i + 1}. {rec}
            </div>
          ))}
        </div>
      )}

      {/* Plain summary */}
      <div style={{ background: '#1a2a1a', borderRadius: 4, padding: 8, marginBottom: 6 }}>
        <div style={{ color: '#1D9E75', fontWeight: 'bold', fontSize: 10, marginBottom: 4 }}>
          Patient Summary
        </div>
        <div style={{ fontSize: 10, color: '#aaa', lineHeight: 1.4 }}>
          {result.plain_summary}
        </div>
      </div>

      {/* Report toggle */}
      <button
        onClick={() => setShowReport(v => !v)}
        style={{
          background: '#333', border: 'none', color: '#aaa',
          padding: '4px 8px', borderRadius: 3, fontSize: 10,
          cursor: 'pointer', width: '100%', textAlign: 'left',
        }}
      >
        {showReport ? '▲ Hide' : '▼ Show'} full report
      </button>
      {showReport && (
        <pre style={{
          background: '#111', color: '#aaa', fontSize: 9,
          padding: 8, borderRadius: 4, marginTop: 4,
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          maxHeight: 200, overflow: 'auto',
        }}>
          {result.report_text}
        </pre>
      )}
    </div>
  );
}


// ── Main Panel ────────────────────────────────────────────────────────────────
export function NeuroscopePanel({ viewportRef }) {
  const [cancerType, setCancerType]   = useState('brain');
  const [patientAge, setPatientAge]   = useState('');
  const [patientSex, setPatientSex]   = useState('');
  const [notes, setNotes]             = useState('');
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [agents, setAgents]           = useState({});
  const wsRef                         = useRef(null);

  const analyze = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    setAgents({});

    const imageB64 = getViewportImageBase64(viewportRef);
    if (!imageB64) {
      setError('No scan loaded in viewport. Open a DICOM study first.');
      setLoading(false);
      return;
    }

    const scanId = Math.random().toString(36).slice(2, 10);

    try {
      // Connect WebSocket for streaming
      const ws = new WebSocket(`${API_BASE.replace('http', 'ws')}/ws/analyze/${scanId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({
          cancer_type : cancerType,
          image_b64   : imageB64,
          patient_age : patientAge ? parseInt(patientAge) : null,
          patient_sex : patientSex || null,
          notes       : notes,
        }));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.error) {
          setError(msg.error);
          setLoading(false);
          return;
        }
        if (msg.stage === 'result') {
          setResult(msg.data);
          setLoading(false);
        } else if (msg.stage) {
          setAgents(prev => ({ ...prev, [msg.stage]: msg.status }));
        }
      };

      ws.onerror = () => {
        // WebSocket fallback: use sync HTTP endpoint
        ws.close();
        analyzeFallback(imageB64, scanId);
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

    } catch (e) {
      analyzeFallback(imageB64, scanId);
    }
  }, [cancerType, patientAge, patientSex, notes, viewportRef]);


  const analyzeFallback = async (imageB64, scanId) => {
    // HTTP fallback when WebSocket unavailable
    try {
      const blob     = await fetch(`data:image/jpeg;base64,${imageB64}`).then(r => r.blob());
      const formData = new FormData();
      formData.append('file', blob, 'scan.jpg');
      formData.append('cancer_type',    cancerType);
      formData.append('clinical_notes', notes);
      if (patientAge) formData.append('patient_age', patientAge);
      if (patientSex) formData.append('patient_sex', patientSex);

      const resp = await fetch(`${API_BASE}/analyze/sync`, {
        method: 'POST', body: formData,
      });
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stop = useCallback(() => {
    wsRef.current?.close();
    setLoading(false);
  }, []);

  return (
    <div style={{
      padding: 12, color: '#ddd', fontSize: 12,
      height: '100%', overflowY: 'auto',
      background: '#0d0d0d',
    }}>
      {/* Header */}
      <div style={{
        fontWeight: 'bold', fontSize: 14, color: '#7F77DD',
        marginBottom: 12, borderBottom: '1px solid #333', paddingBottom: 6,
      }}>
        🧠 NeuroScope AI
      </div>

      {/* Cancer type selector */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ color: '#888', fontSize: 10, display: 'block', marginBottom: 3 }}>
          Cancer Pipeline
        </label>
        <select
          value={cancerType}
          onChange={e => setCancerType(e.target.value)}
          style={{
            width: '100%', background: '#222', color: '#ddd',
            border: '1px solid #444', borderRadius: 4, padding: '4px 6px',
            fontSize: 11,
          }}
        >
          {CANCER_TYPES.map(ct => (
            <option key={ct} value={ct}>{ct.charAt(0).toUpperCase() + ct.slice(1)}</option>
          ))}
        </select>
      </div>

      {/* Patient info */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
        <div style={{ flex: 1 }}>
          <label style={{ color: '#888', fontSize: 10, display: 'block', marginBottom: 3 }}>Age</label>
          <input
            type="number" value={patientAge}
            onChange={e => setPatientAge(e.target.value)}
            placeholder="—"
            style={{
              width: '100%', background: '#222', color: '#ddd',
              border: '1px solid #444', borderRadius: 4,
              padding: '4px 6px', fontSize: 11, boxSizing: 'border-box',
            }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ color: '#888', fontSize: 10, display: 'block', marginBottom: 3 }}>Sex</label>
          <select
            value={patientSex}
            onChange={e => setPatientSex(e.target.value)}
            style={{
              width: '100%', background: '#222', color: '#ddd',
              border: '1px solid #444', borderRadius: 4,
              padding: '4px 6px', fontSize: 11,
            }}
          >
            <option value="">—</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>
      </div>

      {/* Clinical notes */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ color: '#888', fontSize: 10, display: 'block', marginBottom: 3 }}>
          Clinical Notes
        </label>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Symptoms, prior history..."
          rows={2}
          style={{
            width: '100%', background: '#222', color: '#ddd',
            border: '1px solid #444', borderRadius: 4,
            padding: '4px 6px', fontSize: 10, resize: 'vertical',
            boxSizing: 'border-box',
          }}
        />
      </div>

      {/* Analyze button */}
      <button
        onClick={loading ? stop : analyze}
        style={{
          width: '100%', padding: '8px 0',
          background: loading ? '#553322' : '#7F77DD',
          color: 'white', border: 'none', borderRadius: 5,
          fontSize: 12, fontWeight: 'bold', cursor: 'pointer',
          marginBottom: 8,
        }}
      >
        {loading ? '⏹ Stop' : '▶ Analyze Scan'}
      </button>

      {/* Agent progress */}
      {loading && Object.keys(agents).length > 0 && (
        <AgentProgress agents={agents} />
      )}

      {/* Error */}
      {error && (
        <div style={{
          background: '#3a1a1a', border: '1px solid #D4537E',
          borderRadius: 4, padding: 8, fontSize: 10, color: '#D4537E',
          marginTop: 8,
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Result */}
      {result && <ResultCard result={result} />}

      {/* Disclaimer */}
      <div style={{
        marginTop: 12, fontSize: 9, color: '#444',
        borderTop: '1px solid #222', paddingTop: 6,
      }}>
        AI-ASSISTED. CLINICIAN REVIEW REQUIRED. Research platform only.
      </div>
    </div>
  );
}


// ── OHIF Extension Registration ───────────────────────────────────────────────
export default {
  id: 'neuroscope-ai',

  preRegistration({ servicesManager, commandsManager }) {
    console.log('NeuroScope AI extension loaded');
  },

  getPanelModule({ servicesManager, commandsManager }) {
    return [
      {
        name   : 'neuroscope-panel',
        iconName: 'tool-ai',
        iconLabel: 'AI',
        label  : 'NeuroScope AI',
        component: NeuroscopePanel,
      },
    ];
  },
};
