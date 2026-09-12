import React, { useState } from 'react';

export default function MessageBubble({ message }) {
  const isUser = message.sender === 'user';
  const [showTrace, setShowTrace] = useState(false);
  const [activeSnippet, setActiveSnippet] = useState(null);

  const getVerdictClass = (verdict) => {
    switch ((verdict || '').toLowerCase()) {
      case 'correct':
      case 'good':
        return 'badge-success';
      case 'ambiguous':
      case 'regenerate':
        return 'badge-warning';
      case 'incorrect':
      case 'retrieve_more':
        return 'badge-danger';
      default:
        return 'badge-info';
    }
  };

  return (
    <div className="message-bubble-wrapper">
      <div className={`message-bubble ${isUser ? 'user' : 'bot'}`}>
        <div className="message-text">
          {message.text.split('\n').map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>

        {/* Self-Correction Badges and Trace */}
        {!isUser && (message.evaluation || message.reflection || (message.steps && message.steps.length > 0)) && (
          <div className="rag-meta-panel">
            <div className="rag-badges-row">
              {message.evaluation && (
                <div className={`rag-badge ${getVerdictClass(message.evaluation.verdict)}`} title={message.evaluation.reasoning}>
                  <span className="badge-dot" />
                  <span>Retrieval: <strong>{message.evaluation.verdict}</strong> ({message.evaluation.score}/100)</span>
                </div>
              )}

              {message.reflection && (
                <div className={`rag-badge ${getVerdictClass(message.reflection.verdict)}`} title={message.reflection.reasoning}>
                  <span className="badge-dot" />
                  <span>Reflection: <strong>{message.reflection.verdict}</strong> {message.reflection.grounded ? '• Grounded' : '• Ungrounded'}</span>
                </div>
              )}

              {message.steps && message.steps.length > 0 && (
                <button 
                  type="button" 
                  className="rag-trace-toggle-btn"
                  onClick={() => setShowTrace(!showTrace)}
                  title="Toggle Self-Correction Pipeline Steps"
                >
                  <svg 
                    viewBox="0 0 24 24" 
                    width="14" 
                    height="14" 
                    stroke="currentColor" 
                    strokeWidth="2.5" 
                    fill="none" 
                    style={{ transform: showTrace ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                  {showTrace ? 'Hide Trace' : 'View Self-Correction Steps'}
                </button>
              )}
            </div>

            {/* Expandable Execution Steps Trace */}
            {showTrace && message.steps && message.steps.length > 0 && (
              <div className="rag-steps-trace">
                <div className="trace-title">Self-Corrective Pipeline Trace</div>
                <ul className="trace-list">
                  {message.steps.map((step, idx) => (
                    <li key={idx} className="trace-item">
                      <span className="trace-index">{idx + 1}</span>
                      <span className="trace-text">{step.replace(/^\d+\.\s*/, '')}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <div className="sources-title">Retrieved & Refined Sources</div>
            <div className="sources-list">
              {message.sources.map((source, idx) => {
                const sourceName = typeof source === 'object' ? source.name : source;
                const isWeb = typeof source === 'object' && source.type === 'web';
                const pageNum = typeof source === 'object' && source.page != null ? ` (p. ${source.page + 1})` : '';

                return (
                  <div 
                    key={idx} 
                    className="source-tag-container"
                    onMouseEnter={() => typeof source === 'object' && source.snippet && setActiveSnippet(idx)}
                    onMouseLeave={() => setActiveSnippet(null)}
                  >
                    <span className="source-tag">
                      {isWeb ? (
                        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '2px' }}>
                          <circle cx="12" cy="12" r="10" />
                          <line x1="2" y1="12" x2="22" y2="12" />
                          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                        </svg>
                      ) : (
                        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '2px' }}>
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                          <line x1="16" y1="13" x2="8" y2="13" />
                          <line x1="16" y1="17" x2="8" y2="17" />
                          <polyline points="10 9 9 9 8 9" />
                        </svg>
                      )}
                      {sourceName}{pageNum}
                    </span>

                    {/* Hover snippet preview */}
                    {activeSnippet === idx && typeof source === 'object' && source.snippet && (
                      <div className="source-snippet-popover">
                        <div className="snippet-source-name">{sourceName}</div>
                        <div className="snippet-text">{source.snippet}</div>
                        {source.url && (
                          <a href={source.url} target="_blank" rel="noreferrer" className="snippet-link">
                            Open web link &rarr;
                          </a>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
