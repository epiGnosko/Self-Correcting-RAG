import React, { useMemo } from 'react';

export default function DocumentCard({ doc, onDelete }) {
  const thumbnailStyle = useMemo(() => {
    if (doc.previewUrl) {
      return { backgroundImage: `url(${doc.previewUrl})` };
    }
    
    // Abstract frosted gradient options based on file type or index
    const gradients = [
      'linear-gradient(135deg, rgba(255, 46, 81, 0.25) 0%, rgba(76, 29, 149, 0.3) 100%)', // crimson-purple
      'linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(99, 102, 241, 0.3) 100%)', // sky-indigo
      'linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 95, 70, 0.3) 100%)', // emerald
      'linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(220, 38, 38, 0.3) 100%)' // amber-red
    ];
    
    const index = (doc.name || '').length % gradients.length;
    return { background: gradients[index] };
  }, [doc.previewUrl, doc.name]);

  const formatSize = (bytes) => {
    if (!bytes) return '0 KB';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div className="document-card">
      <div className="document-thumbnail-bg" style={thumbnailStyle} />
      <div className="document-card-overlay" />
      <div className="document-card-content">
        <div className="document-info">
          <div className="document-title" title={doc.name}>
            {doc.name}
          </div>
          <div className="document-meta">
            {formatSize(doc.size)}
          </div>
        </div>
        
        <button 
          className="document-delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(doc.id);
          }}
          title="Remove document"
          aria-label="Remove document"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            <line x1="10" y1="11" x2="10" y2="17" />
            <line x1="14" y1="11" x2="14" y2="17" />
          </svg>
        </button>
      </div>
    </div>
  );
}
