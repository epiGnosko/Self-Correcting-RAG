import React, { useState, useRef } from 'react';
import DocumentCard from './DocumentCard';

export default function DocumentPanel({ documents, onAddDocuments, onDeleteDocument }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      onAddDocuments(files);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      onAddDocuments(files);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <aside className="document-panel">
      <div className="panel-header">
        <h3>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
          Knowledge Base
        </h3>
        <span className="panel-header-count">{documents.length}</span>
      </div>

      <div className="dropzone-container">
        <input
          ref={fileInputRef}
          type="file"
          className="file-input-hidden"
          multiple
          onChange={handleFileChange}
        />
        
        <div 
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={onButtonClick}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div className="dropzone-text">Drag & Drop files here</div>
          <div className="dropzone-subtext">or click to browse from device</div>
        </div>

        <div className="document-list-wrapper">
          {documents.length === 0 ? (
            <div className="empty-docs">
              No documents uploaded yet. Upload reference files to enable contextual RAG queries.
            </div>
          ) : (
            <div className="document-list">
              {documents.map((doc) => (
                <DocumentCard 
                  key={doc.id} 
                  doc={doc} 
                  onDelete={onDeleteDocument} 
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
