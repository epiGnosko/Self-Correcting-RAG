import React, { useState, useRef, useEffect } from 'react';

export default function ChatInput({ onSendMessage, theme, onToggleTheme, onUploadClick, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onSendMessage(text.trim());
      setText('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '46px';
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '46px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.max(46, Math.min(scrollHeight - 8, 120))}px`;
    }
  }, [text]);

  return (
    <form className="chat-input-box glass-panel" onSubmit={handleSubmit}>
      <div className="mobile-input-actions">
        <button 
          type="button" 
          className="mobile-input-btn"
          onClick={onUploadClick}
          title="Upload Document"
          aria-label="Upload Document"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </button>
      </div>

      <textarea
        ref={textareaRef}
        className="chat-input-field"
        placeholder="Ask a question about your documents..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />

      <button 
        type="submit" 
        className="send-btn" 
        disabled={!text.trim() || disabled}
        aria-label="Send query"
      >
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="19" x2="12" y2="5" />
          <polyline points="5 12 12 5 19 12" />
        </svg>
      </button>
    </form>
  );
}
