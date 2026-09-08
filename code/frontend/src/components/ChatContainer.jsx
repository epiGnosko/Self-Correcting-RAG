import React, { useRef, useEffect } from 'react';
import ThemeToggle from './ThemeToggle';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

export default function ChatContainer({ 
  messages, 
  isChatStarted, 
  isLoading, 
  theme, 
  onToggleTheme, 
  onSendMessage,
  onAddDocuments 
}) {
  const messagesEndRef = useRef(null);
  const mobileFileInputRef = useRef(null);

  // Scroll to bottom when messages list grows or loading toggles
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleMobileFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      onAddDocuments(files);
    }
  };

  const handleMobileUploadClick = () => {
    if (mobileFileInputRef.current) {
      mobileFileInputRef.current.click();
    }
  };

  return (
    <div className="chat-container">
      {/* Hidden input for mobile uploading, triggered via ref */}
      <input
        ref={mobileFileInputRef}
        type="file"
        className="file-input-hidden"
        multiple
        onChange={handleMobileFileChange}
      />

      <header className="chat-header">
        <div className="chat-header-logo">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
          <h1>Self-Corrective RAG</h1>
        </div>
        <div className="chat-header-actions">
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </header>

      <div className="chat-viewport">
        {/* Welcome Section */}
        <div className={`welcome-container ${isChatStarted ? 'hidden' : ''}`}>
          <h2>Self-Correcting RAG</h2>
          <p>
            Upload reference documents on the right and ask questions. The system will retrieve context, evaluate relevance, and self-correct to give you the most accurate answer.
          </p>
        </div>

        {/* Messages List */}
        <div className={`chat-messages ${isChatStarted ? 'active' : ''}`}>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          
          {isLoading && (
            <div className="chat-loading" key="loading-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className={`chat-input-wrapper ${isChatStarted ? 'docked' : 'centered'}`}>
          <ChatInput 
            onSendMessage={onSendMessage} 
            theme={theme}
            onToggleTheme={onToggleTheme}
            onUploadClick={handleMobileUploadClick}
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
