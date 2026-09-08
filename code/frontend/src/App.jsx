import React, { useState, useEffect } from 'react';
import ChatContainer from './components/ChatContainer';
import DocumentPanel from './components/DocumentPanel';
import './index.css';
import './styles/Chat.css';
import './styles/DocumentPanel.css';

export default function App() {
  const [theme, setTheme] = useState('dark');
  const [isChatStarted, setIsChatStarted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const handleToggleTheme = (event) => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';

    if (!document.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setTheme(nextTheme);
      return;
    }

    const x = event?.clientX ?? window.innerWidth / 2;
    const y = event?.clientY ?? window.innerHeight / 2;
    const endRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    const transition = document.startViewTransition(() => {
      setTheme(nextTheme);
    });

    transition.ready.then(() => {
      document.documentElement.animate(
        {
          clipPath: [
            `circle(0px at ${x}px ${y}px)`,
            `circle(${endRadius}px at ${x}px ${y}px)`
          ]
        },
        {
          duration: 500,
          easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
          pseudoElement: '::view-transition-new(root)'
        }
      );
    });
  };

  const handleAddDocuments = (files) => {
    const newDocs = files.map((file) => {
      const previewUrl = file.type.startsWith('image/')
        ? URL.createObjectURL(file)
        : null;
      return {
        id: `${Date.now()}-${Math.random()}`,
        name: file.name,
        size: file.size,
        type: file.type,
        previewUrl,
        file
      };
    });
    setDocuments((prev) => [...prev, ...newDocs]);
  };

  const handleDeleteDocument = (id) => {
    setDocuments((prev) => {
      const docToDelete = prev.find((d) => d.id === id);
      if (docToDelete && docToDelete.previewUrl) {
        URL.revokeObjectURL(docToDelete.previewUrl);
      }
      return prev.filter((d) => d.id !== id);
    });
  };

  const handleSendMessage = (text) => {
    if (isLoading) return;

    if (!isChatStarted) {
      setIsChatStarted(true);
    }

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
  };

  return (
    <div className="app-container">

      <ChatContainer
        messages={messages}
        isChatStarted={isChatStarted}
        isLoading={isLoading}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        onSendMessage={handleSendMessage}
        onAddDocuments={handleAddDocuments}
      />

      <DocumentPanel
        documents={documents}
        onAddDocuments={handleAddDocuments}
        onDeleteDocument={handleDeleteDocument}
      />
    </div>
  );
}
