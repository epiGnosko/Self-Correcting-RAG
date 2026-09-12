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

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.warn('Backend server not reachable yet for documents:', err);
    }
  };

  const handleAddDocuments = async (files) => {
    if (!files || files.length === 0) return;

    // Build form data to upload to backend
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchDocuments();
      } else {
        console.error('Upload failed with status', res.status);
      }
    } catch (err) {
      console.error('Failed to upload documents:', err);
      // Fallback local preview
      const localDocs = files.map((file) => ({
        id: `${Date.now()}-${Math.random()}`,
        name: file.name,
        size: file.size,
        type: file.type,
        chunks: 'local'
      }));
      setDocuments((prev) => [...prev, ...localDocs]);
    }
  };

  const handleDeleteDocument = async (id) => {
    try {
      await fetch(`/api/documents/${id}`, { method: 'DELETE' });
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      console.error('Failed to delete document from backend:', err);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    }
  };

  const handleSendMessage = async (text) => {
    if (isLoading || !text.trim()) return;

    if (!isChatStarted) {
      setIsChatStarted(true);
    }

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: text.trim()
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: text.trim() }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error ${res.status}`);
      }

      const data = await res.json();
      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: data.answer,
        sources: data.sources || [],
        evaluation: data.evaluation,
        reflection: data.reflection,
        steps: data.steps || []
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error('Failed to execute query:', err);
      const errorMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: `Error processing query: ${err.message || 'Please verify that backend server is running.'}`,
        sources: []
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
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
