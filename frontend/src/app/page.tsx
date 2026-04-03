"use client";

import { useState, useRef, useCallback } from "react";
import { ChatInput } from "@/components/ChatInput";
import { ChatMessage } from "@/components/ChatMessage";
import { ConfidenceBanner } from "@/components/ConfidenceBanner";
import { Disclaimer } from "@/components/Disclaimer";
import { ResearchSessionPanel } from "@/components/ResearchSessionPanel";
import { SuggestedQueries } from "@/components/SuggestedQueries";
import { streamChat } from "@/lib/ai-types";
import type { ChatMessage as ChatMessageType, ConfidenceData } from "@/types";
import { Scale, PanelLeftOpen, PanelLeftClose } from "lucide-react";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<ConfidenceData | null>(null);
  const [researchSessionId, setResearchSessionId] = useState<string | undefined>();
  const [showSidebar, setShowSidebar] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handleSend = useCallback(
    async (message: string) => {
      const userMsg: ChatMessageType = {
        id: crypto.randomUUID(),
        role: "user",
        content: message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setActiveAgent(null);
      setConfidence(null);

      const assistantId = crypto.randomUUID();
      let fullContent = "";

      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: new Date(),
          isLoading: true,
          loadingAgent: undefined,
        },
      ]);

      try {
        for await (const event of streamChat(message, undefined, researchSessionId)) {
          switch (event.type) {
            case "text":
              fullContent += event.content || "";
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: fullContent, isLoading: false } : m
                )
              );
              scrollToBottom();
              break;
            case "agent_event":
              setActiveAgent(event.agent || null);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId && m.isLoading
                    ? { ...m, loadingAgent: event.agent || undefined }
                    : m
                )
              );
              break;
            case "final":
              fullContent = event.content || fullContent;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: fullContent } : m
                )
              );
              break;
          }
        }
      } catch (error) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: "An error occurred. Please try again." }
              : m
          )
        );
      } finally {
        setIsStreaming(false);
        setActiveAgent(null);
        scrollToBottom();
      }
    },
    [researchSessionId, scrollToBottom]
  );

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      {showSidebar && (
        <div className="w-80 border-r border-gray-200 bg-white flex-shrink-0">
          <ResearchSessionPanel
            activeSessionId={researchSessionId}
            onSessionSelect={setResearchSessionId}
          />
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="border-b border-gray-200 bg-white px-4 py-3 flex items-center gap-3">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title="Toggle research sessions"
          >
            {showSidebar ? (
              <PanelLeftClose className="w-5 h-5 text-gray-600" />
            ) : (
              <PanelLeftOpen className="w-5 h-5 text-gray-600" />
            )}
          </button>
          <Scale className="w-6 h-6 text-primary-600" />
          <h1 className="text-lg font-semibold text-gray-900">LexPilot</h1>
          <span className="text-sm text-gray-500">AI Legal Intelligence for Indian Law</span>
        </header>

        {/* Disclaimer */}
        <Disclaimer />

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="max-w-2xl mx-auto mt-12">
              <div className="text-center mb-8">
                <Scale className="w-12 h-12 text-primary-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  Welcome to LexPilot
                </h2>
                <p className="text-gray-600">
                  Ask me about Indian law, contracts, DPDP compliance, or legal
                  precedents.
                </p>
              </div>
              <SuggestedQueries onSelect={handleSend} />
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              {confidence && <ConfidenceBanner confidence={confidence} />}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>


        {/* Input */}
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="max-w-3xl mx-auto">
            <ChatInput onSend={handleSend} isLoading={isStreaming} />
          </div>
        </div>
      </div>
    </div>
  );
}
