"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Scale, Loader2 } from "lucide-react";
import { CitationBadge } from "./CitationBadge";
import type { ChatMessage as ChatMessageType } from "@/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

const CITATION_REGEX =
  /(\(\d{4}\)\s+\d+\s+SCC\s+\d+|AIR\s+\d{4}\s+SC\s+\d+|\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+)/g;

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  const mdProps = { remarkPlugins: [remarkGfm] };

  const renderContent = (content: string) => {
    const citations = message.citations;
    if (citations && citations.length > 0) {
      const parts = content.split(CITATION_REGEX);
      return (
        <div className="prose prose-sm max-w-none prose-headings:font-semibold prose-table:text-sm prose-th:bg-gray-100 prose-td:border prose-th:border">
          {parts.map((part, i) => {
            const citation = citations.find((c) => c.text === part);
            if (citation) {
              return <CitationBadge key={i} citation={citation} />;
            }
            return <ReactMarkdown key={i} {...mdProps}>{part}</ReactMarkdown>;
          })}
        </div>
      );
    }

    return (
      <div className="prose prose-sm max-w-none prose-headings:font-semibold prose-table:text-sm prose-th:bg-gray-100">
        <ReactMarkdown {...mdProps}>{content}</ReactMarkdown>
      </div>
    );
  };

  if (!isUser && message.isLoading) {
    return (
      <div className="flex gap-3 justify-start">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
          <Scale className="w-4 h-4 text-primary-600" />
        </div>
        <div className="flex items-center gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-3 text-sm text-gray-500">
          <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
          <span>
            {message.loadingAgent ? (
              <><span className="font-medium text-gray-700">{message.loadingAgent}</span> is working...</>
            ) : (
              "Working..."
            )}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
          <Scale className="w-4 h-4 text-primary-600" />
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-primary-600 text-white"
            : "bg-white border border-gray-200 text-gray-900"
        }`}
      >
        {message.agentName && (
          <div className="text-xs font-medium text-primary-500 mb-1">
            {message.agentName}
          </div>
        )}
        {isUser ? (
          <p className="text-sm">{message.content}</p>
        ) : (
          renderContent(message.content)
        )}
      </div>
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
          <User className="w-4 h-4 text-gray-600" />
        </div>
      )}
    </div>
  );
}
