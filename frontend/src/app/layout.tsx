import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LexPilot - AI Legal Intelligence",
  description: "AI-powered legal research and analysis for Indian law",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
