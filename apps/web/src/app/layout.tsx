import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic ML",
  description: "AI-powered machine learning agent",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
