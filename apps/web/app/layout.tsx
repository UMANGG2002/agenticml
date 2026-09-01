import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic ML Engineer",
  description: "An autonomous ML engineering agent.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
