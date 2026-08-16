import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/navigation";

export const metadata: Metadata = {
  title: "Luma",
  description: "Evidence-based AI for your documents",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-surface">
        <Navigation />
        <main className="mx-auto max-w-4xl px-6 pb-24 pt-8">{children}</main>
      </body>
    </html>
  );
}
