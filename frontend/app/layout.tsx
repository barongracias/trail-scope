import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "trail-scope",
  description:
    "A qualitative, single-image inference demo of the locked thesis satellite-trail detector. Not a benchmark.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
