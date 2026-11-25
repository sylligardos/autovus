import type { Metadata } from "next";
import { DM_Mono, Public_Sans } from "next/font/google";
import "./globals.css";
import { THEME_BOOT } from "@/lib/theme";

const sans = Public_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-sans" });
const mono = DM_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Protagonist",
  description: "A buffer for every anomaly, chosen by the detector's own lag.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} /></head>
      <body>{children}</body>
    </html>
  );
}
