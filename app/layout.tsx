import type React from "react"
import type { Metadata } from "next"
import { Outfit } from "next/font/google"
import "./globals.css"

const outfit = Outfit({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-outfit",
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
})

export const metadata: Metadata = {
  title: "SkinSavers - Advanced AI Skin Cancer Detection with Staging",
  description:
    "Professional-grade skin cancer detection with AI staging analysis, treatment optimization, and spread prediction. The most advanced consumer skin cancer app available.",
  generator: "SkinSavers",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={outfit.variable}>
      <head>
        <link rel="icon" href="/favicon.png" type="image/png" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js" async />
        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js" async />
      </head>
      <body className="font-outfit antialiased bg-slate-950 text-slate-100">{children}</body>
    </html>
  )
}
