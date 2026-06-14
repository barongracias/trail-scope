"use client";

import React from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-red-50 text-red-800 flex items-center justify-center">
        <div className="max-w-md p-6 rounded-xl bg-white shadow border border-red-200 space-y-4">
          <h1 className="text-lg font-semibold">Something went wrong</h1>
          <p className="text-sm leading-relaxed">{error.message || "An unexpected error occurred."}</p>
          {error.digest && <p className="text-xs text-red-500">Ref: {error.digest}</p>}
          <button
            onClick={reset}
            className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-semibold hover:bg-red-700 transition"
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
