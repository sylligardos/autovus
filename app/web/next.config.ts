import type { NextConfig } from "next";

// The browser reaches the Python side through /api/py/*; the server reaches
// it directly (lib/api.ts). AUTOVUS_API overrides the default port.
const API = process.env.AUTOVUS_API ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/py/:path*", destination: `${API}/:path*` }];
  },
};

export default nextConfig;
