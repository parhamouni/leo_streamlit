/** @type {import('next').NextConfig} */
const nextConfig = {
  // In dev, proxy /api/* to the FastAPI backend on the same host.
  // Avoids needing to tunnel port 8513 separately when 3000 is the only
  // forwarded port. Override the destination via FENCE_API_INTERNAL_URL
  // (e.g. for production deployments where the backend lives elsewhere).
  async rewrites() {
    const dest = process.env.FENCE_API_INTERNAL_URL || "http://127.0.0.1:8513";
    return [
      {
        source: "/api/:path*",
        destination: `${dest}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
