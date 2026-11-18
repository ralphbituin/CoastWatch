/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // "standalone" produces a minimal server bundle for the Docker image.
  output: "standalone",
};

export default nextConfig;
