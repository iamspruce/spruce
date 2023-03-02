/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "www.freecodecamp.org",
        port: "",
        pathname: "/news/content/images/**",
      },
    ],
  },
};

module.exports = nextConfig;
