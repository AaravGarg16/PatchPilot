import type { NextConfig } from "next";

// A fully static site: the replays are bundled JSON, so there is no server, API route or runtime cost.
const config: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default config;
