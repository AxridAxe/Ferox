import "dotenv/config";

function requireEnvList(name: string): string[] {
  const raw = process.env[name];
  if (!raw || raw.trim() === "") {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return raw.split(",").map((v) => v.trim()).filter(Boolean);
}

export const config = {
  port: Number(process.env.PORT ?? 8787),
  openRouterApiKey: process.env.OPENROUTER_API_KEY ?? "",
  apiKeys: requireEnvList("FEROX_API_KEYS"),
  allowedOrigins: (process.env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean),
};
