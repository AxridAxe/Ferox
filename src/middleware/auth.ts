import type { NextFunction, Request, Response } from "express";
import { timingSafeEqual } from "node:crypto";
import { config } from "../config.js";

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

export function requireApiKey(req: Request, res: Response, next: NextFunction): void {
  const provided = req.header("x-api-key");
  if (!provided || !config.apiKeys.some((key) => safeEqual(key, provided))) {
    res.status(401).json({ error: "Invalid or missing API key" });
    return;
  }
  next();
}
