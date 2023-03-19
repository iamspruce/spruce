import { NextRequest, NextResponse } from "next/server";
import { OpenAIFetch } from "lib/openai";

export async function POST(request: NextRequest, response: NextResponse) {
  const body = request.body;
  const {
    topic,
    chapter,
  }: {
    topic: string;
    chapter: string | null;
  } = await new Response(body).json();

  const payload = {
    model: "gpt-3.5-turbo",
    messages: [
      { role: "system", content: "You are a helpful project writer." },
      {
        role: "user",
        content: `write a brief project for me based on this topic: ${topic}`,
      },
      {
        role: "user",
        content: `write only this chapter: ${chapter}`,
      },
    ],
    temperature: 0.7,
    max_tokens: 2048,
    top_p: 1.0,
    frequency_penalty: 0.0,
    stream: true,
    presence_penalty: 0.0,
    n: 1,
  };

  const stream = await OpenAIFetch(payload);
  return new Response(stream);
}
