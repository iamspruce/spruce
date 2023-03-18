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
    temperature: 0.2,
    user: "coursematrs",
  };

  try {
    const completetion = await OpenAIFetch(payload);
    let json = completetion;
    return new Response(json);
  } catch (error: any) {
    console.log(error, error.message);
    return new NextResponse(null, {
      status: 400,
    });
  }
}
