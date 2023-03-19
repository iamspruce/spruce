interface OpenAIResponsePayload {
  messages: { role: string; content: string }[];
  model: string;
  temperature: number;
}

export async function OpenAIFetch(payload: OpenAIResponsePayload) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_APIKEY}`,
    },
    method: "POST",
    body: JSON.stringify(payload),
    cache: "no-cache",
  });
  const json = await res.json();
  // TODO: log error
  return json.choices[0].message.content;
}
