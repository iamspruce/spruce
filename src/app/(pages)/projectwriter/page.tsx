"use client";
import React from "react";
import toast, { Toaster } from "react-hot-toast";
import Image from "next/image";
import replygptIcon from "/public/img/replygpt/android-chrome-192x192.png";

function Page() {
  const [text, SetText] = React.useState("Your project will apppear here");
  let endStream = false;

  const handleForm = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    SetText("");
    const topic = document.getElementById("topic") as HTMLInputElement | null;
    const chapter = document.getElementById(
      "chapter"
    ) as HTMLInputElement | null;

    const payload = {
      topic: topic?.value,
      chapter: chapter?.value,
    };
    toast.loading("Please hang on... I am writing your project");

    const response = await fetch("api", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: {
        "content-type": "application/json",
      },
    });

    if (response.ok) {
      toast.dismiss();
      toast.success("Your project is ready");

      try {
        const data = response.body;
        if (!data) {
          return;
        }
        const reader = data.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { value, done } = await reader.read();
          const chunkValue = decoder.decode(value);
          console.log(value,chunkValue);
          SetText((prev) => prev + chunkValue);
          if (done) {
            endStream = true;
            break;
          }
        }
      } catch (err) {
        toast.dismiss();

        return toast.error("Looks like OpenAI timed out :(");
      }
    } else {
      let error = await response.text();
      console.log(error);
      toast.dismiss();

      return toast.error(
        "Oops, seems I was busy handling other user requests please try again"
      );
    }
    toast.dismiss();
  };

  return (
    <div className="wrapper">
      <div className="wrapper_content">
        <div className="mb_60 flex column align_center text_center justify_center gap_24">
          <div className="h4 border_round round">
            <Image src={replygptIcon} className="height_grow" alt="" />
          </div>
          <h1 className="h4">Project Writer - Your AI Project Writer</h1>
          <p>
            AI Project Writer - This tool is meant to be a guide on how to write
            your project 😁. I was built to guide you not help you write your
            entire project.
          </p>
        </div>
      </div>
      <div className="wrapper_content">
        <div className="form_wrapper">
          <h2 className="form_title h5 mb_24 text_center">
            Use AI to write your project
          </h2>
          <form onSubmit={handleForm} className="form">
            <div className="form_body">
              <p className="form_fields">
                <label htmlFor="topic" className="form_label">
                  What is your project topic?
                </label>
                <input
                  className="form_input"
                  type="text"
                  name="topic"
                  id="topic"
                  placeholder="Investigating the causes of erosion in Auchi"
                  required
                />
              </p>
              <p className="form_fields">
                <label htmlFor="chapter" className="form_label">
                  Select the chapter you want me to write
                </label>
                <select
                  required
                  className="form_input"
                  name="chapter"
                  id="chapter"
                >
                  <option value="Abstract">Write Abstract</option>
                  <option value="Acknowledgement">Write Acknowledgement</option>
                  <option value="Outline">Write Outline</option>
                  <option value="Introduction">Chapter I: Introduction</option>
                  <option value="Review of Literature">
                    Chapter II: Review of Literature
                  </option>
                  <option value="Methodology (Research Design & Methods)">
                    Chapter III: Methodology (Research Design & Methods)
                  </option>
                  <option value="Presentation of Research (Results)">
                    Chapter IV: Presentation of Research (Results)
                  </option>
                  <option value="Conclusions">
                    Chapter V: Summary, Implications, Conclusions (Discussion)
                  </option>
                </select>
              </p>
              <div className="form_fields">
                <label htmlFor="message" className="form_label">
                  Your project
                </label>
                <pre className="textarea form_input" contentEditable>
                  {text}
                </pre>
              </div>
              <p></p>
            </div>
            <div className="form_actions">
              <button className="form_btn" type="submit">
                <span>Start writing</span>
              </button>
            </div>
          </form>
        </div>
      </div>
      <Toaster />
    </div>
  );
}

export default Page;
