"use client";
import React from "react";
import Link from "next/link";
import toast, { Toaster } from "react-hot-toast";
import Image from "next/image";
import replygptIcon from "/public/img/replygpt/android-chrome-192x192.png";

function Page() {
  const [text, SetText] = React.useState("Your project will apppear here");

  async function fetchData(url = "", data = {}) {
    console.log(data);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // 'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: JSON.stringify(data),
    });
    return response.text();
  }

  const handleForm = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const topic = document.getElementById("topic") as HTMLInputElement | null;
    const chapter = document.getElementById(
      "chapter"
    ) as HTMLInputElement | null;
    const message = document.getElementById(
      "message"
    ) as HTMLInputElement | null;

    const payload = {
      topic: topic?.value,
      chapter: chapter?.value,
    };

    toast.loading("Please wait... I am generating your project...");

    fetchData("api", payload)
      .then((data) => {
        console.log(data); // JSON data parsed by `data.json()` call
        toast.dismiss();
        toast.success("Your project is ready.");

        const raw = data;
        SetText(raw);
      })
      .catch((error) => {
        toast.dismiss();
        toast.error("Something bad happened, please try again");
        console.log(error, error.message);
      });
  };

  const handleInput = () => {
    const grower: any = document.querySelector(".grow-wrap");
    const textarea = grower.querySelector("textarea");

    textarea.addEventListener("input", () => {
      grower.dataset.replicatedValue = textarea.value;
    });
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
                <div className="grow-wrap">
                  <textarea
                    className="textarea form_input"
                    name="message"
                    id="message"
                    placeholder="Your project will apppear here"
                    onChange={handleInput}
                    value={text}
                  />
                </div>
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
