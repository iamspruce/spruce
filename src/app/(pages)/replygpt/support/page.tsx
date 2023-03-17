"use client";
import React from "react";

function Page() {
  const handleForm = (event: any) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const name = formData.get("name");
    const email = formData.get("email");
    const message: any = formData.get("message");

    const link = `mailto:replygptdev@gmail.com?cc=${email}&subject=ReplyGPT issue from ${name}&body=${escape(
      message
    )}`;

    window.location.href = link;
  };
  return (
    <div className="wrapper_content">
      <div className="mb_60">
        <div className=" mb_60 flex column align_center text_center justify_center gap_24">
          <h1 className="h4">We&apos;ve been waiting for you.</h1>
          <p>We want to hear from you. Let us know how we can help.</p>
        </div>
      </div>
      <div className="form_wrapper " onSubmit={handleForm}>
        <h2 className="form_title h5 mb_24 text_center">Send us a message</h2>
        <form onSubmit={handleForm} className="form">
          <div className="form_body">
            <p className="form_fields">
              <label htmlFor="name" className="form_label">
                Enter your name
              </label>
              <input
                className="form_input"
                type="text"
                name="name"
                id="name"
                placeholder="John Doe"
                required
              />
            </p>
            <p className="form_fields">
              <label htmlFor="email" className="form_label">
                Enter your email
              </label>
              <input
                className="form_input"
                type="text"
                name="email"
                id="email"
                placeholder="example@gmail.com"
                required
              />
            </p>
            <p className="form_fields">
              <label htmlFor="message" className="form_label">
                Enter your message
              </label>
              <textarea
                className="form_input"
                name="message"
                id="message"
                placeholder="Your message here"
                required
              />
            </p>
          </div>
          <div className="form_actions">
            <button className="form_btn" type="submit">
              <span>Submit</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Page;
