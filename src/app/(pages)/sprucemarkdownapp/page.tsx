"use client";
import Features from "@/components/SpruceMarkdown/Features";
import { Logo } from "./logo";
import Tabs from "@/components/SpruceMarkdown/Tabs";
import Themes from "@/components/SpruceMarkdown/Themes";
import Download from "@/components/SpruceMarkdown/Download";

function Page() {
  return (
    <>
      <div className="wrapper_content">
        <div className=" mb_60 flex column align_center text_center justify_center gap_24">
          <div className="sprucemarkdown_logo">
            <div className="sprucemarkdown_logo_lights"></div>
            <Logo />
          </div>
          <h1 className="h4">Spruce Markdown App</h1>
          <p>
            A Neat and Smart Markdown Editor - Unleash the Power of AI in Your
            Markdown Editing!
          </p>
        </div>
      </div>
      <Features />
      <Tabs />
      <Themes />
      <Download />
    </>
  );
}

export default Page;
