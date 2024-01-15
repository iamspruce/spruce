import { Logo } from "./logo";

function Page() {
  return (
    <div className="wrapper_content">
      <div className=" mb_60 flex column align_center text_center justify_center gap_24">
        <div className="sprucemarkdown_logo">
          <div className="sprucemarkdown_logo_lights"></div>
          <Logo />
        </div>
        <h1 className="h4">Spruce Markdown App</h1>
        <p>A Neat and Smart Markdown Editor</p>
      </div>
    </div>
  );
}

export default Page;
