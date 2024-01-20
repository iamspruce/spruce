import DownloadOptions from "./downloadOptions";
import Logo from "./Logo";

function Download() {
  return (
    <div className="sprucemarkdown_download mt_60 flex column align_center text_center justify_center gap_24">
      <h3>Download Spruce Markdown App</h3>
      <p>
        Free, Neat and Smart Markdown Editor, bring the power of AI into your
        markdown files.
      </p>
      <div className="flex justify_btw gap_24 align_center">
        <Logo />
        <div className="sprucemarkdown_download_text flex gap_4 column justify_start align_start">
          <p>
            <span className="text_small">
              No monthly subscription fee, free for most use.
            </span>{" "}
            <br />
            Your AI Powered Markdown Editor
          </p>
        </div>
      </div>

      <DownloadOptions />
    </div>
  );
}

export default Download;
