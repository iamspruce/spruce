import React from "react";
import {
  Listbox,
  ListboxInput,
  ListboxButton,
  ListboxPopover,
  ListboxList,
  ListboxOption,
} from "@reach/listbox";
import "@reach/listbox/styles.css";

function DownloadOptions() {
  return (
    <div className="flex wrap justify_center align_center gap_36 mt_30">
      <a
        href="https://sprucemarkdownapp.lemonsqueezy.com/checkout"
        className="sprucemarkdown_btn cta"
      >
        Upgrade to Pro
      </a>
      <div className="flex">
        <a className="sprucemarkdown_btn flex gap_12" href="">
          <span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
            >
              <path d="M22 17.607c-.786 2.28-3.139 6.317-5.563 6.361-1.608.031-2.125-.953-3.963-.953-1.837 0-2.412.923-3.932.983-2.572.099-6.542-5.827-6.542-10.995 0-4.747 3.308-7.1 6.198-7.143 1.55-.028 3.014 1.045 3.959 1.045.949 0 2.727-1.29 4.596-1.101.782.033 2.979.315 4.389 2.377-3.741 2.442-3.158 7.549.858 9.426zm-5.222-17.607c-2.826.114-5.132 3.079-4.81 5.531 2.612.203 5.118-2.725 4.81-5.531z" />
            </svg>
          </span>
          Download Now
        </a>
        <select className="sprucemarkdown_select" name="download" id="download">
          <option> </option>
          <option value="macos">Macos</option>
          <option value="macos">Windows(coming soon)</option>
        </select>
      </div>
    </div>
  );
}

export default DownloadOptions;
