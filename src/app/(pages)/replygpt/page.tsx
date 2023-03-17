import Link from "next/link";
import Image from "next/image";
import replygpt from "/public/img/replygpt/replygpt.png";
import replygptIcon from "/public/img/replygpt/android-chrome-192x192.png";

function Page() {
  return (
    <div className="wrapper">
      <div className="wrapper_content">
        <div className="mb_60 flex column align_center text_center justify_center gap_24">
          <div className="h4 border_round round">
            <Image src={replygptIcon} className="height_grow" alt="" />
          </div>
          <h1 className="h4">ReplyGPT - Your AI Email Assistanct Writer</h1>
          <p>
            Introducing ReplyGPT - the ultimate AI email assistant that helps
            you write better, faster, and smarter emails. Say goodbye to
            writer&apos;s block and hello to more productive communication with
            ReplyGPT.
          </p>
        </div>
      </div>
      <video className="shadow hover_shadow" width="100%" autoPlay>
        <source src="/replygpt.mp4" type="video/mp4" />
      </video>
      <div className="wrapper_content">
        <div className=" mt_60 mb_60 flex column  gap_24">
          <p>
            Welcome to ReplyGPT, the ultimate Gmail add-on for effortless and
            effective email communication.
          </p>
          <p>
            Our platform is designed to help you save time and enhance your
            email writing experience, whether you&apos;re composing an email
            from scratch or replying to an existing message.
          </p>
          <p>Some of the features we offer include:</p>
          <ol className="mb_24">
            <li>
              Compose emails from scratch: With our intuitive interface, you can
              easily create a new email and get AI-powered writing assistance as
              you type.
            </li>
            <li>
              Reply to any email: Our AI analyzes the context of the email
              you&apos;re responding to and suggests relevant content to include
              in your reply, saving you time and effort.
            </li>
            <li>
              Choose a tone for your message: Whether you want to sound formal,
              casual, or somewhere in between, our platform allows you to choose
              a tone for your message and provides suggestions to match.
            </li>
            <li>
              Write in any language: Our platform supports multiple languages,
              so you can compose your emails in your preferred language without
              any limitations.
            </li>
            <li>
              Email templates: Save time on repetitive emails with our pre-built
              templates for common types of emails.
            </li>
          </ol>
          <p>
            With our platform, you can streamline your email communication
            process and spend more time on the things that matter most. Try
            ReplyGPT today and experience the power of AI-powered email writing.
          </p>
          <h2>How to Use ReplyGPT - Your AI Email Assistant Writer:</h2>
          <ol style={{ listStyle: "outside", listStyleType: "decimal" }}>
            <li>
              <strong> Download and install the Gmail add-on </strong> <br />
              To get started, download and install the ReplyGPT Gmail add-on
              from the Google Workspace Marketplace. This will allow you to
              access the add-on directly from your Gmail account.
            </li>
            <li>
              <strong>
                {" "}
                Open the Gmail add-on by clicking the ReplyGPT icon{" "}
              </strong>
              <br />
              Once you have installed the add-on, open a new email in Gmail and
              look for the ReplyGPT icon in the add-on toolbar. Click on the
              icon to open the ReplyGPT sidebar.
            </li>
            <li>
              <strong> Write the subject of the email </strong> <br /> Begin by
              writing the subject of your email in the subject line as you would
              normally.
            </li>
            <li>
              <strong> Enter the main points of the email in a list In </strong>{" "}
              <br /> the ReplyGPT sidebar, enter the main points of your email
              in a list format. This will help the AI writer to understand the
              key points of your message.
            </li>
            <li>
              <strong> Select a tone for your email </strong> <br /> Next,
              select a tone for your email using the tone dropdown menu in the
              sidebar. You can choose from professional, friendly, or casual.
            </li>
            <li>
              <strong>Click &quot;Generate Mail&quot;</strong> <br /> Finally,
              click the &quot;Generate Mail&quot; button in the sidebar to have
              ReplyGPT generate a draft email for you. You can then review and
              edit the email before sending it off.
            </li>
          </ol>
          <p className="mb_24">
            <br />
            <br />
            If you&apos;re interested in the ReplyGPT add-on, download it here.
            For information on how we collect and use data through ReplyGPT,
            please see our <Link href="/replygpt/privacy">Privacy Policy</Link>.
            By using ReplyGPT, you agree to our{" "}
            <Link href="/replygpt/terms">Terms of Service</Link>.
          </p>
          <Image src={replygpt} alt="ReplyGPT - Your AI Email Assistant" />
          <hr />
          <p>
            <br />
            If you have any feedback, found any issues or bugs, or have any
            questions at all, please email us at replygptdev@gmail.com or use
            our <Link href="/replygpt/support">Support page</Link> . We will
            respond as soon as possible to resolve any problems / answer any
            inquiries.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Page;
