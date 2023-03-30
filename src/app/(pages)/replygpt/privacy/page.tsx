function Page() {
  return (
    <div className="wrapper">
      <div className="wrapper_content">
        <div className=" mb_60 flex column align_center text_center justify_center gap_24">
          <h1 className="h4">
            Privacy Policy for ReplyGPT - AI Email Writing Assistance
          </h1>
          <p>
            At ReplyGPT, we value your privacy and are committed to protecting
            your personal information. This Privacy Policy outlines how we
            collect, use, and safeguard your information when you use our Gmail
            add-on, ReplyGPT.
          </p>
        </div>
        <h3 className="mb_12">Information Collection And Use</h3>
        <p>
          ReplyGPT's use and transfer to any other app of information received
          from Google APIs will adhere to the{" "}
          <a
            href="https://developers.google.com/terms/api-services-user-data-policy#additional_requirements_for_specific_api_scopes"
            target="_blank"
            rel="noopener noreferrer"
          >
            Google API Services User Data Policy,
          </a>{" "}
          including the Limited Use requirements.
        </p>
        <p>
          The Add-On is an AI-powered email writing assistance tool that works
          with Google Workspace and requires certain permissions to operate:
        </p>

        <ol className="mb_24">
          <li>
            Reads your email address and column headers to access your email
            inbox and messages.
          </li>
          <li>
            Collects your email address and payment information when purchasing
            credits on the Add-On through the LemonSqueezy platform.
          </li>
          <li>
            Sends your email address to a secure, remote server that hosts our
            database of users.
          </li>
        </ol>
        <p>
          We do not collect your email content or any other data that is not
          necessary for providing the services offered by this Add-On. Your
          credit card information is not saved anywhere on our servers, and we
          do not have access to it at any point in time. All payment
          transactions are handled securely by LemonSqueezy.
        </p>
        <p>
          To revoke the permissions granted to the Add-On, you must uninstall
          it. Information gathered by the Add-On will never be sold or shared
          with external third parties.
        </p>
        <h3>Data Storage</h3>
        <h4 className="mb_12">Data We Store:</h4>
        <ol className="mb_12">
          <li>Your Google account email address.</li>
          <li>
            Payment transaction IDs that were returned from the LemonSqueezy
            API.
          </li>
          <li>The date on which your payment was made.</li>
        </ol>
        <h4 className="mb_12">Data We Do NOT Store:</h4>
        <ol className="mb_12">
          <li>Your email content or drafts.</li>
        </ol>
        <h3 className="mb_12">Communications</h3>
        <p>
          We may send you marketing and advertising communications to update and
          relay information about our latest products. You can opt-out of
          receiving these communications at any time.
        </p>

        <h3 className="mb_12">Your Choices</h3>
        <p>
          You have complete control over the data that we have collected and
          stored on our databases. If you want to modify or delete any data we
          have stored, please contact us via email at replygptdev@gmail.com to
          do so, and we will promptly work through your inquiry.
        </p>
        <h3 className="mb_12">Consent</h3>
        <p>
          By using our Add-On, you hereby consent to our Privacy Policy and
          agree to its terms.
        </p>
        <h3 className="mb_12">Changes to ReplyGPT Privacy Policy</h3>
        <p>
          The ReplyGPT Privacy Policy is effective as of March 14th, 2023, and
          will remain in effect except with respect to any changes in its
          provisions in the future, which will be in effect immediately after
          being posted on this page. We reserve the right to update or change
          our Privacy Policy at any time, and you should check this Privacy
          Policy periodically. Your continued use of the Add-On after we post
          any modifications to the Privacy Policy on this page will constitute
          your acknowledgment of the modifications and your consent to abide and
          be bound by the modified Privacy Policy. If we make any material
          changes to this Privacy Policy, we will notify you either through your
          the email address.
        </p>
        <h2 className="mb_12">Contact Us</h2>
        <p>
          If you have any questions or concerns about our Privacy Policy or our
          add-on, please contact us at replygptdev@gmail.com.
        </p>
      </div>
    </div>
  );
}

export default Page;
