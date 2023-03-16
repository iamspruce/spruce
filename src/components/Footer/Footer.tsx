import Link from "next/link";

function Footer() {
  return (
    <footer className="footer">
      <div className="wrapper footer_inner">
        <div>
          <p className="footer_items_title">
            <strong>Links</strong>
          </p>
          <ul className="footer_items">
            <li className="footer_items_list">
              <Link href="/" className="footer_items_link">
                Home
              </Link>
            </li>
            <li className="footer_items_list">
              <Link href="/about" className="footer_items_link">
                About
              </Link>
            </li>
            <li className="footer_items_list">
              <Link href="/contact" className="footer_items_link">
                Contact
              </Link>
            </li>
            <li className="footer_items_list">
              <Link href="/writing" className="footer_items_link">
                Writing
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <p className="footer_items_title">
            <strong>Products</strong>
          </p>
          <ul className="footer_items">
            <li className="footer_items_list">
              <Link href="/replygpt" className="footer_items_link">
                ReplyGPT
              </Link>
            </li>
            <li className="footer_items_list">
              <Link href="#0" className="footer_items_link">
                Feedin.bio
              </Link>
            </li>
          </ul>
        </div>

        <div>
          <p className="footer_items_title">
            <strong>Social</strong>
          </p>
          <ul className="footer_items">
            <li className="footer_items_list">
              <a
                href="https://twitter.com/sprucekhalifa"
                className="footer_items_link"
              >
                Twitter
              </a>
            </li>
            <li className="footer_items_list">
              <a
                href="https://github.com/iamspruce"
                className="footer_items_link"
              >
                Github
              </a>
            </li>
            <li className="footer_items_list">
              <a
                href="https://www.linkedin.com/in/spruceemma/"
                className="footer_items_link"
              >
                Linkedin
              </a>
            </li>
            <li className="footer_items_list">
              <a
                href="https://codepen.io/Spruce_khalifa"
                className="footer_items_link"
              >
                Codepen
              </a>
            </li>
          </ul>
        </div>
        <div>
          <p className="footer_items_title">
            <strong>Publications</strong>
          </p>
          <ul className="footer_items">
            <li className="footer_items_list">
              <a
                href="https://www.freecodecamp.org/news/author/spruce/"
                className="footer_items_link"
              >
                Freecodecamp
              </a>
            </li>
            <li className="footer_items_list">
              <a href="https://dev.to/iamspruce" className="footer_items_link">
                Dev.to
              </a>
            </li>
          </ul>
        </div>
      </div>
    </footer>
  );
}
export default Footer;
