import Link from "next/link";

function Footer() {
  return (
    <footer className="footer">
      <div className="wrapper footer_inner">
        <ul className="footer_items">
          <p className="footer_items_title">
            <strong>Links</strong>
          </p>
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
        <ul className="footer_items">
          <p className="footer_items_title">
            <strong>Tutorials</strong>
          </p>
          <li className="footer_items_list">
            <Link href="" className="footer_items_link">
              JavaScript
            </Link>
          </li>
          <li className="footer_items_list">
            <Link href="" className="footer_items_link">
              React
            </Link>
          </li>
          <li className="footer_items_list">
            <Link href="" className="footer_items_link">
              Node.js
            </Link>
          </li>
          <li className="footer_items_list">
            <Link href="" className="footer_items_link">
              Technical writing
            </Link>
          </li>
        </ul>
        <ul className="footer_items">
          <p className="footer_items_title">
            <strong>Publications</strong>
          </p>
          <li className="footer_items_list">
            <Link
              href="https://www.freecodecamp.org/news/author/spruce/"
              className="footer_items_link"
            >
              Freecodecamp
            </Link>
          </li>
          <li className="footer_items_list">
            <Link href="https://dev.to/iamspruce" className="footer_items_link">
              Dev.to
            </Link>
          </li>
        </ul>
        <ul className="footer_items">
          <p className="footer_items_title">
            <strong>Social</strong>
          </p>
          <li className="footer_items_list">
            <Link
              href="https://twitter.com/sprucekhalifa"
              className="footer_items_link"
            >
              Twitter
            </Link>
          </li>
          <li className="footer_items_list">
            <Link
              href="https://github.com/iamspruce"
              className="footer_items_link"
            >
              Github
            </Link>
          </li>
          <li className="footer_items_list">
            <a
              href="https://www.linkedin.com/in/spruceemma/"
              className="footer_items_link"
            >
              Linkedin
            </a>
          </li>
        </ul>
      </div>
    </footer>
  );
}
export default Footer;
