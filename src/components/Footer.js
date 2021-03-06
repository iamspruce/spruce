import React from "react"
import { Link } from "gatsby"
import { StaticImage } from "gatsby-plugin-image"


const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-content__copy">
          <div className="header-logo">
            <Link title="Spruce" to="/">
            <StaticImage
              src="../img/spruce-logo1.webp"
              alt="Spruce"
              placeholder="blurred"
              layout="fixed"
              width={32}
              height={32}
            />
            </Link>
          </div>
          <p>
             @2021, iamspruce.dev designed and developed with love and more
            of <b>GatsbyJs</b> and hosted on <b>Gatsby Cloud</b>
          </p>
        </div>
      <nav className="footer-nav flex j-btw">
        <ul className="footer-nav__items">
          <li className="footer-nav__list--title">SITE LINKS</li>
          <li className="footer-nav__list">
            <Link to="/" className="footer-nav__link">
              Homepage
            </Link>
          </li>
          <li className="footer-nav__list">
            <Link to="/#about" className="footer-nav__link">
              About Me
            </Link>
          </li>
          <li className="footer-nav__list">
            <Link to="/#contact" className="footer-nav__link">
              Contact Me
            </Link>
          </li>
        </ul>
        <ul className="footer-nav__items">
          <li className="footer-nav__list--title">SOCIAL LINKS</li>
          <li className="footer-nav__list">
            <a href="https://facebook.com/spruce.emma" className="footer-nav__link">
              Facebook
            </a>
          </li>
          <li className="footer-nav__list">
            <a href="https://twitter.com/sprucekhalifa" className="footer-nav__link">
              @sprucekhalifa
            </a>
          </li>
          <li className="footer-nav__list">
            <a href="https://instagram.com/iamspruce" className="footer-nav__link">
              Instagram
            </a>
          </li>
        </ul>
      </nav>
      </div>
    </footer>
  )
}

export default Footer
