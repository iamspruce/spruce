import React, { useState } from "react"
import BlogPost from "./BlogPost"
import { Link, graphql, useStaticQuery } from "gatsby"
import Icon from "./Icon"
import Projects from "./Projects"

const About = () => {
  const [openTab, setOpenTab] = useState("off")
  function toggle() {
    setOpenTab(openTab === "off" ? "on" : "off")
  }
  const data = useStaticQuery(graphql`
    query {
      site {
        siteMetadata {
          about
        }
      }
    }
  `)
  return (
    <section className="section-side-pad section-pad" id="about">
        <div className="section-about">
          <div className="section-content flex flex-col">
            <h4 className="text-xlg">A little more About Me</h4>
            <p>{data.site.siteMetadata.about}</p>
            <ul className="about-link-list" style={{ marginTop: 5 + "em" }}>
              <li className={`about-tab ${openTab}`}>
                <Link to="#0" className={`flex j-bsl`} onClick={toggle}>
                  <span className="span-num">01</span>
                  <span className="line" style={{ marginLeft: 12 + "px" }}>
                    Articles
                  </span>
                </Link>
              </li>
              <li className="about-tab">
                <Link to="#0" className={`flex j-bsl`}  onClick={toggle}>
                  <span className="span-num">02</span>
                  <span className="line" style={{ marginLeft: 12 + "px" }}>
                    Projects
                  </span>
                </Link>
              </li>
            </ul>
            <ul className="icon-list flex j-bsl" style={{ marginTop: "auto" }}>
              <li>
                <a
                  href="https://twitter.com/sprucekhalifa"
                  className="flex j-bsl"
                >
                  <Icon name="icon-twitter" size="18" />
                  <span>Twitter</span>
                  <Icon name="link-external" size="15" />
                </a>
              </li>
              <li>
                <a href="https://github.com/iamspruce" className="flex j-bsl">
                  <Icon name="icon-github" size="18" />
                  <span>Github</span>
                  <Icon name="link-external" size="15" />
                </a>
              </li>
            </ul>
          </div>
          <div className="section-cards">
          <BlogPost openTab={openTab} />
          <Projects />
          </div>
        </div>
    </section>
  )
}

export default About
