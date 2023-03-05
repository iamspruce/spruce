import Image from "next/image";
import coverImage from "../../../../public/img/spruce.png";

function Page() {
  return (
    <div className="flex column gap_24">
      <div className="wrapper_content flex column align_center text_center justify_center gap_24">
        <h1 className="h4">Hi, I&apos;m Spruce Emmanuel 👋</h1>
        <p>
          I am a web developer with experience in technical writing, committed
          to delivering high-quality work and providing exceptional service and
          support to clients, including international companies.
        </p>
      </div>
      <div className="grid mt_60">
        <div className="mb_12" style={{ gridColumn: "1 / 5" }}>
          <Image
            className="shadow hover_shadow"
            src={coverImage}
            width="403"
            height={433}
            alt=""
          />
        </div>
        <div style={{ gridColumn: "5 / 13" }}>
          <h2 className="mb_12 f_700 h4">A little more about me</h2>
          <p>
            I&apos;m a seasoned web developer with many years of experience in
            the field. I&apos;m passionate about building clean, functional, and
            visually appealing websites that deliver results.
          </p>
          <p>
            Over the years, I&apos;ve worked on a wide range of web development
            projects, including e-commerce sites, content management systems,
            and web applications. I specialize in front-end development, but
            I&apos;m also proficient in back-end development and server
            management.
          </p>
          <p>
            Aside from my work as a web developer, I&apos;ve also written many
            technical articles to help students learn web development. Through
            my writing, I&apos;ve been able to help thousands of aspiring
            developers gain a better understanding of web development concepts,
            tools, and technologies.
          </p>
          <p>
            In addition to writing, I&apos;m also an active contributor to
            open-source projects. I believe in the power of collaboration and
            community, and I&apos;m always looking for ways to give back to the
            tech community that has given me so much.
          </p>
          <p>
            When I&apos;m not coding, you can find me reading, or exploring new
            technologies. I&apos;m always looking for ways to expand my
            knowledge and stay on the cutting edge of web development.
          </p>
        </div>
      </div>
      <div className="wrapper_content flex column  gap_12">
        <h3 className="h4">Speaking</h3>
        <p>
          I truly enjoy speaking at events and conferences to share my knowledge
          and engage with like-minded individuals. If you're looking for a web
          development speaker for your upcoming event, don't hesitate to reach
          out to me.
        </p>
      </div>
      <div className="wrapper_content flex column  gap_12">
        <h3 className="h4">Writing</h3>
        <p>
          I author insightful web development content on my personal blog and
          diverse platforms such as freecodecamp and dev.to. You can access some
          of my published articles by following this link.
        </p>
        <ol>
          <li>
            <a
              href="https://www.freecodecamp.org/news/create-full-stack-app-with-nextjs13-and-firebase/"
              target="_blank"
              rel="noopener noreferrer"
            >
              How to Build a Full Stack App with Next.js 13 and Firebase
            </a>
          </li>
          <li>
            <a
              href="https://www.freecodecamp.org/news/javascript-dom-build-a-calculator-app/"
              target="_blank"
              rel="noopener noreferrer"
            >
              JavaScript DOM Tutorial – How to Build a Calculator App in JS
            </a>
          </li>
          <li>
            <a
              href="https://www.freecodecamp.org/news/learn-javascript-by-building-a-project/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn JavaScript Basics by Building a Counter Application
            </a>
          </li>
          <li>
            <a
              href="React Components – How to Create a Search, Filter, and Pagination Component in React"
              target="_blank"
              rel="noopener noreferrer"
            >
              React Components – How to Create a Search, Filter, and Pagination
              Component in React
            </a>
          </li>
          <li>
            <a
              href="https://dev.to/iamspruce/top-5-css-methodologies-in-2021-an1"
              target="_blank"
              rel="noopener noreferrer"
            >
              Top 5 CSS methodologies in 2021
            </a>
          </li>
        </ol>
      </div>
    </div>
  );
}

export default Page;
