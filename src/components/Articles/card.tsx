import Image from "next/image";
import CircleText from "../CircleText";
interface Article {
  title: string;
  url: string;
  description: string;
  image: string;
  author: string;
  platform: string;
  date: string;
}
function Card({ article, index }: { article: Article; index: number }) {
  return (
    <article
      key={index}
      className={`article_cards_card animate slide delay-${index}`}
    >
      <div className="article_cards_card_image">
        <div className="article_cards_card_image_text">
          <CircleText
            width={40}
            height={40}
            radius={20}
            text={article.platform}
          />
        </div>
        <div className="article_cards_card_image_shadow"></div>
        <Image src={article.image} alt="" fill></Image>
      </div>
      <div className="article_cards_card_inner">
        <h2>
          <a
            target="_blank"
            rel="noopener noreferrer"
            className="article_cards_card_link"
            href={article.url}
          >
            {article.title}
          </a>{" "}
        </h2>
        <p>
          <small>{article.date}</small>
        </p>
        <p className="article_cards_card_desc">{article.description}</p>
        <a
          target="_blank"
          rel="noopener noreferrer"
          href={article.url}
          className="article_cards_card_btn"
        >
          Read article
        </a>
      </div>
    </article>
  );
}

export default Card;
