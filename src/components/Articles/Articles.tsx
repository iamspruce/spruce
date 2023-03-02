import Card from "./card";
import { ArticleData } from "@/articles";

function Articles() {
  return (
    <div className="article_wrapper wrapper">
      <p>
        <strong>Featured Posts</strong>{" "}
      </p>
      <div className="article_cards">
        {ArticleData.map((article, i) => {
          return <Card key={i} article={article} index={i} />;
        })}
      </div>
    </div>
  );
}

export default Articles;
