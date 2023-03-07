import Card from "./card";
import { ArticleData } from "@/articles";

function Articles() {
  return (
    <div className="article_wrapper wrapper">
      <p className="mb_12">
        <strong>Featured Posts</strong>{" "}
      </p>
      <div className="article_cards">
        {ArticleData.slice(0, 5).map((article, i) => {
          return <Card key={i} article={article} index={i} />;
        })}
      </div>
    </div>
  );
}

export default Articles;
