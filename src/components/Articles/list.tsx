import { ArticleData } from "@/articles";

function ArticleList() {
  return (
    <ol className="list_none">
      {ArticleData.map((article, i) => {
        return (
          <li key={i} className="mb_60">
            <h2 className="h4">
              <a href={article.url} target="_blank" rel="noopener noreferrer">
                {article.title}
              </a>
            </h2>
            <div className="mt_6 mb_12">
              <time datatype={article.date}>{article.date}</time>
              <span aria-hidden="true">⋅</span>{" "}
              <span className="color_offset">{article.platform}</span>
            </div>
            <p className="color_offset">{article.description}</p>
          </li>
        );
      })}
    </ol>
  );
}

export default ArticleList;
