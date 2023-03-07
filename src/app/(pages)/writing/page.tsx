import ArticleList from "@/components/Articles/list";

function Page() {
  return (
    <div className="wrapper_content">
      <div className=" mb_60 flex column align_center text_center justify_center gap_24">
        <h1 className="h4">A place I publish my thoughts 🧐</h1>
        <p>
          Welcome to my blog page! Here, you&apos;ll find a collection of
          articles and resources covering various topics related to web
          development, design, and technology.
        </p>
      </div>
      <ArticleList />
    </div>
  );
}

export default Page;
