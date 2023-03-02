import Avatar from "../Avatar";

function Hero() {
  return (
    <div className="wrapper ">
      <div className="mt_60 text_center flex column gap_12 align_center justify_center">
        <Avatar />
        <h1 className="text_small animate slide delay-1">
          Hi, I'm Spruce Emmanuel 👋
        </h1>
        <p className="text_large animate slide delay-2">
          Creating Content, <br /> building websites, and <br /> contributing to
          open source.
        </p>
      </div>
    </div>
  );
}
export default Hero;
