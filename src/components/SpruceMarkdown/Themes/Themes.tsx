import Image from "next/image";
import image from "/public/img/features/themes.png";

function Themes() {
  return (
    <div className="sprucemarkdown_themes mt_60 flex column align_center text_center justify_center gap_24">
      <h3>Custom Themes</h3>
      <p>
        Choose the best theme for your editor, plenty of themes to choose from
      </p>
      <div className=" flex wrappe_content">
        <Image
          className="sprucemarkdown_feature_image animate pop delay-2"
          src={image}
          alt="Feature Image"
        />
      </div>
    </div>
  );
}

export default Themes;
