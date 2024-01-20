import Image from "next/image";
import { StaticImageData } from "next/image";

function Avatar({ image }: { image: StaticImageData }) {
  return (
    <Image
      className="sprucemarkdown_feature_image animate pop delay-2"
      src={image}
      alt="Feature Image"
    />
  );
}

export default Avatar;
