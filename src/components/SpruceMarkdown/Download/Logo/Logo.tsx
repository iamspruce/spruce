import Image from "next/image";
import logoImage from "/public/img/features/sprucemarkdownlogo.png";

function Logo() {
  return (
    <Image
      className="border_11x animate slide"
      src={logoImage}
      alt="Spruce Markdown App"
      width="128"
      height="128"
    />
  );
}

export default Logo;
