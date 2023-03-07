import Image from "next/image";
import avatarImage from "../../../public/img/spruce.png";

function Avatar() {
  return (
    <Image
      className="border_round animate slide"
      src={avatarImage}
      alt="Spruce Emmanuel"
      width="100"
      height="100"
    />
  );
}

export default Avatar;
