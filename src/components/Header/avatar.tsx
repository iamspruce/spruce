import Image from "next/image";
import Link from "next/link";

function Avatar() {
  return (
    <Link href="/" className="header_avatar">
      {/*  <Image
        width="45"
        height="45"
        className="header_avatar_img"
        src="/img/spruce.png"
        alt="Spruce Emmanuel"
      /> */}
      <span className="header_avatar_name">Spruce</span>
    </Link>
  );
}
export default Avatar;
