import Image from "next/image";

function Avatar() {
  return (
    <Image
      className="border_round animate slide"
      src="/img/spruce.png"
      alt="Spruce Emmanuel"
      width="100"
      height="100"
    />
  );
}

export default Avatar;
