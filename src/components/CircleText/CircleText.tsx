function CircleText({
  width,
  height,
  radius,
  text,
}: {
  width: number;
  height: number;
  radius: number;
  text: string;
}) {
  return (
    <svg width={width * 2} height={height * 2}>
      <path
        id="circleText"
        fill="none"
        d={`M ${width - radius}, ${height}
        a ${radius},${radius} 0 1,1 ${radius * 2},0
        a ${radius},${radius} 0 1,1 ${radius * -2},0`}
      />
      <text fill="#fff" className="circle_text">
        <textPath xlinkHref="#circleText">{text} -</textPath>
      </text>
    </svg>
  );
}

export default CircleText;
