interface Feature {
  title: string;
  description: string;
}
function Card({ feature, index }: { feature: Feature; index: number }) {
  return (
    <li
      className={`sprucemarkdown__features_list_item animate slide delay-${index}`}
      key={index}
    >
      <h2>{feature.title}</h2>
      <p>{feature.description}</p>
    </li>
  );
}

export default Card;
