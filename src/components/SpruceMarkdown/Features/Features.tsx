import Card from "./card";
import { FeaturesData } from "./featuresdata";

function Features() {
  return (
    <div className="sprucemarkdown_features">
      <p className="mb_12">
        <strong>Features</strong>{" "}
      </p>
      <ul className="sprucemarkdown__features_list article_cards">
        {FeaturesData.slice(0, 6).map((feature, i) => {
          return <Card key={i} feature={feature} index={i} />;
        })}
      </ul>
    </div>
  );
}

export default Features;
