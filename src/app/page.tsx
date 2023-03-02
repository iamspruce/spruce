import Articles from "@/components/Articles";
import Hero from "@/components/Hero";

export default async function Home() {
  return (
    <section>
      <Hero />
      <Articles />
    </section>
  );
}
