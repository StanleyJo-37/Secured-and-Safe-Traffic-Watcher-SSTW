import Image from "next/image";
import HeroCover from "@/assets/hero.jpg";

export default function Page() {
  return (
    <div>
      <section className="full-screen flex flex-col items-start justify-center">
        <Image
          src={HeroCover}
          alt="Hero"
          fill
          className="object-cover -z-50"
          priority
        />

        <div className="absolute inset-0 -z-40 bg-gradient-to-l from-black/20 to-black" />

        <h1 className="text-white text-8xl w-1/2 ml-16">
          Secure and Safe Traffic Watcher
        </h1>
      </section>
    </div>
  );
}
