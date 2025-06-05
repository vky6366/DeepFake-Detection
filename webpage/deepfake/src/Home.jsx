import MediaCard from "./components/MediaCard";

export default function Home() {
  const cards = [
    { title: "Video", path: "/video", color: "bg-blue-400", icon: "🎬" },
    { title: "Audio", path: "/audio", color: "bg-green-400", icon: "🎵" },
    { title: "Text",  path: "/text",  color: "bg-yellow-400", icon: "📝" },
    { title: "Image", path: "/image", color: "bg-pink-400", icon: "🖼️" },
  ];
  return (
    <main className="flex flex-wrap gap-8 justify-center items-center py-16">
      {cards.map(c => (
        <MediaCard key={c.path} {...c} />
      ))}
    </main>
  );
}