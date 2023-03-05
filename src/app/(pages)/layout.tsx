export default function PageLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="wrapper wrapper_page">{children}</div>;
}
