import { notFound } from "next/navigation";

import { DemoInjectionView } from "@/features/demo/DemoInjectionView";

export default function DemoInjectionPage() {
  if (process.env.NEXT_PUBLIC_DEMO_INJECTION_ENABLED !== "true") {
    notFound();
  }
  return <DemoInjectionView />;
}
