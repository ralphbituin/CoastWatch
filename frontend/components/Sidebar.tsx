"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  GaugeCircle,
  LayoutDashboard,
  MapPin,
  Video,
  Waves,
} from "lucide-react";
import clsx from "clsx";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/stations", label: "Stations", icon: MapPin },
  { href: "/tides", label: "Tides", icon: Waves },
  { href: "/cameras", label: "Cameras", icon: Video },
  { href: "/shoreline", label: "Shoreline", icon: GaugeCircle },
  { href: "/anomalies", label: "Anomalies", icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-white/10 bg-deep text-sand">
      <div className="flex items-center gap-2 px-5 py-5">
        <Waves className="h-6 w-6 text-reef" aria-hidden />
        <div>
          <p className="text-lg font-semibold leading-none">CoastWatch</p>
          <p className="text-xs text-sand/60">Shoreline monitoring</p>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-1 px-3">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-reef/20 text-white"
                  : "text-sand/70 hover:bg-white/5 hover:text-white",
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-5 py-4 text-xs text-sand/50">
        Research &amp; decision-support system — not an official tsunami
        warning service.
      </div>
    </aside>
  );
}
