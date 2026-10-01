import type { ReactNode, SVGProps } from "react";

type IconName = "grid" | "sliders" | "help" | "search" | "plus" | "play" | "stop" | "record" | "more" | "lock" | "chevron" | "arrow" | "close" | "edit" | "trash" | "check" | "folder" | "keyboard" | "shield" | "activity" | "refresh" | "spark" | "mouse" | "key" | "scroll" | "dots";

const paths: Record<IconName, ReactNode> = {
  grid: <><rect x="3.5" y="3.5" width="7" height="7" rx="1"/><rect x="13.5" y="3.5" width="7" height="7" rx="1"/><rect x="3.5" y="13.5" width="7" height="7" rx="1"/><rect x="13.5" y="13.5" width="7" height="7" rx="1"/></>,
  sliders: <><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></>,
  help: <><circle cx="12" cy="12" r="9"/><path d="M9.7 9a2.4 2.4 0 0 1 4.6 1c0 1.6-2.3 2-2.3 3.6M12 17.2v.1"/></>,
  search: <><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.2 4.2"/></>,
  plus: <><path d="M12 5v14M5 12h14"/></>,
  play: <path d="m9 6 10 6-10 6z" fill="currentColor" stroke="none"/>,
  stop: <rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor" stroke="none"/>,
  record: <circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>,
  more: <><circle cx="5" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="19" cy="12" r="1" fill="currentColor"/></>,
  lock: <><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
  chevron: <path d="m9 18 6-6-6-6"/>,
  arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
  close: <><path d="m6 6 12 12M18 6 6 18"/></>,
  edit: <><path d="m14 5 5 5M4 20l4-.8L19 8a2.1 2.1 0 0 0-3-3L5 16z"/></>,
  trash: <><path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  folder: <><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M3 10h18"/></>,
  keyboard: <><rect x="2.5" y="5" width="19" height="14" rx="2"/><path d="M6 9h.1M9 9h.1M12 9h.1M15 9h.1M18 9h.1M6 12h.1M9 12h.1M12 12h.1M15 12h.1M18 12h.1M8 15h8"/></>,
  shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/><path d="m9 12 2 2 4-4"/></>,
  activity: <path d="M3 12h4l3-8 4 16 3-8h4"/>,
  refresh: <><path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.6 9a7 7 0 0 1 11.9-2L20 12M4 12l2.5 5a7 7 0 0 0 11.9-2"/></>,
  spark: <><path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7z"/></>,
  mouse: <><rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 3v6M6 9h12"/></>,
  key: <><circle cx="8" cy="10" r="4"/><path d="m11 13 8 8m-3-3 2-2m-5-1 2-2"/></>,
  scroll: <><path d="M4 7v10M8 5v14M12 8v8M16 4v16M20 7v10"/></>,
  dots: <><path d="M4 12h.1M8 12h.1M12 12h.1M16 12h.1M20 12h.1" strokeWidth="3"/></>,
};

export function Icon({ name, size = 18, ...props }: SVGProps<SVGSVGElement> & { name: IconName; size?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...props}>
      {paths[name]}
    </svg>
  );
}
