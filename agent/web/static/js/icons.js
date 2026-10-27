// Icon set (Phosphor-style duotone, drawn on a 24px grid). Paths come straight from the design file L-Icon.dc.html.
// icon(name, size) returns an <svg> element; the second path of an entry is the soft fill drawn at 20 % opacity.

const S = 'M12 3l7 3v5.5c0 4.5-3 8-7 9.5-4-1.5-7-5-7-9.5V6z';
const C = 'M12 3a9 9 0 100 18 9 9 0 000-18z';
const T = 'M12 4l9 16H3z';
const O = 'M8.5 3h7L21 8.5v7L15.5 21h-7L3 15.5v-7z';
const I = {
  check:['M5 12.5l4.5 4.5L19 7.5'], x:['M6 6l12 12M18 6L6 18'], plus:['M12 5v14M5 12h14'],
  chevd:['M6 9l6 6 6-6'], chevr:['M9 6l6 6-6 6'], chevu:['M6 15l6-6 6 6'], arrow:['M5 12h14M13 6l6 6-6 6'],
  shield:[S,S], 'shield-check':[S+'M8.8 12l2.4 2.4 4-4.5',S], 'shield-half':[S+'M12 3v18','M12 3l-7 3v5.5c0 4.5 3 8 7 9.5z'],
  'shield-minus':[S+'M9 12h6'], 'shield-x':[S+'M9.5 9.5l5 5M14.5 9.5l-5 5'], 'shield-q':[S+'M10 10a2 2 0 114 0c0 1.4-2 1.6-2 3M12 16v.01'],
  verified:[C+'M7.5 12.5l2.5 2.5 3-3.5M11 15l1 1 4.5-5.5',C], likely:[C+'M8 12.3l2.8 2.8 5.5-5.6'], single:[C+'M12 10.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z'],
  conflict:['M4 7h6l4 10h6M4 17h6l4-10h6M18 5l2 2-2 2M18 15l2 2-2 2'], lowtrust:[T+'M12 10v4M12 17v.01',T],
  memory:['M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3zM5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3'],
  warn:[T+'M12 10v4M12 17v.01',T], danger:[O+'M12 8v5M12 16v.01',O], blocked:[C+'M5.6 5.6l12.8 12.8'], info:[C+'M12 11v5M12 8v.01'],
  terminal:['M4 5h16v14H4zM7.5 10l3 2.5-3 2.5M13 15h4'], search:['M10.5 4a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM16 16l4.5 4.5'],
  globe:[C+'M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18'], file:['M6 3h8l4 4v14H6zM14 3v4h4','M6 3h8l4 4v14H6z'],
  pencil:['M4 20l1-4L16 5l3 3L8 19zM14 7l3 3'], clip:['M16.5 7.5l-7 7a2 2 0 002.8 2.8l7-7a4 4 0 00-5.7-5.7l-7 7a6 6 0 008.5 8.5l6-6'],
  send:['M12 19V5M6 11l6-6 6 6'], stop:['M7 7h10v10H7z','M7 7h10v10H7z'], copy:['M8 8h11v12H8zM5 16V4h11'],
  pin:['M9 4h6l-1 6 3 3H7l3-3zM12 13v7'], expand:['M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5'], dots:['M5 12v.01M12 12v.01M19 12v.01'],
  clock:[C+'M12 7v5l3 2'], wifioff:['M3 3l18 18M5 10a10 10 0 015-2.5M19 10a10 10 0 00-3.5-2.2M8.5 14a5 5 0 015-1.4M12 18v.01'],
  moon:['M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z'], sun:['M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5'],
  gear:['M12 9a3 3 0 100 6 3 3 0 000-6zM12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1'],
  chat:['M4 5h16v11H9l-5 4z','M4 5h16v11H9l-5 4z'], grid:['M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z'],
  book:['M5 4h10a4 4 0 014 4v12H9a4 4 0 01-4-4z'], cpu:['M7 7h10v10H7zM10 10h4v4h-4zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4'],
  list:['M8 6h12M8 12h12M8 18h12M4 6v.01M4 12v.01M4 18v.01'], folder:['M3 6h6l2 2h10v11H3z','M3 6h6l2 2h10v11H3z'],
  code:['M9 8l-4 4 4 4M15 8l4 4-4 4M13.5 6l-3 12'], trash:['M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13'], undo:['M9 6L4 11l5 5M4 11h10a5 5 0 010 10h-3'],
  play:['M8 5l11 7-11 7z','M8 5l11 7-11 7z'], bolt:['M13 3L5 13h6l-1 8 8-10h-6z','M13 3L5 13h6l-1 8 8-10h-6z'],
  up:['M7 11v9H4v-9zM7 11l4-7c2 0 2.5 1.5 2 3l-.5 2H19a1.5 1.5 0 011.5 1.8l-1.3 6A2 2 0 0117.3 20H7'],
  down:['M7 13V4H4v9zM7 13l4 7c2 0 2.5-1.5 2-3l-.5-2H19a1.5 1.5 0 001.5-1.8l-1.3-6A2 2 0 0017.3 4H7'],
  link:['M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1'],
  sliders:['M4 7h10M18 7h2M4 17h2M10 17h10M14 5v4M6 15v4'], eyeoff:['M3 3l18 18M10 6.2A9 9 0 0121 12a13 13 0 01-3 3.7M6.5 7.5A13 13 0 003 12s3.5 6 9 6a8 8 0 003-.6'],
  sparkle:['M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z'], download:['M12 4v11M7 11l5 5 5-5M5 20h14'],
  lock:['M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 017 0v3','M6 11h12v9H6z'], refresh:['M20 11a8 8 0 10-2 6M20 5v6h-6'],
  image:['M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M9 8.5v.01'], music:['M9 18V6l10-2v12M9 18a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0zM19 16a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z'],
  chart:['M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-3'], menu:['M4 7h16M4 12h16M4 17h16'], power:['M12 3v8M6.5 6.5a8 8 0 1011 0'],
  user:['M12 4a4 4 0 100 8 4 4 0 000-8zM4 21c0-4 3.6-6 8-6s8 2 8 6'], wa:['M4 20l1.5-4.5A8 8 0 1112 20a8 8 0 01-3.5-.8zM9 9c0 3 3 6 6 6l1-2-2-1-1 .8c-.8-.4-1.6-1.2-2-2l.8-1-1-2z'],
  flag:['M5 21V4M5 5h12l-2 4 2 4H5'], eye:['M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6zM12 9.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z'],
  cat:['M5 4l3 3h8l3-3v9a7 7 0 01-14 0z'],
  pause:['M8 5v14M16 5v14'], prev:['M18 5v14l-9-7zM6 5v14'], next:['M6 5v14l9-7zM18 5v14'],
  shuffle:['M4 7h3c5 0 5 10 10 10h3M4 17h3c1.5 0 2.5-1 3.5-2.5M13.5 9.5C14.5 8 15.5 7 17 7h3M17 4l3 3-3 3M17 14l3 3-3 3']
};

const SVGNS = "http://www.w3.org/2000/svg";

export function icon(name, size = 18) {
  const [d, fd = ""] = I[name] || I.info;
  const svg = document.createElementNS(SVGNS, "svg");
  for (const [k, v] of Object.entries({ width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.6,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) svg.setAttribute(k, v);
  svg.style.cssText = "display:block;flex:none";
  if (fd) {
    const f = document.createElementNS(SVGNS, "path");
    f.setAttribute("d", fd); f.setAttribute("fill", "currentColor"); f.setAttribute("fill-opacity", "0.2"); f.setAttribute("stroke", "none");
    svg.append(f);
  }
  const p = document.createElementNS(SVGNS, "path");
  p.setAttribute("d", d);
  svg.append(p);
  return svg;
}

export const ICON_NAMES = Object.keys(I);
