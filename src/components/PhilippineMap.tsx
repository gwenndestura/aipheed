import { useEffect, useRef, useCallback, useMemo, useState } from "react";
import L from "leaflet";
import { leafletLayer } from "protomaps-leaflet";
import { RegionData, MunicipalityData, RISK_COLORS } from "@/data/types";
import { municipalitiesData } from "@/data/mockData";
import "leaflet/dist/leaflet.css";

// Self-hosted open basemap — a single Protomaps .pmtiles file built from
// OpenStreetMap data (ODbL). No API key and no third-party runtime call, so
// the map keeps working without a vendor relationship. The file location is
// deploy-configurable; see docs/basemap.md for how to generate it.
const PMTILES_URL =
  (import.meta.env.VITE_BASEMAP_PMTILES_URL as string | undefined) || "/basemap.pmtiles";
const BASEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://protomaps.com">Protomaps</a>';

interface Props {
  regions: RegionData[];                        // CALABARZON provinces
  selectedRegion: RegionData | null;            // selected province
  selectedMunicipality: MunicipalityData | null;
  onHover: (r: RegionData | null) => void;
  onClick: (r: RegionData) => void;
  onMunicipalityClick: (m: MunicipalityData) => void;
  rejectedProvinceIds?: Set<string>;            // provinces with no forecast for current quarter
}

const AMBER = "hsl(35, 90%, 55%)";

// Basemap flavor + page-background colour per theme. "black" / "white" Protomaps
// flavors are near-monochrome (no cyan water) and match the app's dark/light
// chrome; the bg colour is the flavor's water tone so any tile-boundary is
// invisible against the container.
const BASEMAP_FLAVOR = { dark: "black", light: "white" } as const;
const BASEMAP_PAGE_BG = { dark: "#1a1a1a", light: "#dcdcdc" } as const;

// CALABARZON region overall bounds (overrides flaky polygon bounds for SW Quezon island fragments)
const CALABARZON_BOUNDS: [[number, number], [number, number]] = [[13.40, 120.20], [15.20, 122.95]];

const SVG_NS = "http://www.w3.org/2000/svg";

const hexToRgb = (hex: string) => {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
};

const rgbToHex = ({ r, g, b }: { r: number; g: number; b: number }) =>
  `#${[r, g, b].map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("")}`;

const mixHex = (from: string, to: string, amount: number) => {
  const a = hexToRgb(from);
  const b = hexToRgb(to);
  return rgbToHex({
    r: a.r + (b.r - a.r) * amount,
    g: a.g + (b.g - a.g) * amount,
    b: a.b + (b.b - a.b) * amount,
  });
};

const NO_DATA_COLOR = "#6b7280"; // gray for areas without data

export function PhilippineMap({
  regions,
  selectedRegion,
  selectedMunicipality,
  onHover,
  onClick,
  onMunicipalityClick,
  rejectedProvinceIds,
}: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const provincesLayerRef = useRef<L.GeoJSON | null>(null);
  const municipalitiesLayerRef = useRef<L.GeoJSON | null>(null);
  const provincePulsesRef = useRef<{ marker: L.Marker; provinceId: string }[]>([]);
  const provinceLabelsRef = useRef<L.Marker[]>([]);
  const municipalityLabelsRef = useRef<L.Marker[]>([]);

  const onHoverRef = useRef(onHover);
  const onClickRef = useRef(onClick);
  const onMuniClickRef = useRef(onMunicipalityClick);
  const selectedRef = useRef(selectedRegion);
  const selectedMuniRef = useRef(selectedMunicipality);
  const [isZoomed, setIsZoomed] = useState(false);
  const isZoomedRef = useRef(false);
  const muniGeoDataRef = useRef<GeoJSON.FeatureCollection | null>(null);

  onHoverRef.current = onHover;
  onClickRef.current = onClick;
  onMuniClickRef.current = onMunicipalityClick;
  selectedRef.current = selectedRegion;
  selectedMuniRef.current = selectedMunicipality;

  const regionMap = useMemo(() => {
    const m = new Map<string, RegionData>();
    regions.forEach((r) => m.set(r.id, r));
    return m;
  }, [regions]);
  const regionMapRef = useRef(regionMap);
  regionMapRef.current = regionMap;

  const rejectedRef = useRef<Set<string>>(rejectedProvinceIds ?? new Set());
  rejectedRef.current = rejectedProvinceIds ?? new Set();

  const muniMap = useMemo(() => {
    // Map by lowercased name + province for matching against geojson "name"
    const m = new Map<string, MunicipalityData>();
    municipalitiesData.forEach((mu) => {
      m.set(`${mu.provinceId}::${mu.name.toLowerCase()}`, mu);
    });
    return m;
  }, []);

  const getProvinceStyle = useCallback((id: string | undefined, isSelected: boolean, isDimmed: boolean): L.PathOptions => {
    const region = id ? regionMapRef.current.get(id) : null;
    const isRejected = id ? rejectedRef.current.has(id) : false;
    if (!region || isRejected) {
      return {
        fillColor: NO_DATA_COLOR,
        fillOpacity: isDimmed ? 0.25 : isSelected ? 0.7 : 0.55,
        color: isSelected ? AMBER : "#1f2937",
        weight: isSelected ? 2.5 : 1,
      };
    }
    const fill = RISK_COLORS[region.riskLevel] ?? NO_DATA_COLOR;
    if (isDimmed) {
      return {
        fillColor: fill,
        fillOpacity: 0.25,
        color: "#1f2937",
        weight: 1,
      };
    }
    return {
      fillColor: fill,
      fillOpacity: isSelected ? 0.95 : 0.85,
      color: isSelected ? AMBER : "#1f2937",
      weight: isSelected ? 2.5 : 1,
    };
  }, []);

  const clearMunicipalities = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    if (municipalitiesLayerRef.current) {
      map.removeLayer(municipalitiesLayerRef.current);
      municipalitiesLayerRef.current = null;
    }
    municipalityLabelsRef.current.forEach((l) => map.removeLayer(l));
    municipalityLabelsRef.current = [];
  }, []);

  const showMunicipalities = useCallback((provinceId: string) => {
    const map = mapRef.current;
    if (!map || !muniGeoDataRef.current) return;

    clearMunicipalities();

    const features = muniGeoDataRef.current.features.filter(
      (f) => f.properties?.province_id === provinceId
    );
    if (!features.length) return;

    const isProvRejected = rejectedRef.current.has(provinceId);

    const muniLayer = L.geoJSON(
      { type: "FeatureCollection", features } as GeoJSON.FeatureCollection,
      {
        style: (feature) => {
          const name = (feature?.properties?.name || "").toLowerCase();
          const muni = muniMap.get(`${provinceId}::${name}`);
          const color = isProvRejected ? NO_DATA_COLOR : muni ? (RISK_COLORS[muni.riskLevel] ?? NO_DATA_COLOR) : NO_DATA_COLOR;
          const isSel = !isProvRejected && muni && selectedMuniRef.current?.id === muni.id;
          return {
            fillColor: color,
            fillOpacity: isSel ? 0.98 : 0.9,
            color: isSel ? AMBER : "rgba(255,255,255,0.45)",
            weight: isSel ? 2.5 : 0.6,
            lineJoin: "round",
            lineCap: "round",
          };
        },
        onEachFeature: (feature, layer) => {
          const name: string = feature.properties?.name || "Unknown";
          const muni = muniMap.get(`${provinceId}::${name.toLowerCase()}`);
          const tipColor = muni ? RISK_COLORS[muni.riskLevel] : "#888";
          layer.bindTooltip(
            `<div style="font-size:11px">
              <strong>${name}</strong>
              ${isProvRejected
                ? `<br/><span style="color:#9ca3af;font-style:italic">No Forecast Available</span>`
                : muni ? `<br/><span style="color:${tipColor}">Risk Level ${muni.riskScore.toFixed(2)} • ${muni.riskLevel}</span>` : ""}
            </div>`,
            { sticky: true }
          );
          layer.on({
            mouseover: (e) => {
              const l = e.target as L.Path;
              l.setStyle({ fillOpacity: 1, weight: 2, color: AMBER });
              l.bringToFront();
            },
            mouseout: (e) => {
              const l = e.target as L.Path;
              const isSel = !isProvRejected && muni && selectedMuniRef.current?.id === muni.id;
              l.setStyle({
                fillOpacity: isSel ? 0.98 : 0.9,
                weight: isSel ? 2.5 : 0.8,
                color: isSel ? AMBER : "rgba(255,255,255,0.35)",
              });
            },
            click: () => {
              if (muni && !isProvRejected) onMuniClickRef.current(muni);
            },
          });
        },
      }
    ).addTo(map);
    municipalitiesLayerRef.current = muniLayer;

    muniLayer.eachLayer((layer) => {
      const feature = (layer as any).feature as GeoJSON.Feature;
      const name = (feature?.properties?.name || "").toLowerCase();
      const muni = muniMap.get(`${provinceId}::${name}`);
      const fill = isProvRejected ? NO_DATA_COLOR : muni ? (RISK_COLORS[muni.riskLevel] ?? NO_DATA_COLOR) : NO_DATA_COLOR;
      (layer as L.Path).setStyle({ fillColor: fill });
    });

    // Add muni name labels (only for larger features, capped to avoid clutter)
    const labels: L.Marker[] = [];
    let labelCount = 0;
    muniLayer.eachLayer((layer) => {
      if (labelCount >= 60) return;
      const feature = (layer as any).feature as GeoJSON.Feature;
      const name = feature?.properties?.name || "";
      if (!name) return;
      const center = (layer as L.Polygon).getBounds().getCenter();
      labels.push(
        L.marker([center.lat, center.lng], {
          icon: L.divIcon({
            className: "",
            html: `<div style="
              color: white;
              font-size: 9px;
              font-weight: 600;
              text-shadow: 0 1px 3px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.8);
              white-space: nowrap;
              pointer-events: none;
              transform: translate(-50%, -50%);
            ">${name}</div>`,
            iconSize: [0, 0],
            iconAnchor: [0, 0],
          }),
          interactive: false,
        }).addTo(map)
      );
      labelCount++;
    });
    municipalityLabelsRef.current = labels;
  }, [muniMap, clearMunicipalities]);

  const resetZoom = useCallback(() => {
    const map = mapRef.current;
    const provLayer = provincesLayerRef.current;
    if (!map || !provLayer) return;

    isZoomedRef.current = false;
    setIsZoomed(false);

    clearMunicipalities();

    // Re-show pulses only for non-rejected provinces
    provincePulsesRef.current.forEach(({ marker, provinceId }) => {
      if (!rejectedRef.current.has(provinceId)) marker.addTo(map);
    });

    // Reset all province styles + re-enable interactivity
    provLayer.eachLayer((layer) => {
      const feature = (layer as any).feature as GeoJSON.Feature;
      const id = feature?.properties?.id;
      (layer as L.Path).setStyle(getProvinceStyle(id, false, false));
      const el = (layer as L.Path).getElement() as SVGElement | null;
      if (el) el.style.pointerEvents = "";
    });

    map.flyToBounds(L.latLngBounds(CALABARZON_BOUNDS), { padding: [40, 40], duration: 0.8 });
  }, [getProvinceStyle, clearMunicipalities]);

  const zoomToProvince = useCallback((provinceId: string) => {
    const map = mapRef.current;
    const provLayer = provincesLayerRef.current;
    if (!map || !provLayer) return;

    isZoomedRef.current = true;
    setIsZoomed(true);

    // Hide all province pulses (we'll show muni layer instead)
    provincePulsesRef.current.forEach(({ marker }) => map.removeLayer(marker));

    let targetBounds: L.LatLngBounds | null = null;
    provLayer.eachLayer((layer) => {
      const feature = (layer as any).feature as GeoJSON.Feature;
      const id = feature?.properties?.id;
      const isTarget = id === provinceId;
      if (isTarget) {
        targetBounds = (layer as L.Polygon).getBounds();
        (layer as L.Path).setStyle(getProvinceStyle(id, true, false));
        (layer as L.Path).bringToFront();
      } else {
        (layer as L.Path).setStyle(getProvinceStyle(id, false, true));
      }
    });

    if (targetBounds) {
      map.flyToBounds(targetBounds, { padding: [40, 40], duration: 0.8, maxZoom: 11 });
      // Show municipalities after the zoom animation
      setTimeout(() => {
        showMunicipalities(provinceId);
        // Disable interactivity ONLY on the selected province so lake gaps (Taal, Laguna de Bay)
        // don't capture clicks. Other provinces remain clickable so users can switch directly.
        provLayer.eachLayer((layer) => {
          const feature = (layer as any).feature as GeoJSON.Feature;
          const id = feature?.properties?.id;
          const el = (layer as L.Path).getElement() as SVGElement | null;
          if (el) el.style.pointerEvents = id === provinceId ? "none" : "";
        });
      }, 800);
    }
  }, [getProvinceStyle, showMunicipalities]);

  // ─── Init map once ───
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [14.2, 121.4],
      zoom: 8,
      minZoom: 8,
      maxZoom: 12,
      maxBounds: [[12.8, 119.5], [15.8, 123.6]],
      maxBoundsViscosity: 1.0,
      zoomControl: false,
      attributionControl: false,
      boxZoom: false,
    });

    const onZoomIn = () => map.zoomIn();
    const onZoomOut = () => map.zoomOut();
    window.addEventListener("aipheed:zoom-in", onZoomIn);
    window.addEventListener("aipheed:zoom-out", onZoomOut);
    (map as any).__aipheedZoomCleanup = () => {
      window.removeEventListener("aipheed:zoom-in", onZoomIn);
      window.removeEventListener("aipheed:zoom-out", onZoomOut);
    };

    const isLight = () => document.documentElement.classList.contains("light");
    const applyBg = () => {
      map.getContainer().style.background = isLight() ? BASEMAP_PAGE_BG.light : BASEMAP_PAGE_BG.dark;
    };
    applyBg();

    const makeBasemap = () =>
      leafletLayer({
        url: PMTILES_URL,
        flavor: isLight() ? BASEMAP_FLAVOR.light : BASEMAP_FLAVOR.dark,
        attribution: BASEMAP_ATTRIBUTION,
      }) as unknown as L.GridLayer;

    let tile = makeBasemap().addTo(map);
    tile.on("tileerror", () => {
      if ((map as any).__basemapWarned) return;
      (map as any).__basemapWarned = true;
      console.warn(
        `[aiPHeed] Basemap tiles could not load from "${PMTILES_URL}". ` +
          "Province and municipality outlines still render on the themed background. " +
          "See docs/basemap.md to provision the .pmtiles file for this deployment.",
      );
    });

    const onTheme = () => {
      map.removeLayer(tile);
      tile = makeBasemap().addTo(map);
      applyBg();
    };
    window.addEventListener("aipheed:theme", onTheme);
    (map as any).__aipheedThemeCleanup = () => window.removeEventListener("aipheed:theme", onTheme);

    mapRef.current = map;

    // Load provinces (filter to CALABARZON only) + municipalities
    Promise.all([
      fetch("/ph-provinces.json").then((r) => r.json()),
      fetch("/calabarzon-municities.json").then((r) => r.json()),
    ])
      .then(([provincesGeo, muniGeo]: [GeoJSON.FeatureCollection, GeoJSON.FeatureCollection]) => {
        muniGeoDataRef.current = muniGeo;

        // Filter & re-tag provinces: use province name → our id
        const NAME_TO_ID: Record<string, string> = {
          Batangas: "batangas",
          Cavite: "cavite",
          Laguna: "laguna",
          Quezon: "quezon",
          Rizal: "rizal",
        };
        const calabarzonFeatures = provincesGeo.features
          .filter((f) => f.properties?.region_id === "r4a")
          .map((f) => ({
            ...f,
            properties: {
              ...f.properties,
              id: NAME_TO_ID[f.properties?.name] || f.properties?.name,
              name: f.properties?.name,
            },
          }));

        const provLayer = L.geoJSON(
          { type: "FeatureCollection", features: calabarzonFeatures } as GeoJSON.FeatureCollection,
          {
            style: (feature) => {
              const id = feature?.properties?.id;
              const isSel = selectedRef.current?.id === id;
              return getProvinceStyle(id, isSel, false);
            },
            onEachFeature: (feature, layer) => {
              const id = feature.properties?.id;
              const region = id ? regionMapRef.current.get(id) : null;

              layer.on({
                mouseover: (e) => {
                  const l = e.target as L.Path;
                  if (!isZoomedRef.current || selectedRef.current?.id === id) {
                    l.setStyle({ fillOpacity: 0.85, weight: 2.5, color: AMBER });
                    l.bringToFront();
                    if (municipalitiesLayerRef.current) municipalitiesLayerRef.current.bringToFront();
                  }
                  if (region) onHoverRef.current(region);
                },
                mouseout: (e) => {
                  const l = e.target as L.Path;
                  const sel = selectedRef.current?.id === id;
                  const dimmed = isZoomedRef.current && !sel;
                  l.setStyle(getProvinceStyle(id, sel, dimmed));
                  onHoverRef.current(null);
                },
                click: () => {
                  if (!region) return;
                  if (isZoomedRef.current && selectedRef.current?.id === id) return;
                  onClickRef.current(region);
                },
              });

              if (region) {
                const isRej = rejectedRef.current.has(id);
                layer.bindTooltip(
                  isRej
                    ? `<div style="font-size:12px"><strong>${region.name}</strong><br/><span style="color:#9ca3af;font-style:italic">No Forecast Available</span></div>`
                    : `<div style="font-size:12px"><strong>${region.name}</strong><br/>Risk Level: ${region.riskScore.toFixed(2)}<br/><span style="text-transform:capitalize">${region.riskLevel} risk</span></div>`,
                  { sticky: true }
                );
              }
            },
          }
        ).addTo(map);
        provincesLayerRef.current = provLayer;

        provLayer.eachLayer((layer) => {
          const feature = (layer as any).feature as GeoJSON.Feature;
          const id = feature?.properties?.id;
          const region = id ? regionMapRef.current.get(id) : null;
          if (!region) return;
          // (no texture)
          (layer as L.Path).setStyle(getProvinceStyle(id, selectedRef.current?.id === id, false));
        });

        // ── Rest of the Philippines: visible faint outlines, NOT clickable ──
        const restFeatures = provincesGeo.features.filter(
          (f) => f.properties?.region_id !== "r4a"
        );
        L.geoJSON(
          { type: "FeatureCollection", features: restFeatures } as GeoJSON.FeatureCollection,
          {
            interactive: false,
            style: () => ({
              fillColor: "#1a2230",
              fillOpacity: 0.55,
              color: "rgba(255,255,255,0.18)",
              weight: 0.6,
            }),
          }
        ).addTo(map);
        provLayer.bringToFront();

        // ── Pulse markers + permanent labels per province ──
        const provinceLabelOverrides: Record<string, [number, number]> = {
          quezon: [14.05, 121.85], // Quezon's centroid is in the islands; pin to mainland
        };
        const pulses: { marker: L.Marker; provinceId: string }[] = [];
        const labels: L.Marker[] = [];
        provLayer.eachLayer((layer) => {
          const feature = (layer as any).feature as GeoJSON.Feature;
          const id = feature?.properties?.id;
          const region = id ? regionMapRef.current.get(id) : null;
          if (!region || !id) return;
          const isRejected = rejectedRef.current.has(id);

          const override = provinceLabelOverrides[id];
          const bounds = (layer as L.Polygon).getBounds();
          const center = override ? { lat: override[0], lng: override[1] } : bounds.getCenter();
          const color = isRejected ? NO_DATA_COLOR : RISK_COLORS[region.riskLevel];

          if (!isRejected) {
            // Pulse marker (skipped for rejected forecasts)
            const pulseIcon = L.divIcon({
              className: "",
              html: `<div class="pulse-marker" style="--pulse-color: ${color}">
                <div class="pulse-dot" style="background: ${color}"></div>
                <div class="pulse-ring" style="border-color: ${color}"></div>
              </div>`,
              iconSize: [20, 20],
              iconAnchor: [10, 10],
            });
            const pulse = L.marker([center.lat, center.lng], { icon: pulseIcon, interactive: false }).addTo(map);
            pulses.push({ marker: pulse, provinceId: id });
          }

          // Province name label
          const label = L.marker([center.lat, center.lng], {
            icon: L.divIcon({
              className: "",
              html: `<div style="
                color: white;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-shadow: 0 1px 4px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.8);
                white-space: nowrap;
                pointer-events: none;
                transform: translate(-50%, 14px);
                text-transform: uppercase;
              ">${region.name}</div>`,
              iconSize: [0, 0],
              iconAnchor: [0, 0],
            }),
            interactive: false,
          }).addTo(map);
          labels.push(label);
        });
        provincePulsesRef.current = pulses;
        provinceLabelsRef.current = labels;

        // Fit to CALABARZON bounds
        map.fitBounds(L.latLngBounds(CALABARZON_BOUNDS), { padding: [40, 40] });
      })
      .catch(console.error);

    return () => {
      (map as any).__aipheedZoomCleanup?.();
      map.remove();
      mapRef.current = null;
      provincesLayerRef.current = null;
      municipalitiesLayerRef.current = null;
    };
  }, []);

  // Selection changes
  useEffect(() => {
    if (selectedRegion) {
      zoomToProvince(selectedRegion.id);
    } else {
      resetZoom();
    }
  }, [selectedRegion, zoomToProvince, resetZoom]);

  // Re-style municipalities when selection changes
  useEffect(() => {
    if (!municipalitiesLayerRef.current || !selectedRegion) return;
    const provinceId = selectedRegion.id;
    const isProvRejected = rejectedRef.current.has(provinceId);
    municipalitiesLayerRef.current.eachLayer((layer) => {
      const feature = (layer as any).feature as GeoJSON.Feature;
      const name = (feature?.properties?.name || "").toLowerCase();
      const muni = muniMap.get(`${provinceId}::${name}`);
      if (!muni) return;
      const isSel = !isProvRejected && selectedMunicipality?.id === muni.id;
      (layer as L.Path).setStyle({
        fillColor: isProvRejected ? NO_DATA_COLOR : (RISK_COLORS[muni.riskLevel] ?? NO_DATA_COLOR),
        fillOpacity: isSel ? 0.98 : 0.9,
        color: isSel ? AMBER : "rgba(255,255,255,0.35)",
        weight: isSel ? 2.5 : 0.8,
      });
      if (isSel) (layer as L.Path).bringToFront();
    });
  }, [selectedMunicipality, selectedRegion, muniMap, rejectedProvinceIds]);

  // Apply rejection updates: re-style provinces, re-bind tooltips, hide pulses for rejected.
  useEffect(() => {
    const provLayer = provincesLayerRef.current;
    if (!provLayer) return;
    provLayer.eachLayer((layer) => {
      const feature = (layer as any).feature as GeoJSON.Feature;
      const id = feature?.properties?.id;
      if (!id) return;
      const region = regionMapRef.current.get(id);
      if (!region) return;
      const sel = selectedRef.current?.id === id;
      const dimmed = isZoomedRef.current && !sel;
      (layer as L.Path).setStyle(getProvinceStyle(id, sel, dimmed));
      const isRej = rejectedRef.current.has(id);
      (layer as any).unbindTooltip?.();
      (layer as any).bindTooltip(
        isRej
          ? `<div style="font-size:12px"><strong>${region.name}</strong><br/><span style="color:#9ca3af;font-style:italic">No Forecast Available</span></div>`
          : `<div style="font-size:12px"><strong>${region.name}</strong><br/>Risk Level: ${region.riskScore.toFixed(2)}<br/><span style="text-transform:capitalize">${region.riskLevel} risk</span></div>`,
        { sticky: true }
      );
    });
    // Toggle pulse visibility for rejected provinces
    const map = mapRef.current;
    if (map) {
      provincePulsesRef.current.forEach(({ marker, provinceId }) => {
        const el = (marker as any).getElement?.() as HTMLElement | undefined;
        if (el) el.style.display = rejectedRef.current.has(provinceId) ? "none" : "";
      });
    }
  }, [rejectedProvinceIds, getProvinceStyle]);

  return (
    <div className="absolute inset-0 w-full h-full">
      {isZoomed && (
        <button
          onClick={() => onClick(selectedRegion!)}
          className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] bg-card/90 backdrop-blur border rounded-lg px-4 py-2 text-sm font-medium text-foreground hover:bg-secondary transition-colors shadow-lg"
        >
          ← Back to CALABARZON
        </button>
      )}
      <div ref={mapContainerRef} className="w-full h-full [&:focus]:outline-none" />
    </div>
  );
}
