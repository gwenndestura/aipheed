# Basemap — self-hosted, no API key

The map tiles are a single **Protomaps `.pmtiles`** file built from OpenStreetMap
data (ODbL). It is rendered client-side by `protomaps-leaflet` (see
`src/components/PhilippineMap.tsx`). There is **no API key and no third-party
runtime request** — the file is served from our own origin, so the map keeps
working without a vendor relationship or a free-tier that can change terms.

This replaced the previous CARTO basemap, whose CDN now requires an API key and
stamps unauthenticated tiles with an "API KEY REQUIRED" watermark.

## Where the file lives

`PhilippineMap.tsx` loads it from:

```
import.meta.env.VITE_BASEMAP_PMTILES_URL   // e.g. https://cdn.example.gov.ph/basemap.pmtiles
```

falling back to **`/basemap.pmtiles`** (i.e. `public/basemap.pmtiles`).

- **Dev / small deploys:** drop the file at `public/basemap.pmtiles`. A
  Philippines-clipped extract is a few MB and is fine to serve statically.
- **Production:** host it on the agency CDN / object store and set
  `VITE_BASEMAP_PMTILES_URL` at build time.

`public/basemap.pmtiles` is git-ignored (binary build artifact). Provision it in
the deploy pipeline, or commit it deliberately if reproducibility matters more
than repo size.

If the file is missing the map still renders province/municipality outlines on
the themed background, and logs a one-line `console.warn`.

## Generate the Philippines extract

Requires the [`pmtiles` CLI](https://github.com/protomaps/go-pmtiles/releases)
(single binary, no install).

```sh
# Clip the daily global Protomaps build to a wide Philippines bounding box.
# bbox: minLon,minLat,maxLon,maxLat
pmtiles extract \
  https://build.protomaps.com/20260901.pmtiles \
  public/basemap.pmtiles \
  --bbox=114,4,129,22 \
  --maxzoom=12
```

- This is the extent currently checked in: ~84 MB, covers the whole Philippines
  and surrounding seas so no tile-boundary is ever visible at the map's zoom
  range (`minZoom: 8`, `maxZoom: 12` in `PhilippineMap.tsx`).
- Use a recent date from the [build index](https://build.protomaps.com/) — old
  daily builds are pruned.
- `--maxzoom=12` matches the map's `maxZoom`. z13 roughly doubles the file for
  street-name detail the forecast map doesn't need.
- If the map's `minZoom` is ever lowered, widen the bbox to match (a smaller
  extract would show its rectangular edge when zoomed out).
- The map renders the Protomaps **`black`** flavor in dark mode and **`white`**
  in light mode (near-monochrome, no cyan water).

To self-host the *whole* pipeline instead of pulling from `build.protomaps.com`,
run [`planetiler`](https://github.com/onthegomap/planetiler) against a
Geofabrik Philippines `.osm.pbf` and convert its MBTiles output with
`pmtiles convert`.

## Preferred long-term option: NAMRIA Philippine Geoportal

For a production government site the most institutionally correct basemap is
**NAMRIA's Philippine Geoportal** (<https://www.geoportal.gov.ph/>), the
government's own authoritative geospatial service. Adopting it needs the team to
confirm the tile endpoint, CORS policy, and usage terms with NAMRIA, and to
coordinate access. Until that is arranged, the self-hosted Protomaps file above
is the no-dependency default.
