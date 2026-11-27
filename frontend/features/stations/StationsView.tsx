"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, Link2, Loader2, MapPin, Pencil, Trash2, Unlink, XCircle } from "lucide-react";
import clsx from "clsx";

import { Card } from "@/components/Card";
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { stationsApi } from "@/lib/stations";
import type { Country, MonitoringStation, TideStation } from "@/lib/types";

const UNITS = ["meters", "feet", "centimeters"];

export function StationsView() {
  const [countries, setCountries] = useState<Country[]>([]);
  const [stations, setStations] = useState<MonitoringStation[]>([]);
  const [tideStations, setTideStations] = useState<TideStation[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [linked, setLinked] = useState<TideStation[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editingStationId, setEditingStationId] = useState<number | null>(null);

  const [msForm, setMsForm] = useState({
    code: "",
    name: "",
    country_iso: "PH",
    region: "",
    latitude: "",
    longitude: "",
    timezone: "Asia/Manila",
    vertical_datum: "",
    description: "",
  });
  const [tsForm, setTsForm] = useState({
    code: "",
    name: "",
    country_iso: "PH",
    region: "",
    latitude: "",
    longitude: "",
    timezone: "Asia/Manila",
    data_source: "NAMRIA_CSV",
    measurement_unit: "meters",
    vertical_datum: "",
  });

  const selected = stations.find((s) => s.id === selectedId) ?? null;
  const unlinkableIds = new Set(linked.map((t) => t.id));
  const linkableTideStations = tideStations.filter((t) => !unlinkableIds.has(t.id));
  const [linkTarget, setLinkTarget] = useState<number | "">("");

  const loadAll = useCallback(async () => {
    try {
      const [c, ms, ts] = await Promise.all([
        stationsApi.listCountries(),
        stationsApi.listMonitoringStations(),
        stationsApi.listTideStations(),
      ]);
      setCountries(c);
      setStations(ms);
      setTideStations(ts);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load stations");
    }
  }, []);

  const loadLinked = useCallback(async (stationId: number) => {
    try {
      setLinked(await stationsApi.listLinked(stationId));
    } catch {
      setLinked([]);
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (selectedId) {
      void loadLinked(selectedId);
      setLinkTarget("");
    } else {
      setLinked([]);
    }
  }, [selectedId, loadLinked]);

  const report = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice(null), 4000);
  };

  const resetStationForm = () => {
    setMsForm({
      code: "",
      name: "",
      country_iso: "PH",
      region: "",
      latitude: "",
      longitude: "",
      timezone: "Asia/Manila",
      vertical_datum: "",
      description: "",
    });
    setEditingStationId(null);
  };

  const beginEditStation = (station: MonitoringStation) => {
    setMsForm({
      code: station.code,
      name: station.name,
      country_iso: station.country_iso,
      region: station.region ?? "",
      latitude: String(station.latitude),
      longitude: String(station.longitude),
      timezone: station.timezone,
      vertical_datum: station.vertical_datum ?? "",
      description: station.description ?? "",
    });
    setEditingStationId(station.id);
    setError(null);
    if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const removeStation = async (station: MonitoringStation) => {
    if (
      !window.confirm(
        `Remove monitoring station ${station.code}? This also deletes its cameras, shoreline observations, baselines and anomaly events, and clears tide-station links. This cannot be undone.`,
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await stationsApi.deleteMonitoringStation(station.id);
      if (selectedId === station.id) setSelectedId(null);
      if (editingStationId === station.id) resetStationForm();
      report(`Monitoring station ${station.code} removed.`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove monitoring station");
    } finally {
      setBusy(false);
    }
  };

  const saveMonitoringStation = async () => {
    setBusy(true);
    setError(null);
    try {
      const lat = Number(msForm.latitude);
      const lng = Number(msForm.longitude);
      if (!msForm.code.trim() || !msForm.name.trim() || Number.isNaN(lat) || Number.isNaN(lng)) {
        throw new Error("Code, name and numeric latitude/longitude are required.");
      }
      const payload = {
        code: msForm.code.trim(),
        name: msForm.name.trim(),
        country_iso: msForm.country_iso,
        region: msForm.region.trim() || null,
        latitude: lat,
        longitude: lng,
        timezone: msForm.timezone.trim() || "Asia/Manila",
        vertical_datum: msForm.vertical_datum.trim() || null,
        description: msForm.description.trim() || null,
      };
      if (editingStationId) {
        await stationsApi.updateMonitoringStation(editingStationId, payload);
        report(`Monitoring station ${payload.code} updated.`);
      } else {
        await stationsApi.createMonitoringStation(payload);
        report("Monitoring station created.");
      }
      resetStationForm();
      await loadAll();
      if (selectedId) await loadLinked(selectedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save station");
    } finally {
      setBusy(false);
    }
  };

  const createTideStation = async () => {
    setBusy(true);
    setError(null);
    try {
      const lat = Number(tsForm.latitude);
      const lng = Number(tsForm.longitude);
      if (!tsForm.code.trim() || !tsForm.name.trim() || Number.isNaN(lat) || Number.isNaN(lng)) {
        throw new Error("Code, name and numeric latitude/longitude are required.");
      }
      await stationsApi.createTideStation({
        code: tsForm.code.trim(),
        name: tsForm.name.trim(),
        country_iso: tsForm.country_iso,
        region: tsForm.region.trim() || null,
        latitude: lat,
        longitude: lng,
        timezone: tsForm.timezone.trim() || "Asia/Manila",
        data_source: tsForm.data_source.trim() || "NAMRIA_CSV",
        measurement_unit: tsForm.measurement_unit,
        vertical_datum: tsForm.vertical_datum.trim() || null,
        monitoring_station_id: selectedId,
      });
      setTsForm({
        code: "",
        name: "",
        country_iso: "PH",
        region: "",
        latitude: "",
        longitude: "",
        timezone: "Asia/Manila",
        data_source: "NAMRIA_CSV",
        measurement_unit: "meters",
        vertical_datum: "",
      });
      report("Tide station created.");
      await loadAll();
      if (selectedId) await loadLinked(selectedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create tide station");
    } finally {
      setBusy(false);
    }
  };

  const link = async () => {
    if (linkTarget === "" || !selectedId) return;
    setBusy(true);
    try {
      await stationsApi.linkTideStation(selectedId, Number(linkTarget));
      setLinkTarget("");
      await loadAll();
      await loadLinked(selectedId);
      report("Tide station associated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to associate tide station");
    } finally {
      setBusy(false);
    }
  };

  const unlink = async (tideStationId: number) => {
    if (!selectedId) return;
    setBusy(true);
    try {
      await stationsApi.unlinkTideStation(selectedId, tideStationId);
      await loadAll();
      await loadLinked(selectedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disassociate tide station");
    } finally {
      setBusy(false);
    }
  };

  const countryOptions = countries.map((c) => (
    <option key={c.iso_code} value={c.iso_code}>
      {c.iso_code} · {c.name}
    </option>
  ));

  const input = (
    label: string,
    value: string,
    onChange: (v: string) => void,
    opts: { placeholder?: string; type?: string } = {},
  ) => (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500">{label}</span>
      <input
        className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
        placeholder={opts.placeholder}
        type={opts.type ?? "text"}
        step={opts.type === "number" ? "any" : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Monitoring stations"
        description="Coastal sites, their associated tide stations, countries and per-station units and datums."
      />

      {error && (
        <Card className="border-red-200 bg-red-50 text-sm text-red-700">
          <div className="flex items-start gap-2">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {error}
          </div>
        </Card>
      )}
      {notice && (
        <Card className="border-emerald-200 bg-emerald-50 text-sm text-emerald-700">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            {notice}
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        {/* Monitoring stations list */}
        <Card title="Monitoring stations" subtitle="Click a station to view and manage its tide-station associations">
          {stations.length === 0 ? (
            <p className="text-sm text-slate-400">No monitoring stations yet — create one on the right.</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {stations.map((station) => (
                <li key={station.id}>
                  <button
                    onClick={() => setSelectedId(station.id)}
                    className={clsx(
                      "flex w-full items-center justify-between gap-3 rounded-md px-2 py-3 text-left transition-colors hover:bg-slate-50",
                      selectedId === station.id && "bg-reef/5 ring-1 ring-reef",
                    )}
                  >
                    <span>
                      <span className="block text-sm font-medium text-slate-800">
                        <MapPin className="mr-1 inline h-4 w-4 text-reef" aria-hidden />
                        {station.code} · {station.name}
                      </span>
                      <span className="mt-0.5 block pl-5 text-xs text-slate-500">
                        {station.country_name} · {station.region ?? "region n/a"} ·{" "}
                        {station.timezone} · {station.latitude.toFixed(4)}, {station.longitude.toFixed(4)}
                      </span>
                    </span>
                    <StatusBadge tone={station.tide_station_count > 0 ? "ok" : "neutral"}>
                      {station.tide_station_count} tide station(s)
                    </StatusBadge>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        {/* Create or edit monitoring station */}
        <Card
          title={
            editingStationId
              ? `Edit monitoring station #${editingStationId}`
              : "Create monitoring station"
          }
        >
          <div className="grid grid-cols-2 gap-3">
            {input("Code", msForm.code, (v) => setMsForm({ ...msForm, code: v }), { placeholder: "e.g. PH-MNL-001" })}
            {input("Name", msForm.name, (v) => setMsForm({ ...msForm, name: v }))}
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Country</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={msForm.country_iso}
                onChange={(e) => setMsForm({ ...msForm, country_iso: e.target.value })}
              >
                {countryOptions}
              </select>
            </label>
            {input("Region", msForm.region, (v) => setMsForm({ ...msForm, region: v }), { placeholder: "e.g. NCR" })}
            {input("Latitude", msForm.latitude, (v) => setMsForm({ ...msForm, latitude: v }), { type: "number", placeholder: "14.5995" })}
            {input("Longitude", msForm.longitude, (v) => setMsForm({ ...msForm, longitude: v }), { type: "number", placeholder: "120.9842" })}
            {input("Time zone", msForm.timezone, (v) => setMsForm({ ...msForm, timezone: v }))}
            {input("Vertical datum", msForm.vertical_datum, (v) => setMsForm({ ...msForm, vertical_datum: v }), { placeholder: "e.g. MLWS" })}
            <div className="col-span-2">{input("Description", msForm.description, (v) => setMsForm({ ...msForm, description: v }))}</div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => void saveMonitoringStation()}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {editingStationId ? "Save station changes" : "Create monitoring station"}
            </button>
            {editingStationId && (
              <button
                onClick={resetStationForm}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40"
              >
                Cancel edit
              </button>
            )}
          </div>
        </Card>
      </div>

      {/* Tide station creation */}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Tide stations" subtitle="Datums are kept per station — never blended across stations">
          {tideStations.length === 0 ? (
            <p className="text-sm text-slate-400">No tide stations yet — create one on the right.</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {tideStations.map((station) => (
                <li key={station.id} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
                  <span className="text-slate-700">
                    <span className="mr-2 text-xs text-slate-400">{station.code}</span>
                    {station.name}
                  </span>
                  <span className="flex flex-wrap gap-2 text-xs text-slate-500">
                    <span className="rounded bg-slate-100 px-2 py-0.5">{station.measurement_unit}</span>
                    <span className="rounded bg-slate-100 px-2 py-0.5">{station.vertical_datum ?? "datum n/a"}</span>
                    <span className="rounded bg-slate-100 px-2 py-0.5">{station.data_source}</span>
                    {station.monitoring_station_id && (
                      <StatusBadge tone="ok">linked</StatusBadge>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Create tide station" subtitle="NAMRIA or other sources — each keeps its own datum and unit">
          <div className="grid grid-cols-2 gap-3">
            {input("Code", tsForm.code, (v) => setTsForm({ ...tsForm, code: v }), { placeholder: "e.g. PH-NAMRIA-001" })}
            {input("Name", tsForm.name, (v) => setTsForm({ ...tsForm, name: v }))}
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Country</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={tsForm.country_iso}
                onChange={(e) => setTsForm({ ...tsForm, country_iso: e.target.value })}
              >
                {countryOptions}
              </select>
            </label>
            {input("Region", tsForm.region, (v) => setTsForm({ ...tsForm, region: v }))}
            {input("Latitude", tsForm.latitude, (v) => setTsForm({ ...tsForm, latitude: v }), { type: "number", placeholder: "14.5895" })}
            {input("Longitude", tsForm.longitude, (v) => setTsForm({ ...tsForm, longitude: v }), { type: "number", placeholder: "120.9842" })}
            {input("Time zone", tsForm.timezone, (v) => setTsForm({ ...tsForm, timezone: v }))}
            {input("Data source", tsForm.data_source, (v) => setTsForm({ ...tsForm, data_source: v }))}
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-500">Unit</span>
              <select
                className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm"
                value={tsForm.measurement_unit}
                onChange={(e) => setTsForm({ ...tsForm, measurement_unit: e.target.value })}
              >
                {UNITS.map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            </label>
            {input("Vertical datum", tsForm.vertical_datum, (v) => setTsForm({ ...tsForm, vertical_datum: v }), { placeholder: "e.g. MLWS" })}
          </div>
          <button
            onClick={() => void createTideStation()}
            disabled={busy}
            className="mt-3 inline-flex items-center gap-2 rounded-md bg-deep px-4 py-2 text-sm font-medium text-white hover:bg-lagoon disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Create tide station
          </button>
        </Card>
      </div>

      {/* Associations for the selected monitoring station */}
      {selected && (
        <Card
          title={`Tide-station associations — ${selected.code}`}
          subtitle="Associating never substitutes datums or stations — the tide station keeps its own datum and unit"
        >
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <button
              onClick={() => beginEditStation(selected)}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
            >
              <Pencil className="h-4 w-4" aria-hidden />
              Edit station
            </button>
            <button
              onClick={() => void removeStation(selected)}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md border border-red-200 px-3 py-2 text-sm text-red-600 hover:bg-red-50 disabled:opacity-40"
            >
              <Trash2 className="h-4 w-4" aria-hidden />
              Remove station
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <select
              className="rounded-md border border-slate-200 px-3 py-2 text-sm"
              value={linkTarget}
              onChange={(e) => setLinkTarget(e.target.value ? Number(e.target.value) : "")}
            >
              <option value="">— choose a tide station to associate —</option>
              {linkableTideStations.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.code} · {t.name}
                </option>
              ))}
            </select>
            <button
              onClick={() => void link()}
              disabled={busy || linkTarget === ""}
              className="inline-flex items-center gap-2 rounded-md border border-reef px-3 py-2 text-sm font-medium text-reef hover:bg-reef hover:text-white disabled:opacity-40"
            >
              <Link2 className="h-4 w-4" aria-hidden />
              Associate
            </button>
          </div>

          {linked.length === 0 ? (
            <p className="mt-3 text-sm text-slate-400">No tide stations associated yet.</p>
          ) : (
            <ul className="mt-3 divide-y divide-slate-100 text-sm">
              {linked.map((station) => (
                <li key={station.id} className="flex items-center justify-between gap-3 py-2">
                  <span className="text-slate-700">
                    {station.code} · {station.name}{" "}
                    <span className="text-xs text-slate-400">({station.vertical_datum ?? "datum n/a"}, {station.measurement_unit})</span>
                  </span>
                  <button
                    onClick={() => void unlink(station.id)}
                    disabled={busy}
                    className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-red-200 hover:text-red-600 disabled:opacity-40"
                  >
                    <Unlink className="h-3.5 w-3.5" aria-hidden />
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}
    </div>
  );
}
