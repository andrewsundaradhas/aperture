"use client";

import { useEffect, useState } from "react";

import { ApiKey, Org, Robot, api, waitForJob } from "@/lib/api";
import { ActionButton, CardSkeleton, ErrorNote, PageHeader, SectionHead } from "@/components/ui";

function when(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

export default function SettingsPage() {
  const [org, setOrg] = useState<Org | null>(null);
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [robots, setRobots] = useState<Robot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  const [keyName, setKeyName] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [embodiment, setEmbodiment] = useState("");
  const [policy, setPolicy] = useState("");
  const [classifyNote, setClassifyNote] = useState<string | null>(null);

  async function load() {
    try {
      const [o, k, r] = await Promise.all([api.me(), api.listKeys(), api.listRobots()]);
      setOrg(o);
      setKeys(k);
      setRobots(r);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    load();
  }, []);

  async function act(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  const createKey = () =>
    act("create", async () => {
      const r = await api.createKey(keyName.trim() || "default");
      setNewKey(r.api_key);
      setKeyName("");
      setKeys(await api.listKeys());
    });

  const revoke = (k: ApiKey) => {
    if (!window.confirm(`Revoke key ${k.prefix}… (${k.name})? Anything using it stops working immediately.`)) return;
    act(`revoke-${k.id}`, async () => {
      await api.revokeKey(k.id);
      setKeys(await api.listKeys());
    });
  };

  const register = () =>
    act("robot", async () => {
      await api.registerRobot(embodiment.trim(), policy.trim());
      setEmbodiment("");
      setPolicy("");
      setRobots(await api.listRobots());
    });

  const reclassify = (all: boolean) =>
    act(all ? "reclassify" : "classify", async () => {
      setClassifyNote(null);
      const job = await waitForJob((await api.classifyPending(all)).job_id, { timeoutMs: 60 * 60 * 1000, intervalMs: 1500 });
      setClassifyNote(`${job.result?.classified ?? 0} failed episode(s) classified.`);
    });

  if (loading) return <CardSkeleton />;

  return (
    <div className="space-y-10">
      <PageHeader title="Settings" subtitle="Your organization, its API keys, and its robots." />
      <ErrorNote error={error} />

      {org && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="card p-5">
            <div className="cinetype text-[11px] text-slate-smoke">ORGANIZATION</div>
            <div className="mt-2 text-heading-sm font-medium text-forest-ink">{org.name}</div>
          </div>
          <div className="card p-5">
            <div className="cinetype text-[11px] text-slate-smoke">SLUG</div>
            <div className="muoto mt-2 text-body text-forest-ink">{org.slug}</div>
          </div>
          <div className="card p-5">
            <div className="cinetype text-[11px] text-slate-smoke">PLAN</div>
            <div className="mt-2 text-body text-forest-ink">{org.tier}</div>
          </div>
        </div>
      )}

      <section className="space-y-5">
        <SectionHead left="credentials" right="stored as hashes" title="API keys" />
        <p className="max-w-2xl text-body text-slate-smoke">
          Keys authenticate uploads from your robots and pipelines, and sign-in to this dashboard. A
          key is shown once, when it is created. To rotate: create a new key, move traffic to it, then
          revoke the old one.
        </p>

        {newKey && (
          <div role="status" className="card space-y-2 p-5">
            <p className="text-body font-medium text-forest-ink">New key — copy it now, it will not be shown again.</p>
            <div className="flex flex-wrap items-center gap-2">
              <code className="muoto break-all rounded-tag bg-bone-white px-2 py-1 text-caption text-forest-ink">{newKey}</code>
              <button className="btn py-1" onClick={() => navigator.clipboard?.writeText(newKey)}>
                Copy
              </button>
              <button className="btn-ghost py-1" onClick={() => setNewKey(null)}>
                Done
              </button>
            </div>
          </div>
        )}

        <div className="card overflow-x-auto">
          <table className="w-full text-left text-body">
            <thead className="muoto text-caption text-slate-smoke">
              <tr>
                <th className="px-5 py-3 font-normal">Key</th>
                <th className="px-5 py-3 font-normal">Name</th>
                <th className="px-5 py-3 font-normal">Created</th>
                <th className="px-5 py-3 font-normal">Last used</th>
                <th className="px-5 py-3 font-normal" />
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k.id} className="border-t border-lichen">
                  <td className="muoto px-5 py-3 text-forest-ink">{k.prefix}…</td>
                  <td className="px-5 py-3 text-forest-ink">{k.name}</td>
                  <td className="px-5 py-3 text-slate-smoke">{when(k.created_at)}</td>
                  <td className="px-5 py-3 text-slate-smoke">{when(k.last_used_at)}</td>
                  <td className="px-5 py-3 text-right">
                    {k.revoked_at ? (
                      <span className="tag">revoked</span>
                    ) : (
                      <ActionButton onClick={() => revoke(k)} busy={busy === `revoke-${k.id}`} variant="btn-ghost">
                        Revoke
                      </ActionButton>
                    )}
                  </td>
                </tr>
              ))}
              {keys.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-5 py-6 text-center text-slate-smoke">
                    No issued keys. (A key configured in the server environment is not listed here.)
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            createKey();
          }}
        >
          <label className="flex flex-col gap-1">
            <span className="cinetype text-[11px] text-slate-smoke">NAME</span>
            <input className="field" placeholder="e.g. fleet-uploader" value={keyName} onChange={(e) => setKeyName(e.target.value)} maxLength={100} />
          </label>
          <button type="submit" className="btn" disabled={busy === "create"}>
            Create key
          </button>
        </form>
      </section>

      <section className="space-y-5">
        <SectionHead left="fleet" right={`${robots.length} registered`} title="Robots" />
        <p className="max-w-2xl text-body text-slate-smoke">
          Robots are also created automatically from uploads (one per embodiment + policy pair). Register
          one here to set it up before its first upload.
        </p>
        <div className="card overflow-x-auto">
          <table className="w-full text-left text-body">
            <thead className="muoto text-caption text-slate-smoke">
              <tr>
                <th className="px-5 py-3 font-normal">Embodiment</th>
                <th className="px-5 py-3 font-normal">Policy</th>
                <th className="px-5 py-3 font-normal">ID</th>
              </tr>
            </thead>
            <tbody>
              {robots.map((r) => (
                <tr key={r.id} className="border-t border-lichen">
                  <td className="px-5 py-3 text-forest-ink">{r.embodiment_type}</td>
                  <td className="px-5 py-3 text-forest-ink">{r.policy_name}</td>
                  <td className="muoto px-5 py-3 text-caption text-slate-smoke">{r.id}</td>
                </tr>
              ))}
              {robots.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-5 py-6 text-center text-slate-smoke">
                    No robots yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            register();
          }}
        >
          <label className="flex flex-col gap-1">
            <span className="cinetype text-[11px] text-slate-smoke">EMBODIMENT</span>
            <input className="field" placeholder="e.g. franka_panda" value={embodiment} onChange={(e) => setEmbodiment(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="cinetype text-[11px] text-slate-smoke">POLICY</span>
            <input className="field" placeholder="e.g. openvla-7b-ft" value={policy} onChange={(e) => setPolicy(e.target.value)} />
          </label>
          <button type="submit" className="btn" disabled={busy === "robot" || !embodiment.trim() || !policy.trim()}>
            Register robot
          </button>
        </form>
      </section>

      <section className="space-y-5">
        <SectionHead left="evaluation" right="background job" title="Classification" />
        <p className="max-w-2xl text-body text-slate-smoke">
          New failures are classified automatically on upload. Classify anything left over, or redo every
          verdict after a classifier upgrade or after turning on the learned models.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <ActionButton onClick={() => reclassify(false)} busy={busy === "classify"}>
            Classify pending
          </ActionButton>
          <ActionButton onClick={() => reclassify(true)} busy={busy === "reclassify"} variant="btn-ghost">
            Reclassify all
          </ActionButton>
          {classifyNote && <span className="text-caption text-slate-smoke">{classifyNote}</span>}
        </div>
      </section>
    </div>
  );
}
