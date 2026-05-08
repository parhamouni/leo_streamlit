"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function DashboardPage() {
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    supabase()
      .auth.getSession()
      .then(({ data }) => {
        if (!active) return;
        if (!data.session) {
          router.replace("/login");
          return;
        }
        setEmail(data.session.user.email ?? null);
        setUserId(data.session.user.id);
        setReady(true);
      });
    return () => {
      active = false;
    };
  }, [router]);

  async function onLogout() {
    await supabase().auth.signOut();
    router.replace("/login");
  }

  if (!ready) {
    return (
      <main className="min-h-screen flex items-center justify-center text-gray-500">
        Loading…
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow p-8 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Dashboard</h1>
          <button
            onClick={onLogout}
            className="text-sm text-gray-700 hover:underline"
          >
            Sign out
          </button>
        </div>

        <dl className="text-sm space-y-1">
          <div className="flex gap-2">
            <dt className="text-gray-500 w-24">Email</dt>
            <dd>{email}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="text-gray-500 w-24">User ID</dt>
            <dd className="font-mono text-xs">{userId}</dd>
          </div>
        </dl>

        <p className="text-sm text-gray-500">
          Document list, upload, and job history will live here. (Checkpoints 6.1+)
        </p>
      </div>
    </main>
  );
}
