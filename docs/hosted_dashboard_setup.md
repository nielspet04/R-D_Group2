# Hosted dashboard setup (free prototype)

The website is deployed at <https://training-observatory.pages.dev/> and the three Supabase Edge Functions are deployed. GitHub owner sign-in, VSC API connectivity and imports of six runs (46 epochs at the time of import) have been verified. Automatic uploads require this Git version on VSC and the private credential environment described below. No training job is started by setup.

You need one free Supabase project, one free Cloudflare Pages project and GitHub sign-in. Your PC is only needed for this one-time build/deployment. Afterwards the website runs independently; the uploader runs inside actual VSC training allocations. Free Supabase projects can pause after inactivity; local training files/outbox remain recoverable.

## 1. Create the free database/API project

1. Create a Supabase account at <https://supabase.com/dashboard> and a **Free** project, preferably in an EU region. Keep the database password private. Do not enable paid add-ons.
2. In the project's **SQL Editor**, run these repository files in order:
   - `supabase/migrations/202611190001_monitor.sql`
   - `supabase/migrations/202611190002_read.sql`
3. In project settings, copy the project URL and **publishable** `sb_publishable_...` key. Neither is the database password nor the secret/service-role key.
4. In PowerShell, from this repository, run:

```powershell
python scripts/hosted_setup.py --supabase-url https://YOUR_REF.supabase.co --publishable-key YOUR_PUBLISHABLE_KEY
```

This creates ignored public configuration and `results/monitoring/hosted-bootstrap.sql`. Run that SQL in the Supabase SQL Editor to create the research project/source profile. Preserve the generated project ID. Re-running setup reuses it.

## 2. Deploy the three Edge Functions

The Supabase CLI is a pinned project dependency. On Windows PowerShell, run `npm.cmd ci` once from the repository root, then use `npx.cmd supabase` (the bare `supabase` command need not exist). Official installation reference: <https://supabase.com/docs/guides/local-development/cli/getting-started>. Run these commands from the repository root:

```text
npx.cmd supabase login
npx.cmd supabase link --project-ref YOUR_REF
npx.cmd supabase functions deploy monitor-ingest monitor-read monitor-admin --remote=YOUR_REF --use-api --yes
```

The tracked `supabase/config.toml` sets custom upload authentication on `monitor-ingest` and user-JWT verification on the two browser functions. **Do not disable verification on read/admin to fix a login problem.** Ingestion always verifies the custom upload token in code, despite not requiring a browser JWT.

The functions use the provider's server-only `SUPABASE_URL` and service-role/secret key environment. The normal hosted Edge environment supplies the legacy service-role key; the code also supports the new default secret-key configuration. If your project does not provide a usable server key automatically, set it privately in Edge Function secrets under the supported name. Never put it into frontend config or VSC.

Set the function secret `MONITOR_ALLOWED_ORIGINS` to the final exact Pages URL after step 3. It may contain comma-separated exact approved origins; do not use `*`. Keep it in the provider secret settings, not the website bundle.

## 3. Build and publish the website

Install Node dependencies on the laptop (not on VSC):

```powershell
npm.cmd ci
$env:MONITOR_PUBLIC_CONFIG = "configs/monitoring-hosted.local.json"
npm.cmd run build
```

This builds public website files in `dist/`. SQL, Edge Functions, trainers and uploader scripts are ordinary repository files. No ZIP packaging or patch installer is needed.

Without configured public project settings, the website shows an explicit setup message; it does not pretend to be connected.

The existing Cloudflare Pages project uses Direct Upload. To update it, upload the built `dist/` folder through its deployment screen, or run `npx.cmd wrangler pages deploy dist --project-name training-observatory` after signing in to Cloudflare. Keep builds on your development PC; no Node installation is required on VSC. Updating the Git repository does not redeploy this existing Direct Upload website automatically. A switch to Cloudflare Git integration requires a new Pages project; see <https://developers.cloudflare.com/pages/get-started/direct-upload/>.

After the URL is known, set Supabase Edge secret `MONITOR_ALLOWED_ORIGINS=https://YOUR_NAME.pages.dev` and the Supabase Auth site URL to the same origin. Allow that exact sign-in callback URL. No web server runs on VSC for this setup.

## 4. Configure login and owner membership

In Supabase Auth, enable GitHub sign-in following <https://supabase.com/docs/guides/auth/social-sign-in/auth-github>. Create the GitHub OAuth application with the **Supabase Auth callback URL shown by the provider**, not an improvised callback. Store the GitHub client secret only in Supabase Auth settings. No repository-write scope is required for dashboard viewing.

Open the Pages URL, choose **Sign in with GitHub**, and return to the website. Initially you have no research-project membership. The sign-in panel shows your Supabase **user UUID**. Copy that UUID; it is not a secret key.

Repeat the public setup command with the same project settings and the user UUID:

```powershell
python scripts/hosted_setup.py --supabase-url https://YOUR_REF.supabase.co --publishable-key YOUR_PUBLISHABLE_KEY --owner-id YOUR_USER_UUID
```

Run the regenerated `results/monitoring/hosted-bootstrap.sql` in the Supabase SQL Editor. Then refresh the website. Only explicitly authorized project members can read results; being logged into GitHub alone is insufficient. For teammates, add their Supabase user UUIDs to `project_members` with role `viewer` using the owner/admin route or SQL Editor. Do not grant public table access.

## 5. Create the VSC upload credential

In the website expand **Uploader setup → Manage uploader keys**. Select the VSC source profile, use the **Training uploads only** scope and create a key. Download it using **Download key file**. It is shown only at creation. Do not paste it into this chat, a Slurm script or Git.

Transfer that file to your VSC account using your normal private transfer method, to:

```text
~/.config/rd-dashboard/upload-key
```

In the VSC shell:

```bash
mkdir -p ~/.config/rd-dashboard
chmod 700 ~/.config/rd-dashboard
chmod 600 ~/.config/rd-dashboard/upload-key
export RND_MONITOR_API_URL="https://YOUR_REF.supabase.co/functions/v1/monitor-ingest"
export RND_MONITOR_UPLOAD_KEY_FILE="$HOME/.config/rd-dashboard/upload-key"
```

Only the path is exported. The key is read by the uploader from its file. Revoking a key in the website stops future uploads using it but preserves collected results. Keys currently expire after 180 days; issue a replacement privately when needed.

## 6. Update the VSC code and check connectivity

Use Git to update the VSC checkout. Commit or otherwise preserve any local code edits first. The update includes the monitoring hooks and preserves the newer STA2 final-training support. The private key stays at `~/.config/rd-dashboard/upload-key`, outside the repository. Do not reset/clean the checkout or replace training output.

```bash
cd "$VSC_DATA/R-D_Group2"
git switch STA2_dev
git pull --ff-only origin STA2_dev
export RND_MONITOR_API_URL="https://kdhaypsvbagfsxhgzqmu.supabase.co/functions/v1/monitor-ingest"
export RND_MONITOR_UPLOAD_KEY_FILE="$HOME/.config/rd-dashboard/upload-key"
```

Export these two values in the shell used for submission. To reuse them across terminals, put the exports in `~/.config/rd-dashboard/env` and run `source ~/.config/rd-dashboard/env` before submitting. The file contains an endpoint and credential path, never the key itself. Use the root-level `scripts/` from this checkout; the earlier staging copies under `tmp/` are no longer used.

From the updated VSC project directory and its usual Python environment:

```bash
python scripts/monitor_upload.py --probe
```

Success confirms authenticated HTTPS/API/database access **from that shell's host**. Separately check it from an existing permitted compute allocation: login-node connectivity does not prove training-node connectivity. If site rules/network access prohibit outbound publication, retain local monitoring and use one-shot imports from an approved connected environment. Do not bypass restrictions or start a permanent login-node process.

No cloud packages need installing in the GPU environment. The uploader uses Python's standard library.

## 7. Day-to-day use

Export the two non-secret environment values in the shell that submits training (step 5), then use your existing monitored submission command. An unconfigured upload endpoint/key leaves API publication disabled. Variables do not carry over to unrelated terminals.

**The following commands submit real GPU jobs with their existing resource requests**; use them only when you intend to train:

```bash
python scripts/submit_monitored.py --model gru
# Or the existing STA2 full H100 workflow:
python scripts/submit_monitored.py --job-type sta2-full-h100
```

`--dry-run` inspects submission without creating a job. Do not restart a healthy running job solely for monitoring. New code takes effect on new starts/resumes. The actual allocation owns the uploader; no separate interactive monitoring session stays open. Existing numerical/model/checkpoint settings are unchanged.

Open the Pages URL on your phone and sign in. You can now close your PC. The website's **Refresh view** reads stored cloud observations; it does not force a VSC update or execute scheduler commands.

Job-reported progress and actual scheduler state are separate. A missing final heartbeat is not proof of success or failure. `training_finished` means Python training finished; baseline evaluation and the batch script may still fail later. A batch-exit report is still not a confirmed Slurm accounting result.

## 8. Existing histories, retries and scheduler snapshots

Historical import default is dry run:

```bash
python scripts/monitor_backfill.py --run-dir results/sta2/YOUR_RUN
python scripts/monitor_backfill.py --run-dir results/sta2/YOUR_RUN --upload
```

Import leaves the original history unchanged, stores a persistent mapping under the run's `monitoring/api-import/`, and reuses event IDs on repeat imports. Imported scientific histories are represented by a distinct historical attempt. A history file alone never invents a successful job completion. Import old histories before relying on a new cloud resume chain; a parent attempt absent from the cloud must be uploaded/reconciled first. Legacy imported attempts are deliberately not guessed to be a new runtime attempt's parent.

New attempts store event files under `monitoring/attempts/ATTEMPT/api-outbox/`. On retry the server deduplicates committed IDs. To replay a stopped uploader:

```bash
python scripts/monitor_upload.py --attempt-dir PATH_TO_ATTEMPT --dry-run
python scripts/monitor_upload.py --attempt-dir PATH_TO_ATTEMPT --once
```

The background worker retries network/server/rate-limit failures with backoff. Authentication failures retain the outbox and stop requests; replace/reconfigure credentials and replay. Permanently rejected batches are retained under `api-quarantine/` for diagnosis; do not delete them as a repair. `upload-status.json` contains only sanitized status, not the key.

For an explicit one-shot Slurm snapshot, create a **Scheduler snapshots only** key and use its separate credential file:

```bash
RND_MONITOR_UPLOAD_KEY_FILE="$HOME/.config/rd-dashboard/scheduler-key" \
  python scripts/monitor_scheduler_upload.py --root "$PWD"
```

This queries Slurm only when explicitly run. It does not install cron or create a recurring session. A complete continuously fresh queue is not promised without a separately approved scheduler collector.

## 9. Free-plan maintenance, exports and rollback

The static page stays independently hosted, but free Supabase projects may pause for inactivity. Resume through the provider dashboard, then replay local outboxes. Do not generate fake keepalive traffic. Check provider usage/quotas before importing large histories or adding many viewers.

Scientific epochs/configurations remain stored until explicit cleanup. Raw telemetry can be reduced to hourly aggregates after 7 days, keeping aggregates for 30 days. Run this owner maintenance SQL periodically, or schedule it through Supabase's supported database Cron UI:

```sql
select public.monitor_retention();
```

No cron task is installed automatically. Keep a database export before migrations and periodically during research: use the provider's documented CLI/export procedure (<https://supabase.com/docs/guides/platform/backups>). Do not assume automatic backups are included in Free. Native training histories/checkpoints remain on VSC. Comparisons have a CSV export; the authenticated read API also exposes a portable full experiment JSON export.

To roll back, unset `RND_MONITOR_API_URL` and keep the existing local dashboard workflow. Preserve unsent event files. Restore compatible previous static/function code; do not drop scientific database rows or delete training output. The PC-independent phone view still depends on available provider service and an uploaded source observation; it cannot retrieve data that was never sent.

## 10. Verification record

Run checks from the repository root:

```powershell
python -m unittest discover -s tests -p 'test_monitor*.py' -v
npm.cmd run test:db
npm.cmd run test:contracts
deno check supabase/functions/monitor-ingest/index.ts supabase/functions/monitor-read/index.ts supabase/functions/monitor-admin/index.ts
npm.cmd run build
```

Database tests use an embedded real Postgres engine (PGlite) and exercise SQL migrations, row-level policies, denied writes, idempotency, transactional rollback, resume projection and revocation. They simulate the request's user UUID; they do not verify hosted Supabase JWT gateway/OAuth behavior. Python tests cover durable files, contract parity, failure isolation and replay; Deno validates the shared Edge contracts/types.

Live GitHub owner sign-in, the VSC shell connectivity probe and six historical imports are verified. Compute-node outbound connectivity, free-project pause/restore and actual GPU/DDP integration remain unverified. PyTorch is absent from the local development shell. No real VSC job is submitted by the build/test commands above.
