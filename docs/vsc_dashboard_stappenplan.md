# VSC Training Observatory — stappenplan

Dit dashboard draait op je laptop: één webpagina en één lokale SQLite-database.
Een kort Python-script leest via SSH je eigen Slurm-jobs, runmetadata, epochresultaten
en de laatste 16 KB log per gemonitorde run. Er draait geen webserver op VSC.
De training zelf schrijft alleen lokale bestanden op het gedeelde VSC-bestandssysteem.

Dit is een zelfstandige implementatie, zonder MLflow-installatie. Het geeft direct
de gezamenlijke job- en trainingsweergave uit ons gesprek. MLflow kan later toegevoegd
worden voor uitgebreid experimentbeheer; deze database is geen MLflow-database.

## 1. Probeer het lokaal met demodata

Open PowerShell op je laptop:

```powershell
cd C:\Users\Emiel\Documents\RnD\R-D_Group2
python scripts/monitor_demo.py
python scripts/dashboard.py --snapshot results/monitoring/demo.json --db results/monitoring/demo.sqlite3
```

Open <http://127.0.0.1:8790>. Je ziet synthetische jobs en curves.
Klik op een job, zoek op model/jobnummer en probeer de statusfilter.
De demo dient alleen om de interface te bekijken; er worden geen jobs gestart.
Stop de server met Ctrl+C voordat je de echte versie op dezelfde poort start.
Gebruik de aparte demo-database zodat demojobs niet tussen je echte jobs komen.

Python 3.9 of nieuwer is voldoende voor dashboard, collector en submission wrapper.
Hiervoor hoef je geen packages te installeren. Training gebruikt je bestaande
`rd_epic_cuda`-omgeving met PyTorch en de projectdependencies.

## 2. Zet de gewijzigde code op VSC

Gebruik je bestaande upload- of Git-workflow om deze bestanden naar dezelfde checkout
op VSC te brengen:

```text
src/run_monitor.py
src/sta_trainer.py
scripts/train_sta.py
scripts/submit_monitored.py
scripts/monitor_snapshot.py
slurm/sta_train_p100.slurm
```

Controleer de verschillen voordat je bestaande VSC-bestanden vervangt. In het
bestaande P100-script is alleen de toewijzing van `OUTPUT_ROOT` aangepast naar
`${RND_OUTPUT_ROOT:-results}`; de trainingsinstellingen blijven uit dat script komen.
Het bestaande script verwacht de checkout op `$VSC_DATA/R-D_Group2`.

Log via je normale methode in op de wICE-loginomgeving. Voer daar uit:

```bash
cd "$VSC_DATA/R-D_Group2"
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
test -x "$PYTHON"
"$PYTHON" --version
test -f src/run_monitor.py
test -f scripts/submit_monitored.py
grep RND_OUTPUT_ROOT slurm/sta_train_p100.slurm
printf '%s\n' "$VSC_DATA/R-D_Group2"
printf '%s\n' "$PYTHON"
```

Bewaar de laatste twee paden: de **absolute projectdirectory** en het absolute
Python-pad. Deze heb je straks op je laptop nodig. Controleer ook dat de bestaande
featurebestanden, annotaties en Conda-omgeving beschikbaar zijn; het dashboard
maakt deze niet aan. De wrapper gebruikt de bestaande account/partition/tijdlimiet
van `sta_train_p100.slurm`: `lp_edu_rdlab`, wICE `gpu_p100`, één GPU, vier uur.

## 3. Controleer SSH vanaf je laptop

Gebruik het account en de loginhost waarmee je al op VSC werkt. Hieronder is
`vsc-wice` een **voorbeeldalias**, geen vooraf ingesteld account.
Je kunt in je bestaande SSH-config (`C:\Users\Emiel\.ssh\config`) bijvoorbeeld
een blok toevoegen, met jouw eigen waarden:

```sshconfig
Host vsc-wice
    HostName YOUR_WICE_LOGIN_HOST
    User YOUR_VSC_ACCOUNT
    IdentityFile ~/.ssh/YOUR_EXISTING_VSC_KEY
    ControlMaster auto
    ControlPath ~/.ssh/vsc-%C
    ControlPersist 10m
```

Gebruik een reeds werkende sleutel en de voorgeschreven MFA-methode.
OpenSSH connection sharing werkt op Linux/macOS; de ingebouwde Windows-client
ondersteunt dit niet overal. Op Windows kun je het blok daarom zonder de drie
`Control...`-regels gebruiken. Als jouw account elke nieuwe verbinding een
interactieve MFA-stap laat uitvoeren, gebruik dan de handmatige snapshotroute
in stap 9, of een ondersteunde SSH-client/WSL met connection sharing.
Het dashboard bewaart geen wachtwoorden en omzeilt geen MFA.

Test eerst een normale login en daarna de achtergrondverbinding:

```powershell
ssh vsc-wice
# Log na de normale authenticatie weer uit met: exit
ssh -T -o BatchMode=yes vsc-wice "echo connected"
```

De tweede opdracht moet zonder interactieve vraag `connected` tonen.
Als dit niet lukt, zal de automatische collector ook niet werken. Controleer
sleutel/ssh-agent, VPN, MFA en hostname; gebruik anders stap 9.
Een onbekende host key bevestig je tijdens de normale login nadat je hem via de
officiële toegangsprocedure gecontroleerd hebt, niet door host-keycontrole uit te zetten.

## 4. Start het echte dashboard op je laptop

### Aanbevolen als je al via VS Code Remote SSH verbonden bent

De fout `ssh: connect to host ... port 22: Connection timed out` betekent dat
je laptop de VSC-loginhost niet rechtstreeks bereikt. De dashboardserver werkt
dan wel, maar de collector geraakt niet bij VSC. Dit gebeurt vóór authenticatie;
gebruikersnaam, remote Python en remote paden zijn op dat moment nog niet getest.

Gebruik in dit geval de bestaande VS Code Remote SSH-verbinding. Open de
**remote terminal** in het VS Code-venster dat met VSC verbonden is en voer uit:

```bash
cd /data/leuven/397/vsc39706/R-D_Group2
/data/leuven/397/vsc39706/conda_envs/rd_epic_cuda/bin/python scripts/dashboard.py \
  --local-root "$PWD" \
  --db results/monitoring/vsc.sqlite3 \
  --interval 120
```

Als jouw omgeving op een andere plaats staat, is activeren niet nodig voor het
dashboard zolang `python --version` minstens 3.9 toont. Het dashboard gebruikt
alleen de Python-standaardbibliotheek.

Open daarna in VS Code de **Ports**-weergave, kies **Forward a Port**, voer
`8790` in en open de lokale URL die VS Code toont. Meestal is dat
<http://127.0.0.1:8790>, maar VS Code kan een andere lokale poort kiezen.
Laat de remote terminal open. Stop met Ctrl+C. De Slurm-trainingen blijven lopen.

Deze modus voert de collector direct op de VSC-loginomgeving uit en opent geen
tweede SSH-verbinding. De webserver luistert alleen op VSC-loopback en is via
de bestaande tunnel bereikbaar. Gebruik `--interval 120` alleen tijdens een
actieve monitorsessie; elke update doet één `squeue`- en één `sacct`-aanroep.

### Alternatief: dashboard lokaal met een aparte SSH-verbinding

Vervang de twee voorbeeldpaden door de paden uit stap 2. Gebruik absolute Linux-paden;
typ hier geen letterlijke `$VSC_DATA`, want PowerShell en de remote shell behandelen
variabelen verschillend.

```powershell
cd C:\Users\Emiel\Documents\RnD\R-D_Group2
python scripts/dashboard.py --host vsc-wice --remote-root /ABSOLUTE/VSC_DATA/R-D_Group2 --remote-python /ABSOLUTE/VSC_DATA/conda_envs/rd_epic_cuda/bin/python
```

Open <http://127.0.0.1:8790>. Er wordt één snapshot opgehaald bij het starten.
Met **Sync VSC** vraag je een nieuwe snapshot op; er geldt minimaal 60 seconden
tussen verzoeken. De webpagina leest elke vijf seconden de lokale database;
dat veroorzaakt geen extra Slurm-verzoeken.

Voor automatisch verversen tijdens een trainingssessie:

```powershell
python scripts/dashboard.py --host vsc-wice --remote-root /ABSOLUTE/VSC_DATA/R-D_Group2 --remote-python /ABSOLUTE/VSC_DATA/conda_envs/rd_epic_cuda/bin/python --interval 120
```

Gebruik dit alleen zolang je actief kijkt en volgens de lokale clusterafspraken.
Elke snapshot vraagt één keer `squeue` en één keer `sacct` op voor jouw account.
De collector draait kort op de loginomgeving, zonder permanente VSC-service.
Slurm adviseert schedulerverzoeken tot het noodzakelijke minimum te beperken;
60 seconden is een softwareondergrens, geen door VSC afgegeven toestemming.

## 5. Start een gemonitorde training op VSC

In je gewone VSC-terminal, vanuit de checkout:

```bash
cd "$VSC_DATA/R-D_Group2"
PYTHON="$VSC_DATA/conda_envs/rd_epic_cuda/bin/python"
"$PYTHON" scripts/submit_monitored.py --model mean_pool
```

Voor GRU:

```bash
"$PYTHON" scripts/submit_monitored.py --model gru
```

**Elke uitvoering verstuurt één echte GPU-job en gebruikt de normale projectcredits.**
De wrapper controleert de projectlocatie, maakt `logs/` aan, kiest een unieke
run-ID en doet één `sbatch`-aanroep. Je krijgt job-ID, run-ID en outputpad terug.
Een tweede uitvoering is een nieuw experiment. Bij onverwachte sbatch-uitvoer:
controleer eerst `squeue` voordat je opnieuw indient.

De wrapper exporteert twee variabelen naar de bestaande batchjob:

```text
RND_OUTPUT_ROOT       = absolute results/runs/<unieke-run-id>
RND_MONITOR_RUN_DIR   = dezelfde directory
```

Daardoor kunnen twee runs van hetzelfde model hun checkpoints niet overschrijven.
De evaluatie na training gebruikt automatisch het checkpoint van diezelfde run.
De huidige wrapper is bedoeld voor het P100-trainingsscript en de twee STA-modellen.
Andere jobs verschijnen via Slurm in de tabel; ze hebben zonder instrumentatie
geen curves, GPU-data of logtail in het dashboard.

## 6. Lees de resultaten

Klik na een sync op je job:

| Onderdeel | Betekenis en frequentie |
|---|---|
| State | Slurm-status uit `squeue` / `sacct`; `UNKNOWN` betekent niet bevestigd |
| Epoch | Laatste voltooide epoch, of huidige epoch uit telemetry als er nog geen history is |
| Phase | Initializing, training, validation, training_finished of training_failed |
| Loss/recall | Een nieuw punt na elke volledig afgeronde train+validation-epoch |
| Best epoch/MT5R | Beste validatiescore volgens het bestaande checkpointcriterium |
| GPU | Device utilization en GPU-geheugen voor zichtbare toegewezen GPU's |
| Heartbeat | Tijd sinds laatste telemetry-write; periodiek ongeveer elke 30 seconden |
| Configuration | Configuratie, Git commit en working-tree-info uit training |
| Log | Laatste 16 KB van `logs/sta-train-<jobid>.out` |

Een pending job heeft nog geen trainingstelemetry. Ook featurecontroles en
preprocessing vóór de training kunnen tijd kosten. `training_finished` betekent
dat Python klaar is met trainen; de batchjob kan daarna nog evalueren. Alleen de
Slurm-status bepaalt of de **hele job** geslaagd is. Een OOM, timeout of `scancel`
kan geen finale Python-heartbeat schrijven: de collector haalt de echte eindstatus
uit Slurm-accounting. Accounting kan een korte vertraging hebben.

GPU-samples zijn metingen van het zichtbare device, geen proces-exclusieve
metingen. Ontbrekende `nvidia-smi` of onherkenbare/MIG-device-ID's leveren mogelijk
geen GPU-data op; training loopt verder. CPU/RAM-grafieken, batch-ETA,
checkpointdownloads en het starten/annuleren van jobs vanuit de browser zijn niet
in deze eerste versie opgenomen.

De snapshot neemt maximaal de 200 recentste gemonitorde rundirectories mee, tot
1.000 epochregels per run, en eigen Slurm-jobs van de laatste zeven dagen plus de
actieve queue. Eerder lokaal verzamelde jobs blijven in SQLite staan. Alleen de
laatste 300 lokaal verzamelde telemetry-samples per run worden naar de pagina
gestuurd. Telemetry op VSC bewaart alleen de nieuwste sample; als je laptop uitstaat
blijven epochcurves beschikbaar, maar mis je de tussenliggende GPU-samples.

## 7. Vind checkpoints en tabellen terug

```text
results/runs/<run-id>/
  run.json
  telemetry.json
  checkpoints/<model>/best.pt
  logs/<model>/history.csv
  configs/<model>.json
  tables/<model>_validation.json
  tables/<model>_evaluation.json
```

Het globale Slurm-log blijft `logs/sta-train-<job-id>.out`.
De dashboarddatabase op je laptop staat standaard in
`results/monitoring/dashboard.sqlite3`. Gebruik `--db` voor een ander pad of een
aparte database per VSC-account/cluster. Maak een kopie terwijl de server gestopt is
als je de database wilt back-uppen. Checkpoints blijven op VSC; ze worden niet
gedownload of gedupliceerd door de collector.

## 8. Normaal dagelijks gebruik

1. Maak je gebruikelijke VSC-verbinding en activeer `rd_epic_cuda`.
2. Dien je training in met `submit_monitored.py`.
3. Start lokaal `dashboard.py` en open de pagina.
4. Sync handmatig of gebruik tijdens je sessie `--interval 120`.
5. Open je job voor curves en loguitvoer.
6. Sluit af met Ctrl+C in de lokale dashboardterminal. VSC-training loopt door.
7. Start later dezelfde opdracht opnieuw; opgeslagen data blijft behouden.

Je kunt je normale terminalcommando's blijven gebruiken:

```bash
squeue -M wice -u "$USER"
sacct -M wice -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 80 logs/sta-train-JOB_ID.out
scancel -M wice JOB_ID
```

Vervang `JOB_ID` door het echte nummer. `scancel` stopt daadwerkelijk die job.
Rechtstreeks `sbatch slurm/sta_train_p100.slurm gru` blijft werken met de oude
`results/`-paden, maar geeft geen geïsoleerde gemonitorde run. Voor bestaande jobs
die al draaien verschijnt alleen schedulerinformatie: nieuwe Python-code werkt
pas voor nieuwe processen.

## 9. Alternatief als SSH automatisch ophalen niet ondersteunt

Maak in je interactieve VSC-terminal een snapshot:

```bash
cd "$VSC_DATA/R-D_Group2"
mkdir -p results/monitoring
python scripts/monitor_snapshot.py --root "$PWD" > results/monitoring/vsc-snapshot.json
```

Download dit bestand via je normale SCP/SFTP/OnDemand-workflow naar
`results/monitoring/vsc-snapshot.json` op je laptop. Start lokaal:

```powershell
python scripts/dashboard.py --snapshot results/monitoring/vsc-snapshot.json
```

Je ziet precies die snapshot, geen live verbinding. Vervang het lokale JSON-bestand
na een nieuwe export/download en klik Sync VSC (maximaal eenmaal per minuut).

## 10. Problemen oplossen

| Probleem | Controle/oplossing |
|---|---|
| Permission denied / MFA / timeout | Test `ssh -T -o BatchMode=yes vsc-wice "echo connected"`; controleer VPN, sleutel en sessie. Gebruik stap 9 als interactieve authenticatie nodig blijft. |
| Python niet gevonden | Gebruik `--remote-python` met het volledige Python-pad uit stap 2; versie >=3.9. |
| JSON parse error bij collectie | Remote shellstartup mag geen banner/debugtekst naar stdout schrijven bij een niet-interactieve opdracht. Test de SSH-opdracht; stuur eigen shellmeldingen naar stderr. |
| squeue/sacct niet gevonden | De remote login-shell moet de VSC Slurm-omgeving laden; test deze commando's in dezelfde loginomgeving. Fouten verschijnen in het dashboard. |
| Geen curves | Dien via de wrapper in; wacht op eerste volledig afgeronde epoch; controleer run.json, history.csv, features en Slurm-log. |
| State UNKNOWN | Job ontbreekt uit queue/accounting of collector is onvolledig. Controleer sacct handmatig; dit is geen successtatus. |
| Oude/lege GPU-grafiek | Alleen nieuw opgehaalde heartbeat-samples worden bewaard. Controleer phase, nvidia-smi in de job, CUDA_VISIBLE_DEVICES en collectorverbinding. |
| Port 8790 al in gebruik | Stop de vorige dashboardserver of voeg `--port 8791` toe en open die poort. |
| Demo tussen echte jobs | Start met de echte standaarddatabase; houd `--db results/monitoring/demo.sqlite3` alleen voor demo. |

Bronnen: [VSC job scheduler](https://docs.vscentrum.be/compute/jobs/index.html),
[VSC SSH-toegang en MFA](https://docs.vscentrum.be/compute/terminal/index.html),
[Slurm squeue en performance](https://slurm.schedmd.com/squeue.html),
[Slurm sacct](https://slurm.schedmd.com/sacct.html).

Lokaal verifiëren zonder VSC of GPU:

```powershell
python -m unittest discover -s tests -p test_monitoring.py -v
```

## 11. Combined dashboard and optional W&B

The same dashboard now reads native STA2 `history.json`/run manifests as well as
baseline CSV histories. Select two to six experiment checkboxes to compare curves
and export CSV. Recall is displayed as a percentage; stored values remain fractions.
The summary's verb/noun results come from the epoch selected by best Action MT5R.
Unknown or different split identities produce a comparison warning.

Existing `--snapshot`, `--host`, and `--local-root` commands continue working.
The optional cloud reader has its own background refresh/error state. Browser
polling reads cached data and does not query Slurm or W&B directly. Manual source
refresh has a minimum one-minute cooldown. Existing databases are upgraded in place;
the first upgrade saves a consistent `<database>.before-v2.sqlite3` backup.
Do not run old and new dashboard servers against the same database concurrently.

### Enable cloud reading on the laptop

In PowerShell, from the project root:

```powershell
python -m pip install -r requirements-monitoring.txt
$env:WANDB_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'W&B API key' -AsSecureString)).Password
python scripts/dashboard.py --wandb-only --wandb-project YOUR_TEAM/YOUR_PROJECT
```

Replace the project placeholder. The key is read privately, not put in source code
or browser storage. Close the terminal or remove the environment value afterwards:
`Remove-Item Env:WANDB_API_KEY`. Use a separate database for another cluster/account
or W&B project. W&B-only mode supplies cloud metrics; it cannot supply fresh Slurm
queue state or read VSC log files.

To combine cloud data with VS Code Remote SSH, run in the **remote terminal** with
the key available in that environment, then forward port 8790 as before:

```bash
python scripts/dashboard.py --local-root "$PWD" --interval 120 \
  --wandb-project YOUR_TEAM/YOUR_PROJECT --db results/monitoring/combined.sqlite3
```

Alternatively append `--wandb-project` to the existing laptop `--host` command.
Non-secret defaults can be copied from `configs/monitoring.example.json` to
`configs/monitoring.local.json` and loaded using `--config`; source selection stays
explicit on the command line. The config accepts `wandb_project`, `cluster`,
`interval`, `db` and `port`; credentials are environment-only. `.env` files are not
automatically loaded. With no key or SDK, cloud reads report an error and local
monitoring remains available.

### Enable publication for future training attempts

In your normal VSC shell, install the optional requirements into the training
environment, then set the following before submitting a job:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY; printf '\n'
export WANDB_API_KEY
export RND_WANDB_PROJECT=YOUR_TEAM/YOUR_PROJECT
export RND_WANDB_MODE=online
```

Use `offline` instead of `online` where outbound HTTPS is unavailable. Leave the
mode unset (or set `disabled`) to keep cloud publication disabled. The trainer
writes normal history/checkpoints first. A rank-zero-owned separate publisher
process starts after the first committed epoch; network failures do not change
training/checkpoint selection. Every execution has its own W&B run, grouped by
experiment ID. Local history remains authoritative after a checkpoint resume.
The publisher uploads selected epoch metrics and configuration, not checkpoints,
videos, feature arrays or captured console logs.

The existing baseline submit command is unchanged. STA2 also supports the wrapper:

```bash
# Inspect only: no job is submitted, no output directory is created.
python scripts/submit_monitored.py --job-type sta2-full-h100 --dry-run

# This submits a real job with the existing two-H100 / 72-hour resource request.
python scripts/submit_monitored.py --job-type sta2-full-h100

# Real continuation into an existing output directory and last.pt checkpoint.
python scripts/submit_monitored.py --job-type sta2-full-h100 \
  --resume-run results/sta2/YOUR_EXISTING_RUN
```

Direct `sbatch slurm/sta2_full_h100.slurm` still works. It becomes an instrumented
experiment at Python startup. The monitored wrapper records pending metadata before
submission. If submission is reported as `SUBMISSION_UNKNOWN`, inspect `squeue`
before retrying: a job may already have been created.

Already running Python processes do not acquire new hooks. Their existing native
histories can still appear through the legacy reader. Upload requires the cloud
project to permit your account and the publisher host to have supported outbound
access. Never print the key or add it to a Slurm script or Git.

### Replay and recovery

Attempt data is stored at `<output>/monitoring/attempts/<attempt-id>/`. It includes
`attempt.json`, committed `epochs.json`, publisher status and SDK spool files.
After an interrupted publisher, use the same project/key and online mode to replay:

```bash
python scripts/wandb_publish.py --attempt-dir PATH_TO_ATTEMPT_DIRECTORY --once
```

Online replay checks existing cloud epochs first; the dashboard also deduplicates
epochs. Preserve offline SDK spool data. Use the installed SDK's `wandb sync --help`
for its supported upload command; do not mix obsolete flags from older tutorials.
Offline sync/resume must be validated with your account/SDK before relying on it
for unattended recovery. The canonical local history is available independently.

Sample history is bounded to the latest 300 displayed points and 14 days stored
at full resolution. Scientific epoch history is retained. Collector file reads and
cloud run/history scans have size/count limits; oversized native reads retain
the previous good cache. Discovery currently covers 200 recent native experiments
and 200 cloud attempts per refresh. Import older experiments via separate snapshots
or databases rather than assuming they were all enumerated.

For rollback, stop the new server and point the earlier version at the pre-upgrade
backup. Keep original history/checkpoint files; never delete training output as a
dashboard repair. Remote SQLite uses the default rollback journal; choose a storage
location whose locking is supported by VSC, and keep durable backups if using
node-local temporary storage.

### Validation

```powershell
python -m unittest discover -s tests -p 'test_monitoring*.py' -v
```

Local verification includes mocked VSC/cloud behavior, database migration,
failure isolation, source precedence and resume projection, HTTP checks and a
Chrome demo comparison. Real cloud upload/replay, VSC network access and distributed
GPU training still require the intended account/environment. PyTorch is absent in
the development shell used for these changes; its training suites are not reported
as passing. No real jobs were submitted during implementation.
