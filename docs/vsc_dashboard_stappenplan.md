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
