# Herniation Risk Triage (Research Prototype)

Single-file EEG file-replay pipeline. No simulation, no external sensors.

## The one command you need

    python main.py bootstrap

That's it. Installs `git`/`gh` if missing, authenticates via `gh auth login`
or `GH_TOKEN`, scaffolds the project, detects & removes poisoned git history
(large blobs from previous attempts), commits, creates the GitHub repo,
pushes, tags `v1.0.0`, and GitHub Actions builds desktop exes + Android APK.

## Get artifacts (after bootstrap)

Open:  https://github.com/<you>/<repo>/actions

  - `HerniationTriage-ubuntu-latest` / `-windows-latest`  (desktop)
  - `HerniationTriage-Android`                             (APK)

## Local development (optional)

    python main.py make-sample
    python main.py run --file sample_eeg.csv --cli
    python main.py run --file sample_eeg.csv          # GUI (needs kivy)
    python main.py selftest

## If a previous bootstrap poisoned your history

    python main.py clean-history           # auto-detect + fix
    python main.py bootstrap --nuke-history

## Research prototype only. Not a medical device.
