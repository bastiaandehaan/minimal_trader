# CHATGPT.md – Workflow & Collaboration Guide

## 1. Context van het project
Het project heet **Minimal Trader** en is een vereenvoudigd trading-framework.
De repository staat op GitHub: `bastiaandehaan/minimal_trader`.

Structuur: modules voor backtesting, strategieën, live trading, risk management en ondersteunende utilities.

---

## 2. Huidige status van de repository
- Alle `.pyc` en `__pycache__` bestanden verwijderd uit Git.
- `.gitignore` uitgebreid om build artefacts, virtual environments, logs, IDE-bestanden en datasets te negeren.
- Repo is opgeschoond en gepusht naar `main`.

---

## 3. Workflowregels
- Werk in **kleine stappen**.
- Begin altijd met **documentatie** (wat, waarom, tests).
- Vraag expliciet toestemming voor **grote wijzigingen**.
- Test elke wijziging (`pytest`, `imports`, `git status`).
- Git-workflow: **plan → review → implementatie → commit**.

---

## 4. Toegestane acties
**Zonder toestemming:**
- `ls`, `dir`
- `python`, `pytest`
- `git status`

**Met toestemming:**
- `git add`
- `git commit`
- `git push`

**Verboden:**
- Alles buiten bovenstaande of risicovolle acties.

---

## 5. Filosofie
- **Klein & iteratief**: één component tegelijk.
- **Documenteer** elke stap in `implementation-log.md`.
- **Test-first**: elke wijziging heeft een `pytest`-test.
- **Kritisch**: wijs fouten en aannames uit, niet pleasen.
- **Robuustheid** boven premature optimalisatie.

---

## 6. Verwachtingen bij samenwerking
ChatGPT fungeert als **pair-programmer en code-auditor**.

Taken:
- Repo lezen en problemen analyseren.
- Testbare verbeteringen voorstellen.
- Nieuwe functies of strategieën altijd eerst in documentvorm.
- Implementeren in stapsgewijze blokken met tests.
- Alleen code genereren na expliciet **“GO”**.
- Code-output in blokken die direct in de repo passen.
- Engels voor code/docstrings, Nederlands voor uitleg.

---

## 7. Constraints
- Strategie v1: eenvoudige SMA20 cross + ATR SL/TP.
- Backtest: O(n), geen O(n²) loops.
- CLI: via `argparse` (geen Typer).
- Tests: via `pytest`.
- Config: in `config.yaml`.
- Logging: naar `logs/`.
- Output: trades naar `output/`.

---

## 8. Deliverables per sessie
- `architecture.md` (systeemarchitectuur, datastromen, testplan, keuzes).
- `implementation-log.md` (elke stap).
- `testplan.md` (3–5 pytest-cases).
- Review van repo-structuur (wat is goed, wat moet weg).

---

## 9. To-do lijst voor ChatGPT (eerste acties)
- Lees `CHATGPT.md` en bevestig dat je de workflow begrijpt.
- Controleer dat `.gitignore` correct werkt (geen `.pyc` meer in `git status`).
- Voer `pytest` uit en rapporteer huidige teststatus.
- Schrijf een review van de repo-structuur + voorstel voor herontwerp (max 5–6 kernbestanden).
- Stel `architecture.md` op.
- Schrijf **Step 1** in `implementation-log.md`.
- Code genereren pas na expliciet **“GO”**.
