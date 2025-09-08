\# Minimal Trader – Git Workflow



Deze checklist helpt je om je lokale repo en GitHub (`main` branch) altijd synchroon te houden.



---



\## 1. Status check

git status -sb



\- `## main...origin/main` → alles is gelijk  

\- `ahead` → je hebt commits die nog niet gepusht zijn  

\- `behind` → GitHub heeft nieuwe commits, je moet pullen



---



\## 2. Nieuwe code committen

git add -A

git commit -m "beschrijving van je wijziging"



---



\## 3. Naar GitHub pushen

git push origin main



---



\## 4. Updaten vanaf GitHub

git pull origin main



---



\## 5. Laatste commits bekijken

git log --oneline -10



---



\## 6. Sanity check (af en toe doen)

git fetch origin

git diff HEAD origin/main --stat



\- Geen output = volledig synchroon  

\- Wel output = er zijn verschillen



---



✅ \*\*Belangrijk:\*\*  

\- De repo gebruikt nu alleen nog de branch \*\*main\*\*  

\- `master` bestaat niet meer, gebruik die nooit meer  



