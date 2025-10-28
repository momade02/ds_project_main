"""
daily_prices_update.py
----------------------
Automatisiertes Skript zur Aktualisierung der Preiliste (prices)
aus dem offiziellen Tankerkönig-Azure-Repository

Funktionen:
1. Lädt automatisch die Preis-CSV des Vortags aus dem offiziellen Tankerkönig-Azure-Repository
2. Prüft Spaltennamen und Struktur
3. Führt Typkonvertierungen für numerische Spalten durch
4. Bereinigt fehlende Werte
5. Lädt die bereinigten Daten in die Supabase-Tabelle "prices_test"

Eingaben:
    - Keine (Datum wird automatisch als "gestern" bestimmt)

Ausgaben:
    - CSV-Datei im Ordner ./data/
    - Upload der Daten in Supabase

Erforderliche Umgebungsvariablen:
    SUPABASE_URL
    SUPABASE_SECRET_KEY
"""

import os
import requests
import pandas as pd
from datetime import date, timedelta
from supabase import create_client
from pathlib import Path

# === 1. Datum bestimmen ===
# Bestimmt das gestrige Datum (die Preisdateien liegen immer für den Vortag vor)
yesterday = date.today() - timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")
year = yesterday.strftime("%Y")
month = yesterday.strftime("%m")

# === 2. Lokalen Speicherort für die CSV vorbereiten ===
# Dieses Skript liegt im Unterordner /scripts, daher gehen wir eine Ebene nach oben (Projektverzeichnis)
base_dir = Path(__file__).resolve().parent.parent
save_dir = base_dir / "data"
save_dir.mkdir(exist_ok=True)  # erstellt den Ordner "data", falls er noch nicht existiert
save_path = save_dir / f"{date_str}-prices.csv"

# === 3. Download-URL für die Azure-Rohdaten erzeugen ===
# Tankerkönig hostet tägliche CSV-Dateien im Azure DevOps-Repository
# Die URL wird dynamisch basierend auf Jahr, Monat und Datum zusammengesetzt
url = (
    "https://dev.azure.com/tankerkoenig/362e70d1-bafa-4cf7-a346-1f3613304973/"
    "_apis/git/repositories/0d6e7286-91e4-402c-af56-fa75be1f223d/items"
    f"?path=/prices/{year}/{month}/{date_str}-prices.csv"
    "&versionDescriptor%5BversionOptions%5D=0"
    "&versionDescriptor%5BversionType%5D=0"
    "&versionDescriptor%5Bversion%5D=master"
    "&resolveLfs=true&%24format=octetStream"
    "&api-version=5.0&download=true"
)

# === 4. CSV-Datei herunterladen ===
print(f"\nLade Tankerkönig-Daten für {date_str} ...")

# Sendet eine HTTP-GET-Anfrage an die URL und wartet auf Antwort
resp = requests.get(url, timeout=60)

# Überprüft, ob der Download erfolgreich war (Statuscode 200 = OK)
if resp.status_code != 200:
    raise SystemExit(f"Fehler {resp.status_code}: Datei konnte nicht geladen werden.\nURL: {url}")

# Speichert den Inhalt der Antwort (resp.content = Binärdaten der CSV) lokal als Datei
with open(save_path, "wb") as f:
    f.write(resp.content)

print(f"Datei gespeichert unter: {save_path}")

# === 5. CSV-Datei laden und Struktur prüfen ===
df = pd.read_csv(save_path) # ließt CSV ein
df.columns = [c.strip().lower() for c in df.columns]  # Spaltennamen vereinheitlichen

print(f"\nCSV geladen: {len(df):,} Zeilen, {len(df.columns)} Spalten")
print(f"Spalten: {list(df.columns)}")

# prüft, ob die heruntergeladene CSV-Datei alle erwarteten Spalten enthält
# -> expected_cols = Menge (set) aller Spaltennamen, die im Datensatz vorhanden sein müssen
# -> set(df.columns) = erstellt eine Menge der tatsächlich gefundenen Spaltennamen aus der CSV
# -> expected_cols - set(df.columns) gibt die Spalten zurück, die fehlen
# -> Wenn etwas fehlt: sofortiger Abbruch mit SystemExit, um zu verhindern, dass später fehlerhafte Daten hochgeladen werden

expected_cols = {
    "date", "station_uuid", "diesel", "e5", "e10",
    "dieselchange", "e5change", "e10change"
}
missing = expected_cols - set(df.columns)
if missing:
    raise SystemExit(f"Fehlende Spalten in CSV: {missing}")
else:
    print("Alle erwarteten Spalten vorhanden.")

# === 6. Typkonvertierungen ===
# Datumswerte werden exakt wie in der CSV beibehalten (z. B. "2025-10-26 00:00:49+02").
# Aktuell keine Zeitzonen-Umwandlung -> diese wird abgeschnitten beim Import, d.h. aus "2025-10-26 00:00:49+02" --> "2025-10-26 00:00:49" (final in der Datenbank)
df["date"] = df["date"].astype(str).str.strip()

# Konvertiert Änderungsfelder dieselchange, e5change, e10change zu kleinen Ganzzahlen (int16) für platzsparende Speicherung 
# -> sonst würden sie als text datatype zu Supabase hochgeladen werden
for col in ["dieselchange", "e5change", "e10change"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

# Ersetzt NaN durch None (Supabase-kompatibel)
df = df.where(pd.notnull(df), None)

# === 7. Verbindung zu Supabase herstellen ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Fehlende Umgebungsvariablen: SUPABASE_URL oder SUPABASE_SECRET_KEY.")

# Erstellt einen Supabase-Client mit den Umgebungsvariablen
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 8. Daten in Supabase hochladen ===
records = df.to_dict(orient="records")  # Umwandlung in JSON-kompatible Struktur (Liste von Dictionaries)
chunk_size = 25000  # Anzahl Zeilen pro Upload-Batch

print(f"\nStarte Upload von {len(records):,} Zeilen in Supabase ...")

# Hochladen der CVS-Zeilen in Teilmengen, um Timeouts zu vermeiden
for i in range(0, len(records), chunk_size):
    chunk = records[i:i + chunk_size]
    supabase.table("prices_test").insert(chunk).execute()
    print(f"{i + len(chunk):,} / {len(records):,} hochgeladen")

print("\nUpload abgeschlossen.")
