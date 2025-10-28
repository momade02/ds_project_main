"""
daily_stations_update.py
------------------------
Automatisiertes Skript zur Aktualisierung der Tankstellenliste (stations)
aus dem offiziellen Tankerkönig-Azure-Repository

Funktionen:
1. Lädt automatisch die Stations-CSV-Datei des Vortags mit Tankstelleninformationen herunter
2. Prüft Spaltennamen und Datentypen
3. Wandelt Datums- und JSON-Spalten in geeignete Formate um
4. Bereinigt fehlende Werte
5. Löscht Einträge (des Vortrags) in der Tabelle "stations_test" und lädt aktuelle Daten hoch

Eingaben:
    - Keine (Datum wird automatisch als "gestern" bestimmt)

Ausgaben:
    - CSV-Datei im Ordner ./data/
    - Upload der bereinigten Daten in die Supabase-Tabelle "stations_test"

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
# Bestimmt das gestrige Datum, da Tankerkönig täglich aktualisierte CSVs veröffentlicht
yesterday = date.today() - timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")
year = yesterday.strftime("%Y")
month = yesterday.strftime("%m")

# === 2. Lokalen Speicherort für die CSV vorbereiten ===
# Das Skript liegt im Unterordner /scripts, daher eine Ebene nach oben ins Projektverzeichnis
base_dir = Path(__file__).resolve().parent.parent
save_dir = base_dir / "data"
save_dir.mkdir(exist_ok=True)  # erstellt den Ordner "data", falls er nicht existiert
save_path = save_dir / f"{date_str}-stations.csv"

# === 3. Download-URL für Tankerkönig-Daten dynamisch erzeugen ===
# Tankerkönig speichert tägliche Stationslisten unter /stations/YYYY/MM/DD-stations.csv
url = (
    "https://dev.azure.com/tankerkoenig/362e70d1-bafa-4cf7-a346-1f3613304973/"
    "_apis/git/repositories/0d6e7286-91e4-402c-af56-fa75be1f223d/items"
    f"?path=/stations/{year}/{month}/{date_str}-stations.csv"
    "&versionDescriptor%5BversionOptions%5D=0"
    "&versionDescriptor%5BversionType%5D=0"
    "&versionDescriptor%5Bversion%5D=master"
    "&resolveLfs=true&%24format=octetStream"
    "&api-version=5.0&download=true"
)

# === 4. CSV-Datei herunterladen ===
print(f"\nLade Tankerkönig-Stationsdaten für {date_str} ...")

# Sendet eine HTTP-GET-Anfrage an die URL
resp = requests.get(url, timeout=60)

# Prüft, ob der Download erfolgreich war (HTTP 200 = OK)
if resp.status_code != 200:
    raise SystemExit(f"Fehler {resp.status_code}: Datei konnte nicht geladen werden.\nURL: {url}")

# Schreibt die Binärdaten (resp.content) in eine lokale CSV-Datei
with open(save_path, "wb") as f:
    f.write(resp.content)

print(f"Datei gespeichert unter: {save_path}")

# === 5. CSV laden und Struktur prüfen ===
df = pd.read_csv(save_path)
df.columns = [c.strip().lower() for c in df.columns]  # Spaltennamen vereinheitlichen

print(f"\nCSV geladen: {len(df):,} Zeilen, {len(df.columns)} Spalten")
print(f"Spalten: {list(df.columns)}")

# === 6. Typanpassungen und Formatierung ===
# Die Spalte "first_active" enthält das Datum der Erstaktivierung der Tankstelle
# Sie wird in ein Datetime-Objekt umgewandelt und als UTC-Zeit gespeichert
df["first_active"] = pd.to_datetime(df.get("first_active", None), errors="coerce", utc=True)

# Falls die Spalte "openingtimes_json" existiert, wird sie in String konvertiert,
# um JSON-Strukturen sauber als Text in Supabase zu speichern
if "openingtimes_json" in df.columns:
    df["openingtimes_json"] = df["openingtimes_json"].astype(str)

# === 7. Fehlende Werte bereinigen ===
# Ersetzt NaN durch None, damit Supabase die Werte korrekt als NULL speichert
df = df.where(pd.notnull(df), None)

# === 8. Datumsfelder in ISO-Format umwandeln ===
# Beispiel: 2024-11-10 00:00:00
if "first_active" in df.columns:
    df["first_active"] = df["first_active"].dt.strftime("%Y-%m-%d %H:%M:%S")

print("Datentypen und fehlende Werte bereinigt.")

# === 9. Verbindung zu Supabase herstellen ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Fehlende Umgebungsvariablen: SUPABASE_URL oder SUPABASE_SECRET_KEY.")

# Erstellt Supabase-Client mit den bereitgestellten Zugangsdaten
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 10. Upload vorbereiten ===
# Da sich die Stationsdaten täglich vollständig ändern können, werden alte Datensätze gelöscht
print("\nLösche bestehende Daten in 'stations_test' ...")

# Der Filter 'neq("uuid", "00000000-0000-0000-0000-000000000000")' sorgt dafür,
# dass alle echten Einträge gelöscht werden (die Dummy-UUID existiert nie)
supabase.table("stations_test").delete().neq("uuid", "00000000-0000-0000-0000-000000000000").execute()
print("Bestehende Daten gelöscht.")

# === 11. Neue Daten hochladen ===
# Wandelt DataFrame in JSON-kompatible Struktur um (Liste aus Dictionaries)
records = df.to_dict(orient="records")
chunk_size = 10000  # Anzahl der Zeilen pro Batch für Upload

print(f"\nStarte Upload von {len(records):,} Zeilen in Supabase ...")

# Führt den Upload in mehreren Batches aus, um Timeouts zu vermeiden
for i in range(0, len(records), chunk_size):
    chunk = records[i:i + chunk_size]
    supabase.table("stations_test").insert(chunk).execute()
    print(f"{i + len(chunk):,} / {len(records):,} hochgeladen")

print("\nUpload abgeschlossen. Stationsdaten sind jetzt tagesaktuell.")
