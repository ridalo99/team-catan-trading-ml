# Team Catan – Lernmodell für Handelsentscheidungen

Codeauszug aus einem Teamprojekt zu regelbasierten und lernenden Catan-Bots. Mein Schwerpunkt war das neuronale Netz für Handelsentscheidungen.

Der Auszug enthält Modell, Training, Konverter und Datenaufbereitung. Er benötigt das ursprüngliche Paket `ai_games` und Trainingsdaten, die in diesem Auszug fehlen. Deshalb ist er kein eigenständig lauffähiger Bot.

| Datei | Aufgabe |
| --- | --- |
| `model_trade.py` | PyTorch-Modell für Handelsentscheidungen |
| `train_trade.py` | Training, Validierung und Verlustkurven |
| `converter.py` | Umwandlung von Spielzuständen |
| `data_prepocessing.py` | Aufbereitung von Spielprotokollen |

Für die Datenaufbereitung lässt sich der Ordner über `CATAN_GAMELOG_DIR` setzen. Der Auszug bewahrt die Imports aus dem Teamprojekt; Trainingsdaten und Spiel-Engine sind nicht enthalten.
