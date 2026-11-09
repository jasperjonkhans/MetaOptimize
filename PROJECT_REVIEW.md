# Codequalität, Korrektheit und Reproduzierbarkeit von MetaOptimize

Prüfdatum: 8. November 2026. Repository: `/home/jjonkhans/Code/MetaOptimize`.

Geprüfter Haupt-Commit: `61a57f87f0dea05d6514180d21804eae7f460e30` vom 26. Mai 2025. Die bereits entpackten Experimente sind im äußeren Repository unversioniert. Alle 20 experimentellen Python-Dateien stimmen bytegenau mit `codes/Supervised_tasks.zip` überein. Bestehender Quellcode wurde nicht verändert; dieser Report enthält Empfehlungen, keine durchgeführten Reparaturen.

## 1. Projekt, Prüfmaßstab und Gesamturteil

MetaOptimize passt Lernraten während des Trainings an. Neben dem gewöhnlichen Modellgradienten hält es einen Hilfszustand vor, der Informationen früherer Updates zusammenfasst. Dessen inneres Produkt mit dem aktuellen Gradienten steuert ein Meta-Update. Die Lernrate wird als `alpha = exp(beta)` dargestellt; optimiert wird also ihr Logarithmus. Ein Basisoptimierer verändert Modellgewichte, ein Metaoptimierer verändert Lernraten.

Das Repository enthält nicht eine durchgehend gemeinsame Implementierung, sondern drei Ebenen: eine kleine PyTorch-Bibliotheksklasse, eine JAX-/Optax-Datei mit eigener Demo sowie separat archivierte Forschungsprogramme für CIFAR-10, ImageNet und TinyStories. Die experimentelle Klasse `HF` steht hier für Hessian-free; sie ist nicht mit Hugging Face zu verwechseln. In `export.py` bedeutet `hf_export` dagegen Hugging-Face-Export.

Maßstab dieser Prüfung ist ein kleiner Forschungsprototyp: Eine knappe Implementierung, manuelle Konfiguration und mathematische Approximationen sind nicht an sich Qualitätsmängel. Problematisch sind insbesondere stille Änderungen des tatsächlich trainierten Problems, nicht funktionierende angebotene Optionen sowie fehlende Informationen für einen fairen Vergleich. Popularität des Projekts ist kein Qualitätskriterium. Aus den Befunden folgt keine Unterstellung wissenschaftlicher Manipulation und keine Aussage darüber, welche Dateien oder Konfigurationen tatsächlich die publizierten Kurven erzeugt haben.

| Bereich | Getrennte Gesamtbewertung |
|---|---|
| PyTorch-Package | Der dichte AdamW-Basisschritt ist in einem kleinen echten Framework-Vergleich plausibel und stimmt bei abgeschaltetem Meta-Update numerisch mit `torch.optim.AdamW` überein. Die Klasse ist aber kein voll kompatibler PyTorch-Optimizer. Fehlende Gradienten können Lernraten falsch zuordnen; Validierung und Zustandsserialisierung fehlen. Als eng begrenzter Forschungsbaustein brauchbar, nicht als belastbarer allgemeiner Ersatz ohne Vorbehalte. |
| JAX/Optax | Vor produktivem Einsatz prüf- und reparaturbedürftig. Das innere Produkt ist strukturell falsch. Zusätzlich stehen veraltete Tree-APIs und die Übergabe des Optimizer-Objekts an `jit` einem ausführbaren Beispiel im Weg. Es gab keinen echten JAX-Test; die Bewertung beruht auf Quellcode, überprüften Framework-Verträgen und ausdrücklich gekennzeichneten Ersatzproben. |
| Experimente | CIFAR-10 ist im zentralen Ablauf relativ übersichtlich, hat aber falsches Loss-Logging und eine anders skalierte AdamW-Variante. ImageNet hat defekte optionale Optimizer-/Blockpfade und einen nachgewiesenen HF/DDP-Konflikt. TinyStories trainiert standardmäßig nicht mit der behaupteten akkumulierten Batchgröße; Clipping wirkt nicht auf das Update. Ohne Bereinigung dieser Abläufe und festgehaltene Run-Konfigurationen sind die Programme keine verlässliche Reproduktionsanleitung. |

Die gravierendsten Befunde sind B01–B03. Für verteilte Experimente kommt B07 hinzu. M01–M02 sind bewusst separat aufgeführt: Die Unterschiede sind nachgewiesen, ihre Absicht und vollständige Paper-Äquivalenz jedoch nicht.

### Schwere und Belegsicherheit

**Hoch** bedeutet: mögliche stille Verfälschung des Trainings, unbrauchbarer zentraler Ausführungspfad oder erheblicher Verlust der Vergleichbarkeit. **Mittel** betrifft begrenzte Optionen, Auswertung oder Wiederanlauf. **Niedrig** betrifft vor allem Diagnose und Verständlichkeit. Ein optionaler Fehler kann bei Aktivierung schwerwiegend sein, obwohl er keinen Standardlauf betrifft.

Belegarten:

- **S:** statische Prüfung einschließlich Aufrufern und Konfiguration.
- **F:** echter PyTorch-CPU-Test mit Originalcode, echten Tensoren und gegebenenfalls Autograd; kein vollständiger Datensatz-Trainingslauf.
- **E:** isolierte Probe mit Ersatzobjekten oder nachgebildeter Semantik. Sie beweist nicht automatisch Framework-Kompatibilität.
- **C:** tatsächlich kompilierter und ausgeführter C-Code.
- **Q:** überprüfte externe Quelle oder Framework-Dokumentation.

„Sicherheit hoch“ bezeichnet die Sicherheit der technischen Feststellung, nicht das Wissen um die Motivation ihrer Autoren. „Standard“ setzt voraus, dass erforderliche Abhängigkeiten und lokale Daten vorhanden sind.

## 2. Prüfumfang und nachvollzogene Trainingspfade

Zuerst wurden beide READMEs, `pyproject.toml` und `meta_optimize/__init__.py` geprüft. Danach wurden die 23 Python-Dateien außerhalb von Archivmetadaten untersucht, identische Dateien über Inhaltsvergleich und HF-Kopien über vollständige Differenzen abgeglichen. Zusätzlich wurden Notebook-Kommandos, C-Buildregeln, Tokenizer-Tests sowie wesentliche Lade-, Export- und CLI-Grenzen des übernommenen Inferenzcodes geprüft. Die C-Kernel und die Windows-Portierung wurden nicht vollständig funktional validiert.

- **Package:** Aufrufer erzeugt Loss, führt `backward()` aus und ruft `step()` auf. Die Klasse liest `p.grad`, berechnet korrigierte Momente, aktualisiert Gewichte und Hilfszustände und beendet danach das blockweise Meta-Update. Das MNIST-Beispiel ist durch einen Main-Guard geschützt.
- **CIFAR-10:** CLI → Seed → Netzwerk → HF-/AdamW-Wrapper → CIFAR-Daten → Cross-Entropy → `optimizer.step(net, loss)` → Training-/Testmetriken → TensorBoard → zeitabhängiger Abbruch nach Epochen. HF berechnet Gradienten selbst mit `autograd.grad`; der AdamW-Wrapper führt selbst `backward()` aus. Das falsch geloggte Loss-Mittel wird nicht zurück in die Optimierung geführt.
- **ImageNet:** CLI → CPU/GPU/DataParallel/DDP → Torchvision-Modell → Optimizer-Auswahl → ImageFolder/FakeData → Training und Validierung → TensorBoard → feste Checkpoint-Dateien. Eigene Wrapper erwarten `(model, loss)`, der direkte SGD-Zweig nicht.
- **TinyStories:** Download/Tokenisierung → `.bin`-Shards → IterableDataset → Transformer mit `last_loss` → Microbatch-Schleife → Clipping → Optimizer-Wrapper → Logging/periodische Evaluation/Export. Die Verbindung zwischen Microbatch-Schleife und selbst differenzierendem Wrapper ist fehlerhaft. Datenpfad und zahlreiche Modell-/Laufparameter sind im Quelltext festgelegt.
- **JAX:** Optimizer liefert Optax-Transformation mit `init` und `update`; die Demo erzeugt einen Haiku-MLP und MNIST-Daten. Sie übergibt die Transformation als nicht-statisches Argument an eine jittete Funktion. Bibliothekskern und Demo importieren ihre Abhängigkeiten in derselben Datei.

**Wichtige Einschränkung:** `codes/Continual_CIFAR100_experiments.zip` besteht tatsächlich nur aus den zwei Bytes `0D 0A` beziehungsweise CRLF. Es ist kein ZIP-Archiv. Continual-CIFAR100 kann damit nicht untersucht oder reproduziert werden. `__MACOSX`, `.DS_Store` und verschachtelte `.git`-Verzeichnisse wurden nicht als weitere Forschungsimplementierungen gezählt.

## 3. Bestätigte Fehler

### B01 — TinyStories verwirft die Gradient-Accumulation; Clipping erreicht die Update-Gradienten nicht

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/train.py:35–53,337–360,370–375`; `codes/MetaOptimize/tinystories/HF.py:64–73`; `codes/MetaOptimize/tinystories/build_optimizer.py:48–61`.

**Einstufung:** Hoch; Standardlauf betroffen, HF und AdamW. Technische Sicherheit hoch, S/F. Absicht: wahrscheinlicher Integrationsfehler, nicht bloß andere Mathematik. Kommentar und `tokens_per_iter` versprechen ausdrücklich akkumulierte Gradienten.

**Kontext und Auslöser:** Gradient Accumulation soll die Gradienten mehrerer kleiner Batches zu einem großen Update zusammenführen. Standardmäßig werden acht Microbatches mit je 64 Sequenzen durchlaufen. Jeder Loss wird durch acht geteilt, aber beide Backward-Aufrufe in der Schleife sind auskommentiert. Nur der letzte Loss bleibt für `optimizer.step(model, loss)` übrig.

**Beleg:** HF löscht `.grad` und berechnet mit `autograd.grad` nur den übergebenen Loss. AdamW löscht ebenfalls Gradienten und ruft nur auf diesem Loss `backward()` auf. Die unveränderten AST-Anweisungen aus Zeilen 337–359 wurden mit einem kleinen echten PyTorch-Modell ausgeführt: Microbatch-Gradienten 10 und 100, zwei Akkumulationsschritte. Beide Wrapper erhalten ausschließlich Gradient 50. Der beabsichtigte Mittelwert wäre 55, nach Norm-Clipping auf 1 wäre er 1. Auch mit einem vorher vorhandenen Gradienten 123 entsteht danach wieder der ungeclippte Gradient 50.

**Folgen:** Kein zwingender Absturz. Tatsächlich lernen die Gewichte nur aus dem letzten Microbatch pro Update. Die ersten sieben Microbatches kosten im Standardlauf Vorwärtsrechenzeit, tragen aber keine Gradienten bei. Die nominale globale Batchgröße 512 ist auf einem Prozess effektiv 64 für das Update, dessen Loss zusätzlich durch acht geteilt wurde. Wegen Adam-Normalisierung ist die Wirkung nicht einfach „achtmal kleinere Lernrate“. Token-/MFU-Angaben zählen mehr vollständige Forward-/Backward-Arbeit, als tatsächlich stattfindet; geloggter Trainingsloss ist nur der letzte Microbatch-Loss. Normbegrenzung und bei manuell aktiviertem float16 auch der vorgesehene GradScaler-Pfad fehlen für die neuen Gradienten.

**Kleinster sinnvoller Vorschlag:** Zunächst ausdrücklich einen Microbatch und deaktiviertes wirkungsloses Clipping dokumentieren, falls dies die gewünschte Forschungsvariante war. Für echte Akkumulation müssen alle Microbatch-Gradienten zuerst summiert, dann geclippt und anschließend genau einmal angewandt werden. Nur `backward()` zu entkommentieren reicht nicht: Beide Wrapper würden die Gradienten wieder löschen beziehungsweise neu berechnen.

**Gezielter Test:** Zwei verschieden gerichtete Microbatch-Gradienten gegen einen zusammengefassten Batch vergleichen; zusätzlich einen großen Gradienten mit Clipping prüfen. Je ein Test für HF und AdamW, später auch für AMP und DDP.

### B02 — Fehlende PyTorch-Gradienten verschieben die feste Blockzuordnung

**Bereich/Stellen:** `meta_optimize/optimizer.py:36–46,52–66,95–104`.

**Einstufung:** Hoch bei eingefrorenen, bedingt verwendeten oder anderweitig gradientenlosen Parametern; nicht der vollständig aktive MNIST-MLP. Sicherheit hoch, S/F. Absicht: wahrscheinlicher Fehler; die konfigurierten Blöcke beziehen sich auf Parameter, nicht auf die jeweils aktive Teilmenge.

**Kontext und Auslöser:** Ein Block fasst Parameter-Tensoren zusammen, die eine gemeinsame Lernrate erhalten. `p.grad is None` überspringt den Parameter, bevor der Blockzähler fortgeschrieben wird. Dadurch wird der nächste aktive Parameter als früherer Parameter behandelt.

**Beleg:** Zwei Parameter mit Startwert 1, layerwise-Blöcke mit Lernraten 0.1 und 0.01, nur der zweite mit Gradient 1: Der echte PyTorch-Test ergibt ungefähr 0.9 statt 0.99. In einem skalaren Zweiparameterblock mit nur einem aktiven Parameter, Hilfszustand 1 und Meta-Schrittweite 0.1 bleibt `beta` unverändert: Der Block gilt am Schleifenende noch als unvollständig und das Meta-Update wird nicht ausgeführt.

**Folgen:** Stilles falsches Training durch fremde Lernraten oder ausbleibende Meta-Anpassung. Ein Absturz ist nicht notwendig. Der Fehler hängt von der Gradient-Verfügbarkeit ab und ist deshalb in gewöhnlichen dichten Beispielnetzen leicht zu übersehen. Fehlende Gradienten sind nicht dasselbe wie vorhandene Nullgradienten.

**Kleinster sinnvoller Vorschlag:** Blockmitgliedschaft unabhängig von `grad` festhalten beziehungsweise über feste Parametergruppen iterieren. Für vollständig inaktive Blöcke explizit definieren, welche Zustände unverändert bleiben. Nicht anhand der Zahl aktiver Parameter neue Blöcke bilden.

**Gezielter Test:** Drei Blöcke, jeweils fehlender erster/mittlerer/letzter Gradient; prüfen, dass jede aktive Gewichtskomponente ihre ursprüngliche Lernrate erhält und aktive Blöcke ihr Meta-Update abschließen.

### B03 — JAX berechnet kein inneres Produkt entsprechender PyTree-Blätter

**Bereich/Stellen:** `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:44,103–105`.

**Einstufung:** Hoch; zentraler JAX-Pfad einschließlich Demo betroffen, sobald vorherige Startprobleme überwunden sind. Sicherheit hoch für den Strukturfehler, S/E/Q; kein echter JAX-Lauf. Absicht: wahrscheinlicher Implementierungsfehler.

**Kontext und Auslöser:** Ein PyTree ist ein verschachtelter Container von Array-Blättern. Gesucht ist die Summe der Skalarprodukte jeweils entsprechender Blätter von `h` und `grads`. `tree_map` erzeugt hier jedoch Tupel `(a, b)`, und `tree_leaves` zerlegt diese Tupel wieder in einzelne Array-Blätter. `x[0]` und `x[1]` stammen danach aus demselben Array.

**Beleg:** Die originale Funktion wurde mit NumPy-Arrays und nachgebildeter, dokumentierter Tupel-Flattening-Semantik ausgeführt:

| Blätter                             | Soll |      Ergebnis der Ersatzprobe |
| ----------------------------------- | ---: | ----------------------------: |
| `[1,2]` und `[3,4]`                 |   11 |                            14 |
| `[[1,2],[3,4]]` und `[[5,6],[7,8]]` |   70 |                            94 |
| Nullvektor und `[3,4]`              |    0 |                            12 |
| Skalare 1 und 3                     |    3 | IndexError in der NumPy-Probe |

Die JAX-Dokumentation bestätigt ausdrücklich, dass Tupel standardmäßig Container und keine atomaren Blätter sind [Q3]. Der genaue Exception-Text und etwaige Randfälle der JAX-Indizierung wurden nicht gemessen.

**Folgen:** Falsche Meta-Gradienten und damit Lernraten. Schon bei initialem `h = 0` kann ein künstlicher Meta-Gradient entstehen. Skalare Blätter sind für die verwendete Indizierung ungeeignet; bei mehrdimensionalen Blättern werden Zeilen desselben Arrays statt korrespondierender Zustände multipliziert. Das ist keine alternative sinnvolle Definition des angegebenen inneren Produkts.

**Kleinster sinnvoller Vorschlag:** `vdot` direkt innerhalb des paarweisen Tree-Mappings anwenden und erst dessen skalare Ergebnisse aufsummieren. Dabei eine unterstützte Tree-API verwenden, siehe B08.

**Gezielter Test:** Echte JAX-Tests für skalare, vektorielle und mehrdimensionale Blätter, mehrere Dictionary-/Tuple-Blätter sowie die Invariante `inner_product(zeros, g) == 0`; jeweils eager und unter `jit`.

### B04 — Package-Validierung lässt ungültige Blöcke und numerisch zerstörerische Eingaben passieren

**Bereich/Stellen:** `meta_optimize/optimizer.py:5–24,27–49,74–76`.

**Einstufung:** Mittel bis hoch bei ungültiger Konfiguration; normale dokumentierte Defaultwerte nicht betroffen. Sicherheit hoch, S/F. Absicht: die vergessene Fehlerauslösung ist ein wahrscheinlicher Fehler; weitere fehlende Eingangsprüfungen sind eine unzureichend abgesicherte Vereinfachung.

**Kontext und Auslöser:** Der Konstruktor soll Parameterblöcke und Hyperparameter in einen gültigen Anfangszustand übersetzen. `AssertionError(...)` wird nur erzeugt, nicht ausgelöst.

**Beleg:** Bei zwei Parametern wird `[1]` akzeptiert. Erst im Update tritt ein IndexError auf, nachdem der erste Parameter bereits verändert wurde. `[3]` wird ebenfalls akzeptiert und verhindert den vollständigen Blockabschluss. `[0,2]`, `[-1,3]` und `[0.5,1.5]` werden trotz unbrauchbarer Blockgrößen akzeptiert. Eine leere Parameterliste im skalaren Default scheitert mit `UnboundLocalError`, weil `p` aus der vorherigen Schleife nicht existiert. `alpha0 = -0.1` erzeugt NaN-Gewichte; `alpha0 = 0` erzeugt `beta = -inf` und keine Lernrate; `b1 = 1` oder `b2 = 1` führt zu Division durch null.

**Folgen:** Schlechte Diagnose, teilweise ausgeführte Updates und stille NaNs oder eingefrorene Lernraten. Das erschwert insbesondere automatisierte Sweeps. Negative Weight-Decay-Werte sind zusätzlich inkonsistent: Der direkte Zerfall wird übersprungen, der Wert aber in der Hilfsrekursion verwendet.

**Kleinster sinnvoller Vorschlag:** Vor Zustandsanlage prüfen: nichtleere Parameterliste, positive ganzzahlige Blockgrößen mit passender Summe, endliche positive Startlernrate und definierte Wertebereiche für Momente, Gamma und Decay. Fehler mit aussagekräftigem `ValueError` auslösen. `meta_stepsize = 0` als sinnvollen Fixed-LR-Testfall und `gamma = 1` als angebotene Variante nicht pauschal verbieten.

**Gezielter Test:** Parametrisierte Konstruktor-Negativtests für die genannten Eingaben; sicherstellen, dass kein Gewicht vor dem Konfigurationsfehler geändert wird.

### B05 — ImageNet bietet Optimizer-Pfade an, deren Schnittstellen nicht zum Aufrufer passen

**Bereich/Stellen:** `codes/MetaOptimize/imagenet/train.py:60–65,259–265,421`; `codes/MetaOptimize/imagenet/Optimizers/build_optimizer.py:12–22,53–87`.

**Einstufung:** Mittel; optionale Baselines betroffen, nicht Default-HF. Sicherheit hoch, S/F sowie isolierte Python-Aufrufprobe. Absicht: wahrscheinlicher Integrationsfehler.

**Kontext und Auslöser:** Die Trainingsschleife erwartet einen Wrapper mit `step(model, loss)`. `--optimizer SGD` erzeugt direkt `torch.optim.SGD`, dessen `step` nur eine optionale Closure akzeptiert.

**Beleg:** Ein echter PyTorch-Aufruf `sgd.step(model, loss)` erzeugt `TypeError: SGD.step() takes from 1 to 2 positional arguments but 3 were given`. Außerdem lesen die Builder-Zweige `SGDm` und `Adam` `args.base_lr`; der Parser speichert `--base-lr` ausdrücklich als `args.lr`. SGDm erwartet zusätzlich `base_momentum` statt `momentum`. Beide Builder-Zweige scheiterten mit einem entsprechend aufgebauten Namespace bereits an `base_lr`.

**Folgen:** Sofortiger Laufzeitabbruch, keine nur leicht veränderte Mathematik. Vorhandene SGD-Wrapper werden vom direkten SGD-Zweig gerade nicht verwendet. Ein bloßes Ersetzen durch `optimizer.step()` genügt ebenfalls nicht, weil der aktuelle Trainingsloop nicht selbst `backward()` ausführt.

**Kleinster sinnvoller Vorschlag:** Entweder die vorhandenen Wrapper konsistent nutzen und Argumentnamen korrigieren oder den normalen Zero-Grad-/Backward-/Step-Ablauf für native Optimizer sauber abgrenzen. Nicht funktionierende Optionen bis dahin aus der angebotenen Oberfläche entfernen.

**Gezielter Test:** Jeden angebotenen `--optimizer`-Wert mit Parser-Defaults, einem kleinen Modell und einem synthetischen Batch einmal initialisieren und aktualisieren.

### B06 — Experimentelle Blockoptionen sind unterschiedlich und teilweise nicht ausführbar

**Bereich/Stellen:** `codes/MetaOptimize/imagenet/Optimizers/HF.py:127–168`; `codes/MetaOptimize/cifar10/Optimizers/HF.py:149–165`; `codes/MetaOptimize/tinystories/HF.py:7–9,130–171`.

**Einstufung:** Mittel; optionale Gruppen betroffen. Sicherheit hoch, S/F. Absicht: wahrscheinliche Implementierungsdrift beziehungsweise unvollständig übernommene Optionen.

**Kontext und Auslöser:** Experimentelles HF bietet skalare oder blockweise Lernraten. ImageNet ruft im Blockzweig `torch.log(alpha0)` direkt mit dem CLI-Float auf, während CIFAR-10 und TinyStories vorher einen Tensor erzeugen.

**Beleg:** Ein Einparameter-Modell und `stepsize_groups='[1]'`, `alpha0=0.1`: CIFAR-10 und TinyStories initialisieren; ImageNet scheitert mit „input must be Tensor, not float“. TinyStories nennt `layerwise`, `nodewise` und `weightwise` in der Schnittstellenbeschreibung, implementiert in `init_meta` aber nur scalar/blockwise. `layerwise` führt dort und in CIFAR-10 zu `AttributeError: ... no attribute 'beta'`. ImageNet behandelt `layerwise` als ungültigen Blockausdruck und scheitert mit Division durch null; es ist dort keine funktionierende layerwise-Option.

**Folgen:** Laufzeitfehler bei ansonsten plausibel wirkender Konfiguration. Ein in einer Implementierung gültiger Gruppenname ist nicht automatisch in der anderen unterstützt. Das beeinträchtigt blockweise Vergleiche, nicht jedoch die Standardkonfiguration scalar.

**Kleinster sinnvoller Vorschlag:** Den Float im ImageNet-Blockzweig konsistent konvertieren. Nur tatsächlich implementierte Gruppen anbieten; unbekannte Optionen früh mit erklärender Fehlermeldung ablehnen. Unterstützte Namen pro Implementierung dokumentieren.

**Gezielter Test:** Jeden dokumentierten Gruppenmodus mit passender Modellstruktur initialisieren und mindestens zwei Updates ausführen; Fehlermeldungen für unbekannte Namen und falsche Blocklängen prüfen.

### B07 — HF umgeht die Gradientensynchronisation von DistributedDataParallel

**Bereich/Stellen:** `codes/MetaOptimize/imagenet/train.py:213–231,419–421`; `codes/MetaOptimize/imagenet/Optimizers/HF.py:64–73`; `codes/MetaOptimize/tinystories/train.py:248–254,337–359`; `codes/MetaOptimize/tinystories/HF.py:64–73`.

**Einstufung:** Hoch bei verteiltem HF-Training; nicht Einzelprozess-Standardlauf. Sicherheit hoch, S/F/Q. Absicht: wahrscheinlicher Integrationsfehler zwischen übernommenem DDP-Trainingsgerüst und eigenem Optimizer.

**Kontext und Auslöser:** DDP soll Modellgradienten über Prozesse mitteln, damit alle Repliken identische Updates erhalten. Die HF-Wrapper benutzen `torch.autograd.grad` statt in `.grad` akkumulierender Backpropagation. Die PyTorch-Dokumentation warnt ausdrücklich, dass DDP diesen Weg nicht unterstützt [Q4].

**Beleg:** Zwei echte CPU-Prozesse mit Gloo und DDP, identischem Startgewicht 1, lokalen Gradienten +1 und −1, HF-Basis SGD, Lernrate 0.1, kein Decay und festes Meta-Update: Nach zwei Schritten stehen die Gewichte bei ungefähr 0.8 und 1.2. Bei synchronisiertem Mittelgradienten null müssten beide 1 bleiben. Die ursprüngliche ImageNet-HF-Klasse wurde verwendet; kein nachgebildetes DDP.

**Folgen:** Stilles Auseinanderlaufen von Repliken statt Training eines gemeinsam synchronisierten Modells. Ergebnisse und effektive Batchgröße entsprechen nicht dem üblichen DDP-Vertrag. Die Aussage gilt für den getesteten HF-Gradientenpfad; der AdamW-Wrapper benutzt `backward()` und hat diesen konkreten Fehler nicht, bleibt in TinyStories aber von B01 betroffen.

**Kleinster sinnvoller Vorschlag:** HF/DDP vorerst ausdrücklich ablehnen. Für Unterstützung muss HF synchronisierte Gradienten konsumieren oder eine bewusst getestete explizite Reduktion durchführen; dabei Akkumulation und Meta-Zustände gemeinsam berücksichtigen.

**Gezielter Test:** Zwei Prozesse mit unterschiedlichen lokalen Daten; Gewichte und sämtliche Meta-Zustände müssen nach jedem Schritt übereinstimmen und dem entsprechenden globalen Batch folgen.

### B08 — JAX-Demo und angebotene API haben zusätzliche Ausführungsbarrieren

**Bereich/Stellen:** `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:18–24,30–56,71–105,178–183,224`.

**Einstufung:** Hoch für das Starten mit neueren JAX-Versionen beziehungsweise der Demo; Mittel für optionale Masken. Sicherheit hoch aus S/Q; Masken zusätzlich E. Kein echter JAX-Test. Absicht: wahrscheinlicher API-/Integrationsfehler.

**Kontext und Auslöser:** Drei voneinander unabhängige Probleme bestehen zusätzlich zu B03.

- Die Datei benutzt wiederholt `jax.tree_map`. Dieses Top-Level-Alias wurde in JAX 0.6.0 entfernt; das ist im offiziellen Quelltext dieses Tags ausdrücklich hinterlegt [Q3]. Ohne Versionsgrenze kann schon `optimizer.init` scheitern.
- `train_step` ist mit einfachem `@jax.jit` dekoriert und erhält `optimizer` als normales Argument. Eine Optax-GradientTransformation enthält Funktionen, keine Array-Blätter. Der dokumentierte JIT-Vertrag verlangt für solche Argumente eine statische Behandlung; diese fehlt [Q3].
- `weight_decay_mask` ist als Baum oder Callable typisiert, wird aber immer aufgerufen. Eine Dictionary-Maske führt in der Originalhilfsfunktion zur Ausnahme „dict object is not callable“; eine Funktion, die denselben Baum liefert, funktioniert in der Ersatzprobe.

**Folgen:** Laufzeitabbruch vor oder beim ersten Trainingsschritt beziehungsweise nur bei Aktivierung einer Baum-Maske. Diese Probleme können den mathematischen Fehler B03 zunächst verdecken. Ein Import-/Init-Erfolg allein wäre deshalb kein Funktionsnachweis.

**Kleinster sinnvoller Vorschlag:** Unterstützte Tree-API durchgehend verwenden; Optimizer in der jitteten Funktion einschließen oder als statisches Argument kennzeichnen. Maskenvertrag entweder auf Callables einschränken oder beide Formen korrekt behandeln. Für fehlende erforderliche `params` einen expliziten Fehler statt des derzeit möglichen nicht gesetzten Rückgabewerts vorsehen.

**Gezielter Test:** Import, Init, zwei eager Updates und zwei jittete Updates in einer dokumentierten Versionskombination; Tree- und Callable-Maske getrennt testen. Nach Behebung der Startprobleme B03 erneut ausführen.

### B09 — Loss-/Lernraten-Logging liefert teilweise falsche oder fehlende Auswertungsdaten

**Bereich/Stellen:** `codes/MetaOptimize/cifar10/train.py:121,140–161`; `codes/MetaOptimize/imagenet/train.py:356–358,410–416,431,525–529`; `codes/MetaOptimize/tinystories/train.py:30–33,293–318`.

**Einstufung:** Mittel; CIFAR-/ImageNet-Logging im Standardlauf, W&B nur nach Aktivierung. Sicherheit hoch, S; arithmetische Proben und isolierter W&B-Kontrollfluss E. Absicht: wahrscheinliche Auswertungsfehler.

**Kontext und Beleg:**

- CIFAR-10 summiert bereits batchweise gemittelte Cross-Entropy-Werte und dividiert durch die Beispielzahl. Zwei Batches zu je 100 Beispielen mit Loss 2 und 4 liefern 0.03 statt 3. Bei gleich großen Batches ist der geloggte Loss um die Batchgröße zu klein; bei ungleichen Größen ist zusätzlich die Gewichtung falsch.
- ImageNet sammelt mit `AverageMeter.update(loss, batch_size)` korrekt. TensorBoard erhält danach aber `losses.val`, also den letzten Batch, statt `losses.avg`, dem Epochenmittel. Das interne Mittel ist vorhanden; die Training-Accuracy wird anders als der Loss tatsächlich gemittelt zurückgegeben.
- TinyStories referenziert im W&B-Dictionary `lr`, das nirgends gesetzt wird. Bei `wandb_log=True` wird schon die Dictionary-Erzeugung abgebrochen. Die umgebende Ausnahmebehandlung fängt den NameError ab und druckt „logging to wandb failed“; das ist kein Absturz des gesamten Trainings, sondern Verlust des vorgesehenen Logeintrags. Default ist W&B aus.

**Folgen:** Irreführende Kurven, falsche Größenordnung und erschwerte Vergleiche. **Der an den CIFAR-Optimizer übergebene Loss wird durch den Loggingfehler nicht verändert.** Auch ImageNets `.val`-Verwechslung verändert nicht den optimierten Loss. TinyStories' falsche Akkumulation ist dagegen ein separater Trainingsfehler B01.

**Kleinster sinnvoller Vorschlag:** CIFAR-Summe mit tatsächlicher Batchgröße gewichten; bei ImageNet `.avg` loggen. W&B entweder die tatsächlich verwendete(n) Lernrate(n) übergeben oder das nicht verfügbare Feld entfernen.

**Gezielter Test:** Batches ungleicher Größe mit analytisch bekannten Loss-Mitteln; Recorder statt TensorBoard/W&B verwenden und die übergebenen Werte prüfen. W&B-Zweig einmal aktivieren, ohne einen externen Dienst aufzurufen.

### B10 — Übernommene Exportfunktionen haben begrenzte, aber konkrete Fehler

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/export.py:75–88,182–210,245–247,262,492–510,554–567`.

**Einstufung:** Mittel für betroffene Exportoptionen; niedrig für die latente Fehlerdiagnose. Standardtraining mit gebundenen Embedding-/Output-Gewichten und Version 0 ist von den genannten Auslösern nicht betroffen. Sicherheit hoch, S/F/E. Absicht: wahrscheinliche Implementierungsfehler im übernommenen Code, nicht in den MetaOptimize-Gleichungen.

**Kontext und Belege:**

- Bei ungebundenen Output-Gewichten verändert `legacy_export` `model.params.vocab_size` selbst, um das Vorzeichen im Binärheader zu setzen. Zwei echte Exporte desselben kleinen Modells erzeugten die Headerwerte −16 und +16, obwohl die Gewichte in beiden Fällen ungebunden waren. Der zweite Header kennzeichnet dadurch fälschlich gebundene Gewichte. Der Export sollte das lebende Modell nicht umkonfigurieren.
- `model_export(..., version=-1, dtype=...)` übergibt `dtype` als drittes Positionsargument an `hf_export`; dort ist diese Position `group_size`, nicht `dtype`. Der eigentliche dtype bleibt float32. Die CLI berechnet zwar einen Torch-dtype, reicht später aber den ursprünglichen String weiter. Die Dispatch-Probe belegt die falsche Argumentposition; ein vollständiger Transformers-Export wurde nicht ausgeführt. Betroffen ist der angebotene HF-fp16-Export, nicht die bewusst festen Binärformate 0–2.
- Ruff meldet `i` in Zeile 210. Der Name wird nur benötigt, wenn die vorangehende Assert-Bedingung fehlschlägt; `i` wird erst später in derselben Funktion gebunden. Ein absichtlich unpassendes Ersatzmodell löste `UnboundLocalError` statt der erwarteten verständlichen Assertion aus. Für reguläre Transformer-Gewichte teilt die zuvor angepasste Gruppengröße üblicherweise schon `dim` und damit die relevanten Tensorgrößen. Der Fehler ist somit eine latente kaputte Diagnose, kein belegter Standardexport-Absturz.

**Folgen:** Falsche Exportmetadaten bei wiederholtem ungebundenem Export, ignorierter Präzisionswunsch sowie schlechte Diagnose bei ungültigen Formen. Keine Aussage über falsches Standardtraining. Alle genannten Funktionen sind bytegleich zum im Archiv enthaltenen llama2.c-Stand [Q2].

**Kleinster sinnvoller Vorschlag:** Vorzeichen nur in einer lokalen Header-Variable berechnen; dtype als benanntes Argument und den bereits konvertierten Wert übergeben; den Prüfloop bei Bedarf mit `enumerate` versehen oder `i` aus dem Fehlertext entfernen.

**Gezielter Test:** Export darf Modellparameter/-konfiguration nicht verändern; zwei Exporte müssen dieselben Header liefern. Für Version −1 gespeicherte Tensor-dtypes prüfen. Ungültiges Gewicht gezielt in die Assertion führen.

### B11 — Fehler einzelner TinyStories-Tokenisierungsjobs können als Erfolg erscheinen

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/tinystories.py:127–153,159–172`.

**Einstufung:** Mittel; Datenvorbereitung betroffen, wenn ein Worker scheitert. Sicherheit hoch, S/E mit echtem ProcessPoolExecutor. Absicht: wahrscheinlicher Fehler im übernommenen Datenpfad.

**Kontext und Auslöser:** `pretokenize` startet parallel `process_shard` für JSON-Shards. `executor.map(...)` liefert einen Ergebnisiterator, der nicht konsumiert wird. Ausnahmen der Worker werden erst beim Abrufen ihrer Ergebnisse zum Aufrufer transportiert.

**Beleg:** Die unveränderte Funktion wurde mit einem einzelnen temporären Shard und einem gezielt fehlschlagenden Ersatz-Worker ausgeführt. Der echte Prozesspool beendete seine Arbeit; die Funktion druckte dennoch `Done.` und kehrte erfolgreich zurück. Das war eine Fehler-Injektionsprobe, keine ausgeführte SentencePiece-Tokenisierung.

**Folgen:** Beispielsweise ungültiges JSON, ein fehlendes Tokenizer-Modell oder Schreibfehler können fehlende oder unvollständige `.bin`-Shards hinterlassen, ohne dass der CLI-Aufruf fehlschlägt. Training kann später auf einem unbeabsichtigten Teilbestand laufen oder erst dort scheitern. Das verändert den Datensatz und die Vergleichbarkeit, nicht die Optimizer-Gleichung.

**Kleinster sinnvoller Vorschlag:** Den Ergebnisiterator vollständig konsumieren, damit Worker-Fehler propagiert werden. Vor Trainingsfreigabe außerdem die erwartete Zahl beziehungsweise ein Manifest der Token-Shards prüfen.

**Gezielter Test:** Zwei Shards, davon einer absichtlich defekt: Das Kommando muss mit Fehler enden und darf keinen vollständigen Erfolg melden. Danach einen fehlerfreien Mini-Shard-End-to-End-Test ergänzen.

### B12 — C-Chat akzeptiert überlange CLI-Prompts in feste Puffer

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/run.c:797–850,906–961`; `codes/MetaOptimize/tinystories/runq.c:916–969,1025–1080`.

**Einstufung:** Hoch bei langen oder fremden Chat-Eingaben; keine Auswirkung auf das Python-Standardtraining oder den standardmäßigen Generate-Modus. Sicherheit hoch, S/C. Absicht: der ausdrücklich als unsicher markierte Proof-of-Concept-Status ist belegt; der Speicherfehler selbst bleibt real.

**Kontext und Auslöser:** Die C-Programme bieten `-m chat`, `-i` und `-y` an. `chat` kopiert CLI-Strings mit `strcpy` in 512-Byte-Puffer, ohne deren Länge zu prüfen. Die Kommentare benennen den Code ausdrücklich als „not safely implemented“ und Proof of Concept.

**Beleg:** Ein kleiner C-Aufrufer der unveränderten `chat`-Funktion mit 600 Zeichen User-Prompt, leerem System-Prompt und einem Schritt brach unter GCC mit `_FORTIFY_SOURCE=2` mit `*** buffer overflow detected ***` und SIGABRT ab. Der Fehler liegt vor eigentlicher Modellinferenz. Ein AddressSanitizer-Build wurde versucht, scheiterte aber an der fehlenden Runtime; daraus wird kein ASan-Ergebnis abgeleitet.

**Folgen:** Speicherüberschreibung beziehungsweise Prozessabbruch bei der angebotenen Chat-Option. Eine konkrete Ausnutzung oder Auswirkung auf publizierte Experimente wurde nicht untersucht. Die Dateien sind unverändert aus llama2.c übernommen; dies ist kein spezifischer MetaOptimize-Algorithmusfehler.

**Kleinster sinnvoller Vorschlag:** Längen vor dem Kopieren prüfen und zu lange Eingaben verständlich ablehnen; bis dahin den Chatmodus nur für begrenzte vertrauenswürdige Eingaben ausweisen, nicht als sichere allgemeine Schnittstelle.

**Gezielter Test:** User- und System-Prompts der Längen 0, 511, 512 und 600 mit aktivierten Speicherprüfungen. Danach auch Dateiformat- und EOF-Grenzen separat prüfen; diese sind durch den vorliegenden Test nicht abgedeckt.

## 4. Mathematische oder experimentelle Abweichungen mit unklarer Absicht

Die folgenden Unterschiede sind keine bloßen Vermutungen, aber „abweichend von Standard-AdamW/Lion“ ist nicht gleichbedeutend mit „mathematisch unzulässig“. Zu prüfen ist, welche konkrete Variante beabsichtigt und mit welchen Vergleichslern­raten evaluiert wurde.

### Notation für den Vergleich

`g_t` ist der Gradient am aktuellen Gewicht `w_t`; `alpha_t = exp(beta_t)` die Lernrate; `kappa` Weight Decay; `eta` die Meta-Schrittweite. Ein Moment ist ein geglätteter Gradient beziehungsweise ein geglättetes Gradientenquadrat. Bias-Korrektur gleicht dessen Start bei null aus. `h_t` ist der zusätzliche Lernraten-Hilfszustand. Die Formeln beziehen sich auf einen Block, dichte Gradienten und gültige positive Parameter. Indizes bezeichnen hier den tatsächlich im Code benutzten Zustand, nicht eine ungeprüfte Gleichsetzung mit Paper-Indizes.

| Aspekt | PyTorch-Package | JAX, abgesehen vom defekten inneren Produkt | Experimentelle HF-Klassen |
|---|---|---|---|
| Basis-AdamW | Korrigierte erste und zweite Momente | EMA plus explizite Korrektur beider Momente | Unnormalisiertes erstes Moment; korrigiertes zweites Moment |
| Lernrate für aktuellen Gewichtsschritt | `exp(beta_t)` vor Meta-Update | `exp(beta_{t+1})` nach Meta-Update | `exp(beta_t)` vor Meta-Update |
| Lion-Entscheidung | Mit bereits aktualisiertem Meta-Moment | Mit bereits aktualisiertem Meta-Moment | Mit altem Meta-Moment |
| Vorzeichen des Hilfszustands | Positiver Gradiententrace, positives Meta-Vorzeichen | Wie Package | Negativer Gewichtsschritt-Trace, negatives Meta-Vorzeichen |
| Neue Einspeisung in `h` | `alpha*g/(sqrt(v_hat)+eps)` | Wie Package, aber neue Lernrate | Tatsächlicher Gewichtsschritt einschließlich Momentum und Decay |
| Numerischer Schutz | `sqrt(v_hat) + 1e-8` | `sqrt(v_hat) + 1e-8` | `sqrt(v_hat + 1e-10)` |
| Lernratengruppen | scalar, layerwise, Größenliste | Eine skalare Lernrate für den ganzen Tree | scalar und je nach Kopie unterschiedlich implementierte Blöcke |

### M01 — Experimentelles „AdamW“ verwendet eine andere Erst-Moment-Skalierung

**Bereich/Stellen:** `codes/MetaOptimize/cifar10/Optimizers/HF.py:239–255`; `codes/MetaOptimize/imagenet/Optimizers/HF.py:235–251`; `codes/MetaOptimize/tinystories/HF.py:242–258`; `meta_optimize/optimizer.py:74–85`; `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:34–42,71–83`.

**Einstufung:** Hoch für Vergleichbarkeit und Übertragung von Hyperparametern; Standard-HF mit AdamW betroffen. Sicherheit hoch für die Gleichungen und Zahlen, S/F/Q. Absicht unklar; eine andere Momentum-Konvention oder experimentelles Tinkering ist plausibel. Eine ungekennzeichnete Gleichsetzung mit Standard-AdamW wäre jedoch falsch.

**Kontext und konkrete Abweichung:** HF berechnet `m_t = b1*m_{t-1} + g_t` und `v_t = b2*v_{t-1} + g_t²`. Mit `q_t = (1-b2)/(1-b2^t)` verwendet es `alpha*m_t/sqrt(q_t*v_t + eps)`. Die Normierung `(1-b1)` und die Erst-Moment-Bias-Korrektur fehlen im Update. Dass `m` unnormalisiert gespeichert wird, wäre allein unproblematisch, wenn der passende Faktor im Gewichtsschritt auftauchte.

Beim Package sind dagegen die rekursiven Faktoren `(1-b)/(1-b^t)` eine gültige direkte Fortschreibung der bereits korrigierten Momente. Diese ungewöhnliche Schreibweise ist **kein festgestellter Fehler**. Ein echter Vergleich mit `torch.optim.AdamW` über vier verschiedenartige Gradienten, float64-Gewichte, Weight Decay 0.1 und ausgeschaltetes Meta-Update ergab eine maximale Gewichtsabweichung von nur `3.83e-10`; die Lernrate selbst wird im Package als float32 gespeichert.

**Beleg:** Startgewicht 1, Gradient dauerhaft 1, Lernrate 0.1, kein Weight Decay, festes Meta-Update: Alle drei Original-HF-Klassen liefern etwa `1 → 0.9 → 0.71`. Standard-AdamW liefert `1 → 0.9 → 0.8`. Ohne die unterschiedlichen Epsilon-Terme ist das Verhältnis des HF-Momentumanteils zum bias-korrigierten AdamW-Anteil `(1-b1^t)/(1-b1)`: für `b1=0.9` zuerst 1, dann 1.9, langfristig 10.

**Paper-Einordnung:** Appendix A.3.1 der geprüften Versionen v5 und v6 zeigt ebenfalls unnormalisiert gespeicherte Momente, aber einen Faktor `mu_t`, der `(1-rho)` enthält. Die Erst-Moment-Bias-Korrektur ist grafisch abgeschwächt dargestellt. Das ist kein überprüfter Beleg dafür, dass auch der konstante Normierungsfaktor im Code absichtlich fehlen soll. Algorithmus 3 und Appendix unterscheiden zudem Darstellungen und Zeitindizes. Eine vollständige Äquivalenz wurde nicht bewiesen [Q1].

**Folgen:** Andere effektive Lernraten und andere Anfangsdynamik; keine automatische Aussage über schlechtere Konvergenz. Ein konstanter Lernratenfaktor kann einen Teil der Skalierung absorbieren, aber nicht die zeitabhängige Anfangskorrektur. Bei Weight Decay ändert eine Lernratenreskalierung zusätzlich den Zerfallsanteil. Die Meta-Adam-Varianten haben dasselbe Skalierungsthema. Faire Baseline-Vergleiche benötigen daher die tatsächlich verwendete Konvention und passende Suchbereiche.

**Kleinster sinnvoller Vorschlag:** Die Forschungsvariante benennen und Gleichung, Normierung, Bias-Korrektur und Bedeutung von `alpha0` dokumentieren. Bestehende Experimentkurven nicht durch eine stille „Standardisierung“ nachträglich umdeuten. Erst nach Festlegung des Sollverhaltens Code angleichen.

**Gezielter Test:** Handgerechnete konstante und wechselnde Gradienten, mehrere `b1`-Werte, `eta=0`, mit/ohne Decay. Ein Vergleich zu Standard-AdamW muss entweder die dokumentierte Transformation berücksichtigen oder ausdrücklich eine erwartete Abweichung testen.

### M02 — Lion-Reihenfolge, Lernratenzeitpunkt und Hilfszustände sind nicht dieselbe Variante

**Bereich/Stellen:** `meta_optimize/optimizer.py:59–66,87–104`; `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:44–56,85–96`; `codes/MetaOptimize/cifar10/Optimizers/HF.py:64–73,182–188,220–247`; `codes/MetaOptimize/imagenet/Optimizers/HF.py:223–243`; `codes/MetaOptimize/tinystories/HF.py:229–250`.

**Einstufung:** Hoch für behauptete Implementierungsäquivalenz; keine pauschale Einstufung als Trainingsfehler. Standardvarianten betroffen. Sicherheit hoch für Codeunterschiede, S/F/Q; beabsichtigte Approximation unklar.

**Kontext und Beleg:**

**Lion-Zeitpunkt:** Package/JAX aktualisieren zuerst das Meta-Moment `M_neu = r*M_alt + (1-r)*z` und entscheiden danach anhand von `sign(c*M_neu + (1-c)*z)`. HF entscheidet anhand von `M_alt` und aktualisiert es erst danach. Die erste Form ist `sign(c*r*M_alt + (1-c*r)*z)`, also als Lion mit effektivem Mischkoeffizienten `c_eff = c*r` interpretierbar. Sie ist nicht per se sinnlos, aber bei denselben angegebenen Koeffizienten nicht dieselbe Regel. Bei `M_alt=1`, `z=-8.5`, `c=.9`, `r=.99` ergeben sich entgegengesetzte Vorzeichen. Der echte Package-Test änderte `beta` um etwa −0.1; die Alt-Moment-Form ergäbe +0.1 in derselben Vorzeichenkonvention. Das Paper verwendet für die Lion-Entscheidung das alte Moment [Q1].

**Hilfszustand:** Package nutzt `h_neu = gamma*(1-alpha*kappa)*h_alt + alpha*g/(sqrt(v_hat)+eps)`. HF nutzt `h_neu = gamma*(1-alpha*kappa)*h_alt - delta`, wobei `delta` den gesamten abgezogenen Gewichtsschritt einschließlich Moment und Weight Decay enthält. Nur das Vorzeichen umzudrehen erklärt den Unterschied nicht. Umgekehrt ist das unterschiedliche Vorzeichen allein kein Fehler: Ein negativer Trace mit negativem Meta-Update kann dieselbe Richtung wie ein positiver Trace mit positivem Update ausdrücken. Algorithmus 3 des Papers zeigt die Einspeisung des tatsächlichen Gewichtsschritts; die Package-Abweichung ist im Repository nicht hergeleitet.

**Lernratenzeitpunkt:** Package/HF verwenden die alte Lernrate für das aktuelle Gewichtsupdate. JAX berechnet zunächst `beta_neu` und verwendet bereits `exp(beta_neu)`. Bei identischem positivem Meta-Signal unterscheiden sich die aktuellen Schritte dadurch um den Faktor `exp(eta)`. Ob dies als andere Online-Reihenfolge gewollt ist, bleibt offen.

**Weitere Grenzen:** Eine JAX-Weight-Decay-Maske wird auf das Gewichtsupdate angewandt, aber nicht auf den Zerfallsfaktor von `h`. Bei einem ausgeschlossenen Blatt mit `kappa>0` werden also unterschiedliche Decay-Annahmen benutzt. Auch dies braucht eine Herleitung oder eine Anpassung der Rekursion. Experimentelles SGDm verwendet zunächst das alte Momentum und kann ohne Decay mit einem Nullschritt beginnen. Das passt zumindest zur geschriebenen Alt-Moment-Regel von Algorithmus 3 und ist nicht allein deshalb ein Bug. Unterschiedliche Epsilon-Platzierungen sind ebenfalls numerische Varianten, kein Beweis falscher Mathematik.

**Folgen:** Unterschiedliche Lernratenverläufe trotz identischer Namen und Zahlenwerte. Ein Gleichheitstest zwischen Package, JAX und Experimenten würde derzeit legitime Konventionsunterschiede mit echten Implementierungsfehlern vermischen. Die Auswirkung auf endgültige Modellgüte wurde nicht gemessen.

**Kleinster sinnvoller Vorschlag:** Eine kurze versionierte Sollbeschreibung mit Reihenfolge, Vorzeichen, Momentkonvention, Trace-Rekursion und Maskenverhalten erstellen. Varianten bei Bedarf getrennt benennen. Erst danach gemeinsame Implementierung oder Äquivalenztests festlegen.

**Gezielter Test:** Ein-/Zweiparameterprobleme mit wechselnden Gradienten und per Schritt protokollierten `w`, `alpha`, Momenten und `h`; zusätzlich Decay null/nichtnull und maskierte Blätter. Für als Paper-Variante bezeichnete Updates die jeweilige Paper-Gleichung und ihre Indexkonvention ausdrücklich als Referenz verwenden.

## 5. Reproduzierbarkeit und Vergleichbarkeit

### R01 — Seed- und Zeitbudgetversprechen werden nicht einheitlich eingehalten

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/train.py:65–70,138–161,381–383`; `codes/MetaOptimize/tinystories/tinystories.py:185–195`; `codes/MetaOptimize/cifar10/train.py:105–110,166–167`; `codes/MetaOptimize/imagenet/train.py:148–157,345–347,381`; `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:194–205`.

**Einstufung:** Hoch für Seed-Sweeps und budgetierte Vergleiche; kein pauschaler Laufzeitfehler. Sicherheit hoch, S. Absicht in TinyStories: wahrscheinliche unvollständige CLI-Integration.

**Kontext und Auslöser:** TinyStories parst `--seed` und `--max-time`, liest die resultierenden Werte danach aber nicht. Modellinitialisierung verwendet `1337 + seed_offset`; der Dataset-Iterator verwendet unabhängig davon `42 + worker_id + 1337*rank`. Abbruch erfolgt über `max_iters`, nicht die Uhr.

**Beleg:** Eine vollständige Verwendungssuche bestätigt die fehlende Verbindung von `args.seed`/`args.max_time` zur Initialisierung beziehungsweise Terminierung. Bei gleichem Rang und Worker ändert ein CLI-Seed-Sweep weder den beabsichtigten Modellseed noch die Datenreihenfolge. Auch `--max-time 00:00:01` aktiviert keinen Timer. Die Bedingung `iter_num > max_iters` erlaubt ab Start null nebenbei 100001 statt 100000 Updates.

**Abgrenzung:** CIFAR-10 und ImageNet verwenden ihre Seeds und Zeitargumente tatsächlich. Ihre Budgetprüfung erfolgt aber erst nach einer ganzen Epoche und reserviert pauschal 60 Sekunden plus eine geschätzte weitere Epochenzeit. Dies ist ein weicher Planungsmechanismus, kein harter Timeout. In ImageNet ist deterministisches cuDNN bewusst nicht eingeschaltet. In JAX ist der Initialisierungs-Key fest, das NumPy-Shuffling jedoch nicht separat geseedet.

**Folgen:** Wiederholungen mit verschiedenen TinyStories-CLI-Seeds sind keine gezielt unabhängig initialisierten Wiederholungen. GPU-Nichtdeterminismus kann trotzdem Unterschiede erzeugen; daraus folgt nicht, dass alle Runs bitidentisch wären. Die drei Datensatzprogramme setzen denselben Zeitbudgetbegriff nicht um. Streuungsmaße und Laufzeitvergleiche können dadurch fehlinterpretiert werden.

**Kleinster sinnvoller Vorschlag:** Seed bis Modell, Dateniterator und gegebenenfalls Worker weiterreichen. Zeitlimit entweder tatsächlich prüfen oder als nicht unterstützt zurückweisen. Weiche Epochenbudgets als solche dokumentieren; für Vergleiche gemessene Zeit und tatsächliche Updates/Tokens speichern. Determinismus ausdrücklich als Wahl mit Leistungsfolgen behandeln.

**Gezielter Test:** Gleicher Seed reproduziert auf CPU Initialgewichte und erste Batch-IDs; unterschiedlicher Seed verändert sie. Eine kontrollierbare Uhr testet Abbruchgrenzen, einschließlich eines Budgets unterhalb einer Epoche.

### R02 — Run-Ausgaben kollidieren; „Resume“ lädt keine vollständige Trainingshistorie

**Bereich/Stellen:** `codes/MetaOptimize/imagenet/train.py:188–190,272–291,373–379,496–499`; `codes/MetaOptimize/tinystories/train.py:23–29,204–238,320–331`; `meta_optimize/optimizer.py:26–46,52–112`.

**Einstufung:** Hoch bei mehreren Runs oder Fortsetzung; Modellinferenz aus gültigen Checkpoints ist davon zu unterscheiden. Sicherheit hoch, S/F hinsichtlich fehlender Package-API. Absicht: vereinfachtes Speichern von Modellständen plausibel, als vollständiges Resume aber nicht ausreichend.

**Kontext und konkrete Probleme:**

**Run-Trennung:** ImageNet speichert unabhängig von `--run-name` und `--save-directory` im Arbeitsverzeichnis als `checkpoint.pth.tar` und `model_best.pth.tar`. Zwei Runs im selben Verzeichnis überschreiben diese Dateien. TinyStories verwendet gemeinsam `out/ckpt.pt` und `out/model.bin`; CLI-Runname/-Save-Directory steuern nur TensorBoard. Standardname `test` kann zudem Logserien verschiedener Runs im selben Logverzeichnis vermischen, auch wenn TensorBoard unterschiedliche Eventdateien anlegt.

**Fortsetzung:** ImageNet stellt Modell, Epoche und bisheriges Best-Ergebnis wieder her; Speichern/Laden des Optimizers und Schedulers ist auskommentiert. TinyStories speichert Modell, Modellargumente, Iteration, Best-Loss und leere `config`, initialisiert danach jedoch einen neuen Optimizer. Momente, `beta`, Hilfszustände, Schedulerposition, RNG-Zustände und Dateniteratorposition fehlen. TinyStories speichert standardmäßig zudem nur bei verbesserter Validierung nach einer positiven Evaluationsiteration, nicht automatisch am Programmende.

**Package-Vertrag:** Die Klasse hält durchaus relevante Zustände, bietet aber weder `state_dict()` noch `load_state_dict()`. Das ist direkt geprüft. Gewichte allein reichen für MetaOptimize noch weniger als für gewöhnliches Momentum-Training.

**Folgen:** Verlust früherer Run-Artefakte; ein scheinbar fortgesetzter Lauf hat eine andere Optimierungsdynamik. Ein Weight-Warmstart kann gewollt und nützlich sein, ist aber nicht dasselbe wie unterbrechungsfreies Training. Ein selbst gebauter vollständiger Snapshot wäre theoretisch möglich; die angebotenen Pfade leisten ihn nicht.

**Kleinster sinnvoller Vorschlag:** Alle Artefakte unter ein eindeutiges Run-Verzeichnis legen. Bis vollständige Zustandsrestauration implementiert ist, den aktuellen Weg als Weight-Warmstart kennzeichnen. Für echtes Resume Modell, Optimizer-/Meta-Zustände, Scheduler, Konfiguration und Zufalls-/Datenzustände gemeinsam versioniert speichern.

**Gezielter Test:** Zwei Run-Namen dürfen keine Artefaktpfade teilen. Einen kleinen deterministischen Lauf nach N Schritten speichern und M Schritte fortsetzen; gegen N+M ununterbrochene Schritte sämtliche Gewichte, Lernraten und Zustände vergleichen.

### R03 — Wirksame Konfiguration, Baseline-Schedule und Ausführungsmodus sind nicht zuverlässig festgehalten

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/train.py:23–60,116–129,204–238,243–246,275–286,322–328`; `codes/MetaOptimize/tinystories/model.py:115–122,298–322`; `codes/MetaOptimize/tinystories/build_optimizer.py:48–61`; `codes/MetaOptimize/imagenet/Optimizers/build_optimizer.py:130–144`; `codes/MetaOptimize/cifar10/Optimizers/build_optimizer.py:48–61`; `codes/MetaOptimize/tinystories/tinystories.py:23,41–59`; `codes/MetaOptimize/imagenet/train.py:37–38,299–320,342`.

**Einstufung:** Hoch für exakte Reproduktion und faire Konfigurationsvergleiche, nicht automatisch falsches Training. Sicherheit hoch, S/F für Scheduler-Zustand. Absicht: plausibles manuelles Forschungstinkering; Dokumentation des wirksamen Zustands unzureichend.

**Kontext und Beleg:** TinyStories übernimmt zahlreiche globale Nano-/Llama-Trainingsoptionen, deaktiviert aber den Configurator und setzt `config = {}`. Modellgröße, Batchgröße, `device`, `dtype`, `init_from` und ähnliche Optionen sind über die aktuelle CLI nicht verfügbar. Die lokalen AdamW-Angaben `learning_rate`, `beta2`, `warmup_iters` und `get_lr` bestimmen den tatsächlich gebauten Wrapper nicht. Dieser verwendet in allen drei Kopien feste Betas `(0.9, 0.999)`, Decay 0.1 sowie 10000 Warmup- und 422000 Gesamtschritte. Insbesondere passt das nicht zu TinyStories' sichtbaren 1000 Warmup-/100000 Iterationswerten. Der echte Scheduler-Zustand bestätigt 10000/422000; ein Standard-TinyStories-Lauf erreicht damit das Ende des Cosine-Schedules nicht.

`compile=True` druckt eine Kompilierungsmeldung, aber `torch.compile` ist auskommentiert. `Attention.flash` ist ausdrücklich `False`; die Warnung über benötigtes PyTorch >=2.0 erklärt daher den tatsächlichen Grund nicht. Diese Deaktivierungen können für das Experiment sinnvoll gewesen sein und sind keine Belege falscher Numerik. Sie beeinflussen aber Zeitvergleiche.

Datenpfade sind maschinenspezifisch. Die README fordert ihre Anpassung für ImageNet und TinyStories ausdrücklich; dies ist also teilweise dokumentierte manuelle Einrichtung. Es fehlen trotzdem ein ausführbares vollständiges Run-Manifest, Datensatz-/Tokenizer-Fingerprints und eine Zuordnung konkreter Konfigurationen zu den Paper-Kurven. Downloads referenzieren bewegliche `main`-URLs. Die im Paper beschriebene TinyStories-Batchgröße 128 und die jetzigen Defaultwerte sind beispielsweise keine identische, dokumentierte Run-Konfiguration [Q1].

**Folgen:** Leser können sichtbare, aber unwirksame Werte für die tatsächliche Versuchskonfiguration halten. HF und Baselines sind ohne Aufzeichnung ihrer effektiven Gleichungen/Schedules nicht fair allein anhand gleich benannter CLI-Argumente vergleichbar. Es ist nicht belegt, dass die publizierten Experimente mit diesen Defaults liefen.

**Kleinster sinnvoller Vorschlag:** Zuerst wirksame Konfiguration beim Start vollständig ausgeben und in jedem Run speichern, einschließlich verwendeter Optimizer-Variante, Scheduler-Grenzen, Präzision, Compile-/Attention-Modus, Datenfingerprints und tatsächlicher Batchgröße. Nicht benutzte Einstellungen entfernen oder klar kennzeichnen. Je Datensatz einen getesteten Referenzaufruf mit Umgebung und Datenlayout dokumentieren; keine neue Konfigurationsplattform nötig.

**Gezielter Test:** Jede angebotene Option muss entweder das beobachtbare Verhalten beziehungsweise gespeicherte Manifest ändern oder ausdrücklich abgelehnt werden. Scheduler an Warmup-/Endgrenzen testen; Startprotokoll muss den tatsächlich ausgeführten Compile- und Attention-Pfad widerspiegeln.

## 6. Wartbarkeit, Verständlichkeit und Stil

### W01 — Bibliotheksvertrag, Demo-Abhängigkeiten und Installationsanleitung passen nur teilweise zusammen

**Bereich/Stellen:** `meta_optimize/__init__.py:1`; `pyproject.toml:1–15`; `meta_optimize/optimizer.py:1–4,107–112,130–138,158–208`; `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py:2–6,124–153`; `README.md:19–31,94–144`; `codes/MetaOptimize/cifar10/train.py:169–173`; `codes/MetaOptimize/tinystories/train.py:94–123`; `codes/MetaOptimize/tinystories/requirements.txt:1–7`.

**Einstufung:** Mittel; vor allem neue Nutzer und Integration in vorhandene Werkzeuge. Sicherheit hoch, S/F. Absicht: einfache Forschungs-API plausibel; Name und Dokumentation grenzen ihre Einschränkungen nicht ausreichend ab.

**Kontext und Beleg:** Der öffentliche Export ist genau die spezielle PyTorch-Klasse `MetaOptimize_AdamW_Lion_hg`, kein generischer Wrapper um einen beliebigen vorhandenen Optimizer. Sie erbt nicht von `torch.optim.Optimizer` und besitzt weder `param_groups` noch `add_param_group` oder Zustands-API. Ein echter `StepLR`-Konstruktor lehnt sie mit „is not an Optimizer“ ab. Ihre eigene manuelle `zero_grad`/`step`-Nutzung funktioniert dagegen im untersuchten kleinen Fall. Ein Anspruch auf nahtlose Scheduler-/Trainer-Kompatibilität wäre zu weitgehend.

`pyproject.toml` deklariert nur `torch`. Das genügt für den eigentlichen Package-Import: Torchvision wird dort erst im Datenloader importiert; es gibt keinen MNIST-Download beim Import. Die README-Demo benötigt aber zusätzlich Torchvision. Die JAX-Datei importiert neben JAX/Optax/Chex auch Haiku und TensorFlow für die Demo auf Modulebene; keine dieser JAX-Abhängigkeiten ist deklariert. Der qualifizierte Import über `meta_optimize` lädt zuvor außerdem den PyTorch-Export aus `__init__.py`. JAX importiert nicht automatisch MNIST-Daten, verlangt aber unnötig die Demo-Bibliotheken schon für den Optimizer-Kern.

Die Experimente benötigen TensorBoard über `SummaryWriter`, das weder in der Root-Requirements-Liste noch in der TinyStories-Requirements-Datei steht. Die mitgelieferten `tinystories/run`- und `runq`-Binärdateien sind zudem keine portablen Linux-Builds: Ihr ELF-Interpreter verweist auf `/cvmfs/soft.computecanada.ca/gentoo/2020/lib64/ld-linux-x86-64.so.2`. Ein lokaler Neubau gelang; die fremden Binärdateien sollten nicht als Ersatz für einen dokumentierten Build vorausgesetzt werden. Transformers ist nur für bestimmte Export-/Importfunktionen optional und wird dort mit einer Installationsmeldung behandelt; es ist kein notwendiges Package-Core-Dependency.

Die README verweist auf `code`, `Supervised_task.zip` und `main.py`; vorhanden sind `codes`, `Supervised_tasks.zip` und `train.py`. Ein Trainingaufruf muss im passenden Experimentverzeichnis erfolgen. Der aktuelle CIFAR-Trainingsimport scheitert nach den importierten Abhängigkeiten an `writer.close()` außerhalb des Main-Guards, falls `writer` nicht erzeugt wurde. TinyStories `train.py` parst und startet bewusst auf Modulebene; das macht isolierte Tests schwerer, ist bei einem Skript aber nicht automatisch ein Bibliotheksfehler.

**Folgen:** Vermeidbare Installations- und Integrationsabbrüche, versteckte schwere Demo-Abhängigkeiten und erschwerte Testbarkeit. Die manuelle README-PyTorch-Trainingslogik ist für einen vollständig aktiven MLP plausibel; sie wurde nicht mit echtem MNIST end-to-end ausgeführt. Die globale Beschreibung des MetaOptimize-Forschungsframeworks ist nicht als Implementierungsnachweis für beliebige Optimizer zu lesen.

**Kleinster sinnvoller Vorschlag:** Die derzeit kleine API und ihre Grenzen explizit dokumentieren. Demo-Code und optionale Dependencies trennen, passende Extras beziehungsweise Requirements anbieten und einen Referenz-Python-/Framework-Stand festlegen. Nur wenn native Tool-Kompatibilität gewünscht ist, den PyTorch-Optimizer-Vertrag vollständig umsetzen; bloßes Umbenennen genügt nicht. README-Pfade und Main-Guards korrigieren.

**Gezielter Test:** Minimaler Package-Import nur mit deklarierten Kernabhängigkeiten; separat Demo- und JAX-Importtests. Dokumentierte Aufrufe in temporären Arbeitsverzeichnissen mit kleinen synthetischen Daten prüfen. Für zugesagte PyTorch-Kompatibilität Scheduler- und Zustandsroundtrip testen.

### W02 — Duplikation führt zu konkreter Drift; voreilige Zusammenführung würde Semantik verdecken

**Bereich/Stellen:** `codes/MetaOptimize/cifar10/Optimizers/HF.py:1–264`; `codes/MetaOptimize/imagenet/Optimizers/HF.py:1–260`; `codes/MetaOptimize/tinystories/HF.py:1–267`; `codes/MetaOptimize/cifar10/build_network.py:1–243` und `codes/MetaOptimize/imagenet/build_network.py:1–243`; die drei `build_optimizer.py`-Dateien; Package-/JAX-Dateien. Provenienz: `codes/MetaOptimize/tinystories/.git`.

**Einstufung:** Mittel für Wartung und Verständlichkeit. Sicherheit hoch, Inhaltsvergleich/S. Absicht: plausible Snapshot-Kopien für unabhängige Forschungsversuche; nicht allein wegen der Duplikatmenge abzulehnen.

**Kontext und Beleg:** Die HF-Klassen sind ähnlich, aber nicht identisch. ImageNets Blockinitialisierung hat den Float-Logarithmusfehler B06; TinyStories hat zusätzlich `transformer_blocks_15m`; erkannte Gruppen unterscheiden sich. ImageNets Block-Logging verwendet in `codes/MetaOptimize/imagenet/Optimizers/HF.py:86–89` für Alpha und Beta denselben `Optimizer_blockwise/beta_block...`-Tag, während die anderen Kopien getrennte Tags benutzen. Beim normalen CLI-Aufruf wird dieser zusätzliche Loggingfehler zunächst durch die defekte Blockinitialisierung verdeckt. Der Skalierungsunterschied M01 ist hingegen in allen drei HF-Klassen gemeinsam und müsste bei einer beabsichtigten Änderung dreifach berücksichtigt werden.

Die beiden `build_network.py`-Dateien sind byteidentisch. Die ImageNet-Trainingsdatei verwendet diese Kopie aber nicht, sondern `torchvision.models`; ebenso liegt der dortige separate `load_data.py` nicht auf dem nachvollzogenen Hauptpfad. Repositoryweite Referenzsuche stützt diese Aussage. TinyStories' `model.configure_optimizers` hat ebenfalls keinen aktiven Aufrufer im Repository: Maßgeblich ist der externe Builder, nicht diese scheinbar zentrale Methode. Der TinyStories-/CIFAR-Builder unterscheidet sich im Wesentlichen im HF-Importpfad, teilt aber dieselben festen Scheduler-Konstanten.

Das innere TinyStories-Git enthält einen konkreten Upstream-Stand: llama2.c `d9862069e7ef665fe6309e3c17398ded2f121bf5` vom 9. Oktober 2023. `export.py`, `tokenizer.py`, `sample.py`, `test_all.py`, `test.c`, `run.c` und `runq.c` stimmen damit bytegenau überein. `train.py`, `model.py` und der Datenpfad wurden geändert. Diese Zuordnung ist aussagekräftiger als eine pauschale Verantwortung aller Fehler beim MetaOptimize-Optimizer. Die verschachtelte Git-Historie ist keine zusätzliche Forschungsimplementierung.

**Folgen:** Mehrfach nötige Korrekturen, unterschiedliche verfügbare Optionen und Verwechslung aktiver mit liegengebliebenen Pfaden. Ein großer gemeinsamer Optimizer ohne vorherige Klärung von M01/M02 würde die Unterschiede eher verstecken. `0/0` als Fehlersignal und große auskommentierte Umbauabschnitte verschlechtern konkrete Diagnosen, sind aber gegenüber den Trainingsfehlern nachrangig.

**Kleinster sinnvoller Vorschlag:** Zuerst aktive Pfade und Varianten beschriften, Upstream-Commit und lokale Änderungen in einer kurzen Provenienznotiz festhalten. Identische, nachweislich nicht verwendete Kopien erst nach Klärung externer Nutzung entfernen. Nur semantisch identische Teile gemeinsam nutzen; experimentelle Varianten bewusst getrennt lassen. Keine abstrakte Architektur allein zur Reduktion einer Duplikatkennzahl einführen.

**Gezielter Test:** Kleine gemeinsame Vektorfälle für ausdrücklich gleiche Semantik und separate Erwartungswerte für benannte Varianten. Builder-/CLI-Matrix gegen dieselbe dokumentierte Optionsliste prüfen. Bei Upstream-Updates gezielt lokale Diffs und Export-/Inferenz-Smokes ausführen.

## 7. Testabdeckung und Prüfgrenzen

### T01 — Vorhandene Tests sichern Inferenz und Tokenisierung, nicht den Forschungsbeitrag

**Bereich/Stellen:** `codes/MetaOptimize/tinystories/test_all.py:18–89`; `codes/MetaOptimize/tinystories/test.c:11–84`; `codes/MetaOptimize/tinystories/Makefile:60–78`; die in B01–M02 aufgeführten Optimizer-/Trainingsstellen. Prüfung über das gesamte Arbeitsverzeichnis, ohne `.git` und Archivmetadaten.

**Einstufung:** Hoch als Absicherungslücke, nicht als eigenständiger Beweis falscher Updates. Sicherheit hoch, AST-/Dateiinventar/S/C. Absicht: übernommene Upstream-Tests sind erkennbar; fehlende neue Optimizer-Tests sind für einen Prototyp nachvollziehbar, für belastbare Wiederverwendung aber unzureichend.

**Kontext und Beleg:** Es existieren tatsächlich die zwei Python-Tests `test_runc` und `test_python` für bekannte Textgenerierung. Die Ausgangshypothese „nur zwei Tests im gesamten Repository“ ist dennoch zu eng: Zusätzlich enthält `test.c` einen Tokenizer-Testlauf mit fünf Prompt-Fixtures, erreichbar über `make testcc`. Dieser Lauf wurde lokal aus dem Quelltext kompiliert und bestand mit `ALL OK`.

Keine Tests für die MetaOptimize-Updategleichungen, Parameterblöcke, fehlende Gradienten, Akkumulation, CLI-Optionen oder Resume wurden gefunden. Alle 23 Python-Dateien konnten syntaktisch geparst werden. Ruff mit `F821,F822,F823,E9` meldete genau die beiden besprochenen Namen `i` und `lr`; deren sehr unterschiedliche Erreichbarkeit ist in B09/B10 eingeordnet. Eine umfangreiche allgemeine Lint-Liste wäre hier weniger hilfreich als kleine semantische Tests.

**Folgen:** Inferenztests können grün bleiben, obwohl das Training eine falsche Batchgröße verwendet oder Lernraten falschen Parametern zuordnet. Die beiden Python-Tests laden bei fehlenden Fixtures Modelle/Tokenizer von einer beweglichen Hugging-Face-URL; der C-Inferenztest erwartet zusätzlich ein lokales `./run`. Sie sind keine hermetische Offline-Optimizer-Suite.

**Kleinster sinnvoller Vorschlag:** Eine kleine datensatzfreie Kern-Testgruppe priorisieren: Handrechnungen, Blöcke/None, Akkumulation/Clipping, CLI-Smokes und Resume-Roundtrip. Vorhandene Inferenztests behalten; externe Fixtures mit Version/Hash festhalten und separat markieren. Ein großer Trainingsbenchmark ist für die gefundenen Fehler zunächst nicht nötig.

**Gezielter Test:** Die in B01–M02 angegebenen Minimalfälle als wiederholbar ausführbare Tests aufnehmen; als Positivkontrolle den bereits erfolgreichen AdamW-Vergleich und die fünf Tokenizer-Fixtures behalten.

### Tatsächlich verwendete Umgebung

Die erste Untersuchung erfolgte mit `/usr/bin/python3`, Python 3.14.7. Dort fehlten Torch, Torchvision, JAX, JAXlib, Optax und pytest. Ruff war als eigenständiges Programm vorhanden, Version 0.15.16.

Zusätzlich wurde eine bereits vorhandene Umgebung gefunden und ohne Installation verwendet:

- Interpreter: `/home/jjonkhans/Code/IDBD-optimizer/.venv/bin/python`.
- Python 3.14.7, PyTorch `2.14.1+cu130`, NumPy `2.5.3`.
- CUDA im Testprozess nicht verfügbar; sämtliche Tensor-/Autograd-Tests auf CPU.
- Auch dort kein JAX, Optax, Chex, pytest, Torchvision, Haiku, TensorFlow, SentencePiece oder Transformers.
- C-Compiler: GCC 15.3.1.

Das ist eine **Prüfumgebung**, kein rekonstruierter historischer Paper-Stack. Insbesondere wurde nicht gegen die README-Pins Torch 2.0.1/Torchvision 0.15.2 getestet. Es wurden weder Frameworks installiert noch Trainingsdatensätze oder Modellgewichte heruntergeladen.

### Ausgeführte Prüfungen und ihre Aussagekraft

| Prüfung | Ergebnis / Grenze |
|---|---|
| Package-Import und kleine echte Updates | Erfolgreich; beweist weder native Optimizer-Kompatibilität noch vollständiges Training. |
| Package gegen Torch-AdamW, Meta-Schrittweite 0 | Vier Schritte mit wechselnden Gradienten, maximale Differenz `3.83e-10`. Positivkontrolle für korrigierte Basismomente. |
| Fehlende Gradienten und Blockvalidierung | B02/B04 mit Originalklasse reproduziert; teilweise falsche Werte, teilweise späte Fehler. |
| Drei experimentelle AdamW-HF-Klassen | Jeweils `0.9 → 0.71` nach den beiden konstanten Gradientenschritten; bestätigt die gemeinsame Variante. |
| TinyStories-Microbatchsegment | Original-AST plus originale HF-/AdamW-Wrapper und echtes Toy-Modell: nur Gradient 50 statt Mittel 55 beziehungsweise geclippt 1. Kein vollständiger Transformer-Trainingslauf. |
| HF mit zwei Gloo-/DDP-Prozessen | Gewichte nach zwei Schritten `0.8` und `1.2` statt synchron `1.0`. |
| Kleiner originaler Transformer | Forward mit Cross-Entropy sowie kurze greedy Generation auf synthetischen Token-IDs erfolgreich. Keine Qualitätsbewertung eines Sprachmodells. |
| Export | Wiederholter ungebundener Legacy-Export mit echten Tensoren; dtype-Dispatch und Assert-Fehler isoliert geprüft. Keine vollständige Transformers-/Llama-Konversion. |
| JAX-inneres Produkt und Masken | Originalfunktionen mit expliziten Ersatzobjekten; kein JAX-Import, keine JIT-Ausführung. |
| Preprocessing-Fehlerweitergabe | Originalfunktion, echter Prozesspool, absichtlich fehlschlagender Ersatz-Worker; Erfolgsmeldung trotz Fehler. |
| C-Build | `run.c`, `runq.c` und `test.c` erfolgreich nach `/tmp` kompiliert; mitgelieferte Binärdateien nicht überschrieben. |
| C-Tokenizer | Fünf mitgelieferte Fixtures bestanden. |
| C-Chatgrenze | Fortify-Abbruch bei 600 Zeichen. ASan wegen fehlender Runtime nicht ausführbar. |
| Repository-Syntax/Lint | 23 Python-Dateien parsebar; die zwei relevanten undefinierten Namen verifiziert. |

### Kleine reproduzierbare Anker

Die folgenden Kommandos setzen das Repository als Arbeitsverzeichnis und einen Interpreter mit den jeweils genannten Abhängigkeiten voraus. Sie verändern keinen Repository-Quellcode. Für die ursprünglichen Framework-Proben wurde der oben genannte alternative Interpreter mit `-B` beziehungsweise `PYTHONDONTWRITEBYTECODE=1` verwendet.

**B02, echte Torch-Blockprobe:**

`python -B -c 'import torch; from meta_optimize import MetaOptimize_AdamW_Lion_hg as O; p=torch.nn.Parameter(torch.ones(1)); q=torch.nn.Parameter(torch.ones(1)); o=O([p,q], stepsize_blocks="layerwise", meta_stepsize=0); o.meta_state[0]["beta"]=torch.tensor(.1).log(); o.meta_state[1]["beta"]=torch.tensor(.01).log(); q.grad=torch.ones_like(q); o.step(); print(q.item())'`

Ergebnis im geprüften Stand: etwa `0.9`; blocktreu wäre `0.99`.

**B04, ungültige Blocksumme:**

`python -B -c 'import torch; from meta_optimize import MetaOptimize_AdamW_Lion_hg as O; ps=[torch.nn.Parameter(torch.ones(1)) for _ in range(2)]; o=O(ps, stepsize_blocks=[1]); print("Konstruktor akzeptiert"); [setattr(p,"grad",torch.ones_like(p)) for p in ps]; o.step()'`

Erst `step()` scheitert, nachdem ein Teilupdate erfolgt ist.

**Archiveinschränkung:**

`python -B -c 'from pathlib import Path; import zipfile; p=Path("codes/Continual_CIFAR100_experiments.zip"); print(p.read_bytes(), zipfile.is_zipfile(p))'`

Ergebnis: `b'\r\n' False`.

**C-Tokenizer ohne Download und ohne Überschreiben von `run`:**

`gcc -O2 -o /tmp/metaopt-tokenizer-test codes/MetaOptimize/tinystories/test.c -lm && (cd codes/MetaOptimize/tinystories && /tmp/metaopt-tokenizer-test)`

Ergebnis: `ALL OK`.

**Noch auszuführender echter JAX-Test, nicht als erfolgte Prüfung zu lesen:** Die Originalfunktion `inner_product` allein per AST laden, mit echtem `jax`/`jax.numpy` binden und die vier Fälle aus B03 sowohl eager als auch jittet prüfen. Ein normaler Modulimport würde vorher die zusätzlichen Torch-/Haiku-/TensorFlow-Abhängigkeiten aus W01 benötigen und kann B08 verdecken.

## 8. Quellen und Provenienz

Alle lokalen Zeilenangaben beziehen sich auf den oben genannten Arbeitsstand. Archiv-Hashes sichern zusätzlich die unversionierten experimentellen Quellen ab:

- `codes/Supervised_tasks.zip`, 1849452 Bytes, SHA-256 `99714221f7ff75fa3a0658313f2b9d0b9815e3d944f7736d261e24530172cdda`.
- `codes/Continual_CIFAR100_experiments.zip`, 2 Bytes, SHA-256 `7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6`.

**Q1 — Paper:** Sharifnassab, Salehkaleybar, Sutton, „MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters“. Geprüft wurden die versionierten HTML-Fassungen [arXiv v5 vom 4. Oktober 2024](https://arxiv.org/html/2402.02342v5) und [v6 vom 9. Juli 2025](https://arxiv.org/html/2402.02342v6), insbesondere Abschnitt 6, Algorithmus 3, Appendix A.1.2/A.1.3/A.3.1/A.3.2 sowie die Experimentbeschreibung und Tabellen in Appendix C. V5 liegt vor dem Repository-Commit, v6 danach. Diese Prüfung stützt die Einordnung der Konventionen; sie ist kein vollständiger Beweis der Paper-Äquivalenz oder Reproduktion seiner Ergebnisse.

**Q2 — Übernommener TinyStories-/Llama-Code:** Lokal enthaltenes Git-HEAD und Dateivergleich mit [karpathy/llama2.c, Commit d9862069e7ef665fe6309e3c17398ded2f121bf5](https://github.com/karpathy/llama2.c/tree/d9862069e7ef665fe6309e3c17398ded2f121bf5), 9. Oktober 2023. Keine Gleichsetzung mit dem heutigen Upstream-HEAD. Die Root-README benennt außerdem `pytorch/examples` als ImageNet-Vorlage, aber keinen genauen Stand; dieser wurde nicht rekonstruiert.

**Q3 — JAX-Verträge:** Offizieller versionierter Quelltext, nicht nur Suchmaschinenzusammenfassungen:

- [JAX 0.6.0, `jax/__init__.py`](https://github.com/jax-ml/jax/blob/jax-v0.6.0/jax/__init__.py): explizite Entfernung von `jax.tree_map`.
- [JAX 0.4.38, `jax/_src/api.py`](https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/api.py): `jit`-Dokumentation; nicht-arrayartige Argumente müssen statisch markiert sein.
- [JAX 0.4.38, `docs/pytrees.md`](https://github.com/jax-ml/jax/blob/jax-v0.4.38/docs/pytrees.md): Listen, Tupel und Dictionaries sind standardmäßig PyTree-Container.

**Q4 — PyTorch-DDP-Vertrag:** Warnung in der tatsächlich installierten `DistributedDataParallel.__doc__` von Torch 2.14.1 sowie [offizielle DDP-Dokumentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): keine Unterstützung für `torch.autograd.grad`, sondern Akkumulation in Parameter-`.grad`. Der Befund wurde zusätzlich unabhängig mit zwei echten Prozessen reproduziert.

Direkte Abrufe einiger Framework-Dokumentationsseiten wurden mit HTTP 403 abgewiesen. Für JAX wurden deshalb die genannten versionierten offiziellen Raw-Quellen gelesen, für DDP zusätzlich die installierte Dokumentation. Externe Quellen wurden am Prüfdatum abgerufen.

## 9. Priorisierte Maßnahmen

1. **Vor neuen langen Läufen die Trainingssemantik absichern:** TinyStories-Akkumulation und Clipping B01; Package-Blockzuordnung und Validierung B02/B04; JAX-inneres Produkt und Startpfad B03/B08. Zuerst kleine Tests, erst danach Datensatztraining.
2. **Nicht unterstützte Modi sichtbar sperren:** HF/DDP B07 sowie defekte ImageNet-Optimizer-/Blockoptionen B05/B06 dürfen nicht scheinbar regulär angeboten werden. Für fremde Chat-Eingaben B12 unmittelbar begrenzen oder den unsicheren Modus deaktivieren.
3. **Mathematische Sollvariante festlegen:** M01/M02 mit Gleichungen, Zeitindizes und effektiven Koeffizienten dokumentieren. Historische Varianten nicht stillschweigend vereinheitlichen. Baseline-Lernraten und Decay-Konventionen dazu passend festhalten.
4. **Reproduzierbare Runs ermöglichen:** Seeds und Budgetbegriff R01, eindeutige Artefaktpfade und korrekte Resume-Bezeichnung R02, vollständige effektive Konfiguration R03. Ein einziger kleiner dokumentierter Referenzlauf pro Datensatz ist wertvoller als viele nicht zuordenbare Ausgaben.
5. **Auswertung und Randpfade korrigieren:** Loss-/W&B-Logging B09, Export B10 und Tokenisierungsfehlerweitergabe B11. Diese Arbeiten mit gezielten Tests absichern, nicht nur durch Lint.
6. **Installation und Wartung vereinfachen:** Öffentliche API und Extras klären W01; aktive Pfade und Upstream-Provenienz dokumentieren W02; erst dann wirkungsgleiche Duplikate zusammenführen. Die datensatzfreie Optimizer-Testsuite T01 sollte jede dieser Änderungen begleiten.

## 10. Was nicht geprüft oder nicht ausgeführt werden konnte

- Keine vollständigen MNIST-, CIFAR-10-, ImageNet- oder TinyStories-Trainingsläufe und keine Reproduktion von Paper-Kurven, Endmetriken oder Konfidenzintervallen.
- Keine Prüfung der Continual-CIFAR100-Implementierung: das gelieferte Zwei-Byte-„Archiv“ enthält keinen prüfbaren Code.
- Kein echtes JAX/Optax, keine JIT-Ausführung und kein Framework-übergreifender numerischer Gleichheitsbeweis. Die JAX-Zahlenproben sind ausdrücklich Ersatzproben.
- Keine GPU-/CUDA-/MPS-, AMP-/float16-, NCCL- oder Mehrknotenprüfung. DDP wurde nur in dem beschriebenen kleinen Zwei-CPU-Prozess-Test geprüft.
- Kein Lauf unter den historischen README-Framework-Pins; keine frische Paketinstallation, kein vollständiger Wheel-/Install-Smoke und keine vollständige README-Datendemo.
- Keine Ausführung der zwei downloadabhängigen Python-Inferenztests mit pytest. Die C-Tokenizer-Fixtures wurden dagegen tatsächlich ausgeführt.
- Keine vollständige SentencePiece-Datenaufbereitung, keine Prüfung vorhandener echter Trainingsshards und keine Validierung der publizierten Daten-/Tokenizer-Versionen.
- Kein vollständiger Hugging-Face-/Meta-Llama-Import/Export-Roundtrip und keine numerische Gleichheit zwischen C-Inferenz und Python-Modell. C-Programme wurden kompiliert, aber nicht mit heruntergeladenen Sprachmodellgewichten evaluiert.
- Keine vollständige Sicherheitsprüfung untrusted Checkpoints, Tokenizerdateien, Tar-Archive oder C-Kernel; keine Windows-Portierungsprüfung. Der konkrete Chat-Pufferfehler ist nachgewiesen, ASan war wegen fehlender Runtime nicht verfügbar.
- Keine Laufzeit-/Speicherbenchmarks, keine Bewertung wissenschaftlicher Absichten und keine Aussage, dass ein bestimmter Fehler tatsächlich in den publizierten Messungen vorkam.

Eine abschließende SHA-256-Prüfung bestätigte: Keine der 43 zu Beginn erfassten bestehenden Nutzdateien wurde verändert. Der äußere Git-Diff für bereits getrackte Dateien ist leer; hinzugekommen ist ausschließlich dieser Report innerhalb des untersuchten Projektumfangs.

**Abschließendes Urteil:** Das Repository enthält nachvollziehbare Forschungsbausteine und einige funktionierende kleine Pfade. Seine wesentlichen Risiken liegen aber in stillen Trainings- und Integrationsfehlern sowie unklar abgegrenzten Algorithmusvarianten, nicht im Formatierungsstil. Vor belastbaren Vergleichen sollten die kurzen semantischen Tests, die wirksame Konfiguration und die mathematische Sollbeschreibung Vorrang vor größeren Refactorings oder neuen langen Trainingsläufen haben.
