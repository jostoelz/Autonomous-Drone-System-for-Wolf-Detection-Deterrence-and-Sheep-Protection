# Read Me

## Allgemeines

Dieses Template kann verwendet werden um selbständige Arbeiten (MA, SA, SLA) an der Kantonsschule Romanshorn zu schreiben.

Bei Fragen, Unklarheiten oder falls Fehler gefunden werden, bitte melden bei sca@ksr.ch

## Arbeiten mit dem Template

### LaTeX einrichten

   1. Zuerst muss **LaTeX installiert** werden:
      * Windows: https://www.tug.org/texlive/windows.html (klicke aus `install-tl-windows.exe')
      * Mac https://www.tug.org/mactex/mactex-download.html (klicke auf `MacTeX.pkg.')
    \end{itemize}
   1. Es wird empfohlen, mit dem Editor **Visual Studio Code** (kurz VSCode) zu arbeiten: https://code.visualstudio.com
   1. **LaTeX-Workshop-Extension** für VSCode installieren:
      1. Extensions-Reiter öffnen
      1. Nach "LaTeX Workshop" von James Yu suchen ...
      1. ... und installieren

### Erste Schritte mit LaTeX

   1. Entzippe den Ordner mit dem Template und verschiebe diesen an einen passenden Speicherort.
   1. Öffne den Ordner in VSCode:
      1. VSCode öffnen
      1. Datei / Ordner öffnen / Ordner auswählen
   1. Auf der linken Seite sollte nun der Inhalt des ganzen Ordners angezeigt werden.
   1. Das wichtigste File ist **"main.tex"**. Dieses kannst du passende umbenennen (z.B: "maturaarbeit.tex", "sla.tex" oder ...).
   1. Öffne nun das File: In dieses schreibst du den Inhalt deiner Arbeit. Nehme Änderungen vor.
   1. TeX-File **Kompilieren**:
      * Nun muss das File kompiliert, also in ein PDF umgewandelt, werden.
      * Dies sollte immer geschehen, wenn du speicherst.
      * Alternativ kann man auch manuell kompilieren:
        1. Command palette öffnen: Ctrl + Shift + P (Win), Shift + Command + P (Mac)
        1. "LaTeX Workshop: Build with recipe"
        1. "latexmk" oder "pdflatex -> bibtex -> pdflatex x2" (falls Quellen nicht richtig kompiliert werden)

### Wichtigste Files im Template

   * **main.tex:** Das **Hauptfile**, hier wird $95\%$ der Arbeit erledigt.
   * **biblio.bib**:
      * Hier werden alle **Quellen** erfasst.
      * Achte darauf, dass die Notation genau stimmt.
      * Oft kann man bei Artikeln oder Büchern den entsprechenden Eintrag aus dem Internet kopieren.
      * Es gibt gute Programme (sollte direkt mit LaTeX installiert werden), um die Quellen in diesem File zu verwalten.
   * **commands.tex**: Hier sind einige Kommandos definiert, die es einem erlauben, den Code kürzer zu halten, z.B. \f anstelle \frac um Brüche zu schreiben. Diese können ignoriert, angepasst und erweitert werden.
   * **code_formatting.tex**: Nur relevant für Arbeiten, in denen Programmiert wird. Hier wird definiert, wie Programmiercode dargestellt werden soll. Aktuell erfasst Programmiersprachen sind: Python, C#, JavaScript. Dieses File kann verändert und für andere Programmiersprachen erweitert werden. Falls du dies machst, bin ich dir dankbar, wenn du mir deine Version zukommenlassen kannst (sca@ksr.ch), damit ich das Template damit erweitern kann.
   * **makefile**: Erlaubt es einem, im Terminal mit "make clean" alle Files, die beim Kompilieren erstellt werden, wieder zu löschen und nur die ursprünglichen zu behalten.