# DQN_Project

Denne README er en kort oversigt over, hvad man kan finde i vores repository DQN_Project, som er brugt under ITIS-eksamen januar 2025.

Der optræder 5 mapper:

  - Prompts til DTU HPC:
    Her ligger alt det, vi har opgivet til DTU Data Centers High Performance Computer. Herunder findes     
    også vores spil 'Balls' samt vores 3 DQN's, som vi har trænet vores tre modeller på. I denne mappe er tre undermapper:
      - FloatPrompt: Prompten vi har opgivet for at træne Float-state-modellen.
        - BallsFloat.py: Noget med class osv.
        - DQNBallsFloat.py: Vores neurale netværk, environment osv, som vi har trænet på.
        - Float.sh: Den præcise activation file brugt i ssh cluster.
        - PlayBallsGameFloat.py: Scriptet, der kører spillet, hvis man vil prøve det:)
      - GridPrompt: --||--
      - OneHotPrompt --||--

  - Rå data (og få figurer): 30 .csv-filer fordelt ud på 3 modeller samt et plot for hver. Vi har primært brugt 'Episode' og 'Score'.

  - Scripts og notebooks:
    - graphthat.py: Brugt til at lave de tre plots af hver model.
    - SuccesCrit.py: Brugt til at identificere vores succeskriterie, som er et score-gennemsnit over 10 episoder, som er mindst 90 score.
    - means.txt: Tekstfil, hvor vi har noteret middelværdier.
    - Stats - Basic.ipynb: Statistikværktøjer til beregning af middelværdi, spredning og konfidensintervaller, som i meget stor grad kan findes i pensum. 
    - Stats - Advanced.ipynb: Statistikværktøjer til beregning af proportionsforskel, variansforskel og simulation, der låner fra 02402 Statistics pensum.

  - Mikkels løsning: Indeholder Mikkels løsning på spillet SpaceInvaders. Vi har draget stor inspiration herfra. Især når koden har drillet. Hvilken den har gjort. Meget.

  - t-table.pdf: Brugt i udregninger.

  - README.txt: Denne fil :)
