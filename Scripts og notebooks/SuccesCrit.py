import pandas as pd

# Indlæs CSV-filen
cat = "OneHot"
no = "10"

file_name = f"C:/Users/Magnu/OneDrive - Danmarks Tekniske Universitet/Documents/GitHub/ITIS-Project/New/1000episodes/OneHot/{cat}{no}.csv"  # Skift til navnet på din fil
df = pd.read_csv(file_name)

# Tjek, at de nødvendige kolonner findes
if "Episode" in df.columns and "Score" in df.columns:
    # Sorter data efter Episode, hvis nødvendigt
    df = df.sort_values("Episode").reset_index(drop=True)

    # Loop gennem episoder, og tjek gennemsnit af de seneste 10 Scores
    for i in range(len(df)):
        if i >= 9:  # Start først fra den 10. episode
            recent_scores = df.loc[i-9:i, "Score"]  # Udvælg de seneste 10 scores (loc inkluderer også i)
            avg_score = recent_scores.mean()       # Beregn gennemsnittet
            
            if avg_score >= 90:
                episode = df.loc[i, "Episode"]  # Få episodenummeret
                print(f"I {cat}{no}: Gennemsnittet overstiger 90 ved Episode {episode}.")
                break  # Stop loopet, da vi har fundet en episode, der opfylder kriteriet
