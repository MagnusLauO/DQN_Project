import pandas as pd
import matplotlib.pyplot as plt

# Definer modermappen
parent_folder = "C:/Users/Magnu/OneDrive - Danmarks Tekniske Universitet/Documents/GitHub/ITIS_DQN_Project/Rå data"

# Indlæs CSV-filerne og plottes data
for i in range(1, 11):  # Antag at filerne hedder 1.csv, 2.csv, ...,
    file_path = f"{parent_folder}/OneHot{i}.csv"
    df = pd.read_csv(file_path)  # Læs filen
    
    # Antag at kolonnenavnene er "Episode" og "Score"
    plt.scatter(df["Episode"], df["Score"], label=f"OneHot{i}", s=10)

# Tilpas plottet
plt.title("OneHot")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

# Vis plottet
plt.show()