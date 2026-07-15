"""
Personalized Cross-Language Music Recommender (using YOUR Spotify data)
------------------------------------------------------------------------
Connects to your Spotify account via OAuth, pulls your top tracks + their
audio features, then recommends similar-sounding songs in a DIFFERENT
language using the same genre-to-language + cosine similarity approach
as the dataset-based version.

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth


CLIENT_ID = "26f6e5d144b149109fc7e93561256fc1"
CLIENT_SECRET = "02ed3157e1d2485fa0c7af9608bd9de3"
REDIRECT_URI = "http://127.0.0.1:8501"

# Scopes = what data we're allowed to request
SCOPE = "user-top-read"

# ---------------------------------------------------------
# 2. Authenticate (opens a browser tab the first time)
# ---------------------------------------------------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    open_browser=True,
))

# ---------------------------------------------------------
# 3. Fetch your top tracks (try "medium_term" = last ~6 months)
# ---------------------------------------------------------
def fetch_top_tracks(time_range="medium_term", limit=50):
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks = []
    for item in results["items"]:
        tracks.append({
            "track_id": item["id"],
            "track_name": item["name"],
            "artists": ", ".join(a["name"] for a in item["artists"]),
        })
    return pd.DataFrame(tracks)

print("Fetching your top tracks from Spotify...")
my_tracks = fetch_top_tracks()
print(f"Got {len(my_tracks)} tracks.\n")
print(my_tracks[["track_name", "artists"]].to_string(index=False))

# ---------------------------------------------------------
# 4. Fetch audio features for those tracks
# ---------------------------------------------------------
def fetch_audio_features(track_ids):
    features = sp.audio_features(track_ids)
    return pd.DataFrame(features)

print("\nFetching audio features...")
features_df = fetch_audio_features(my_tracks["track_id"].tolist())
my_tracks = my_tracks.merge(features_df, left_on="track_id", right_on="id")

# ---------------------------------------------------------
# 5. Merge with the public genre-tagged dataset to find cross-language matches
#    (your own tracks give the "taste profile"; the big dataset supplies
#    the pool of candidate songs to recommend, including non-English ones)
# ---------------------------------------------------------
FEATURES = ["danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "speechiness", "loudness"]

catalogue = pd.read_csv("spotify_dataset.csv", index_col=0)
catalogue = catalogue.drop_duplicates(subset="track_name").reset_index(drop=True)

GENRE_TO_LANGUAGE = {
    "french": "French", "spanish": "Spanish", "latin": "Spanish", "latino": "Spanish",
    "reggaeton": "Spanish", "salsa": "Spanish", "german": "German", "swedish": "Swedish",
    "turkish": "Turkish", "iranian": "Persian", "k-pop": "Korean", "mandopop": "Mandarin",
    "cantopop": "Cantonese", "j-pop": "Japanese", "j-rock": "Japanese", "j-idol": "Japanese",
    "j-dance": "Japanese", "anime": "Japanese", "indian": "Hindi", "brazil": "Portuguese",
    "mpb": "Portuguese", "sertanejo": "Portuguese", "forro": "Portuguese",
    "pagode": "Portuguese", "samba": "Portuguese",
}
catalogue["language"] = catalogue["track_genre"].map(GENRE_TO_LANGUAGE).fillna("English")

scaler = StandardScaler()
catalogue_scaled = scaler.fit_transform(catalogue[FEATURES])

# Build YOUR taste profile as the average of your top tracks' features
my_profile = my_tracks[FEATURES].mean().values.reshape(1, -1)
my_profile_scaled = scaler.transform(my_profile)

sims = cosine_similarity(my_profile_scaled, catalogue_scaled)[0]
catalogue["similarity"] = sims

# Prioritize non-English results since that's the discovery goal
catalogue["adjusted_score"] = catalogue["similarity"] + \
    (catalogue["language"] != "English").astype(float) * 0.2

recs = catalogue.sort_values("adjusted_score", ascending=False).head(15)

print("\n" + "=" * 70)
print("Recommended cross-language songs based on YOUR taste profile:")
print("=" * 70)
print(recs[["track_name", "artists", "track_genre", "language", "tempo"]].to_string(index=False))
