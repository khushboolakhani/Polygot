# 🎧 Cross-Language Music Recommender

A personalized music recommendation system that connects to your real Spotify
listening history and suggests similar-sounding songs in **other languages**
(French, Japanese, Spanish, Korean, and more) — built with an interactive
Streamlit interface.

## How it works

1. **Authenticates with Spotify** via OAuth 2.0 to read your top tracks
   (`user-top-read` scope only — no write access, no playback control).
2. **Builds a taste profile** by averaging audio features (danceability,
   energy, valence, tempo, etc.) across your top tracks.
3. **Compares your profile** against a 114,000-track reference dataset using
   cosine similarity to find sonically similar songs.
4. **Boosts non-English matches** using a genre-to-language mapping, so
   recommendations lean toward music discovery across languages rather than
   just "more of what you already listen to."
5. Results are shown as filterable, styled cards in a Streamlit UI — filter
   by language, adjust how many results, and choose your listening timeframe.

## A real engineering challenge I hit

Spotify deprecated its `/audio-features` endpoint for all apps created after
November 2024 — so this app **can't call that endpoint directly** (it returns
a permanent 403 for new developer apps). Instead of dropping the feature,
I worked around it: your top track *names* are still fetched live from
Spotify, then matched against the reference dataset (which already includes
audio features) to reconstruct your taste profile. Tracks not found in the
reference dataset are gracefully excluded rather than breaking the app.

## Setup

1. **Get Spotify API credentials**
   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) → Create app
   - Set Redirect URI to exactly: `http://127.0.0.1:8501`
   - Copy your Client ID and Client Secret

2. **Create a `.env` file** in the project folder (see `.env.example`):
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

3. **Install dependencies**
   ```
   pip install streamlit spotipy pandas scikit-learn python-dotenv
   ```

4. **Run it**
   ```
   streamlit run app.py
   ```
   Open `http://127.0.0.1:8501` in your browser, click "Log in with Spotify,"
   and approve access.

## Tech stack
Python · Streamlit · Spotify Web API (OAuth 2.0) · pandas · scikit-learn

## Dataset
[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
— 114,000 tracks across 114 genres with Spotify's audio features.

## Known limitations
- No release-year/decade data in the reference dataset, so decade-based
  filtering isn't implemented.
- Language tags are inferred from genre labels (e.g. `spanish`, `j-pop`),
  not a verified language field — Spotify doesn't publish one.
- Your top tracks are only included in the taste profile if they also
  exist in the reference dataset (a limitation of the audio-features
  workaround described above).

## Possible extensions
- Add a decade dimension via a dataset that includes release year
- Swap the reference-dataset workaround for a live feature-extraction
  service (e.g. Essentia on audio previews) for full track coverage
- Deploy publicly (current version assumes local use due to the loopback
  redirect URI)
