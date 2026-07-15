"""
Cross-Language Music Recommender -- Streamlit GUI
---------------------------------------------------
Run with: streamlit run app.py

SETUP (same as before):
1. Spotify Developer Dashboard -> Create app -> Redirect URI: http://127.0.0.1:8888/callback
2. Paste Client ID / Secret below
3. pip install streamlit spotipy pandas scikit-learn
4. spotify_dataset.csv must be in the same folder
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ---------------------------------------------------------
# CREDENTIALS -- paste yours here
# ---------------------------------------------------------
CLIENT_ID = "PASTE_YOUR_CLIENT_ID_HERE"
CLIENT_SECRET = "PASTE_YOUR_CLIENT_SECRET_HERE"
# Must exactly match the Redirect URI set in your Spotify Dashboard app settings
REDIRECT_URI = "http://127.0.0.1:8501"
SCOPE = "user-top-read"

FEATURES = ["danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "speechiness", "loudness"]

GENRE_TO_LANGUAGE = {
    "french": "French", "spanish": "Spanish", "latin": "Spanish", "latino": "Spanish",
    "reggaeton": "Spanish", "salsa": "Spanish", "german": "German", "swedish": "Swedish",
    "turkish": "Turkish", "iranian": "Persian", "k-pop": "Korean", "mandopop": "Mandarin",
    "cantopop": "Cantonese", "j-pop": "Japanese", "j-rock": "Japanese", "j-idol": "Japanese",
    "j-dance": "Japanese", "anime": "Japanese", "indian": "Hindi", "brazil": "Portuguese",
    "mpb": "Portuguese", "sertanejo": "Portuguese", "forro": "Portuguese",
    "pagode": "Portuguese", "samba": "Portuguese",
}

# A small color per language, used for the chip/badge in each result card
LANGUAGE_COLORS = {
    "English": "#4A90D9", "French": "#7B68EE", "Spanish": "#E07A5F", "German": "#6B8E23",
    "Swedish": "#3E92CC", "Turkish": "#D62839", "Persian": "#B5838D", "Korean": "#F26419",
    "Mandarin": "#C1121F", "Cantonese": "#780000", "Japanese": "#EF476F",
    "Hindi": "#F4A261", "Portuguese": "#2A9D8F",
}

st.set_page_config(page_title="Cross-Language Music Recommender", page_icon="🎧", layout="wide")


# ---------------------------------------------------------
# Cached data + auth (so we don't re-fetch on every filter change)
# ---------------------------------------------------------
@st.cache_resource
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        open_browser=False,  # we handle the redirect manually inside Streamlit
        cache_path=".spotify_cache",
    )


def get_spotify_client():
    """
    Handles Spotify login inside Streamlit's request/response model:
    1. If we already have a cached token, use it.
    2. Otherwise, if Spotify just redirected back here with a `code` in the URL,
       exchange that code for a token.
    3. Otherwise, show a login link and stop until the user clicks it.
    """
    auth_manager = get_auth_manager()

    token_info = auth_manager.get_cached_token()

    if not token_info:
        query_params = st.query_params
        code = query_params.get("code")
        if code:
            token_info = auth_manager.get_access_token(code, as_dict=True, check_cache=False)
            st.query_params.clear()  # clean the URL after using the code
        else:
            auth_url = auth_manager.get_authorize_url()
            st.title("🎧 Cross-Language Music Recommender")
            st.markdown("### Connect your Spotify account to get started")
            st.link_button("Log in with Spotify", auth_url, type="primary")
            st.caption("You'll be redirected back here automatically after logging in.")
            st.stop()

    return spotipy.Spotify(auth=token_info["access_token"])


@st.cache_data
def load_catalogue():
    df = pd.read_csv("spotify_dataset.csv", index_col=0)
    df = df.drop_duplicates(subset="track_name").reset_index(drop=True)
    df["language"] = df["track_genre"].map(GENRE_TO_LANGUAGE).fillna("English")
    return df


@st.cache_data(show_spinner="Fetching your top tracks from Spotify...")
def fetch_my_tracks(_sp, time_range):
    results = _sp.current_user_top_tracks(limit=50, time_range=time_range)
    tracks = [{"track_name": i["name"],
               "artists": ", ".join(a["name"] for a in i["artists"])}
              for i in results["items"]]
    return pd.DataFrame(tracks)


def match_features_from_catalogue(my_tracks, catalogue):
    """
    Spotify's audio-features endpoint is deprecated for new apps (permanent
    403 as of Nov 2024) -- there's no official replacement. As a workaround,
    we match each of your top tracks against the public catalogue by track
    name (case-insensitive) to reuse its audio features instead.
    Tracks with no match in the catalogue are dropped, since we have no
    feature data for them.
    """
    catalogue_lookup = catalogue.drop_duplicates(subset="track_name").set_index(
        catalogue["track_name"].str.lower()
    )
    matched_rows = []
    unmatched = []
    for _, row in my_tracks.iterrows():
        key = row["track_name"].lower()
        if key in catalogue_lookup.index:
            match = catalogue_lookup.loc[key]
            if isinstance(match, pd.DataFrame):  # more than one match, take first
                match = match.iloc[0]
            matched_rows.append(match[FEATURES])
        else:
            unmatched.append(row["track_name"])

    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=FEATURES)
    return matched_df, unmatched


def compute_recommendations(my_tracks, catalogue, n, boost_non_english):
    scaler = StandardScaler()
    catalogue_scaled = scaler.fit_transform(catalogue[FEATURES])
    my_profile = scaler.transform(my_tracks[FEATURES].mean().values.reshape(1, -1))

    sims = cosine_similarity(my_profile, catalogue_scaled)[0]
    result = catalogue.copy()
    result["similarity"] = sims
    boost = 0.2 if boost_non_english else 0.0
    result["adjusted_score"] = result["similarity"] + (result["language"] != "English").astype(float) * boost
    return result.sort_values("adjusted_score", ascending=False)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.title("🎧 Filters")
time_range = st.sidebar.selectbox(
    "Your taste profile based on:",
    ["short_term", "medium_term", "long_term"],
    index=1,
    format_func=lambda x: {"short_term": "Last 4 weeks", "medium_term": "Last 6 months", "long_term": "All time"}[x],
)
boost_non_english = st.sidebar.checkbox("Prioritize non-English recommendations", value=True)
num_results = st.sidebar.slider("Number of recommendations", 5, 30, 12)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
st.title("🎧 Cross-Language Music Recommender")
st.caption("Built from your real Spotify listening history — discover similar-sounding songs in new languages.")

catalogue = load_catalogue()
all_languages = sorted(catalogue["language"].unique())
selected_languages = st.sidebar.multiselect("Show only these languages", all_languages, default=all_languages)

sp = get_spotify_client()

try:
    my_tracks = fetch_my_tracks(sp, time_range)
except Exception as e:
    st.error(f"Couldn't fetch your Spotify data: {e}")
    st.stop()

with st.expander(f"Your top tracks ({time_range.replace('_', ' ')})", expanded=False):
    st.dataframe(my_tracks[["track_name", "artists"]], use_container_width=True, hide_index=True)

matched_features, unmatched = match_features_from_catalogue(my_tracks, catalogue)

if matched_features.empty:
    st.warning(
        "None of your top tracks were found in the reference dataset used for audio "
        "features (Spotify's own audio-features API is no longer available to new apps). "
        "Try a different listening timeframe in the sidebar, or this account's top tracks "
        "may skew toward songs outside the 114k-track reference dataset."
    )
    st.stop()

if unmatched:
    st.caption(f"Note: {len(unmatched)} of your top {len(my_tracks)} tracks weren't found in the "
               f"reference dataset and were excluded from your taste profile.")

my_profile_source = matched_features
recs = compute_recommendations(my_profile_source, catalogue, num_results, boost_non_english)
recs = recs[recs["language"].isin(selected_languages)].head(num_results)

st.subheader(f"Recommended for you ({len(recs)} tracks)")

# ---------------------------------------------------------
# Render as attractive cards, 3 per row
# ---------------------------------------------------------
cols = st.columns(3)
for idx, (_, row) in enumerate(recs.iterrows()):
    color = LANGUAGE_COLORS.get(row["language"], "#888888")
    with cols[idx % 3]:
        st.markdown(
            f"""
            <div style="border:1px solid #333; border-radius:12px; padding:16px; margin-bottom:16px; background:#1a1a1a;">
                <div style="display:inline-block; background:{color}; color:white; padding:2px 10px;
                            border-radius:999px; font-size:12px; font-weight:600; margin-bottom:8px;">
                    {row['language']}
                </div>
                <h4 style="margin:4px 0 2px 0; color:#fff;">{row['track_name']}</h4>
                <p style="margin:0; color:#bbb; font-size:14px;">{row['artists']}</p>
                <p style="margin:6px 0 0 0; color:#888; font-size:12px;">
                    {row['track_genre']} · {row['tempo']:.0f} BPM · match {row['similarity']*100:.0f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
