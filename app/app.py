import os
import sys
import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import random

# ────────────────────────────────
#   Make parent folder importable
# ────────────────────────────────
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# project-level helpers
from app_helpers import (
    add_new_user,
    add_rating,
    expand_model_for_new_users,
    personalize_model_for_user,
    save_trained_model,
)
from utils.data_loader import (
    load_cleaned_books,
    load_cleaned_users,
    load_cleaned_ratings,
    load_trained_model,
)


# ────────────────────────────────
#   Streamlit caching wrappers
# ────────────────────────────────
@st.cache_data
def get_books_df():
    return load_cleaned_books()


@st.cache_data
def get_users_df():
    return load_cleaned_users()


@st.cache_data
def get_ratings_df():
    return load_cleaned_ratings()


@st.cache_resource
def get_model():
    return load_trained_model()


def refresh_caches():
    """Clear cached @st.cache_data objects after we append to CSVs."""
    get_users_df.clear()
    get_ratings_df.clear()


# ────────────────────────────────
#   Authentication helper
# ────────────────────────────────

def authenticate(username: str, password: str) -> int | None:
    users = get_users_df()
    hit = users[
        (users["User-Name"] == username) &
        (users["Password"] == password)
        ]
    return None if hit.empty else int(hit.iloc[0]["User-ID"])


# ────────────────────────────────
#   UI helper : show_book_card
# ────────────────────────────────

def show_book_card(
        book: dict,
        idx: int | None = None,
        *,
        with_rating_slider: bool = False,
        slider_key: str | None = None,
        predicted_rating: float | None = None
):
    """
    Display a book cover + its metadata.
    """

    def gv(key: str, default="Unknown"):
        val = book.get(key) or book.get(key.replace('-', '_'))
        if val in ("", None, np.nan, "unknown", "Unknown"):
            return default
        return val

    cols = st.columns([1, 3])
    # choose the best cover
    cover_url = gv("Image-URL-L", "") or gv("Image-URL-M", "") or gv("Image-URL-S", "")
    if not cover_url:
        cover_url = f"https://picsum.photos/120/180?random={idx}"  # fallback

    with cols[0]:
        st.image(cover_url, use_column_width=True)

    with cols[1]:
        st.markdown(f"### {gv('Book-Title')}")
        st.markdown(f"**Author:** {gv('Book-Author')}")
        st.markdown(f"**Year:** {gv('Year-Of-Publication')}")
        st.markdown(f"**Publisher:** {gv('Publisher')}")
        if predicted_rating is not None:
            st.markdown(f"**Predicted rating:** {predicted_rating:.1f} / 10")
        if with_rating_slider:
            return st.slider("Your rating", 1, 10, 5, key=slider_key)

    st.markdown("---")
    return None


# ────────────────────────────────
#   Wrapper for model.recommend signature
# ────────────────────────────────

def get_recommendations(model, user_id: int, ratings_df, n: int = 10):
    try:
        return model.recommend(user_id, N=n, df=ratings_df)
    except TypeError:
        return model.recommend(user_id, N=n)


# ────────────────────────────────
#   Get popular books with caching
# ────────────────────────────────
@st.cache_data
def get_popular_books(ratings_df, books_df, n=500):
    """Get top n popular books by rating count."""
    pop = (ratings_df.groupby("ISBN")
           .size().reset_index(name="count")
           .sort_values("count", ascending=False))
    pop = pop.merge(books_df, on="ISBN").dropna(subset=["Book-Title", "Book-Author"])
    return pop.head(n)


# ────────────────────────────────
#   Streamlit application
# ────────────────────────────────
def main():
    st.set_page_config(page_title="📚 Book Recommender", layout="wide")
    st.title("📚 Book Recommendation System")

    # ------- Initialise session state -------
    defaults = {
        "username": "",
        "user_id": None,
        "selected_isbn": None,
        "nav_page": "Dashboard",
        "previous_page": None,
        "displayed_popular_books": []  # Track displayed popular books
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # =========================================================
    #   NOT LOGGED-IN  →  Sign-In / Register
    # =========================================================
    if not st.session_state.username:
        st.info("Please sign in or register to continue")
        tab_si, tab_reg = st.tabs(["Sign In", "Register"])
        with tab_si:
            u = st.text_input("Username", key="si_user")
            p = st.text_input("Password", type="password", key="si_pass")
            if st.button("Sign In", key="btn_signin"):
                uid = authenticate(u.strip(), p.strip())
                if uid:
                    st.session_state.update(
                        username=u.strip(),
                        user_id=uid,
                        nav_page="Dashboard"
                    )
                    st.success(f"Welcome back, {u}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
        with tab_reg:
            ru = st.text_input("Choose a username", key="reg_user")
            rp = st.text_input("Choose a password", type="password", key="reg_pass")
            rloc = st.text_input("Location (optional)")
            rage = st.number_input("Age", 1, 120, 25)
            if st.button("Register", key="btn_register"):
                users = get_users_df()
                if not ru or not rp:
                    st.error("Username & password cannot be empty")
                elif ru in users["User-Name"].values:
                    st.error("Username already taken")
                elif len(rp) < 4:
                    st.error("Password must be ≥ 4 characters")
                else:
                    new_id = int(users["User-ID"].max()) + 1
                    ok = add_new_user(
                        ru.strip(), rp.strip(),
                        user_id=new_id, loc=rloc.strip() or "Unknown",
                        Age=int(rage) if rage else None
                    )
                    if ok:
                        refresh_caches()
                        model = get_model()
                        model = expand_model_for_new_users(model, get_ratings_df())
                        save_trained_model(model)
                        st.session_state.update(
                            username=ru.strip(),
                            user_id=new_id,
                            nav_page="Dashboard"
                        )
                        st.success("Registered – welcome!")
                        st.experimental_rerun()
        return

    # =========================================================
    #   LOGGED-IN  →  Sidebar navigation
    # =========================================================
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Sign Out"):
        st.session_state.clear()
        st.experimental_rerun()

    # Store current page in session state before navigation
    current_page = st.session_state.get("nav_page", "Dashboard")

    nav_options = ["Dashboard", "Popular Books For You", "Recommendations", "Book Details"]
    page = st.sidebar.radio("Navigate", nav_options,
                            index=nav_options.index(current_page),
                            key="nav_radio")

    # Update navigation state if changed
    if page != current_page:
        st.session_state.previous_page = current_page
        st.session_state.nav_page = page
        if page != "Book Details":
            st.session_state.selected_isbn = None
        st.experimental_rerun()

    books_df = get_books_df()
    ratings_df = get_ratings_df()
    user_id = st.session_state.user_id

    # ---------------------------------------------------------
    #   Dashboard
    # ---------------------------------------------------------
    if page == "Dashboard":
        st.header("📊 Dashboard")

        # ISBN Search Section
        st.subheader("Search by ISBN")
        isbn_search = st.text_input("Enter ISBN to search for a book:", key="isbn_search")
        if isbn_search:
            match = books_df[books_df["ISBN"] == isbn_search.strip()]
            if not match.empty:
                book = match.iloc[0].to_dict()
                show_book_card(book, idx=0)
                if st.button("View Details", key=f"isbn_dtl_{isbn_search}"):
                    st.session_state.selected_isbn = isbn_search.strip()
                    st.session_state.previous_page = page
                    st.session_state.nav_page = "Book Details"
                    st.experimental_rerun()
            else:
                st.error("No book found with that ISBN.")

        # Regular search bar
        search = st.text_input("Search books by title or author", key="dashboard_search")
        if search:
            mask = (
                    books_df["Book-Title"].str.contains(search, case=False, na=False) |
                    books_df["Book-Author"].str.contains(search, case=False, na=False)
            )
            results = books_df[mask]
            if results.empty:
                st.info("No books found matching your search.")
            else:
                st.markdown(f"**Search results ({len(results)})**")
                for _, row in results.head(10).iterrows():
                    book = row.to_dict()
                    show_book_card(book, idx=0)
                    if st.button("View Details", key=f"search_dtl_{row.ISBN}"):
                        st.session_state.selected_isbn = row.ISBN
                        st.session_state.previous_page = page
                        st.session_state.nav_page = "Book Details"
                        st.experimental_rerun()

        st.markdown(f"- **User ID:** {user_id}")
        user_ratings = ratings_df[ratings_df["User-ID"] == user_id]
        st.markdown(f"- **Books rated so far:** {len(user_ratings)}")
        st.markdown("- **Actions**:\n  - Rate popular books for you\n  - See your recommendations")

    # ---------------------------------------------------------
    #   Popular Books For You
    # ---------------------------------------------------------
    elif page == "Popular Books For You":
        st.header("📚 Popular Books For You")

        # Get top 500 popular books
        popular_books = get_popular_books(ratings_df, books_df, n=500)

        # Get user's rated books
        user_rated = ratings_df[ratings_df["User-ID"] == user_id]["ISBN"].tolist()

        # Filter out books the user has already rated
        available_books = popular_books[~popular_books["ISBN"].isin(user_rated)]

        # Select random books to display (5 at a time)
        if "displayed_popular_books" not in st.session_state or not st.session_state.displayed_popular_books:
            if len(available_books) > 0:
                sample_size = min(5, len(available_books))
                st.session_state.displayed_popular_books = random.sample(available_books["ISBN"].tolist(), sample_size)

        if not st.session_state.displayed_popular_books:
            st.info("You've rated all the popular books we have! Check out your recommendations.")
        else:
            st.info("Rate these popular books to get better recommendations!")

            for isbn in st.session_state.displayed_popular_books:
                book_row = books_df[books_df["ISBN"] == isbn].iloc[0].to_dict()
                rating = show_book_card(book_row, idx=0, with_rating_slider=True, slider_key=f"rate_{isbn}")

                if st.button(f"Submit Rating for {book_row['Book-Title']}", key=f"btn_rate_{isbn}"):
                    if add_rating(user_id, isbn, rating):
                        refresh_caches()
                        model = get_model()
                        model = expand_model_for_new_users(model, get_ratings_df())
                        model = personalize_model_for_user(model, user_id, get_ratings_df())
                        save_trained_model(model)
                        st.success("Rating saved and model updated!")

                        # Remove this book from displayed books
                        st.session_state.displayed_popular_books.remove(isbn)

                        # Clear the cache to get fresh recommendations
                        get_popular_books.clear()

                        st.experimental_rerun()

                if st.button(f"View Details for {book_row['Book-Title']}", key=f"pop_dtl_{isbn}"):
                    st.session_state.selected_isbn = isbn
                    st.session_state.previous_page = page
                    st.session_state.nav_page = "Book Details"
                    st.experimental_rerun()

            # Button to refresh the displayed books
            if st.button("Show me different books"):
                st.session_state.displayed_popular_books = []
                st.experimental_rerun()

    # ---------------------------------------------------------
    #   Recommendations
    # ---------------------------------------------------------
    elif page == "Recommendations":
        st.header("🎯 Your Recommendations")
        model = get_model()
        if not hasattr(model, "_is_personalized") or not getattr(model, "_is_personalized"):
            model = expand_model_for_new_users(model, ratings_df)
            model = get_model()
            model = expand_model_for_new_users(model, get_ratings_df())
            model = personalize_model_for_user(model, user_id, get_ratings_df())
            model._is_personalized = True
        rec_isbns = get_recommendations(model, user_id, ratings_df, n=10)
        if not rec_isbns:
            st.info("No recommendations yet — please rate more books first.")
        else:
            for isbn in rec_isbns:
                if isbn not in books_df["ISBN"].values:
                    continue
                book_row = books_df[books_df["ISBN"] == isbn].iloc[0].to_dict()
                pred = model.predict(user_id, isbn)
                show_book_card(book_row, predicted_rating=pred)
                if st.button("View Details", key=f"dtl_{isbn}"):
                    st.session_state.selected_isbn = isbn
                    st.session_state.previous_page = page
                    st.session_state.nav_page = "Book Details"
                    st.experimental_rerun()

    # ---------------------------------------------------------
    #   Book Details
    # ---------------------------------------------------------
    elif page == "Book Details":
        st.header("📖 Book Details")
        isbn = st.session_state.get("selected_isbn")

        if not isbn:
            st.info("Use the search or navigation to select a book to view details.")
            if st.button("← Back to Dashboard"):
                st.session_state.nav_page = "Dashboard"
                st.experimental_rerun()
            return

        if isbn:
            match = books_df[books_df["ISBN"] == isbn]
            if not match.empty:
                book = match.iloc[0].to_dict()
                show_book_card(book, idx=0)
                st.markdown(f"**ISBN:** {isbn}")

                # Back button
                if st.button("← Back"):
                    previous_page = st.session_state.get("previous_page", "Dashboard")
                    st.session_state.nav_page = previous_page
                    st.session_state.selected_isbn = None
                    st.experimental_rerun()

                # Rating section
                already = isbn in ratings_df[ratings_df["User-ID"] == user_id]["ISBN"].values
                if not already:
                    r = st.slider("Your rating", 1, 10, 5, key="detail_rate")
                    if st.button("Rate this book", key="btn_rate_detail"):
                        if add_rating(user_id, isbn, r):
                            refresh_caches()
                            model = get_model()
                            model = expand_model_for_new_users(model, get_ratings_df())
                            model = personalize_model_for_user(model, user_id, get_ratings_df())
                            save_trained_model(model)
                            st.success("Rating saved and model updated!")
                            # Stay on the same page after rating
                            st.session_state.nav_page = "Book Details"
                            st.experimental_rerun()
                else:
                    st.info("You have already rated this book.")
            else:
                st.error("No book found with that ISBN.")


if __name__ == "__main__":
    main()