# Content-Based Movie Recommender

Homework 2 project built with the MovieLens 100K dataset. The application recommends Top-5 unseen movies using genre features and displays a score for every result.

**[Open the live demo](https://predprof.github.io/movie-recommender-system/)**

## Recommendation modes

- Item-to-item cosine similarity
- Item-to-item Jaccard similarity
- Profile-based cosine similarity using a user's three top-rated movies
- Single-item vs. aggregated-profile comparison with a chronological 5-star holdout

## Run locally

Start a static web server in this directory:

```bash
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).
