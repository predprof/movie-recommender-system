# Graph-Based Movie Recommender

Homework 6 project built with the MovieLens 100K dataset. The application recommends Top-10 unseen movies using positive training ratings and compares four recommendation methods.

**[Open the live demo](https://predprof.github.io/movie-recommender-system/)**

## Recommendation modes

- Popularity based on positive training ratings
- Global PageRank with uniform restart
- Personalized PageRank (PPR) with restart at the selected user
- Matrix factorization (MF) baseline
- Validation alpha sweep, Recall@10 evaluation, and a step-by-step study fixture

## Run locally

From this directory, download D3 7.9.0 if `vendor/d3.min.js` is missing, then start a static web server:

```bash
mkdir -p vendor
curl -fL https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js -o vendor/d3.min.js
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). Internet access is required for TensorFlow.js 4.22.0, loaded from a CDN.
