const PROFILE_MOVIE_COUNT = 3;
const TOP_RECOMMENDATION_COUNT = 5;
const COMPARISON_HELD_OUT_RATING = 5;

// Use engine-independent code-point ordering instead of locale-sensitive collation.
function compareTitles(firstTitle, secondTitle) {
    if (firstTitle < secondTitle) return -1;
    if (firstTitle > secondTitle) return 1;
    return 0;
}

// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        populateUsersDropdown();
        updateInputVisibility();
        document.getElementById('similarity-select')
            .addEventListener('change', updateInputVisibility);
        resultElement.textContent = "Data loaded. Select a recommendation method and input.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) =>
        compareTitles(a.title, b.title) || a.id - b.id
    );
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Populate the user dropdown from the rating-history data.
function populateUsersDropdown() {
    const selectElement = document.getElementById('user-select');
    const ratingCounts = new Map();

    ratings.forEach(rating => {
        ratingCounts.set(rating.userId, (ratingCounts.get(rating.userId) || 0) + 1);
    });

    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    [...ratingCounts.keys()]
        .sort((a, b) => a - b)
        .forEach(userId => {
            const option = document.createElement('option');
            option.value = userId;
            option.textContent = `User ${userId} (${ratingCounts.get(userId)} ratings)`;
            selectElement.appendChild(option);
        });
}

// Profile and comparison modes use a user history; the other modes use one movie.
function updateInputVisibility() {
    const selectedMethod = document.getElementById('similarity-select').value;
    const usesUserHistory = selectedMethod === 'profile' || selectedMethod === 'compare';
    document.getElementById('movie-control').hidden = usesUserHistory;
    document.getElementById('user-control').hidden = !usesUserHistory;

    const resultElement = document.getElementById('result');
    const instructions = {
        profile: 'Select a MovieLens user to build a profile from their top 3 ratings.',
        compare: 'Select a MovieLens user to hide their latest 5-star rating and compare a single active item with a three-movie profile.'
    };
    resultElement.textContent = instructions[selectedMethod] ||
        'Select a movie to calculate recommendations.';
    resultElement.className = '';

    document.getElementById('recommend-btn').textContent = selectedMethod === 'compare'
        ? 'Run Comparison'
        : 'Get Recommendations';
}

// Calculate Jaccard similarity for two sets of binary genre features.
function calculateJaccardSimilarity(firstGenres, secondGenres) {
    const intersectionSize = [...firstGenres]
        .filter(genre => secondGenres.has(genre))
        .length;
    const unionSize = new Set([...firstGenres, ...secondGenres]).size;

    return unionSize > 0 ? intersectionSize / unionSize : 0;
}

// Calculate cosine similarity for two binary genre vectors.
// The dot product is the number of shared genres, and each vector's
// magnitude is the square root of its number of active genres.
function calculateCosineSimilarity(firstGenres, secondGenres) {
    const dotProduct = [...firstGenres]
        .filter(genre => secondGenres.has(genre))
        .length;
    const magnitudeProduct = Math.sqrt(firstGenres.size * secondGenres.size);

    return magnitudeProduct > 0 ? dotProduct / magnitudeProduct : 0;
}

function calculateSimilarity(firstGenres, secondGenres, method) {
    if (method === 'cosine') {
        return calculateCosineSimilarity(firstGenres, secondGenres);
    }

    if (method === 'jaccard') {
        return calculateJaccardSimilarity(firstGenres, secondGenres);
    }

    throw new Error(`Unsupported recommendation method: ${method}`);
}

// Select the highest-rated movies from a rating collection.
// Newer ratings and then smaller movie IDs break equal-rating ties.
function selectProfileMoviesFromRatings(sourceRatings, limit = PROFILE_MOVIE_COUNT) {
    return [...sourceRatings]
        .sort((a, b) =>
            b.rating - a.rating ||
            b.timestamp - a.timestamp ||
            a.itemId - b.itemId
        )
        .slice(0, limit)
        .map(rating => ({
            rating,
            movie: movies.find(movie => movie.id === rating.itemId)
        }))
        .filter(entry => entry.movie);
}

function selectProfileMovies(userId, limit = PROFILE_MOVIE_COUNT) {
    return selectProfileMoviesFromRatings(
        ratings.filter(rating => rating.userId === userId),
        limit
    );
}

// Average the selected movies' binary genre vectors into one user-profile vector.
function buildUserProfile(profileMovies) {
    if (profileMovies.length === 0) {
        return Array(genreNames.length).fill(0);
    }

    const profileVector = Array(genreNames.length).fill(0);

    profileMovies.forEach(({ movie }) => {
        const movieGenres = new Set(movie.genres);
        genreNames.forEach((genre, index) => {
            profileVector[index] += movieGenres.has(genre) ? 1 : 0;
        });
    });

    return profileVector.map(value => value / profileMovies.length);
}

// Compare a numeric user-profile vector with a movie's binary genre vector.
function calculateProfileCosineSimilarity(profileVector, candidateGenres) {
    const candidateVector = genreNames.map(genre => candidateGenres.has(genre) ? 1 : 0);
    const dotProduct = profileVector.reduce(
        (sum, value, index) => sum + value * candidateVector[index],
        0
    );
    const profileMagnitude = Math.sqrt(
        profileVector.reduce((sum, value) => sum + value * value, 0)
    );
    const candidateMagnitude = Math.sqrt(
        candidateVector.reduce((sum, value) => sum + value * value, 0)
    );
    const magnitudeProduct = profileMagnitude * candidateMagnitude;

    return magnitudeProduct > 0 ? dotProduct / magnitudeProduct : 0;
}

function getProfileRecommendations(userId) {
    const userHistory = ratings.filter(rating => rating.userId === userId);
    const profileMovies = selectProfileMovies(userId);

    if (profileMovies.length < PROFILE_MOVIE_COUNT) {
        throw new Error(`User ${userId} has fewer than ${PROFILE_MOVIE_COUNT} usable ratings.`);
    }

    const profileVector = buildUserProfile(profileMovies);
    const watchedMovieIds = new Set(userHistory.map(rating => rating.itemId));
    const recommendations = movies
        .filter(movie => !watchedMovieIds.has(movie.id))
        .map(movie => ({
            ...movie,
            score: calculateProfileCosineSimilarity(
                profileVector,
                new Set(movie.genres)
            )
        }))
        .filter(movie => movie.score > 0)
        .sort((a, b) =>
            b.score - a.score ||
            compareTitles(a.title, b.title) ||
            a.id - b.id
        )
        .slice(0, TOP_RECOMMENDATION_COUNT);

    return { profileMovies, recommendations };
}

// Select a positive held-out item and retain only earlier ratings for training.
function chooseComparisonSplit(userRatings) {
    const chronologicalRatings = [...userRatings].sort((a, b) =>
        a.timestamp - b.timestamp || a.itemId - b.itemId
    );

    for (
        let index = chronologicalRatings.length - 1;
        index >= PROFILE_MOVIE_COUNT;
        index -= 1
    ) {
        if (chronologicalRatings[index].rating === COMPARISON_HELD_OUT_RATING) {
            return {
                trainingRatings: chronologicalRatings.slice(0, index),
                testRating: chronologicalRatings[index]
            };
        }
    }

    return null;
}

function rankComparisonCandidates(excludedMovieIds, scoreMovie) {
    return movies
        .filter(movie => !excludedMovieIds.has(movie.id))
        .map(movie => ({ ...movie, score: scoreMovie(movie) }))
        .filter(movie => movie.score > 0)
        .sort((a, b) =>
            b.score - a.score ||
            compareTitles(a.title, b.title) ||
            a.id - b.id
        );
}

function getTargetEvaluation(rankedCandidates, heldOutMovie, scoreMovie) {
    const index = rankedCandidates.findIndex(movie => movie.id === heldOutMovie.id);

    return {
        score: scoreMovie(heldOutMovie),
        positiveSimilarityRank: index === -1 ? null : index + 1
    };
}

// Reproduce the report experiment for one selected user.
function getComparisonRecommendations(userId) {
    const userRatings = ratings.filter(rating => rating.userId === userId);
    const split = chooseComparisonSplit(userRatings);

    if (!split) {
        throw new Error(
            `User ${userId} has no ${COMPARISON_HELD_OUT_RATING}-star rating with at least ` +
            `${PROFILE_MOVIE_COUNT} earlier ratings.`
        );
    }

    const profileMovies = selectProfileMoviesFromRatings(split.trainingRatings);
    if (profileMovies.length < PROFILE_MOVIE_COUNT) {
        throw new Error(`User ${userId} has fewer than ${PROFILE_MOVIE_COUNT} usable training movies.`);
    }

    const activeMovie = profileMovies[0].movie;
    const heldOutMovie = movies.find(movie => movie.id === split.testRating.itemId);
    if (!heldOutMovie) {
        throw new Error(`Held-out movie ${split.testRating.itemId} was not found.`);
    }

    const trainingMovieIds = new Set(
        split.trainingRatings.map(rating => rating.itemId)
    );
    const activeGenres = new Set(activeMovie.genres);
    const profileVector = buildUserProfile(profileMovies);
    const itemScoreMovie = movie =>
        calculateCosineSimilarity(activeGenres, new Set(movie.genres));
    const profileScoreMovie = movie =>
        calculateProfileCosineSimilarity(profileVector, new Set(movie.genres));
    const itemRanking = rankComparisonCandidates(
        trainingMovieIds,
        itemScoreMovie
    );
    const profileRanking = rankComparisonCandidates(
        trainingMovieIds,
        profileScoreMovie
    );
    const itemRecommendations = itemRanking.slice(0, TOP_RECOMMENDATION_COUNT);
    const profileRecommendations = profileRanking.slice(0, TOP_RECOMMENDATION_COUNT);
    const itemTargetEvaluation = getTargetEvaluation(
        itemRanking,
        heldOutMovie,
        itemScoreMovie
    );
    const profileTargetEvaluation = getTargetEvaluation(
        profileRanking,
        heldOutMovie,
        profileScoreMovie
    );
    const itemMovieIds = new Set(itemRecommendations.map(movie => movie.id));
    const overlapCount = profileRecommendations
        .filter(movie => itemMovieIds.has(movie.id))
        .length;

    return {
        trainingCount: split.trainingRatings.length,
        heldOutMovie,
        heldOutRating: split.testRating,
        activeMovie,
        profileMovies,
        itemRecommendations,
        profileRecommendations,
        itemTargetEvaluation,
        profileTargetEvaluation,
        itemHit: itemMovieIds.has(heldOutMovie.id),
        profileHit: profileRecommendations.some(movie => movie.id === heldOutMovie.id),
        overlapCount
    };
}

function displayRecommendationList(resultElement, introductionText, recommendations, profileText = '') {
    resultElement.textContent = '';

    const introduction = document.createElement('p');
    introduction.className = 'recommendation-intro';
    introduction.textContent = introductionText;
    resultElement.appendChild(introduction);

    if (profileText) {
        const profileSource = document.createElement('p');
        profileSource.className = 'profile-source';
        profileSource.textContent = profileText;
        resultElement.appendChild(profileSource);
    }

    const recommendationList = document.createElement('ol');
    recommendationList.className = 'recommendation-list';

    recommendations.forEach(movie => {
        const listItem = document.createElement('li');
        listItem.textContent = `${movie.title} — score: ${movie.score.toFixed(3)}`;
        recommendationList.appendChild(listItem);
    });

    resultElement.appendChild(recommendationList);
    resultElement.className = 'success';
}

function createComparisonList(recommendations, heldOutMovieId) {
    const list = document.createElement('ol');
    list.className = 'recommendation-list comparison-list';

    recommendations.forEach(movie => {
        const listItem = document.createElement('li');
        const isTarget = movie.id === heldOutMovieId;
        listItem.textContent = `${movie.title} — score: ${movie.score.toFixed(3)}`;

        if (isTarget) {
            listItem.className = 'held-out-hit';
            const targetLabel = document.createElement('span');
            targetLabel.className = 'target-label';
            targetLabel.textContent = ' held-out target';
            listItem.appendChild(targetLabel);
        }

        list.appendChild(listItem);
    });

    return list;
}

function createComparisonCard(
    title,
    sourceText,
    recommendations,
    heldOutMovieId,
    targetEvaluation,
    isHit
) {
    const card = document.createElement('section');
    card.className = 'comparison-card';

    const heading = document.createElement('h2');
    heading.textContent = title;
    card.appendChild(heading);

    const source = document.createElement('p');
    source.className = 'comparison-source';
    source.textContent = sourceText;
    card.appendChild(source);

    const targetScore = document.createElement('p');
    targetScore.className = 'comparison-target-evaluation';
    const rankText = targetEvaluation.positiveSimilarityRank === null
        ? 'no positive-similarity rank'
        : `positive-similarity rank: ${targetEvaluation.positiveSimilarityRank}`;
    targetScore.textContent =
        `Target cosine similarity: ${targetEvaluation.score.toFixed(3)}; ${rankText}.`;
    card.appendChild(targetScore);

    const outcome = document.createElement('p');
    outcome.className = `comparison-outcome ${isHit ? 'hit' : 'miss'}`;
    outcome.textContent = isHit ? 'Hit@5: yes' : 'Hit@5: no';
    card.appendChild(outcome);

    card.appendChild(createComparisonList(recommendations, heldOutMovieId));
    return card;
}

function displayComparison(resultElement, userId, comparison) {
    resultElement.textContent = '';

    const heading = document.createElement('p');
    heading.className = 'recommendation-intro';
    heading.textContent = `User ${userId}: single active item vs. aggregated profile`;
    resultElement.appendChild(heading);

    const protocol = document.createElement('p');
    protocol.className = 'comparison-protocol';
    protocol.textContent =
        `Held-out positive: "${comparison.heldOutMovie.title}" ` +
        `(actual user rating ${comparison.heldOutRating.rating}). ` +
        `The algorithms assign cosine similarity scores from 0 to 1, not predicted stars. ` +
        `Both methods use only ${comparison.trainingCount} earlier ratings.`;
    resultElement.appendChild(protocol);

    const grid = document.createElement('div');
    grid.className = 'comparison-grid';
    grid.appendChild(createComparisonCard(
        'Single active item',
        `Active movie: "${comparison.activeMovie.title}".`,
        comparison.itemRecommendations,
        comparison.heldOutMovie.id,
        comparison.itemTargetEvaluation,
        comparison.itemHit
    ));
    grid.appendChild(createComparisonCard(
        'Aggregated profile',
        `Profile movies: ${comparison.profileMovies
            .map(({ movie, rating }) => `"${movie.title}" (rating ${rating.rating})`)
            .join(', ')}.`,
        comparison.profileRecommendations,
        comparison.heldOutMovie.id,
        comparison.profileTargetEvaluation,
        comparison.profileHit
    ));
    resultElement.appendChild(grid);

    const summary = document.createElement('p');
    summary.className = 'comparison-summary';
    summary.textContent =
        `Shared recommendations: ${comparison.overlapCount} of ` +
        `${TOP_RECOMMENDATION_COUNT} ` +
        `(${((comparison.overlapCount / TOP_RECOMMENDATION_COUNT) * 100).toFixed(1)}%).`;
    resultElement.appendChild(summary);
    resultElement.className = 'success comparison-result';
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const userElement = document.getElementById('user-select');
        const similarityElement = document.getElementById('similarity-select');
        const selectedMethod = similarityElement.value;
        const selectedMethodLabel = similarityElement.options[similarityElement.selectedIndex].text;

        if (selectedMethod === 'profile' || selectedMethod === 'compare') {
            const selectedUserId = parseInt(userElement.value);

            if (isNaN(selectedUserId)) {
                resultElement.textContent = "Please select a user first.";
                resultElement.className = 'error';
                return;
            }

            if (selectedMethod === 'compare') {
                resultElement.textContent = "Running the chronological comparison...";
                resultElement.className = 'loading';

                setTimeout(() => {
                    try {
                        const comparison = getComparisonRecommendations(selectedUserId);
                        displayComparison(resultElement, selectedUserId, comparison);
                    } catch (error) {
                        console.error('Error in comparison calculation:', error);
                        resultElement.textContent = `Comparison error: ${error.message}`;
                        resultElement.className = 'error';
                    }
                }, 100);
                return;
            }

            resultElement.textContent = "Building the user profile and calculating recommendations...";
            resultElement.className = 'loading';

            setTimeout(() => {
                try {
                    const { profileMovies, recommendations } = getProfileRecommendations(selectedUserId);
                    const profileText = `Profile movies: ${profileMovies
                        .map(({ movie, rating }) => `"${movie.title}" (rating ${rating.rating})`)
                        .join(', ')}.`;

                    if (recommendations.length > 0) {
                        displayRecommendationList(
                            resultElement,
                            `Using ${selectedMethodLabel} for User ${selectedUserId}, we recommend these unseen movies:`,
                            recommendations,
                            profileText
                        );
                    } else {
                        resultElement.textContent = `No profile-based recommendations found for User ${selectedUserId}.`;
                        resultElement.className = 'error';
                    }
                } catch (error) {
                    console.error('Error in profile recommendation calculation:', error);
                    resultElement.textContent = `Profile recommendation error: ${error.message}`;
                    resultElement.className = 'error';
                }
            }, 100);
            return;
        }

        const selectedMovieId = parseInt(selectElement.value);

        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedGenres = new Set(likedMovie.genres);
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                
                // Step 4: Calculate similarity scores using the selected method
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateGenres = new Set(candidate.genres);
                    const score = calculateSimilarity(
                        likedGenres,
                        candidateGenres,
                        selectedMethod
                    );
                    
                    return {
                        ...candidate,
                        score: score
                    };
                });
                
                // Step 5: Sort by score in descending order
                scoredMovies.sort((a, b) =>
                    b.score - a.score ||
                    compareTitles(a.title, b.title) ||
                    a.id - b.id
                );
                
                // Step 6: Select top recommendations
                const topRecommendations = scoredMovies.slice(0, TOP_RECOMMENDATION_COUNT);
                
                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    displayRecommendationList(
                        resultElement,
                        `Using ${selectedMethodLabel}, because you liked "${likedMovie.title}", we recommend:`,
                        topRecommendations
                    );
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
