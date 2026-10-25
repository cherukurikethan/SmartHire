
"""Calculate similarity between resume text and job text."""

from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_vector, job_vectors):
    """Return similarity scores between one resume and job vectors."""

    scores = cosine_similarity(resume_vector, job_vectors).flatten()
    return scores


def get_top_matches(resume_vector, job_vectors, top_n=5):
    """Return indices of the top matching jobs and their scores."""

    scores = calculate_similarity(resume_vector, job_vectors)
    top_indices = scores.argsort()[::-1][:top_n]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]
                       