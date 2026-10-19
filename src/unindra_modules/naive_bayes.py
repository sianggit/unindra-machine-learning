"""
Module: naive_bayes.py
Author: Lab Machine Learning - Universitas Indraprasta PGRI (Unindra)
Description:
    Implementasi Naive Bayes (Gaussian & Multinomial) dari scratch (Numpy),
    lengkap dengan formula Laplace Smoothing dan penjelasan langkah manual
    untuk bank soal UPM/UAS.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def laplace_smoothing_prob(
    count_word_in_class: int,
    total_words_in_class: int,
    vocab_size: int,
    alpha: float = 1.0,
) -> float:
    """
    Menghitung probabilitas bersyarat dengan Laplace (Additive) Smoothing:
    P(w_i | c) = (count(w_i, c) + alpha) / (count(c) + alpha * |V|)
    
    Analogi Perkuliahan:
    Seperti memberi 'modal awal' 1 suara kepada kandidat yang belum pernah terlihat,
    agar probabilitas total tidak langsung anjlok menjadi nol (Zero-Frequency Problem).
    """
    numerator = count_word_in_class + alpha
    denominator = total_words_in_class + (alpha * vocab_size)
    return float(numerator / denominator)


class ManualGaussianNB:
    """
    Gaussian Naive Bayes dari Scratch untuk data fitur kontinu.
    Menghitung Mean (mu) dan Varian (sigma^2) tiap kelas, lalu
    mengevaluasi Gaussian Probability Density Function (PDF).
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = np.array([])
        self.priors_: Dict[Any, float] = {}
        self.means_: Dict[Any, np.ndarray] = {}
        self.vars_: Dict[Any, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualGaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = len(y)

        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / n_samples
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    def _pdf(self, class_label: Any, x: np.ndarray) -> np.ndarray:
        mean = self.means_[class_label]
        var = self.vars_[class_label]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def explain_prediction(self, x_query: np.ndarray) -> Dict[str, Any]:
        x_q = np.asarray(x_query, dtype=float)
        posteriors = {}
        likelihood_breakdown = {}

        for c in self.classes_:
            prior = self.priors_[c]
            likelihoods = self._pdf(c, x_q)
            # Log posterior untuk kestabilan numerik
            log_prior = np.log(prior)
            log_likelihood = np.sum(np.log(likelihoods))
            posteriors[c] = float(log_prior + log_likelihood)
            likelihood_breakdown[c] = {
                "prior": float(prior),
                "feature_likelihoods": likelihoods.tolist(),
                "log_posterior": float(posteriors[c]),
            }

        predicted_class = max(posteriors, key=posteriors.get)

        return {
            "query_sample": x_q.tolist(),
            "classes": self.classes_.tolist(),
            "class_analysis": likelihood_breakdown,
            "predicted_class": predicted_class,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.explain_prediction(x)["predicted_class"] for x in X])


class ManualMultinomialNB:
    """
    Multinomial Naive Bayes dari Scratch untuk klasifikasi teks/frekuensi diskrit.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        self.priors_: Dict[Any, float] = {}
        self.word_probs_: Dict[Any, np.ndarray] = {}
        self.vocab_size: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualMultinomialNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, self.vocab_size = X.shape

        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / n_samples
            total_counts_per_word = np.sum(X_c, axis=0)
            total_words_in_class = np.sum(total_counts_per_word)
            
            # Laplace smoothing
            probs = (total_counts_per_word + self.alpha) / (
                total_words_in_class + (self.alpha * self.vocab_size)
            )
            self.word_probs_[c] = probs

        return self

    def explain_prediction(self, x_query: np.ndarray) -> Dict[str, Any]:
        x_q = np.asarray(x_query, dtype=float)
        log_posteriors = {}

        for c in self.classes_:
            log_prior = np.log(self.priors_[c])
            log_likelihood = np.sum(x_q * np.log(self.word_probs_[c]))
            log_posteriors[c] = float(log_prior + log_likelihood)

        predicted_class = max(log_posteriors, key=log_posteriors.get)

        return {
            "query_vector": x_q.tolist(),
            "log_posteriors": log_posteriors,
            "predicted_class": predicted_class,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.explain_prediction(x)["predicted_class"] for x in X])
