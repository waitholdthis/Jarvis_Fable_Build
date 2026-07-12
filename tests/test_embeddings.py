from jarvis.embeddings import HashEmbedder, cosine, tokenize


def test_tokenize_drops_stopwords():
    tokens = tokenize("The quick brown fox is in the yard")
    assert "the" not in tokens
    assert "quick" in tokens and "fox" in tokens


def test_hash_embedder_deterministic_and_normalized():
    emb = HashEmbedder(dim=256)
    a = emb.embed("reboot the staging server at midnight")
    b = emb.embed("reboot the staging server at midnight")
    assert a == b
    assert len(a) == 256
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_similar_texts_score_higher_than_unrelated():
    emb = HashEmbedder()
    query = emb.embed("how do I restart the postgres database")
    close = emb.embed("to restart postgres run: systemctl restart postgresql")
    far = emb.embed("grandma's lasagna recipe with extra basil")
    assert cosine(query, close) > cosine(query, far)


def test_empty_text_gives_zero_vector():
    emb = HashEmbedder(dim=64)
    vec = emb.embed("")
    assert vec == [0.0] * 64


def test_cosine_dimension_mismatch_is_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
