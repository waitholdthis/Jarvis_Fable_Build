from jarvis.voice import strip_wake_word


def test_wake_word_with_command():
    assert strip_wake_word("Jarvis, what time is it?") == "what time is it?"
    assert strip_wake_word("jarvis open my notes") == "open my notes"


def test_wake_word_tolerates_asr_slips():
    # Whisper often mishears names; fuzzy matching should absorb near-misses.
    assert strip_wake_word("Jarvus, hello there") == "hello there"
    assert strip_wake_word("Jervis what's the weather") == "what's the weather"


def test_wake_word_not_first_but_early():
    assert strip_wake_word("Hey Jarvis, play some music") == "play some music"
    assert strip_wake_word("OK Jarvis turn it off") == "turn it off"


def test_bare_wake_word_returns_empty_string():
    assert strip_wake_word("Jarvis.") == ""
    assert strip_wake_word("Jarvis") == ""


def test_unaddressed_speech_returns_none():
    assert strip_wake_word("what a nice day outside") is None
    assert strip_wake_word("") is None
    assert strip_wake_word("I was talking to my friend about music") is None


def test_custom_assistant_name():
    assert strip_wake_word("Friday, run diagnostics", name="Friday") == "run diagnostics"
    assert strip_wake_word("Jarvis, run diagnostics", name="Friday") is None
