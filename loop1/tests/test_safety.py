"""Phase 6 — safety substring matching tests."""
from loop1.safety import EMERGENCY_PHRASES, check_safety


# ── check_safety: positive cases ─────────────────────────────────────────────

def test_exact_phrase_match():
    assert check_safety("I am having a seizure") == "seizure"


def test_case_insensitive_match():
    assert check_safety("I am SUICIDAL right now") == "suicidal"


def test_phrase_embedded_in_sentence():
    assert check_safety("I can't breathe at all, please help") == "can't breathe"


def test_seizure_match():
    assert check_safety("He's having a seizure") == "seizure"


def test_overdose_match():
    assert check_safety("I think I took an overdose") == "overdose"


def test_throat_swelling():
    assert check_safety("My throat swelling is getting worse") == "throat swelling"


def test_want_to_die():
    assert check_safety("I want to die") == "want to die"


def test_kill_myself():
    assert check_safety("I'm going to kill myself") == "kill myself"


def test_face_drooping():
    assert check_safety("Her face drooping on one side") == "face drooping"


# ── check_safety: negative cases ─────────────────────────────────────────────

def test_normal_input_returns_none():
    assert check_safety("I have a mild headache for two days") is None


def test_partial_word_not_matched():
    # "orchestral" contains "chest" but not "chest pain"
    assert check_safety("I attended an orchestral painting class") is None


def test_empty_input_returns_none():
    assert check_safety("") is None


def test_benign_symptom_returns_none():
    assert check_safety("I feel dizzy when I stand up too fast") is None


def test_quit_command_returns_none():
    assert check_safety("quit") is None


# ── EMERGENCY_PHRASES list sanity checks ─────────────────────────────────────

def test_all_phrases_are_strings():
    assert all(isinstance(p, str) for p in EMERGENCY_PHRASES)


def test_no_duplicate_phrases():
    assert len(EMERGENCY_PHRASES) == len(set(p.lower() for p in EMERGENCY_PHRASES))


def test_common_symptoms_not_in_list():
    # these are normal diagnostic symptoms that must not terminate a session
    for phrase in ["chest pain", "difficulty breathing", "arm weakness",
                   "speech difficulty", "stroke", "unconscious", "anaphylaxis"]:
        assert phrase not in EMERGENCY_PHRASES, f"{phrase!r} should not be an emergency trigger"


def test_suicidal_in_list():
    assert "suicidal" in EMERGENCY_PHRASES
