"""Tests pour data_loader.py"""
import pytest
from rag.data_loader import load_events, clean_html, chunk_event_text


def test_load_events_real():
    """Test chargement événements réels"""
    events = load_events(source="real")
    assert len(events) == 100
    assert "uid" in events[0]
    assert "title_fr" in events[0]


def test_load_events_dummy():
    """Test chargement événements dummy"""
    events = load_events(source="dummy")
    assert len(events) == 15
    assert "uid" in events[0]


def test_clean_html_removes_tags():
    """Test nettoyage HTML"""
    html = "<p>Bonjour <strong>le monde</strong></p>"
    result = clean_html(html)
    assert result == "Bonjour le monde"
    assert "<" not in result


def test_clean_html_empty():
    """Test HTML vide"""
    assert clean_html("") == ""
    assert clean_html(None) == ""


def test_chunk_event_text():
    """Test chunking événement"""
    event = {
        "uid": "123",
        "title_fr": "Test Event",
        "description_fr": "Description test " * 100,  # Texte long
        "location_name": "Paris",
        "firstdate_begin": "2024-11-10"
    }
    
    chunks = chunk_event_text(event)
    
    # Doit avoir au moins 1 chunk
    assert len(chunks) >= 1
    
    # Chaque chunk a la bonne structure
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["uid"] == "123"