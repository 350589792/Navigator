import pytest
from app.utils import add_bac_token, add_drunk_tokens, custom_tokenizer

def test_add_bac_token():
    # Test high BAC detection
    text = "被告人血液酒精含量为135.6mg/100ml"
    result = add_bac_token(text)
    assert "BAC_HIGH" in result
    
    # Test BAC below threshold
    text = "被告人血液酒精含量为20.5mg/100ml"
    result = add_bac_token(text)
    assert "BAC_HIGH" not in result
    
    # Test no BAC mention
    text = "被告人超速行驶"
    result = add_bac_token(text)
    assert "BAC_HIGH" not in result

def test_add_drunk_tokens():
    # Test various drunk driving keywords
    keywords = ["酒后驾驶", "酒驾", "醉酒驾驶", "饮酒驾驶"]
    for keyword in keywords:
        text = f"被告人{keyword}导致事故"
        result = add_drunk_tokens(text)
        assert "DRUNK_DRIVING" in result
    
    # Test non-drunk driving text
    text = "被告人超速行驶"
    result = add_drunk_tokens(text)
    assert "DRUNK_DRIVING" not in result

def test_custom_tokenizer():
    # Test combined functionality
    text = "被告人酒后驾驶，血液酒精含量为135.6mg/100ml"
    tokens = custom_tokenizer(text)
    
    # Check for both BAC and drunk driving tokens
    assert any("BAC_HIGH" in token for token in tokens)
    assert any("DRUNK_DRIVING" in token for token in tokens)
    
    # Check for normal word tokenization
    assert "被告人" in tokens
    assert "驾驶" in tokens
