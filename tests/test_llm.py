from langchain_core.messages import HumanMessage

from docqa.config import load_settings
from docqa.llm import get_llm


def payload(llm):
    return llm._get_request_payload([HumanMessage(content="hi")])


def test_no_key_means_no_model():
    assert get_llm(load_settings({})) is None
    assert get_llm(load_settings({"ANTHROPIC_API_KEY": "your-api-key-here"})) is None


def test_key_builds_a_claude_model_with_the_default_model_and_low_effort():
    llm = get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test"}))
    sent = payload(llm)
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"]["effort"] == "low"


def test_no_sampling_parameters_are_sent():
    """The newest Claude models reject temperature / top_p / top_k, so none may be sent."""
    sent = payload(get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test"})))
    assert not {"temperature", "top_p", "top_k"} & set(sent)


def test_max_tokens_is_explicit_and_small_enough_for_a_normal_request():
    sent = payload(get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_MAX_TOKENS": "4000"})))
    assert sent["max_tokens"] == 4000
    assert payload(get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test"})))["max_tokens"] == 8000


def test_models_without_effort_support_do_not_receive_it():
    sent = payload(get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_MODEL": "claude-haiku-4-5"})))
    assert sent["model"] == "claude-haiku-4-5" and "output_config" not in sent


def test_model_and_effort_can_be_overridden():
    sent = payload(get_llm(load_settings({"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_MODEL": "claude-sonnet-5-5", "ANTHROPIC_EFFORT": "high"})))
    assert sent["model"] == "claude-sonnet-5-5" and sent["output_config"]["effort"] == "high"
