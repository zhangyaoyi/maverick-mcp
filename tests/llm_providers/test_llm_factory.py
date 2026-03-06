"""Tests for llm_factory — verifies provider priority and fallback behaviour."""

import os
import unittest
from unittest.mock import MagicMock, patch

from maverick_mcp.providers.openrouter_provider import TaskType

# Import the module once; individual tests patch names within it.
import maverick_mcp.providers.llm_factory as factory

_NO_KEYS = {"ALIYUN_API_KEY": "", "OPENROUTER_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}


def _clear_provider_keys():
    for k in ("ALIYUN_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)


class TestLlmFactoryProviderPriority(unittest.TestCase):
    """get_llm() should pick the highest-priority provider whose key is set."""

    @patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_bailian_only_returns_bailian_llm_directly(self, mock_bailian):
        """Single provider → no fallback chain, returns LLM directly."""
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "aliyun-key"
        mock_llm = MagicMock()
        mock_bailian.return_value = mock_llm

        result = factory.get_llm()

        mock_bailian.assert_called_once()
        self.assertEqual(result, mock_llm)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_openrouter_llm")
    def test_openrouter_only_returns_directly(self, mock_openrouter):
        """Single provider → no fallback chain."""
        _clear_provider_keys()
        os.environ["OPENROUTER_API_KEY"] = "or-key"
        mock_llm = MagicMock()
        mock_openrouter.return_value = mock_llm

        result = factory.get_llm()

        mock_openrouter.assert_called_once()
        self.assertEqual(result, mock_llm)

    @patch("maverick_mcp.providers.llm_factory.get_openrouter_llm")
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_both_keys_builds_fallback_chain(self, mock_bailian, mock_openrouter):
        """Both keys → Bailian is primary, OpenRouter is fallback."""
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "aliyun-key"
        os.environ["OPENROUTER_API_KEY"] = "or-key"

        bailian_llm = MagicMock()
        openrouter_llm = MagicMock()
        chained_llm = MagicMock()
        mock_bailian.return_value = bailian_llm
        mock_openrouter.return_value = openrouter_llm
        bailian_llm.with_fallbacks.return_value = chained_llm

        result = factory.get_llm()

        mock_bailian.assert_called_once()
        mock_openrouter.assert_called_once()
        # Bailian is primary; OpenRouter is in fallback list
        bailian_llm.with_fallbacks.assert_called_once()
        fallback_list = bailian_llm.with_fallbacks.call_args[0][0]
        self.assertIn(openrouter_llm, fallback_list)
        self.assertEqual(result, chained_llm)

        os.environ.pop("ALIYUN_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)

    def test_fakellm_used_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=False):
            _clear_provider_keys()

            from langchain_community.llms import FakeListLLM
            result = factory.get_llm()
            self.assertIsInstance(result, FakeListLLM)

    @patch("maverick_mcp.providers.llm_factory.get_openrouter_llm")
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_auth_errors_passed_to_with_fallbacks(self, mock_bailian, mock_openrouter):
        """with_fallbacks should receive _AUTH_ERRORS so 401s trigger failover."""
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "bad-key"
        os.environ["OPENROUTER_API_KEY"] = "or-key"

        bailian_llm = MagicMock()
        mock_bailian.return_value = bailian_llm
        mock_openrouter.return_value = MagicMock()
        bailian_llm.with_fallbacks.return_value = MagicMock()

        factory.get_llm()

        call_kwargs = bailian_llm.with_fallbacks.call_args.kwargs
        self.assertIn("exceptions_to_handle", call_kwargs)
        self.assertTrue(len(call_kwargs["exceptions_to_handle"]) > 0)

        os.environ.pop("ALIYUN_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)


class TestLlmFactoryArgumentForwarding(unittest.TestCase):
    """get_llm() should forward all parameters to the selected provider."""

    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_task_type_forwarded_to_bailian(self, mock_bailian):
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "aliyun-key"
        mock_bailian.return_value = MagicMock()

        factory.get_llm(task_type=TaskType.DEEP_RESEARCH)

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertEqual(call_kwargs["task_type"], TaskType.DEEP_RESEARCH)
        os.environ.pop("ALIYUN_API_KEY", None)

    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_prefer_quality_forwarded_to_bailian(self, mock_bailian):
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "aliyun-key"
        mock_bailian.return_value = MagicMock()

        factory.get_llm(prefer_quality=True)

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertTrue(call_kwargs["prefer_quality"])
        os.environ.pop("ALIYUN_API_KEY", None)

    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_model_override_forwarded_to_bailian(self, mock_bailian):
        _clear_provider_keys()
        os.environ["ALIYUN_API_KEY"] = "aliyun-key"
        mock_bailian.return_value = MagicMock()

        factory.get_llm(model_override="glm-5")

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertEqual(call_kwargs["model_override"], "glm-5")
        os.environ.pop("ALIYUN_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
