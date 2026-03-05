"""Tests for llm_factory — verifies provider priority and fallback behaviour."""

import os
import unittest
from unittest.mock import MagicMock, patch

from maverick_mcp.providers.openrouter_provider import TaskType

# Import the module once; individual tests patch names within it.
import maverick_mcp.providers.llm_factory as factory


class TestLlmFactoryProviderPriority(unittest.TestCase):
    """get_llm() should pick the highest-priority provider whose key is set."""

    @patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_bailian_used_when_aliyun_key_present(self, mock_bailian):
        os.environ.pop("OPENROUTER_API_KEY", None)
        mock_llm = MagicMock()
        mock_bailian.return_value = mock_llm

        result = factory.get_llm()

        mock_bailian.assert_called_once()
        self.assertEqual(result, mock_llm)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_openrouter_llm")
    def test_openrouter_used_when_no_aliyun_key(self, mock_openrouter):
        os.environ.pop("ALIYUN_API_KEY", None)
        mock_llm = MagicMock()
        mock_openrouter.return_value = mock_llm

        result = factory.get_llm()

        mock_openrouter.assert_called_once()
        self.assertEqual(result, mock_llm)

    @patch.dict(
        os.environ,
        {"ALIYUN_API_KEY": "aliyun-key", "OPENROUTER_API_KEY": "or-key"},
        clear=False,
    )
    @patch("maverick_mcp.providers.llm_factory.get_openrouter_llm")
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_bailian_takes_priority_over_openrouter(self, mock_bailian, mock_openrouter):
        mock_bailian.return_value = MagicMock()

        factory.get_llm()

        mock_bailian.assert_called_once()
        mock_openrouter.assert_not_called()

    def test_fakellm_used_when_no_keys(self):
        keys = ("ALIYUN_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        env_patch = {k: "" for k in keys}
        # Remove the keys from the environment entirely
        with patch.dict(os.environ, {}, clear=False):
            for k in keys:
                os.environ.pop(k, None)

            from langchain_community.llms import FakeListLLM
            result = factory.get_llm()
            self.assertIsInstance(result, FakeListLLM)


class TestLlmFactoryArgumentForwarding(unittest.TestCase):
    """get_llm() should forward all parameters to the selected provider."""

    @patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_task_type_forwarded_to_bailian(self, mock_bailian):
        os.environ.pop("OPENROUTER_API_KEY", None)
        mock_bailian.return_value = MagicMock()

        factory.get_llm(task_type=TaskType.DEEP_RESEARCH)

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertEqual(call_kwargs["task_type"], TaskType.DEEP_RESEARCH)

    @patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_prefer_quality_forwarded_to_bailian(self, mock_bailian):
        os.environ.pop("OPENROUTER_API_KEY", None)
        mock_bailian.return_value = MagicMock()

        factory.get_llm(prefer_quality=True)

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertTrue(call_kwargs["prefer_quality"])

    @patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-key"}, clear=False)
    @patch("maverick_mcp.providers.llm_factory.get_bailian_llm")
    def test_model_override_forwarded_to_bailian(self, mock_bailian):
        os.environ.pop("OPENROUTER_API_KEY", None)
        mock_bailian.return_value = MagicMock()

        factory.get_llm(model_override="glm-5")

        call_kwargs = mock_bailian.call_args.kwargs
        self.assertEqual(call_kwargs["model_override"], "glm-5")


if __name__ == "__main__":
    unittest.main()
