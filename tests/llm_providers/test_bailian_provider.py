"""Tests for the Bailian (Aliyun) LLM provider."""

import unittest
from unittest.mock import MagicMock, patch

from maverick_mcp.providers.bailian_provider import (
    BAILIAN_MODEL_PROFILES,
    BailianProvider,
    get_bailian_llm,
)
from maverick_mcp.providers.openrouter_provider import TaskType


class TestBailianModelProfiles(unittest.TestCase):
    """Verify model profile data is sane."""

    def test_all_expected_models_present(self):
        expected = {"qwen3.5-plus", "qwen3-max-2026-01-23", "glm-5"}
        self.assertEqual(set(BAILIAN_MODEL_PROFILES.keys()), expected)

    def test_default_model_covers_general_task(self):
        profile = BAILIAN_MODEL_PROFILES["qwen3.5-plus"]
        self.assertIn(TaskType.GENERAL, profile.best_for)

    def test_qwen3_max_covers_deep_research(self):
        profile = BAILIAN_MODEL_PROFILES["qwen3-max-2026-01-23"]
        self.assertIn(TaskType.DEEP_RESEARCH, profile.best_for)
        self.assertIn(TaskType.COMPLEX_REASONING, profile.best_for)
        self.assertIn(TaskType.MULTI_AGENT_ORCHESTRATION, profile.best_for)

    def test_all_profiles_have_positive_costs(self):
        for name, profile in BAILIAN_MODEL_PROFILES.items():
            with self.subTest(model=name):
                self.assertGreater(profile.cost_per_million_input, 0)
                self.assertGreater(profile.cost_per_million_output, 0)

    def test_ratings_in_valid_range(self):
        for name, profile in BAILIAN_MODEL_PROFILES.items():
            with self.subTest(model=name):
                self.assertGreaterEqual(profile.speed_rating, 1)
                self.assertLessEqual(profile.speed_rating, 10)
                self.assertGreaterEqual(profile.quality_rating, 1)
                self.assertLessEqual(profile.quality_rating, 10)


class TestBailianProviderModelSelection(unittest.TestCase):
    """Test BailianProvider.get_llm() model selection logic (no real API calls)."""

    def setUp(self):
        self.provider = BailianProvider(api_key="test-key")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_default_task_uses_qwen35_plus(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.GENERAL)

        # First call should be the primary model
        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "qwen3.5-plus")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_deep_research_uses_qwen3_max(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.DEEP_RESEARCH)

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "qwen3-max-2026-01-23")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_complex_reasoning_uses_qwen3_max(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.COMPLEX_REASONING)

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "qwen3-max-2026-01-23")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_prefer_quality_uses_qwen3_max(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.GENERAL, prefer_quality=True)

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "qwen3-max-2026-01-23")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_model_override_respected(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(model_override="glm-5")

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "glm-5")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_temperature_override(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(temperature=0.7)

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["temperature"], 0.7)

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_uses_correct_base_url(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm()

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertIn("dashscope.aliyuncs.com", first_call_kwargs["openai_api_base"])

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_api_key_passed_to_client(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm()

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["openai_api_key"], "test-key")

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_fallback_chain_built(self, mock_chat):
        """Primary + fallbacks should be wired up via with_fallbacks."""
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.GENERAL)

        mock_instance.with_fallbacks.assert_called_once()
        fallback_list = mock_instance.with_fallbacks.call_args[0][0]
        self.assertGreater(len(fallback_list), 0)

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_fallback_does_not_include_primary(self, mock_chat):
        """The primary model should not appear in its own fallback list."""
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        self.provider.get_llm(task_type=TaskType.GENERAL)

        # Collect all model IDs passed to ChatOpenAI after the first call
        all_calls = mock_chat.call_args_list
        primary_model = all_calls[0].kwargs["model"]
        fallback_models = [c.kwargs["model"] for c in all_calls[1:]]
        self.assertNotIn(primary_model, fallback_models)


class TestGetBailianLlmConvenience(unittest.TestCase):
    """Test the module-level get_bailian_llm() convenience function."""

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_returns_llm_instance(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        result = get_bailian_llm(api_key="test-key")

        self.assertIsNotNone(result)

    @patch("maverick_mcp.providers.bailian_provider.ChatOpenAI")
    def test_forwards_task_type(self, mock_chat):
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance
        mock_instance.with_fallbacks.return_value = mock_instance

        get_bailian_llm(api_key="test-key", task_type=TaskType.DEEP_RESEARCH)

        first_call_kwargs = mock_chat.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], "qwen3-max-2026-01-23")


if __name__ == "__main__":
    unittest.main()
