import unittest
from llm_utils import OAI_LM, LMClassifier, display_chat_messages_as_html, get_conversation_one_turn

class TestOAILM(unittest.TestCase):
    def test_initialization(self):
        lm = OAI_LM(model="gpt-3.5-turbo")
        self.assertEqual(lm.model, "openai/gpt-3.5-turbo")

    def test_get_session(self):
        from llm_utils import LMAgent
        agent = LMAgent(lm=OAI_LM('gpt-4.1-nano'), system_prompt="You are a helpful assistant.")

class TestLMClassifier(unittest.TestCase):
    def test_initialization(self):
        from pydantic import BaseModel

        class InputModel(BaseModel):
            input_field: str

        class OutputModel(BaseModel):
            output_field: str

        classifier = LMClassifier(
            system_prompt="Classify this input.",
            input_model=InputModel,
            output_model=OutputModel,
            model="gpt-3.5-turbo",
        )
        self.assertEqual(classifier.system_prompt, "Classify this input.")

class TestChatFormat(unittest.TestCase):
    def test_display_chat_messages_as_html(self):
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
            {"role": "assistant", "content": "Assistant message."},
        ]
        html = display_chat_messages_as_html(messages, return_html=True)
        self.assertIn("System message.", html)
        self.assertIn("User message.", html)
        self.assertIn("Assistant message.", html)

    def test_get_conversation_one_turn(self):
        messages = get_conversation_one_turn(
            system_msg="System message.",
            user_msg="User message.",
            assistant_msg="Assistant message.",
        )
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")

if __name__ == "__main__":
    unittest.main()
