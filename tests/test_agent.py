import pytest
from llm_utils.lm.oai_lm import OAI_LM

@pytest.fixture
def lm():
    return OAI_LM('gpt-4.1', cache=False)

def test_lm_response(lm):
    response = lm('say this is a test')
    assert 'a test' in response.lower()

def test_lm_inspect_history(lm):
    lm.inspect_history()



def test_lm_agent(lm):
    from llm_utils.lm.lm_agent import LMAgent
    # Fix: Pass a proper model name as first parameter, and system message as second parameter
    agent = LMAgent(lm=lm, system_prompt='You are an asisstant name Alex')
    answer = agent('What is your name?')
    print(answer)
    assert 'Alex' in answer