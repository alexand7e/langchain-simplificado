from agente_edu.chains.prompt import PERSONAS, montar_prompt


def test_personas_defined():
    assert "crianca" in PERSONAS
    assert "vestibular" in PERSONAS
    assert "ensino_medio" in PERSONAS
    assert "simples" in PERSONAS


def test_montar_prompt_default():
    prompt = montar_prompt()
    template = prompt.template
    assert "{contexto}" in template
    assert "{pergunta}" in template
    assert "{regra}" in template
    assert "{persona}" in template


def test_montar_prompt_persona_especifica():
    prompt = montar_prompt("crianca")
    expected = PERSONAS["crianca"]
    assert prompt.partial_variables["persona"] == expected
