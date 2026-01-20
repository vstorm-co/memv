from agent_memory import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "AgentMemory" in captured.out
    assert "examples/" in captured.out
