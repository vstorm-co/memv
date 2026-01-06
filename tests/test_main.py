from agentmemory import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from agentmemory!\n"
