import subprocess

def test_transmission_main_example(capfd):
    res = subprocess.run(['./examples/transmission_main_example.py'])
    assert res.returncode == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert 'SNR total (signal bw, dB): 26.27' in captured.out

def test_path_request_run(capfd):
    res = subprocess.run(['./examples/path_requests_run.py'])
    assert res.returncode == 0
    captured = capfd.readouterr()
    assert captured.err == ""

def test_transmission_main_example_raman(capfd):
    res = subprocess.run(['./examples/transmission_main_example.py',
                          'examples/raman_edfa_example_network.json',
                          '--sim', 'examples/sim_params.json',
                          '--show-channels'])
    assert res.returncode == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert 'SNR total (signal bw, dB): 26.48' in captured.out
